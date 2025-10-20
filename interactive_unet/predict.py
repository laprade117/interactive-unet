import os
import glob
import zarr
import time
import copy
import shutil
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
import torch.multiprocessing as mp

from . import utils, unet

def predict_slice(image_slice, num_channels=1, num_classes=2, return_probabilities=False):

    torch.set_float32_matmul_precision('medium')
    
    # Get CUDA device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if os.path.isfile('model/model.ckpt'):
        model = unet.UNet.load_from_checkpoint(checkpoint_path='model/model.ckpt').to(device)
    else:
        model = unet.UNet(num_channels=num_channels, num_classes=num_classes).to(device)
    model.eval()

    # Convert image_slice to tensor in form BxCxWxH
    X = (image_slice[None,None,:,:] / 255).astype('float32')
    X = torch.tensor(X).to(device) 

    # Predict slice
    with torch.inference_mode():
        y_prob = model(X).cpu().detach().numpy()

    y_prob = np.moveaxis(y_prob, 1, -1)
    y_pred = np.argmax(y_prob[0,:,:,:num_classes], axis=-1)
    y_pred = np.stack([y_pred == i for i in range(num_classes)], -1)
    y_pred = (y_pred * 255).astype('uint8')

    y_pred = utils.categorical_to_colored(y_pred)

    if return_probabilities:
        return y_prob
    else:
        return y_pred
        
def find_max_batch_size(model, input_size=256, start=4, max_limit=512):
    
    batch_size = start
    best = start

    device = model.device
    
    while batch_size <= max_limit:
        try:
            with torch.inference_mode():
                # Make a fake batch to test memory use
                test_batch = torch.zeros((batch_size, 1, input_size, input_size), dtype=torch.float32, device=device)
                _ = model(test_batch)
            
            best = batch_size
            batch_size *= 2  # Try next larger
            torch.cuda.empty_cache()
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                break  # Too big, stop searching
            else:
                raise e  # Unexpected error

    del test_batch, model
    torch.cuda.empty_cache()

    return best

def predict_block(model, block, num_classes=2, batch_size=8, axes=[0,1,2]):

    input_size = block.shape[0]
    
    device = model.device
    
    block_prediction = np.zeros((input_size, input_size, input_size, num_classes), dtype=np.float32)
        
    for axis in axes:

        with torch.inference_mode():

            block = torch.moveaxis(block, axis, 0)
            
            for i in range(0, input_size, batch_size):
                
                batch = block[i:i+batch_size].unsqueeze(1)
                
                batch_prediction = model(batch.to(device))
                batch_prediction = batch_prediction.permute(0, 2, 3, 1).cpu().numpy()

                # Accumulate predictions into correct orientation depending on axis
                if axis == 0:   # Z axis
                    block_prediction[i:i+batch_size, :, :, :] += batch_prediction
                elif axis == 1: # Y axis
                    block_prediction[:, i:i+batch_size, :, :] += batch_prediction.transpose(1, 0, 2, 3)
                elif axis == 2: # X axis
                    block_prediction[:, :, i:i+batch_size, :] += batch_prediction.transpose(1, 2, 0, 3)

            block = torch.moveaxis(block, 0, axis)
            
    block_prediction /= len(axes)

    return block_prediction

def predict_volumes(input_size=256, num_channels=1, num_classes=2, overlap=0.25, chunk_size=128, shard_size=256, batch_size=8, axes=[0,1,2]):

    import signal, os, sys

    def handle_sigint(sig, frame):
        print("\nCaught Ctrl+C → force exit")
        os._exit(1)  # bypass Python, kill immediately

    signal.signal(signal.SIGINT, handle_sigint)

    torch.set_float32_matmul_precision('medium')
    
    # Get CUDA device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    if os.path.isfile('model/model.ckpt'):
        model = unet.UNet.load_from_checkpoint(checkpoint_path='model/model.ckpt')
    else:
        model = unet.UNet(num_channels=num_channels, num_classes=num_classes)
    model.eval()

    # # Get number of GPU's available
    # num_gpus = torch.cuda.device_count()

    # # Assign a copy of the model to each GPU
    # models = []
    # for gpu_id in range(num_gpus):
    #     m = copy.deepcopy(model)         
    #     m = m.to(f'cuda:{gpu_id}')
    #     m.eval()
    #     models.append(m)
    # del model
    
    # Get list of volume files to predict
    volume_files = np.sort(glob.glob('data/image_volumes/*.zarr'))
    
    # Precompute blending window for block size
    window = gaussian_3d(input_size, sigma=0.125)
    # window = hanning_3d(input_size)

    batch_size = find_max_batch_size(model, input_size=input_size, start=4, max_limit=input_size)
    print(f'Found optimal inference batch size of {batch_size}.')

    # Predict volumes
    for f in volume_files:

        start_time = time.time()
        
        volume = zarr.open(f, mode='r')['0'] # Load highest resolution
        input_volume_shape = np.array(volume.shape)
        output_volume_shape = np.append(input_volume_shape, num_classes)

        # Initial prediction volume
        save_path = f.replace('image_volumes', 'predicted_volumes')
        root = zarr.open(save_path, mode='w')
        final_predictions = root.create_array(name='0',
                                              shape=output_volume_shape.astype(int).tolist(),
                                              chunks=(chunk_size, chunk_size, chunk_size, num_classes),
                                              shards=(shard_size, shard_size, shard_size, num_classes),
                                              dtype='uint8',
                                              overwrite=True)
        
        # Initialize temporary prediction volume
        pred_root = zarr.open('temp/pred.zarr', mode='w')
        pred = pred_root.create_array(name='0',
                                      shape=output_volume_shape.astype(int).tolist(),
                                      chunks=(chunk_size, chunk_size, chunk_size, num_classes),
                                      shards=(shard_size, shard_size, shard_size, num_classes),
                                      dtype='float32',
                                      overwrite=True)

        
        # Initialize temporary weight volume
        weight_root = zarr.open('temp/weight.zarr', mode='w')
        weight = weight_root.create_array(name='0',
                                          shape=input_volume_shape.astype(int).tolist(),
                                          chunks=(chunk_size, chunk_size, chunk_size),
                                          shards=(shard_size, shard_size, shard_size),
                                          dtype='float32',
                                          overwrite=True)
        # Get block coordinates
        block_coords, padded_block_coords, local_block_coords = get_block_coordinates(input_volume_shape, input_size=input_size, overlap=overlap)
        num_blocks = len(padded_block_coords)

        # print(f'\nSegmenting {f.split('/')[-1]}...')
        # pbar = tqdm(total=num_blocks)
        # for i in range(0, num_blocks, num_gpus):

        #     # Build tasks for a parallelization
        #     tasks = []
        #     for j in range(num_gpus):
        #         if i + j < num_blocks:
        #             idx = i + j
        #             gpu_id = j
        #             padded_block = torch.tensor(get_padded_block(volume, *padded_block_coords[idx]).astype('float32') / 255.0)
        #             tasks.append((models[gpu_id], padded_block, num_classes, batch_size, axes, gpu_id))

        #     # Begin tasks, 1 block per GPU
        #     with mp.Pool(processes=len(tasks)) as pool:
        #         predicted_blocks = pool.starmap(predict_block, tasks)

        #     # Write predictions into Zarr files
        #     for j in range(len(predicted_blocks)):
        #         if i + j < num_blocks:
        #             idx = i + j

        #             i0, j0, k0, i1, j1, k1 = block_coords[idx]
        #             l_i0, l_j0, l_k0, l_i1, l_j1, l_k1 = local_block_coords[idx]
                    
        #             pred[i0:i1, j0:j1, k0:k1] += predicted_blocks[j][l_i0:l_i1, l_j0:l_j1, l_k0:l_k1, :] * window[l_i0:l_i1, l_j0:l_j1, l_k0:l_k1, None]
        #             weight[i0:i1, j0:j1, k0:k1] += window[l_i0:l_i1, l_j0:l_j1, l_k0:l_k1]

        #     pbar.update(len(predicted_blocks))

        print(f'\nSegmenting {f.split('/')[-1]}...')
        for i in tqdm(range(num_blocks)):

            padded_block = torch.tensor(get_padded_block(volume, *padded_block_coords[i]).astype('float32') / 255.0)
        
            predicted_block = predict_block(model, padded_block, num_classes=num_classes, batch_size=batch_size, axes=axes)

            i0, j0, k0, i1, j1, k1 = block_coords[i]
            l_i0, l_j0, l_k0, l_i1, l_j1, l_k1 = local_block_coords[i]
            
            pred[i0:i1, j0:j1, k0:k1] += predicted_block[l_i0:l_i1, l_j0:l_j1, l_k0:l_k1, :] * window[l_i0:l_i1, l_j0:l_j1, l_k0:l_k1, None]
            weight[i0:i1, j0:j1, k0:k1] += window[l_i0:l_i1, l_j0:l_j1, l_k0:l_k1]

        del volume
        
        print('Postprocessing and generating multiscale pyramid...')

        # Threaded normalization by shard rather than chunk to prevent two threads from writing to same file.
        shard_coordinates = get_shard_coordinates(input_volume_shape, shard_size=shard_size)
        def normalize_shard(final_predictions, pred, weight, coords, eps=1e-3):
            i0, j0, k0, i1, j1, k1 = coords
            final_predictions[i0:i1, j0:j1, k0:k1] = (255 * pred[i0:i1, j0:j1, k0:k1] / np.maximum(weight[i0:i1, j0:j1, k0:k1], eps)[...,None]).astype('uint8')
        Parallel(n_jobs=-1, prefer='threads')(delayed(normalize_shard)(final_predictions, pred, weight, coords) for coords in shard_coordinates)
        
        del pred, weight, final_predictions
        shutil.rmtree('temp')

        utils.add_multiscales(save_path, scale=0.5)

        time_elapsed = time.time() - start_time
        print(f'Completed volume {f.split('/')[-1]} {tuple(input_volume_shape.astype(int).tolist())} in {time_elapsed}.')

    print(f'\nAll volumes segmented.\n')

#---------------Helper functions --------------------------------------------------------------

def reflect_index(idx, size):
    '''
    Reflect indices into valid [0, size-1] range
    '''
    if size == 1:
        return np.zeros_like(idx)

    period = 2 * size - 2
    idx = np.abs(idx) % period
    return np.where(idx < size, idx, period - idx)

def get_padded_block(volume, i0, j0, k0, i1, j1, k1):
    '''
    Gets a block with edges outside the range of the volume via reflect padding along edges.
    '''
    ii = reflect_index(np.arange(i0, i1), volume.shape[0])
    jj = reflect_index(np.arange(j0, j1), volume.shape[1])
    kk = reflect_index(np.arange(k0, k1), volume.shape[2])

    return volume[np.ix_(ii, jj, kk)]

def get_padded_block(volume, i0, j0, k0, i1, j1, k1):
    '''
    Extracts a block from a volume with reflection padding at the boundaries.
    '''

    volume_shape = volume.shape

    pad_before = [max(0, -i0), max(0, -j0), max(0, -k0)]
    pad_after  = [max(0, i1 - volume_shape[0]), max(0, j1 - volume_shape[1]), max(0, k1 - volume_shape[2])]

    # Clip indices into valid range
    c_i0, c_i1 = max(i0, 0), min(i1, volume_shape[0])
    c_j0, c_j1 = max(j0, 0), min(j1, volume_shape[1])
    c_k0, c_k1 = max(k0, 0), min(k1, volume_shape[2])

    # Load only the needed block from the zarr volume
    block = volume[c_i0:c_i1, c_j0:c_j1, c_k0:c_k1]

    # Pad to desired shape with reflection
    padding = ((pad_before[0], pad_after[0]),
               (pad_before[1], pad_after[1]),
               (pad_before[2], pad_after[2]))

    padded = np.pad(block, pad_width=padding, mode='reflect')

    return padded

def get_shard_coordinates(volume_shape, shard_size=128):
    '''
    Returns coordinates of all shards in the volume.
    '''
    starts = [np.arange(0, s, shard_size) for s in volume_shape]
    chunk_coordinates = np.stack(np.meshgrid(*starts, indexing='ij'), -1).reshape(-1, 3)
    chunk_coordinates = np.concatenate([chunk_coordinates,np.minimum(chunk_coordinates + shard_size, volume_shape)], axis=1)
    return chunk_coordinates
    
def gaussian_3d(input_size, sigma=0.125, eps=1e-3):
    """
    Create a 3D Gaussian window for edge weighting.
    """

    # Adjust sigma based on input size
    sigma *= input_size
    
    # 1D Gaussian
    coords = np.arange(input_size, dtype=np.float32) - (input_size - 1) / 2.0
    g = np.exp(-(coords**2) / (2 * sigma**2)).astype(np.float32)
    g /= g.max()

    # 3D gaussian
    gaussian = g[:, None, None] * g[None, :, None] * g[None, None, :]

    # Normalize and clip
    gaussian /= gaussian.max()
    gaussian = np.clip(gaussian, max(gaussian.min(), eps), 1.0)
    
    return gaussian

def hanning_3d(input_size, eps=1e-3):
    """
    Create a 3D Hanning window for edge weighting.
    """
    
    h = np.hanning(input_size)
    hanning = h[:, None, None] * h[None, :, None] * h[None, None, :]
    
    hanning /= hanning.max()
    hanning = np.clip(hanning, max(hanning.min(), eps), 1.0)

    return hanning.astype('float32')

def get_block_coordinates(volume_shape, input_size=256, overlap=0.25):
    
    blocks_per_axis = np.ceil((volume_shape - overlap * input_size) / (input_size - overlap * input_size)).astype(int)
    padded_volume_shape = np.round(blocks_per_axis * input_size - (blocks_per_axis - 1) * input_size * overlap).astype(int)

    padding_shift = (padded_volume_shape - volume_shape) // 2
    padding_shift = np.array(list(padding_shift) + list(padding_shift))

    block_coords = []
    padded_block_coords = []
    local_block_coords = []
    
    for i in range(blocks_per_axis[0]):
        
        p_i0 = i * input_size * (1 - overlap)
        p_i1 = p_i0 + input_size
        
        for j in range(blocks_per_axis[1]):
            
            p_j0 = j * input_size * (1 - overlap)
            p_j1 = p_j0 + input_size
            
            for k in range(blocks_per_axis[2]):
                
                p_k0 = k * input_size * (1 - overlap)
                p_k1 = p_k0 + input_size

                # padded block coords (outside of volume range)
                coords = np.array([p_i0, p_j0, p_k0, p_i1, p_j1, p_k1]) - padding_shift
                coords = coords.astype(int)
                padded_block_coords.append(coords)

                # block coords (clipped to volume)
                i0, j0, k0, i1, j1, k1 = coords
                i0_c, i1_c = max(0, i0), min(volume_shape[0], i1)
                j0_c, j1_c = max(0, j0), min(volume_shape[1], j1)
                k0_c, k1_c = max(0, k0), min(volume_shape[2], k1)
                block_coords.append([i0_c, j0_c, k0_c, i1_c, j1_c, k1_c])

                # local indices within block
                l_i0, l_i1 = i0_c - i0, i1_c - i0
                l_j0, l_j1 = j0_c - j0, j1_c - j0
                l_k0, l_k1 = k0_c - k0, k1_c - k0
                local_block_coords.append([l_i0, l_j0, l_k0, l_i1, l_j1, l_k1])

    padded_block_coords = np.array(padded_block_coords)
    block_coords = np.array(block_coords)
    local_block_coords = np.array(local_block_coords)

    return block_coords, padded_block_coords, local_block_coords