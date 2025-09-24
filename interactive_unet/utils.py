import cv2
import glob
import time
import zarr
import shutil
import urllib
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
from scipy import ndimage
from numba import njit, prange
from skimage.io import imsave, imread

from interactive_unet import metrics, volumedata

def read_volume(path, level=0):

    # Load root node 
    root = zarr.open(path, mode='r')

    # Ensure the requested multiscale level exists
    num_scales = len(np.sort(list(root.array_keys())))
    level = int(np.clip(level, 0, num_scales))

    return root[str(level)]

def resize_volume(src_vol, dst_vol, scale=0.5, block_size=512, order=0):
    
    src_shape = np.array(src_vol.shape).astype(int)
    
    for i in range(0, src_shape[0], block_size):
            
        i0, i1 = i, min(i + block_size, src_shape[0])
        t_i0, t_i1 = int(i0 * scale), int(i1 * scale)
        
        for j in range(0, src_shape[1], block_size):
            
            j0, j1 = j, min(j + block_size, src_shape[1])
            t_j0, t_j1 = int(j0 * scale), int(j1 * scale)
            
            for k in range(0, src_shape[2], block_size):
                
                k0, k1 = k, min(k + block_size, src_shape[2])
                t_k0, t_k1 = int(k0 * scale), int(k1 * scale)

                dst_vol[t_i0:t_i1, t_j0:t_j1, t_k0:t_k1] = ndimage.zoom(src_vol[i0:i1, j0:j1, k0:k1], scale, order=order)

def add_multiscales(src_file, scale=0.5):

    # Load root node 
    root = zarr.open(src_file, mode='r+')

    volume_shape = root['0'].shape
    chunk_shape = root['0'].chunks
    shard_shape = root['0'].shards
    
    # Number of downscale steps until the final size fits inside a chunk
    num_steps = int(np.floor(np.log((np.array(volume_shape) / np.array(chunk_shape)).max()) / np.log(1 / scale)))

    # Create multiscale volume
    for i in range(num_steps):
        
        z0 = root[str(i)]
        
        z1_shape = tuple(int(x * scale) for x in z0.shape)
        z1 = root.create_array(name=str(i+1),
                               shape=z1_shape,
                               chunks=chunk_shape,
                               shards=shard_shape,
                               dtype=z0.dtype,
                               overwrite=True)
        resize_volume(z0, z1, scale=scale, block_size=shard_shape[0], order=0)
        
    # Clear some memory
    del root, z0, z1

def create_multiscale_zarr(volume, dst_file, scale=0.5, chunk_size=128, shard_size=256):

    # Create chunk and shard shapes
    chunk_shape = (chunk_size, chunk_size, chunk_size)
    shard_shape = (shard_size, shard_size, shard_size)

    # Copy original resolution to first level of the multiscale volume
    root = zarr.open(dst_file, mode='w')
    z0 = root.create_array(name='0',
                        shape=volume.shape,
                        chunks=chunk_shape,
                        shards=shard_shape,
                        dtype=volume.dtype,
                        overwrite=True)
    z0[:] = volume

    # Clear some memory
    del root, z0, volume

    add_multiscales(dst_file, scale=scale)

def download_example_data():

    print('No volumetric data found. Downloading sample volume...')
    start_time = time.time()

    Path("temp").mkdir(parents=True, exist_ok=True)

    url = 'https://filestash.qim.dk/api/files/cat?path=%2Fsample_data.npy&share=57lVz63'
    urllib.request.urlretrieve(url, 'temp/sample_volume.npy')

    end_time = time.time()
    print(f'Download completed in {end_time - start_time:.02f} seconds. \n')

    print('Creating multiscale zarr...')
    volume = np.load('temp/sample_volume.npy')
    
    create_multiscale_zarr(volume, 'data/image_volumes/sample_volume.zarr')

    shutil.rmtree('temp')   
    print('Done!')


# def load_dataset(annotations=False):

#     image_volume_files = np.sort(glob.glob('data/image_volumes/*.npy'))

#     dataset = []
#     if len(image_volume_files) > 0:
#         dataset = [volumedata.VolumeData(f, annotations=annotations) for f in image_volume_files]

#     return dataset

def load_dataset(annotations=False):

    image_volume_files = np.sort(glob.glob('data/image_volumes/*.zarr'))

    dataset = []
    if len(image_volume_files) > 0:
        dataset = [volumedata.VolumeData(f, annotations=annotations) for f in image_volume_files]

    return dataset

def build_annotation_volumes(dataset):
    for i in range(len(dataset)):
        print(f'{i}/{len(dataset)} - Rebuilding annotation volumes for {dataset[i].filename}')
        dataset[i].build_annotation_volumes()
    print(f'Rebuilding complete.')

def get_input_size():

    input_size = 512

    train_masks = glob.glob('data/train/masks/*.tiff')  

    if len(train_masks) > 0:
        mask = imread(train_masks[0])
        input_size = mask.shape[0]
    
    return input_size

def get_num_classes():

    num_classes = 2

    train_masks = glob.glob('data/train/masks/*.tiff')

    if len(train_masks) > 0:
        mask = imread(train_masks[0])
        num_classes = np.unique(mask.reshape(-1, mask.shape[-1]), axis=0).shape[0] - 1

    return num_classes

def normalize(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return x

def save_sample(image_slice, mask_slice, slice_data, num_classes=None):
    
    # Set the corner pixels to ensure the mask always has at least one pixel for each class to prevent divide by zero errors.
    # I should find a better method, this is just stupid.
    if num_classes is not None:
        colors = np.array([[0,0,0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                           [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [170, 255, 195]])
        for i in range(num_classes + 1):
            mask_slice[0,i,:] = colors[i]

    _, weight_slice = colored_to_categorical(mask_slice)
    weight_slice[0,:num_classes+1] = 0

    # Generate noise to define train-val split
    noise = normalize(ndimage.gaussian_filter(np.random.rand(image_slice.shape[0],image_slice.shape[1]), 2)) > 0.5

    train_weight_slice = weight_slice * noise
    val_weight_slice = weight_slice * (1 - noise)

    image_slice = np.round(image_slice).astype('uint8')
    mask_slice = np.round(mask_slice).astype('uint8')
    train_weight_slice = np.round(train_weight_slice).astype('uint8')
    val_weight_slice = np.round(val_weight_slice).astype('uint8')

    # Save training sample
    n_samples = len(glob.glob("data/train/images/*.tiff"))
    imsave(f'data/train/images/{n_samples:04d}.tiff', image_slice)
    imsave(f'data/train/masks/{n_samples:04d}.tiff', mask_slice)
    imsave(f'data/train/weights/{n_samples:04d}.tiff', train_weight_slice)
    np.save(f'data/train/slices/{n_samples:04d}.npy', slice_data)

    # Save validation sample
    n_samples = len(glob.glob("data/val/images/*.tiff"))
    imsave(f'data/val/images/{n_samples:04d}.tiff', image_slice)
    imsave(f'data/val/masks/{n_samples:04d}.tiff', mask_slice)
    imsave(f'data/val/weights/{n_samples:04d}.tiff', val_weight_slice)
    np.save(f'data/val/slices/{n_samples:04d}.npy', slice_data)

# Folder and data functions -------------------------------------------------------------------------------------

def create_directories():

    Path("data/image_volumes").mkdir(parents=True, exist_ok=True)
    Path("data/mask_volumes").mkdir(parents=True, exist_ok=True)
    Path("data/weight_volumes").mkdir(parents=True, exist_ok=True)
    Path("data/predicted_volumes").mkdir(parents=True, exist_ok=True)

    Path("data/train/images").mkdir(parents=True, exist_ok=True)
    Path("data/train/masks").mkdir(parents=True, exist_ok=True)
    Path("data/train/weights").mkdir(parents=True, exist_ok=True)
    Path("data/train/slices").mkdir(parents=True, exist_ok=True)

    Path("data/val/images").mkdir(parents=True, exist_ok=True)
    Path("data/val/masks").mkdir(parents=True, exist_ok=True)
    Path("data/val/weights").mkdir(parents=True, exist_ok=True)
    Path("data/val/slices").mkdir(parents=True, exist_ok=True)

    Path("model").mkdir(parents=True, exist_ok=True)

    # Download sample data if no volumes exist
    if len(glob.glob('data/image_volumes/*')) == 0:
        download_example_data()

def clear_annotations(): 

    shutil.rmtree('./data/mask_volumes')     
    shutil.rmtree('./data/weight_volumes')
    shutil.rmtree('./data/predicted_volumes')
    shutil.rmtree('./data/train')
    shutil.rmtree('./data/val')
    create_directories()

def clear_model(): 

    shutil.rmtree('./model')   
    create_directories()

def reset_all():

    shutil.rmtree('./data/mask_volumes')     
    shutil.rmtree('./data/weight_volumes')
    shutil.rmtree('./data/predicted_volumes')
    shutil.rmtree('./data/train')
    shutil.rmtree('./data/val')
    shutil.rmtree('./model')  
    create_directories()


# Data representation functions -------------------------------------------------------------------------------------
    
# def get_unique_colors(colored_mask):
#     '''
#     Gets list of unique colors in correct order corresponding to the class colors.
#     '''
    
#     colors = np.array([[0,0,0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
#                        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [170, 255, 195]])
    
#     unique_colors = np.unique(colored_mask.reshape(-1, colored_mask.shape[-1]), axis=0)
#     idx_fixed = np.nonzero(np.all(colors[:, None] == unique_colors, axis=2))[1]
#     unique_colors = unique_colors[idx_fixed]
    
#     return unique_colors


# def colored_to_categorical(colored_mask, include_background=True):
#     h, w, c = colored_mask.shape
#     unique_colors = get_unique_colors(colored_mask)  # (num_colors, 3)
#     num_colors = len(unique_colors)

#     # Convert colors to single integers for fast comparison
#     factor = np.array([256*256, 256, 1], dtype=np.int32)
#     flat_mask = colored_mask.reshape(-1, 3).dot(factor)  # shape (H*W,)
#     flat_colors = unique_colors.dot(factor)              # shape (num_colors,)

#     # Compare all pixels to all colors efficiently
#     mask = (flat_mask[:, None] == flat_colors[None, :]).astype(np.uint8) * 255  # (H*W, num_colors)
#     mask = mask.reshape(h, w, num_colors)

#     # Compute weight for background/unlabeled pixels
#     weight = 255 - mask[:, :, 0]

#     # Remove background channel
#     mask = mask[:, :, 1:]

#     return mask, weight

COLORS = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25],
                   [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
                   [240, 50, 230], [210, 245, 60], [170, 255, 195]], dtype=np.uint8)

def get_unique_colors(colored_mask):
    '''
    Gets list of unique colors in correct order corresponding to the class colors.
    '''

    # Flatten mask into Nx3
    flat = colored_mask.reshape(-1, 3)

    # Convert to a 1D integer key (avoid np.unique(axis=0))
    flat_keys = flat[:, 0].astype(np.uint32) << 16 | flat[:, 1].astype(np.uint32) << 8 | flat[:, 2]
    color_keys = COLORS[:, 0].astype(np.uint32) << 16 | COLORS[:, 1].astype(np.uint32) << 8 | COLORS[:, 2]

    # Mark which predefined colors exist in the mask
    present = np.isin(color_keys, flat_keys)

    return COLORS[present]

@njit(parallel=True)
def _colored_to_categorical_ultrafast(flat_mask, flat_colors, h, w, num_colors):
    mask = np.zeros((h, w, num_colors), dtype=np.uint8)
    for i in prange(h):
        for j in range(w):
            pixel = flat_mask[i*w + j]
            for k in range(num_colors):
                if pixel == flat_colors[k]:
                    mask[i, j, k] = 255
                    break  # stop at first match
    weight = 255 - mask[:, :, 0]
    return mask[:, :, 1:], weight

def colored_to_categorical(colored_mask, include_background=True):
    h, w, c = colored_mask.shape
    unique_colors = get_unique_colors(colored_mask)
    num_colors = len(unique_colors)

    # Convert colors to single integers
    factor = np.array([256*256, 256, 1], dtype=np.int32)
    flat_mask = colored_mask.reshape(-1, 3).dot(factor).astype(np.int32)
    flat_colors = unique_colors.dot(factor).astype(np.int32)

    # Call fully compiled Numba function with parallelism
    return _colored_to_categorical_ultrafast(flat_mask, flat_colors, h, w, num_colors)

def categorical_to_colored(mask):

    colored_mask = np.zeros((mask.shape[0],mask.shape[1],3), dtype='uint8')
    for i in range(mask.shape[-1]):
        colored_mask[mask[:,:,i] == 255,:] = COLORS[i+1]
    
    return colored_mask

def colored_to_class(colored_mask):

    categorical_mask, _ = colored_to_categorical(colored_mask)
    categorical_mask = (categorical_mask > 0).astype('uint8')
    mask = np.zeros((categorical_mask.shape[0], categorical_mask.shape[1]), dtype='uint8')

    for i in range(categorical_mask.shape[-1]):
        mask[categorical_mask[...,i] > 0] = i

    return mask

def class_to_categorical(class_mask, num_classes, weight=None):

    if weight is None:
        weight = np.ones(class_mask.shape)
    
    categorical_mask = np.zeros((class_mask.shape[0], class_mask.shape[1], num_classes), dtype='uint8')
    
    for i in range(num_classes):
        categorical_mask[:,:,i] = (class_mask == i) * weight
        
    return categorical_mask

# Plotly training history functions -------------------------------------------------------------------------------------

def get_training_history(metric='Loss'):
    
    log_files = np.sort(glob.glob('model/history/*/version_0/metrics.csv'))
    
    epochs = []
    train = []
    val = []
    
    for log_file in log_files:
        
        df = pd.read_csv(log_file)
        
        epochs_i = df['epoch'].unique()
        train_i = df.groupby('epoch')[f'train/{metric}'].mean().values
        val_i = df.dropna(subset=[f'val/{metric}'])[f'val/{metric}'].values
        
        epochs_i += len(epochs)
        
        epochs += list(epochs_i)
        train += list(train_i)
        val += list(val_i)
    
    epochs = np.array(epochs)
    train = np.array(train)
    val = np.array(val)
    
    return epochs, train, val

def get_training_history_figure(metric):
    
    epochs, train, val = get_training_history(metric)

    train_curve = {
        'x': epochs,
        'y': train,
        'mode': 'lines+markers',
        'type': 'scatter',
        'name': 'Train'
    }
    val_curve = {
        'x': epochs,
        'y': val,
        'mode': 'lines+markers',
        'type': 'scatter',
        'name': 'Validation'
    }

    data = [train_curve, val_curve]

    layout = {
        'legend': {'x': 0.7, 'y': 0.5},
        'margin': {'l': 40, 'r': 0, 't': 30, 'b': 40},
        'xaxis': {
            'title': {
                'text': 'Epoch'
            }
        },
        'yaxis': {
            'title': {
                'text': f'{metric}'
            }
        },
    }
    
    fig = {
        'data': data,
        'layout': layout
    }

    return fig


# Miscellaneous helper functions -------------------------------------------------------------------------------------

def loss_name_to_function(loss_function_name):

    if loss_function_name == 'Crossentropy (CE)':
        loss_function = metrics.crossentropy_loss
    elif loss_function_name == 'Dice':
        loss_function = metrics.dice_loss
    elif loss_function_name == 'Intersection over Union (IoU)':
        loss_function = metrics.iou_loss
    elif loss_function_name == 'Matthews correlation coefficient (MCC)':
        loss_function = metrics.mcc_loss
    elif loss_function_name == 'Dice + CE':
        loss_function = metrics.dice_ce_loss
    elif loss_function_name == 'IoU + CE':
        loss_function = metrics.iou_ce_loss
    elif loss_function_name == 'MCC + CE':
        loss_function = metrics.mcc_ce_loss

    return loss_function
