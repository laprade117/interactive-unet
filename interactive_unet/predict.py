import os
import glob
import numpy as np

import torch

from monai.inferers import sliding_window_inference
from monai.inferers import SliceInferer

from interactive_unet import utils, unet

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

# def predict_volumes(input_size=256, num_channels=1, num_classes=2):

#     torch.set_float32_matmul_precision('medium')

#     # Get CUDA device if available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Load model
#     if os.path.isfile('model/model.ckpt'):
#         model = unet.UNet.load_from_checkpoint(checkpoint_path='model/model.ckpt').to(device)
#     else:
#         model = unet.UNet(num_channels=num_channels, num_classes=num_classes).to(device)
#     model.eval()

#     # Get list of volume files to predict
#     volume_files = np.sort(glob.glob('data/image_volumes/*.npy'))

#     # Predict volumes
#     for f in volume_files:
        
#         volume = np.load(f) / 255
        
#         final_prediction = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2], num_classes))

#         for i in range(3):

#             print(f'Predicting volume {f}, view {i}.', flush=True)

#             X = np.moveaxis(volume, i, 0)
#             X = torch.tensor(X, dtype=torch.float32).to(device)

#             y_pred = np.zeros((X.shape[0], num_classes, X.shape[1], X.shape[2]))

#             for j in range(X.shape[0]):

#                 print(j, flush=True)

#                 input_array = X[j][None,None,:,:]
#                 output_array = sliding_window_inference(predictor=model,
#                                                        inputs=input_array,
#                                                        roi_size=input_size,
#                                                        overlap=0.25,
#                                                        mode='gaussian',
#                                                        padding_mode='reflect',
#                                                        sw_batch_size=4).cpu().detach().numpy()[0]
#                 y_pred[j] = output_array

#             y_pred = np.moveaxis(y_pred, 1, -1)
#             final_prediction += np.moveaxis(y_pred, 0, i)

#         final_prediction = ((final_prediction / 3) * 255).astype('uint8')
        
#         # Save binary mask for each class
#         for i in range(num_classes):
#             save_path = f.replace('image_volumes', 'predicted_volumes')[:-4]
#             np.save(f'{save_path}_{i}.npy', final_prediction[...,i])

def predict_volumes(input_size=256, num_channels=1, num_classes=2):

    torch.set_float32_matmul_precision('medium')

    # Get CUDA device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if os.path.isfile('model/model.ckpt'):
        model = unet.UNet.load_from_checkpoint(checkpoint_path='model/model.ckpt').to(device)
    else:
        model = unet.UNet(num_channels=num_channels, num_classes=num_classes).to(device)
    model.eval()

    # Get list of volume files to predict
    volume_files = np.sort(glob.glob('data/image_volumes/*.npy'))

    with torch.no_grad():

        # Predict volumes
        for f in volume_files:
            
            volume = torch.tensor(np.load(f)[None,None,...] / 255.0, dtype=torch.float32)
            
            final_prediction = np.zeros((num_classes, volume.shape[-3], volume.shape[-2], volume.shape[-1]))

            for i in range(3):
            
                print(f'Predicting volume {f}, view {i}.', flush=True)

                inferer = SliceInferer(spatial_dim=i,
                                       roi_size=(input_size, input_size),
                                       sw_batch_size=8,
                                       overlap=0.5,
                                       mode='gaussian',
                                       padding_mode='reflect',
                                       sw_device=device,
                                       device=torch.device('cpu'))

                prediction = inferer(volume, model)[0].detach().cpu().numpy()

                final_prediction += prediction

            final_prediction = ((final_prediction / 3) * 255).astype('uint8')
            
            print(f'Saving predicted volume {f}...')

            # Save binary mask for each class
            for i in range(num_classes):
                save_path = f.replace('image_volumes', 'predicted_volumes')[:-4]
                np.save(f'{save_path}_{i}.npy', final_prediction[i])