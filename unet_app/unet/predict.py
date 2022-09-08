import os
import glob
import numpy as np

import torch

from monai.inferers import sliding_window_inference

from unet_app.unet import unet2d, utils

def predict(image_slice, n_channels=1, n_classes=2, return_probabilities=False):

    # Get CUDA device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert image_slice to tensor in form BxCxWxH
    X = np.expand_dims(np.expand_dims(image_slice, axis=0), axis=0)
    X = torch.tensor(X.astype(np.float32)).to(device) 

    # Load model
    if os.path.isfile('model/model.ckpt'):
        model = unet2d.UNet.load_from_checkpoint(checkpoint_path='model/model.ckpt').to(device)
    else:
        model = unet2d.UNet(num_channels=n_channels, num_classes=n_classes).to(device)
    model.eval()

    # Predict slice
    y_prob = model(X).cpu().detach().numpy()

    y_prob = np.moveaxis(y_prob, 1, -1)
    y_pred = np.argmax(y_prob[0,:,:,:n_classes], axis=-1)

    y_pred = utils.make_categorical(y_pred, class_values=np.arange(n_classes))
    y_pred = utils.categorical_to_colored(y_pred, num_classes=n_classes)

    if return_probabilities:
        return y_prob
    else:
        return y_pred

def predict_volumes(input_size=256, n_channels=1, n_classes=2):

    # Get CUDA device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if os.path.isfile('model/model.ckpt'):
        model = unet2d.UNet.load_from_checkpoint(checkpoint_path='model/model.ckpt').to(device)
    else:
        model = unet2d.UNet(num_channels=n_channels, num_classes=n_classes).to(device)
    model.eval()

    # Get list of volume files to predict
    volume_files = np.sort(glob.glob('data/image_volumes/*.npy'))

    # Predict volumes
    for f in volume_files:
        
        volume = np.load(f) / 255
        
        final_prediction = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2], n_classes))

        for i in range(3):

            print(f'Predicting volume {f}, view {i}.', flush=True)

            X = np.moveaxis(volume, i, 0)
            X = torch.tensor(X, dtype=torch.float32).to(device)

            y_pred = np.zeros((X.shape[0], n_classes, X.shape[1], X.shape[2]))

            for j in range(X.shape[0]):

                print(j, flush=True)

                input_array = X[j][None,None,:,:]
                output_array = sliding_window_inference(predictor=model,
                                                       inputs=input_array,
                                                       roi_size=input_size,
                                                       overlap=0.25,
                                                       mode='gaussian',
                                                       padding_mode='reflect',
                                                       sw_batch_size=4).cpu().detach().numpy()[0]
                y_pred[j] = output_array

            y_pred = np.moveaxis(y_pred, 1, -1)
            final_prediction += np.moveaxis(y_pred, 0, i)

        final_prediction = ((final_prediction / 3) * 255).astype('uint8')

        np.save(f.replace('image_volumes', 'predicted_volumes'), final_prediction[:,:,:,1])

