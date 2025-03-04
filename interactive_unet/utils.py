import cv2
import glob
import shutil
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from scipy import ndimage

from interactive_unet import metrics

def normalize(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return x

def get_mask(annotations, mask_size):

    oversampled_mask_size = 8 * mask_size
    
    mask = np.zeros((oversampled_mask_size, oversampled_mask_size, 3), dtype='uint8')

    for i in range(len(annotations)):

        path = annotations[i]

        for j in range(len(path)):

            x0, y0, x1, y1, brush_size, color = path[j]
            x0 = int(np.round(x0 * oversampled_mask_size))
            y0 = int(np.round(y0 * oversampled_mask_size))
            x1 = int(np.round(x1 * oversampled_mask_size))
            y1 = int(np.round(y1 * oversampled_mask_size))
            brush_size = brush_size * oversampled_mask_size

            color = color.split('(')[-1].split(')')[0].split(',')
            color = (int(color[2]), int(color[1]), int(color[0]))
            
            cv2.circle(mask, (x0,y0), int(np.round(brush_size/2)), color, -1)
            cv2.line(mask, (x0,y0), (x1,y1), color, int(np.round(brush_size)))

            if j == len(path) - 1:
                cv2.circle(mask, (x1,y1), int(np.round(brush_size/2)), color, -1)

    mask = cv2.resize(mask, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)[...,::-1]


    return mask

def save_sample(image_slice, mask_slice, slice_data):

    _, weight_slice = colored_to_categorical(mask_slice)

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
    io.imsave(f'data/train/images/{n_samples:04d}.tiff', image_slice)
    io.imsave(f'data/train/masks/{n_samples:04d}.tiff', mask_slice)
    io.imsave(f'data/train/weights/{n_samples:04d}.tiff', train_weight_slice)
    np.save(f'data/train/slices/{n_samples:04d}.npy', slice_data)

    # Save validation sample
    n_samples = len(glob.glob("data/val/images/*.tiff"))
    io.imsave(f'data/val/images/{n_samples:04d}.tiff', image_slice)
    io.imsave(f'data/val/masks/{n_samples:04d}.tiff', mask_slice)
    io.imsave(f'data/val/weights/{n_samples:04d}.tiff', val_weight_slice)
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

def colored_to_categorical(colored_mask):

    colors = np.array([[0,0,0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                       [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [170, 255, 195]])
    
    unique_colors = np.unique(colored_mask.reshape(-1, colored_mask.shape[-1]), axis=0)

    idx_fixed = np.nonzero(np.all(colors[:, None] == unique_colors, axis=2))[1]
    unique_colors = unique_colors[idx_fixed]
    
    mask = np.stack([np.all(colored_mask == color, axis=-1).astype(np.uint8) for color in unique_colors], axis=-1) * 255
    
    weight = 255 - mask[:,:,0]
    mask = mask[:,:,1:]

    return mask, weight

def categorical_to_colored(mask):

    colors = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [170, 255, 195]])

    colored_mask = np.zeros((mask.shape[0],mask.shape[1],3), dtype='uint8')
    for i in range(mask.shape[-1]):
        colored_mask[mask[:,:,i] == 255,:] = colors[i]
    
    return colored_mask


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