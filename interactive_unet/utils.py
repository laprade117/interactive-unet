import cv2
import glob
import time
import shutil
import urllib
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
from scipy import ndimage
from skimage.io import imsave, imread

from interactive_unet import metrics, volumedata

def download_example_data():
    url = 'https://filestash.qim.dk/api/files/cat?path=%2Fsample_data.npy&share=evZRE58'
    urllib.request.urlretrieve(url, 'data/image_volumes/sample_volume.npy')

def load_dataset(annotations=False):

    image_volume_files = np.sort(glob.glob('data/image_volumes/*.npy'))

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
        print('No volumetric data found. Downloading sample volume...')
        start_time = time.time()
        download_example_data()
        end_time = time.time()
        print(f'Download completed in {end_time - start_time:.02f} seconds. \n')

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

def get_unique_colors(colored_mask):
    '''
    Gets list of unique colors in correct order corresponding to the class colors.
    '''
    
    colors = np.array([[0,0,0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                       [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [170, 255, 195]])
    
    unique_colors = np.unique(colored_mask.reshape(-1, colored_mask.shape[-1]), axis=0)
    idx_fixed = np.nonzero(np.all(colors[:, None] == unique_colors, axis=2))[1]
    unique_colors = unique_colors[idx_fixed]
    
    return unique_colors

def colored_to_categorical(colored_mask, include_background=True):

    unique_colors = get_unique_colors(colored_mask)
    
    mask = np.stack([np.all(colored_mask == color, axis=-1).astype(np.uint8) for color in unique_colors], axis=-1) * 255

    # Check if contains unlabeled pixels
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