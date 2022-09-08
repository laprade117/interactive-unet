import os
import glob
import shutil
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation as R

from unet_app.unet import predict


# Volume and sampling functions -------------------------------------------------------------------------------------

def pad_to_input_shape(ndarray, input_shape=512):
    """
    Pads input ndarray so that each dimension has a minimum length of input_size.
    """
    
    ndarray_shape = np.array(ndarray.shape)

    lower_pad = np.floor((input_shape - ndarray_shape) / 2).clip(min=0).astype(int)
    upper_pad = np.ceil((input_shape - ndarray_shape) / 2).clip(min=0).astype(int)
    padding = np.stack([lower_pad, upper_pad]).T

    return np.pad(ndarray, padding)

def crop_to_input_shape(ndarray, input_shape=512):
    """
    Crops input ndarray so that each dimension has a maximum length of input_size.
    """

    ndarray_shape = np.array(ndarray.shape)

    start = np.floor((ndarray_shape - input_shape) / 2).clip(min=0).astype(int)
    end = start + input_shape
    slices = tuple([slice(start[i], end[i]) for i in range(len(ndarray_shape))])

    return ndarray[slices]

def generate_uniformly_random_unit_vector(ndim=3):
    """
    Generates a uniformly random unit vector. Uses one of the methods
    outlined in http://corysimon.github.io/articles/uniformdistn-on-sphere/.
    """

    # Initial vector
    u = np.random.normal(size=ndim)
    
    # Regenerate to avoid rounding issues
    while np.linalg.norm(u) < 0.0001:
        u = np.random.normal(size=ndim)
        
    # Make unit vector
    u = u / np.linalg.norm(u)
    
    return u

def compute_rotation_matrix_from_vectors(src, dst):
    """
    Calculates the rotation matrix that rotates the source vector to the destination vector.
    """

    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)
    
    v = np.cross(src, dst)
    s = np.linalg.norm(v)
    c = np.dot(src, dst)
    
    v_mat = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
    
    rotation_matrix = np.eye(3) + v_mat + np.dot(v_mat, v_mat) * ((1 - c) / (s ** 2))
    
    return rotation_matrix

def rotate_volume(volume, rotation_vector, reverse=False, order=0, eps=np.finfo(float).eps):
    '''
    Rotates a cubic volume along a random vector
    '''    
    
    rotation_vector = rotation_vector.astype(float) + np.ones(3) * eps
    rotation_matrix = compute_rotation_matrix_from_vectors(np.array([1,0,0]), rotation_vector)
    rotation_matrix = np.around(rotation_matrix, 15)
    
    if order < 2:
        prefilter = False
    else:
        prefilter = True
    
    if reverse:
        rotation_matrix = rotation_matrix.T

    volume_shape = np.array(volume.shape)

    x = np.arange(volume_shape[0])
    y = np.arange(volume_shape[1])
    z = np.arange(volume_shape[2])
    
    coords = np.meshgrid(x, y, z)
    coords = np.vstack([coords[i].reshape(-1) - volume_shape[i] / 2 for i in range(3)])

    rotated_coords = np.dot(rotation_matrix, coords)
    
    x = rotated_coords[0, :] + volume_shape[0] / 2
    y = rotated_coords[1, :] + volume_shape[1] / 2
    z = rotated_coords[2, :] + volume_shape[2] / 2

    x = x.reshape((volume_shape[0], volume_shape[1], volume_shape[2]))
    y = y.reshape((volume_shape[0], volume_shape[1], volume_shape[2]))
    z = z.reshape((volume_shape[0], volume_shape[1], volume_shape[2]))
    
    rotated_coords = np.array([x, y, z])
    
    rotated_volume = map_coordinates(volume, rotated_coords, order=order, mode='constant', prefilter=prefilter).reshape(volume_shape)
    rotated_volume = np.rollaxis(rotated_volume, 1, 0)
    
    return rotated_volume

# Encoding/decoding functions -------------------------------------------------------------------------------------
    
def decode_to_numpy(base64_image):
    base64_decoded = base64.b64decode(base64_image)
    numpy_image = np.array(Image.open(BytesIO(base64_decoded)))
    return numpy_image

def encode_to_base64(numpy_image):
    rawBytes = BytesIO()
    Image.fromarray(numpy_image).save(rawBytes, "JPEG")
    rawBytes.seek(0)
    base64_image = str(base64.b64encode(rawBytes.read()))
    return base64_image

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
    Path(f"model/history").mkdir(parents=True, exist_ok=True)

def clear_data(): 

    shutil.rmtree('./data/mask_volumes')     
    shutil.rmtree('./data/weight_volumes') 
    shutil.rmtree('./data/predicted_volumes')    
    shutil.rmtree('./data/train')   
    shutil.rmtree('./data/val') 
    shutil.rmtree('./model')   
    
    create_directories()

def setup_volumes():
    image_volume_files = np.sort(glob.glob('./data/image_volumes/*'))

    images = []
    for i in range(len(image_volume_files)):
        image_volume = np.load(image_volume_files[i])
        mask_filename = f'./data/mask_volumes/{Path(image_volume_files[i]).stem}.npy'
        weight_filename = f'./data/weight_volumes/{Path(image_volume_files[i]).stem}.npy'

        if not os.path.isfile(mask_filename):
            np.save(mask_filename, np.zeros((image_volume.shape[0],
                                            image_volume.shape[1],
                                            image_volume.shape[2],
                                            2), dtype='uint8'))
        if not os.path.isfile(weight_filename):
            np.save(weight_filename, np.zeros((image_volume.shape[0],
                                            image_volume.shape[1],
                                            image_volume.shape[2]), dtype='uint8'))

        image_volume = np.load(image_volume_files[i])
        images.append(image_volume)

    mask_volume_files = np.sort(glob.glob('./data/mask_volumes/*'))
    weight_volume_files = np.sort(glob.glob('./data/weight_volumes/*'))

    return images, image_volume_files, mask_volume_files, weight_volume_files

def get_random_volume_index(fewest_first=True):
    
    """
    Returns the index of a randomly chosen volume. When fewest_first is True 
    the function returns the index of a one of the volumes with the fewest samples.
    """
    
    image_vols = np.sort(glob.glob('data/image_volumes/*'))

    train_slices = list(glob.glob('data/train/slices/*'))
    val_slices = list(glob.glob('data/val/slices/*'))
    slices = np.sort(train_slices + val_slices)

    if (fewest_first == False) or (len(slices) == 0):
        
        return int(np.random.randint(len(image_vols)))
    
    else:
    
        counts = np.zeros(len(image_vols))

        for i in range(len(slices)):
            slice_data = np.load(slices[i], allow_pickle=True)
            counts[slice_data[0]] += 1


        min_count = np.min(counts)
        vols_to_sample = image_vols[np.argwhere(counts == min_count).ravel()]

        ind = np.argwhere(image_vols == vols_to_sample[np.random.randint(len(vols_to_sample))]).ravel()[0].astype('uint8')

        return int(ind)

# Data representation functions -------------------------------------------------------------------------------------

def make_categorical(data, class_values=None):
    if class_values is None:
        class_values = np.unique(data)
    data = np.moveaxis(np.array([data == v for v in class_values]), 0, -1).astype('uint8')
    return data

def colored_to_categorical(colored_mask, num_classes=2):
    
    colors = np.array([[0,0,0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                       [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [170, 255, 195]])

    mask = np.argmin([np.linalg.norm(colored_mask - colors[i], axis=-1) for i in range(len(colors))], axis=0)
    
    class_values = np.arange(num_classes + 1)
    
    categorical_mask = make_categorical(mask, class_values=class_values)

    weight_slice = 1 - categorical_mask[:,:,0]
    mask_slice = categorical_mask[:,:,1:]

    return mask_slice, weight_slice

def categorical_to_colored(categorical_mask, num_classes=2):
    
    colors = np.array([[0,0,0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                       [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [170, 255, 195]])

    colored_mask = np.sum([categorical_mask[...,i][:,:,None] * colors[i + 1] for i in range(num_classes)], axis=0)
    
    return colored_mask
                             
def display_predictions():

    # Get lists of training samples
    train_images = np.sort(glob.glob('data/train/images/*.tiff'))
    train_weights = np.sort(glob.glob('data/train/weights/*.tiff'))
    train_masks = np.sort(glob.glob('data/train/masks/*.tiff'))

    train_indices = np.arange(len(train_images))
    np.random.shuffle(train_indices)

    # Get lists of validation samples
    val_images = np.sort(glob.glob('data/val/images/*.tiff'))
    val_weights = np.sort(glob.glob('data/val/weights/*.tiff'))
    val_masks = np.sort(glob.glob('data/val/masks/*.tiff'))

    val_indices = np.arange(len(val_images))
    np.random.shuffle(val_indices)

    n_samples = np.min([6, len(train_images)+len(val_images)])

    fig, ax = plt.subplots(3, n_samples, figsize=(n_samples*3, 9))

    for i in range(n_samples):
        if i >= len(val_images):
            index = train_indices[i-len(val_images)]
            title = 'training'
            image = io.imread(train_images[index]) / 255
            mask = np.argmax(io.imread(train_masks[index]), axis=-1)
        else:    
            index = val_indices[i]
            title = 'validation'
            image = io.imread(val_images[index]) / 255
            mask = np.argmax(io.imread(val_masks[index]), axis=-1)

        weight = (1 - (image == 0))
        pred = np.argmax(predict.predict(image, return_probabilities=True)[0,:,:,:-1], axis=-1) * weight

        ax[0,i].set_title(title)
        ax[0,i].imshow(image, cmap='gray', vmin=0, vmax=1)
        ax[0,i].axis('off')
        ax[1,i].set_title(title)
        ax[1,i].imshow(pred, cmap='gray')
        ax[1,i].axis('off')
        ax[2,i].set_title(title)
        ax[2,i].imshow(mask, cmap='gray')
        ax[2,i].axis('off')

    plt.tight_layout()
    plt.show()