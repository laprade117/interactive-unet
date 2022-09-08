import glob
import numpy as np
from skimage import io
from scipy import ndimage

from torch.utils.data import Dataset, DataLoader

def get_data_loader(data_folder, batch_size, augment=False):

    dataset = UNetDataset(data_folder, augment=augment)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)

    return loader

def augment_sample(image_slice, mask_slice, weight_slice):

    
    # Random flipping
    if np.random.rand() > 0.5:
        image_slice = np.flip(image_slice, axis=0)
        mask_slice = np.flip(mask_slice, axis=0)
        weight_slice = np.flip(weight_slice, axis=0)
    if np.random.rand() > 0.5:
        image_slice = np.flip(image_slice, axis=1)
        mask_slice = np.flip(mask_slice, axis=1)
        weight_slice = np.flip(weight_slice, axis=1)

    # Random rotations
    rot_angle = np.random.rand() * 360
    image_slice = ndimage.rotate(image_slice, rot_angle, reshape=False, order=0)
    mask_slice = ndimage.rotate(mask_slice, rot_angle, reshape=False, order=0)
    weight_slice = ndimage.rotate(weight_slice, rot_angle, reshape=False, order=0)
    
    # Random shifting
    # shift_amount = np.random.rand(2) * image_slice.shape[0] / 2 - image_slice.shape[0] / 4
    # image_slice = ndimage.shift(image_slice, [shift_amount[0], shift_amount[1]], order=0)
    # mask_slice = ndimage.shift(mask_slice, [shift_amount[0], shift_amount[1], 0], order=0)
    # weight_slice = ndimage.shift(weight_slice, [shift_amount[0], shift_amount[1]], order=0)
    
    # # Random intensity inversion
    # gamma_value = np.random.rand()*0.5 + 0.75
    # image_slice = image_slice ** gamma_value

    # Random brightness adjustments
    brightness_shift = np.random.rand() * 1 - 0.5
    image_slice = image_slice + brightness_shift

    return image_slice, mask_slice, weight_slice

class UNetDataset(Dataset):

    def __init__(self, data_folder, augment=False):
        
        image_filenames = np.sort(glob.glob(f'{data_folder}/images/*'))    
        mask_filenames = np.sort(glob.glob(f'{data_folder}/masks/*'))    
        weight_filenames = np.sort(glob.glob(f'{data_folder}/weights/*'))

        self.data = []
        for i in range(len(image_filenames)):
            self.data.append([io.imread(image_filenames[i]),
                              io.imread(mask_filenames[i]),
                              io.imread(weight_filenames[i])])
        np.random.shuffle(self.data)

        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image_slice, mask_slice, weight_slice = self.data[idx]
	
        image_slice = image_slice / 255
        mask_slice = mask_slice / 255
        weight_slice = weight_slice / 255
	            
        # Augment sample
        if self.augment:
            image_slice, mask_slice, weight_slice = augment_sample(image_slice, mask_slice, weight_slice)

        image_slice = (np.expand_dims(image_slice, 0)).astype(np.float32)
        mask_slice = (np.moveaxis(mask_slice, -1, 0)).astype(np.float32)
        weight_slice = (np.expand_dims(weight_slice, 0)).astype(np.float32)

        # Make weight into multi-class class for softmax
        weight_slice = np.array([weight_slice[0] for i in range(mask_slice.shape[0])])

        # print(image_slice.shape, np.min(image_slice), np.max(image_slice))
        # print(mask_slice.shape, np.min(mask_slice), np.max(mask_slice))
        # print(weight_slice.shape, np.min(weight_slice), np.max(weight_slice))

        return (image_slice, mask_slice, weight_slice)
