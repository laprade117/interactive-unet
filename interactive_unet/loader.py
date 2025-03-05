import glob
import numpy as np
from skimage import io
from scipy import ndimage

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader

from interactive_unet import utils

def get_data_loader(data_folder, batch_size, augment=False, shuffle=False):

    dataset = UNetDataset(data_folder, augment=augment)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=4,
		        persistent_workers=True)

    return loader

class UNetDataset(Dataset):

    def __init__(self, data_folder, augment=False):
        
        image_filenames = np.sort(glob.glob(f'{data_folder}/images/*'))    
        mask_filenames = np.sort(glob.glob(f'{data_folder}/masks/*'))    
        weight_filenames = np.sort(glob.glob(f'{data_folder}/weights/*'))

        self.data = []
        for i in range(len(image_filenames)):
            image = io.imread(image_filenames[i])
            mask, _ = utils.colored_to_categorical(io.imread(mask_filenames[i]))
            weight = io.imread(weight_filenames[i])
            self.data.append([image, mask, weight])

        self.augment = augment

        self.transforms = v2.Compose([v2.RandomHorizontalFlip(p=0.5),
                                      v2.RandomVerticalFlip(p=0.5),
                                      v2.RandomRotation(degrees=(-360,360), interpolation=InterpolationMode.NEAREST),
                                    #   v2.RandomResizedCrop(size=(512,512), scale=(0.3,1), interpolation=InterpolationMode.NEAREST),
                                      v2.ElasticTransform(interpolation=InterpolationMode.NEAREST),
                                      v2.ColorJitter(brightness=(0.75, 1.25), contrast=(0.5, 2.0)),
                                      v2.RandomChoice([v2.GaussianBlur(kernel_size=5),
                                                       v2.GaussianNoise()], [0.5,0.5]),
                                      ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image_slice, mask_slice, weight_slice = self.data[idx]
        if len(image_slice.shape) == 2:
            image_slice = image_slice[:,:,None]
        weight_slice = np.repeat(weight_slice[:,:,None], mask_slice.shape[-1], axis=2)

        image_slice = (np.moveaxis(image_slice, -1, 0) / 255).astype('float32')
        mask_slice = (np.moveaxis(mask_slice, -1, 0) / 255).astype('float32')
        weight_slice = (np.moveaxis(weight_slice, -1, 0) / 255).astype('float32')
	
        image_slice = tv_tensors.Image(image_slice)
        mask_slice = tv_tensors.Mask(mask_slice)
        weight_slice = tv_tensors.Mask(weight_slice)
	            
        # Augment sample
        if self.augment:
            image_slice, mask_slice, weight_slice = self.transforms(image_slice, mask_slice, weight_slice)

        image_slice = image_slice.to(torch.float16)
        mask_slice = mask_slice.to(torch.float16)
        weight_slice = weight_slice.to(torch.float16)

        return (image_slice, mask_slice, weight_slice)
