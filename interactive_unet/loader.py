import glob
import numpy as np
from skimage import io
from scipy import ndimage

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader

from . import utils

def load_annotations(set_type='train'): 

    if set_type == 'train':
        data_folder = os.path.join('data', 'train')
    else:
        data_folder = os.path.join('data', 'val')

    image_filenames = np.sort(glob.glob(os.path.join(data_folder, 'images', '*')))  
    mask_filenames = np.sort(glob.glob(os.path.join(data_folder, 'masks', '*'))) 
    weight_filenames = np.sort(glob.glob(os.path.join(data_folder, 'weights', '*'))) 

    annotations = []
    for i in range(len(image_filenames)):
        image_slice = io.imread(image_filenames[i])
        mask_slice, _ = utils.colored_to_categorical(io.imread(mask_filenames[i]))
        weight_slice = io.imread(weight_filenames[i])

        if len(image_slice.shape) == 2:
            image_slice = image_slice[:,:,None]
        weight_slice = np.repeat(weight_slice[:,:,None], mask_slice.shape[-1], axis=2)

        image_slice = (np.moveaxis(image_slice, -1, 0) / 255).astype('float32')
        mask_slice = (np.moveaxis(mask_slice, -1, 0) / 255).astype('float32')
        weight_slice = (np.moveaxis(weight_slice, -1, 0) / 255).astype('float32')

        for c in range(weight_slice.shape[0]):
            weight_slice[c][image_slice[0] == 0] = 0.0
            mask_slice[c][image_slice[0] == 0] = 0.0

        annotations.append([image_slice, mask_slice, weight_slice])

    return annotations

def load_resliced_annotations(set_type='train', count=100, num_classes=2): 

    dataset = utils.load_dataset(annotations=True)

    annotations = []
    for i in range(count):
        
        vol_idx = np.random.randint(len(dataset))
        if set_type == 'train':
            image_slice, mask_slice, weight_slice = dataset[vol_idx].sample(weight_channel=0)
        else:
            image_slice, mask_slice, weight_slice = dataset[vol_idx].sample(weight_channel=1)
        mask_slice = utils.class_to_categorical(mask_slice, weight_slice, num_classes)

        while (np.max(mask_slice) != 255) and (np.max(weight_slice) != 255):
            vol_idx = np.random.randint(len(dataset))
            if set_type == 'train':
                image_slice, mask_slice, weight_slice = dataset[vol_idx].sample(weight_channel=0)
            else:
                image_slice, mask_slice, weight_slice = dataset[vol_idx].sample(weight_channel=1)
            mask_slice = utils.class_to_categorical(mask_slice, weight_slice, num_classes)

        if len(image_slice.shape) == 2:
            image_slice = image_slice[:,:,None]
        weight_slice = np.repeat(weight_slice[:,:,None], mask_slice.shape[-1], axis=2)

        image_slice = (np.moveaxis(image_slice, -1, 0) / 255).astype('float32')
        mask_slice = (np.moveaxis(mask_slice, -1, 0) / 255).astype('float32')
        weight_slice = (np.moveaxis(weight_slice, -1, 0) / 255).astype('float32')

        annotations.append([image_slice, mask_slice, weight_slice])

        print(np.max(image_slice), np.max(mask_slice), np.max(weight_slice))

    return annotations

def get_data_loader(set_type='train', num_classes=2, batch_size=2, reslice=False, reslice_factor=2, augment=True, shuffle=True):

    annotations = load_annotations(set_type=set_type)
    # annotations = load_resliced_annotations(set_type=set_type, num_classes=num_classes)

    if reslice:
        resliced_annotations = load_resliced_annotations()
    else:
        resliced_annotations = None

    dataset = UNetDataset(annotations, resliced_annotations, reslice=reslice, reslice_factor=reslice_factor, augment=augment)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=0,
                        persistent_workers=False)

    return loader

class UNetDataset(Dataset):

    def __init__(self, annotations, resliced_annotations, reslice=False, reslice_factor=2, augment=False):
        
        self.annotations = annotations
        self.resliced_annotations = resliced_annotations

        self.reslice = reslice
        self.reslice_factor = reslice_factor

        self.augment = augment

        # self.transforms = v2.Compose([v2.RandomHorizontalFlip(p=0.5),
        #                               v2.RandomVerticalFlip(p=0.5),
        #                               v2.RandomRotation(degrees=(-360,360), interpolation=InterpolationMode.NEAREST),
        #                             #   v2.RandomResizedCrop(size=(512,512), scale=(0.3,1), interpolation=InterpolationMode.NEAREST),
        #                               v2.ElasticTransform(interpolation=InterpolationMode.NEAREST),
        #                               v2.ColorJitter(brightness=(0.75, 1.25), contrast=(0.5, 2.0)),
        #                               v2.RandomChoice([v2.GaussianBlur(kernel_size=5),
        #                                                v2.GaussianNoise()], [0.5,0.5]),
        #                               ])

        self.transforms = v2.Compose([v2.RandomHorizontalFlip(p=0.5),
                                      v2.RandomVerticalFlip(p=0.5),
                                      v2.RandomRotation(degrees=(-360,360), interpolation=InterpolationMode.NEAREST),
                                      v2.RandomResizedCrop(size=(512,512), scale=(0.3,1), interpolation=InterpolationMode.NEAREST),
                                    #   v2.ElasticTransform(interpolation=InterpolationMode.NEAREST),
                                    #   v2.ColorJitter(brightness=(0.75, 1.25), contrast=(0.5, 2.0)),
                                    #   v2.RandomChoice([v2.GaussianBlur(kernel_size=5),
                                    #                    v2.GaussianNoise()], [0.5,0.5]),
                                      ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        image_slice, mask_slice, weight_slice = self.annotations[idx]
	
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
