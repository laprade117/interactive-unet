import os
import glob
import zarr
import numpy as np
from skimage import io

from . import utils
from .slicer import Slicer

"""
A VolumeData object stores the image volume and an associated slicer object. Can also hold annotation information.
"""

class VolumeData(object):

    def __init__(self, file, annotations=False):

        # self.filename = file.split('/')[-1].split('.npy')[0]
        self.filename = file.split('/')[-1].split('.zarr')[0]

        # self.image_volume = np.load(f'data/image_volumes/{self.filename}.npy')
        self.image_volume = zarr.open(f'data/image_volumes/{self.filename}.zarr', mode='r')['0']

        self.slicer = Slicer(self.image_volume.shape)
        
        if annotations:
            self.mask_volume = np.load(f'data/mask_volumes/{self.filename}.npy')
            self.weight_volume = np.load(f'data/weight_volumes/{self.filename}.npy')
            self.candidates, self.class_weights = self.slicer.get_origin_candidates(self.mask_volume)

    def build_annotation_volumes(self):

        current_slicer_state = self.slicer.to_dict()
        
        mask_volume = np.zeros((self.image_volume.shape[0],self.image_volume.shape[1],self.image_volume.shape[2]), dtype='uint8')
        weight_volume = np.zeros((self.image_volume.shape[0],self.image_volume.shape[1],self.image_volume.shape[2], 2), dtype='uint8')

        slice_files = np.sort(glob.glob('data/train/slices/*.npy'))

        for i in range(len(slice_files)):

            slice_data = np.load(slice_files[i], allow_pickle=True).ravel()[0]

            if slice_data['volume'] == self.filename:

                # Load sample
                mask = io.imread(slice_files[i].replace('slices','masks').replace('.npy', '.tiff'))
                weight_train = io.imread(slice_files[i].replace('slices','weights').replace('.npy', '.tiff'))
                weight_val = io.imread(slice_files[i].replace('slices','weights').replace('.npy', '.tiff').replace('train', 'val'))

                mask = utils.colored_to_class(mask)

                # Load slicer info
                self.slicer.from_dict(slice_data['slicer'])

                # Replace 
                mask_volume = self.slicer.update_volume(mask, mask_volume)
                weight_volume[...,0] = self.slicer.update_volume(weight_train, weight_volume[...,0])
                weight_volume[...,1] = self.slicer.update_volume(weight_val, weight_volume[...,1])

        np.save(f'data/mask_volumes/{self.filename}.npy', mask_volume)
        np.save(f'data/weight_volumes/{self.filename}.npy', weight_volume)

        self.slicer.from_dict(current_slicer_state)

    def sample(self, weight_channel=0, slice_width=512, origin_shift_range=0.8, sampling_mode='random', sampling_axis='random', order=1):

        self.slicer.randomize(candidates=self.candidates,
                              class_weights=self.class_weights, 
                              origin_shift_range=origin_shift_range,
                              sampling_mode=sampling_mode, 
                              sampling_axis=sampling_axis)

        image_slice = self.slicer.get_slice(self.image_volume, slice_width=slice_width, order=order)
        mask_slice = self.slicer.get_slice(self.mask_volume, slice_width=slice_width, order=0)
        weight_slice = self.slicer.get_slice(self.weight_volume[...,weight_channel], slice_width=slice_width, order=0)

        return image_slice, mask_slice, weight_slice

    # Slicer functions-----------------------------------------------------------------------------------------------------------------------

    def randomize(self, candidates=None, class_weights=None, origin_shift_range=0.8, sampling_mode='random', sampling_axis='random'):
        self.slicer.randomize(candidates=candidates, class_weights=class_weights,
                              origin_shift_range=origin_shift_range, sampling_mode=sampling_mode,
                              sampling_axis=sampling_axis)
 
    def shift_origin(self, shift_amount=[0,0,0]):        
        self.slicer.shift_origin(shift_amount=shift_amount)

    def get_slice(self, axis=0, slice_width=256, order=0):
        return self.slicer.get_slice(self.image_volume, axis=axis, slice_width=slice_width, order=order)    
    
    # ---------------------------------------------------------------------------------------------------------------------------------------