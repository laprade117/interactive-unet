import numpy as np
from scipy.ndimage import map_coordinates

"""
A tool used for extracting and replacing randomly (or grid) oriented and positioned slices from a 3D volume.
"""

class Slicer(object):

    def __init__(self, volume_shape=[512,512,512]):

        self.volume_shape = np.array(volume_shape)

        self.update_orientation_vectors(np.array([1,0,0]))

        self.origin = self.volume_shape / 2

        self._normalize_vectors()

    def _normalize_vectors(self):
        """
        Converts orientation vectors to unit vectors.
        """
        
        self.rot_vec = np.around(self.rot_vec, decimals=15)
        self.u = np.around(self.u, decimals=15)
        self.v = np.around(self.v, decimals=15)
        self.w = np.around(self.w, decimals=15)

        self.rot_vec = self.rot_vec / np.linalg.norm(self.rot_vec)
        self.u = self.u / np.linalg.norm(self.u)
        self.v = self.v / np.linalg.norm(self.v)
        self.w = self.w / np.linalg.norm(self.w)

    def _generate_uniformly_random_unit_vector(self, ndim=3):
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

    def _compute_rotation_matrix_from_vectors(self, src, dst):
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

    def to_dict(self):
 
        slicer_dict = {}
        slicer_dict['RotationVector'] = self.rot_vec.tolist()
        slicer_dict['RotationMatrix'] = self.rot_mat.tolist()
        slicer_dict['Origin'] = self.origin.tolist()
        slicer_dict['VolumeShape'] = self.volume_shape.tolist()

        return slicer_dict

    def from_dict(self, slicer_dict):

        self.rot_vec = np.array(slicer_dict['RotationVector'])
        self.rot_mat = np.array(slicer_dict['RotationMatrix'])
        self.origin = np.array(slicer_dict['Origin'])
        self.volume_shape = np.array(slicer_dict['VolumeShape'])
        
        self.update_orientation_vectors(self.rot_vec)

    def get_interpolation_coords(self, slice_width=256):
        """
        Computes the interpolation coordinates for extracting a randomly orientated slice.
        """

        start =  int(-np.floor(slice_width / 2))
        end = start + slice_width

        index_range = np.linspace(start, end-1, slice_width)

        x_coords = (self.v[:, None, None] * index_range[None, :, None] +
                    self.w[:, None, None] * index_range[None, None, :] + self.origin[:,None,None])

        y_coords = (self.u[:, None, None] * index_range[None, :, None] +
                    self.w[:, None, None] * index_range[None, None, :] + self.origin[:,None,None])

        z_coords = (self.u[:, None, None] * index_range[None, :, None] +
                    self.v[:, None, None] * index_range[None, None, :] + self.origin[:,None,None])

        coords = np.array([x_coords, y_coords, z_coords])

        return coords

    def get_origin_candidates(self, volume):
        """
        Computes origin candidates positions and weights for class balancing the extracted slices. 
        """

        classes = np.unique(volume)
            
        candidates = [np.argwhere(volume == class_index) for class_index in classes]
        counts = np.array([candidates[i].shape[0] for i in range(len(classes))])
        class_weights = np.max(counts) / counts
        class_weights = class_weights / np.sum(class_weights)
        
        return candidates, class_weights

    def update_orientation_vectors(self, rotation_vector, eps=np.finfo(float).eps):
        """
        Updates the orientation vectors from the given rotation vector.
        """
        self.rot_vec = rotation_vector.astype(float)

        rotation_vector = rotation_vector.astype(float) + np.ones(3) * eps
        rotation_matrix = self._compute_rotation_matrix_from_vectors(np.array([1,0,0]), rotation_vector)
        rotation_matrix = np.around(rotation_matrix, decimals=15)

        self.u = rotation_vector
        self.v = np.dot(rotation_matrix, np.array([0,1,0]))
        self.w = np.dot(rotation_matrix, np.array([0,0,1]))
        self.rot_mat = rotation_matrix

        self._normalize_vectors()

    def randomize(self, candidates=None, class_weights=None, origin_shift_range=0.8, sampling_mode='random', sampling_axis='random'):
        """
        Randomizes the orientation vectors and origin.
        """
        if sampling_mode == 'grid':
            if sampling_axis == 'random':
                rotation_vector = np.zeros(3)
                rotation_vector[np.random.randint(3)] = 1
            elif sampling_axis == 'x':
                rotation_vector = np.array([1,0,0])
            elif sampling_axis == 'y':
                rotation_vector = np.array([0,1,0])
            elif sampling_axis == 'z':
                rotation_vector = np.array([0,0,1])
        elif sampling_mode == 'random':
            rotation_vector = self._generate_uniformly_random_unit_vector()
        else:
            raise ValueError("sampling_mode must be either \"random\" or \"grid\".")

        self.update_orientation_vectors(rotation_vector)

        if candidates is not None:
            n_classes = len(candidates)
            if class_weights is None:
                class_weights = np.ones(n_classes) / n_classes
            candidate_class = np.random.choice(np.arange(n_classes), p=class_weights)
            ind = np.random.randint(candidates[candidate_class].shape[0])
            self.origin = candidates[candidate_class][ind]
        else:
            half_shape = self.volume_shape / 2
            self.origin = half_shape + origin_shift_range * (np.random.rand(3) * self.volume_shape - half_shape)
            self.origin = np.clip(self.origin, a_min=np.zeros(3), a_max=self.volume_shape-1)

        return self.rot_vec, self.u, self.v, self.w, self.origin

    def get_slice(self, volume, axis=0, slice_width=256, order=0):
        """
        Extracts slice from the given volume.
        """
        
        if len(volume.shape) > 3:
            return np.moveaxis(np.array([map_coordinates(volume[...,i],
                                                         self.get_interpolation_coords(slice_width=slice_width)[axis],
                                                         order=order) for i in range(volume.shape[-1])]), 0, -1)
        else:
            coords = self.get_interpolation_coords(slice_width=slice_width)
            return map_coordinates(volume, coords[axis], order=order)

    def update_volume(self, data, volume, axis=0):
        """
        Updates the volume with slice information.
        """

        # Retrieve the coordinates
        coords = self.get_interpolation_coords(slice_width=data.shape[0])

        # Round coordinates to nearest integer values and flatten
        slice_coords = np.round(coords[axis]).reshape((3,np.prod(coords[axis].shape[1:]))).astype(int)

        slice_coords = np.array([np.clip(slice_coords[i], 0, volume.shape[i]-1) for i in range(3)])
        
        if len(data.shape) == 2:
            data = data.ravel()
        if len(data.shape) == 3:
            data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
            
        volume[slice_coords[0,:], slice_coords[1,:], slice_coords[2,:]] = data

        # # Keep only the coordinates that lie within the volume
        # ind = (np.sum([(slice_coords[i] >= 0).astype(int) & (slice_coords[i] < volume.shape[i]) for i in range(3)], axis=0) == 3)
        # slice_coords = np.array([slice_coords[i][ind] for i in range(3)])

        # # Assign values
        # volume[slice_coords[0,:], slice_coords[1,:], slice_coords[2,:]] = data.ravel()[ind]

        return volume

    def shift_origin(self, shift_amount=[0,0,0]):        
        """
        Updates the position of the slice in the volume.

        Parameters
        ----------
        shift_amount : list, optional
            Amount to shift the origin along each axis.
        """

        self.origin += np.dot(self.rot_mat, shift_amount)
