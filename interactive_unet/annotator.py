import cv2
import numpy as np
from scipy.ndimage import map_coordinates

class Annotator(object):

    def __init__(self, canvas_size, scale_factor=1.1):
        
        self.canvas_size = canvas_size
        self.scale_factor = scale_factor
        
        self.roi = np.array([0.0,0.0,1.0,1.0])
        self.scale = 1.0

        self.annotations = []
        self.deleted_annotations = []

    def new_path(self, x0, y0, brush_size, color):

        x0, y0 = self.get_roi_mouse_pos(x0, y0)
        brush_size = brush_size / self.canvas_size * self.scale 

        self.annotations += [[]]
        self.annotations[-1] += [[x0, y0, x0, y0, brush_size, color]]

    def continue_path(self, x0, y0, x1, y1, brush_size, color):

        x0, y0 = self.get_roi_mouse_pos(x0, y0)
        x1, y1 = self.get_roi_mouse_pos(x1, y1)
        brush_size = brush_size / self.canvas_size * self.scale 

        self.annotations[-1] += [[x0, y0, x1, y1, brush_size, color]]

    def undo_annotation(self):

        if len(self.annotations) > 0:
            self.deleted_annotations += [self.annotations[-1]]
            self.annotations = self.annotations[:-1]

    def redo_annotation(self):

        if len(self.deleted_annotations) > 0:
            self.annotations += [self.deleted_annotations[-1]]
            self.deleted_annotations = self.deleted_annotations[:-1]

    def get_mask_overlay(self):

        mask = ''

        for i in range(len(self.annotations)):

            path = self.annotations[i]

            for j in range(len(path)):

                x0, y0, x1, y1, brush_size, color = path[j]
                x0 = (x0 - self.roi[0]) * self.canvas_size / self.scale 
                y0 = (y0 - self.roi[1]) * self.canvas_size / self.scale
                x1 = (x1 - self.roi[0]) * self.canvas_size / self.scale
                y1 = (y1 - self.roi[1]) * self.canvas_size / self.scale
                brush_size = brush_size * self.canvas_size / self.scale

                mask += f'<circle cx="{x0}" cy="{y0}" r="{brush_size/2}" fill="{color}" stroke="{color}" />'
                mask += f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="{brush_size}" fill="none" />'

                if j == len(path) - 1:
                    mask += f'<circle cx="{x1}" cy="{y1}" r="{brush_size/2}" fill="{color}" stroke="{color}" />'

        return mask
    
    def get_num_unique_colors(self):

        colors = []

        for i in range(len(self.annotations)):

            path = self.annotations[i]

            for j in range(len(path)):

                _, _, _, _, _, color = path[j]
                colors.append(color)

        return len(np.unique(colors).ravel())

    def get_roi_center_pos(self):
        
        roi_center_x = self.roi[0] + self.scale / 2
        roi_center_y = self.roi[1] + self.scale / 2
        
        return roi_center_x, roi_center_y
        
    def get_roi_mouse_pos(self, mouse_x, mouse_y):
        
        roi_mouse_x = self.roi[0] + (mouse_x / self.canvas_size) * self.scale 
        roi_mouse_y = self.roi[1] + (mouse_y / self.canvas_size) * self.scale
        
        return roi_mouse_x, roi_mouse_y

    def translate(self, x0, y0, x1, y1):

        translate_x = - self.scale * (x1 - x0) / self.canvas_size
        translate_y = - self.scale * (y1 - y0) / self.canvas_size

        self.roi += np.array([translate_x, translate_y, translate_x, translate_y])

    def zoom_in(self, mouse_x, mouse_y):
        
        # Center point in roi space
        roi_center_x, roi_center_y = self.get_roi_center_pos()
        
        # Mouse coordinates in roi space
        roi_mouse_x, roi_mouse_y = self.get_roi_mouse_pos(mouse_x, mouse_y)

        # Update ROI
        self.scale = self.scale * (1 / self.scale_factor)
        roi_start_x = roi_center_x - self.scale / 2
        roi_start_y = roi_center_y - self.scale / 2
        self.roi = np.array([roi_start_x, roi_start_y, roi_start_x + self.scale, roi_start_y + self.scale])
        
        # Mouse coordinates in updated roi space
        new_roi_mouse_x, new_roi_mouse_y = self.get_roi_mouse_pos(mouse_x, mouse_y)
        
        # Distance from mouse coordinates to center point in roi space
        roi_shift_x = roi_mouse_x - new_roi_mouse_x
        roi_shift_y = roi_mouse_y - new_roi_mouse_y
        
        # Shift roi
        self.roi += np.array([roi_shift_x, roi_shift_y, roi_shift_x, roi_shift_y])
        
    def zoom_out(self, mouse_x, mouse_y):
        
        # Center point in roi space
        roi_center_x, roi_center_y = self.get_roi_center_pos()
        
        # Mouse coordinates in roi space
        roi_mouse_x, roi_mouse_y = self.get_roi_mouse_pos(mouse_x, mouse_y)

        # Update ROI
        self.scale = self.scale * self.scale_factor
        roi_start_x = roi_center_x - self.scale / 2
        roi_start_y = roi_center_y - self.scale / 2
        self.roi = np.array([roi_start_x, roi_start_y, roi_start_x + self.scale, roi_start_y + self.scale])
        
        # Mouse coordinates in updated roi space
        new_roi_mouse_x, new_roi_mouse_y = self.get_roi_mouse_pos(mouse_x, mouse_y)
        
        # Distance from mouse coordinates to center point in roi space
        roi_shift_x = roi_mouse_x - new_roi_mouse_x
        roi_shift_y = roi_mouse_y - new_roi_mouse_y
        
        # Shift roi
        self.roi += np.array([roi_shift_x, roi_shift_y, roi_shift_x, roi_shift_y])

    def get_roi_image(self, image, size=None):
        '''
        Extract a region of interest from an image.
        '''
        if size is None:
            x_grid, y_grid = np.meshgrid(np.linspace(self.roi[0], self.roi[2], self.canvas_size) * (image.shape[0] - 1),
                                        np.linspace(self.roi[1], self.roi[3], self.canvas_size) * (image.shape[1] - 1))
        else:
            x_grid, y_grid = np.meshgrid(np.linspace(self.roi[0], self.roi[2], size) * (image.shape[0] - 1),
                                        np.linspace(self.roi[1], self.roi[3], size) * (image.shape[1] - 1))
        
        if len(image.shape) > 2:
            roi_image = np.concatenate([map_coordinates(image[:,:,i], np.array([y_grid, x_grid]), order=0)[:,:,None] for i in range(image.shape[2])], axis=2)
        else:
            roi_image = map_coordinates(image, np.array([y_grid, x_grid]), order=0)

        return roi_image
        
    def reset(self):
        '''
        Resets region of interest and scale.
        '''
        self.roi = np.array([0.0,0.0,1.0,1.0])
        self.scale = 1.0
        self.annotations = []
        self.deleted_annotations = []