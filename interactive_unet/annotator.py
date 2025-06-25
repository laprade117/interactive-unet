import cv2
import numpy as np
from scipy.ndimage import map_coordinates

class Annotator(object):

    def __init__(self, canvas_size):

        self.canvas_size = canvas_size

        self.scale_factor = 1.1
        self.roi = np.array([0.0,0.0,1.0,1.0])
        self.scale = 1.0

        self.annotations = []
        self.deleted_annotations = []

        self.input_size = 256
        self.image = None
        self.mask = None
        self.overlays = {}
        self.display_image = None

    def set_image(self, image):

        self.input_size = image.shape[0]
        self.image = image
        self.mask = np.zeros((self.input_size, self.input_size, 3), dtype='uint8')
        self.overlays = {}
        self.display_image = image

    def new_path(self, x0, y0, brush_size, color, mode='paint', overlay=None):

        x0, y0 = self.get_roi_mouse_pos(x0, y0)
        brush_size = brush_size / self.canvas_size * self.scale 

        self.annotations += [[]]
        self.annotations[-1] += [[x0, y0, x0, y0, brush_size, color, mode, overlay]]

    def continue_path(self, x0, y0, x1, y1, brush_size, color, mode='paint', overlay=None):

        x0, y0 = self.get_roi_mouse_pos(x0, y0)
        x1, y1 = self.get_roi_mouse_pos(x1, y1)
        brush_size = brush_size / self.canvas_size * self.scale 

        self.annotations[-1] += [[x0, y0, x1, y1, brush_size, color, mode, overlay]]

    def undo_annotation(self):

        if len(self.annotations) > 0:
            self.deleted_annotations += [self.annotations[-1]]
            self.annotations = self.annotations[:-1]
            self.rebuild_mask()

    def redo_annotation(self):

        if len(self.deleted_annotations) > 0:
            self.annotations += [self.deleted_annotations[-1]]
            self.deleted_annotations = self.deleted_annotations[:-1]
            self.rebuild_mask()

    def get_current_path_overlay(self, mode='paint'):
        
        path_svg = ''
        
        if len(self.annotations) > 0:
            
            path = self.annotations[-1]

            for j in range(len(path)):

                x0, y0, x1, y1, brush_size, color, path_mode, overlay = path[j]
                x0 = (x0 - self.roi[0]) * self.canvas_size / self.scale 
                y0 = (y0 - self.roi[1]) * self.canvas_size / self.scale
                x1 = (x1 - self.roi[0]) * self.canvas_size / self.scale
                y1 = (y1 - self.roi[1]) * self.canvas_size / self.scale
                brush_size = brush_size * self.canvas_size / self.scale

                if path_mode == mode:

                    path_svg += f'<circle cx="{x0}" cy="{y0}" r="{brush_size/2}" fill="{color}" stroke="{color}" />'
                    path_svg += f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="{brush_size}" fill="none" />'

                    if j == len(path) - 1:
                        path_svg += f'<circle cx="{x1}" cy="{y1}" r="{brush_size/2}" fill="{color}" stroke="{color}" />'

        return path_svg

    def apply_current_path(self, idx=-1):

        path = self.annotations[idx]

        for j in range(len(path)):

            x0, y0, x1, y1, brush_size, color, path_mode, overlay = path[j]
            x0 = int(x0 * self.input_size)
            y0 = int(y0 * self.input_size)
            x1 = int(x1 * self.input_size)
            y1 = int(y1 * self.input_size)
            brush_size = brush_size * self.input_size

            if path_mode == 'paint':

                color = color.split('(')[-1].split(')')[0].split(',')
                color = (int(color[0]), int(color[1]), int(color[2]))
                
                cv2.circle(self.mask, (x0,y0), int(brush_size/2), color, -1)
                cv2.line(self.mask, (x0,y0), (x1,y1), color, int(brush_size))

                if j == len(path) - 1:
                    cv2.circle(self.mask, (x1,y1), int(brush_size/2), color, -1)
                
            elif path_mode == 'capture_overlay':

                overlay_mask = self.overlays[overlay]

                overlay_capture = np.zeros((overlay_mask.shape[0], overlay_mask.shape[1]))

                color = (255)
                
                cv2.circle(overlay_capture, (x0,y0), int(brush_size/2), color, -1)
                cv2.line(overlay_capture, (x0,y0), (x1,y1), color, int(brush_size))

                if j == len(path) - 1:
                    cv2.circle(overlay_capture, (x1,y1), int(brush_size/2), color, -1)

                overlay_capture_region = overlay_capture == 255
                self.mask[overlay_capture_region] = overlay_mask[overlay_capture_region]
        

    def update_display(self, annotation_opacity=0.25, overlay_opacity=0.25, overlay=None):

        image = self.image / 255
        mask = self.mask / 255

        if (len(self.overlays) > 0) and (overlay_opacity > 0) and (overlay is not None):
            overlay = self.overlays[overlay] / 255
            image = image * (1 - overlay_opacity) + overlay * overlay_opacity

        if annotation_opacity > 0:
            mask_overlay_region = mask[:,:,1] > 0
            image[mask_overlay_region] = image[mask_overlay_region] * (1 - annotation_opacity) + mask[mask_overlay_region] * annotation_opacity

        self.display_image = np.round(255 * image).astype('uint8')
    
    def get_num_unique_colors(self):

        colors = []

        for i in range(len(self.annotations)):

            path = self.annotations[i]

            for j in range(len(path)):

                _, _, _, _, _, color, _, _ = path[j]
                colors.append(color)

        return len(np.unique(colors).ravel())

    def rebuild_mask(self):

        self.mask = np.zeros((self.input_size, self.input_size, 3), dtype='uint8')

        for i in range(len(self.annotations)):
            self.apply_current_path(i)


    def reset(self):
        '''
        Resets region of interest and clears annotations.
        '''
        self.roi = np.array([0.0,0.0,1.0,1.0])
        self.scale = 1.0
        self.annotations = []
        self.deleted_annotations = []

        self.mask = np.zeros((self.input_size, self.input_size, 3), dtype='uint8')
        self.overlays = {}
        self.display_image = self.image


# Zooming, scale and translation functions-----------------------------------------------------------------------

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

    def get_roi_image(self, size=None):
        '''
        Extract a region of interest from an image.
        '''

        if size is None:
            x_grid, y_grid = np.meshgrid(np.linspace(self.roi[0], self.roi[2], self.canvas_size) * (self.display_image.shape[0] - 1),
                                        np.linspace(self.roi[1], self.roi[3], self.canvas_size) * (self.display_image.shape[1] - 1))
        else:
            x_grid, y_grid = np.meshgrid(np.linspace(self.roi[0], self.roi[2], size) * (self.display_image.shape[0] - 1),
                                        np.linspace(self.roi[1], self.roi[3], size) * (self.display_image.shape[1] - 1))
        
        if len(self.display_image.shape) > 2:
            roi_image = np.concatenate([map_coordinates(self.display_image[:,:,i], np.array([y_grid, x_grid]), order=0)[:,:,None] for i in range(self.display_image.shape[2])], axis=2)
        else:
            roi_image = map_coordinates(self.display_image, np.array([y_grid, x_grid]), order=0)

        return roi_image