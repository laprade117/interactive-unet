import os
import cv2
import glob
import time
import pickle
import asyncio
import threading
import numpy as np
from PIL import Image
from skimage import io

import plotly.graph_objects as go

from nicegui import ui, events, run
from nicegui.events import KeyEventArguments

import segmentation_models_pytorch as smp

from .slicer import Slicer
from .annotator import Annotator
from . import utils, trainer, predict, suggestor

def run_app(reload=False):
    app = InteractiveSegmentationTool()
    ui.run(port=9090, show=False, reload=reload)

if __name__ in {"__main__", "__mp_main__"}:
    run_app(True)

class InteractiveSegmentationTool:

    def __init__(self):

        # Creates initial directory structure if not already created
        utils.create_directories()

        # Load data
        self.dataset = utils.load_dataset()
        if len(self.dataset) > 0:
            self.volume_index = np.random.randint(len(self.dataset))

        self.train_samples = glob.glob('data/train/images/*.tiff')

        # Data parameters
        self.num_classes = utils.get_num_classes()
        self.input_size = utils.get_input_size()

        # Color parameters
        self.colors = ['rgba(230, 25, 75, 1)', 'rgba(60, 180, 75, 1)',
                       'rgba(255, 225, 25, 1)', 'rgba(0, 130, 200, 1)',
                       'rgba(245, 130, 48, 1)', 'rgba(145, 30, 180, 1)',
                       'rgba(70, 240, 240, 1)', 'rgba(240, 50, 230, 1)',
                       'rgba(210, 245, 60, 1)', 'rgba(170, 255, 195, 1)']
        self.color_idx = 1
        self.color_idx_prev = 1

        # Plotly figure parameters
        self.metric = 'Loss'
        self.fig = utils.get_training_history_figure(self.metric)

        # Sampling parameters
        self.sampling_mode = 'random'
        self.sampling_axis = 'random'

        # UI/Canvas parameters
        self.canvas_size = 700
        self.annotator = Annotator(self.canvas_size)

        self.training = False
        self.predicting = False
        self.suggesting = False
        self.interacting = False
        self.updated = True
        self.last_interaction = time.time()

        ui.add_body_html("""
                            <script>
                            document.addEventListener('DOMContentLoaded', function() {
                        
                                document.addEventListener('mousedown', function(event) {
                                    event.preventDefault();
                                });
                        
                                document.addEventListener('wheel', function(event) {
                                    if (event.ctrlKey) {
                                        event.preventDefault();
                                    }
                                }, { passive: false });
                        
                                document.addEventListener('mouseup', function(event) {
                                    event.preventDefault();
                                });
                        
                                document.addEventListener('keydown', function(event) {
                                    event.preventDefault();
                                });
                        
                                document.addEventListener('keyup', function(event) {
                                    event.preventDefault();
                                });
                        
                                document.addEventListener("contextmenu", function (event) {
                                    event.preventDefault();
                                });
                        
                                const elements = document.querySelectorAll('button');
                        
                                elements.forEach(element => {
                                    element.addEventListener('keyup', function() {
                                        this.blur();
                                        document.body.focus();
                                        event.preventDefault();
                                    });
                                    element.addEventListener('keydown', function() {
                                        this.blur();
                                        document.body.focus();
                                        event.preventDefault();
                                    });
                                    element.addEventListener('click', function() {
                                        this.blur();
                                        document.body.focus();
                                        event.preventDefault();
                                    });
                                    element.addEventListener('change', function() {
                                        this.blur();
                                        document.body.focus();
                                        event.preventDefault();
                                    });
                                    element.addEventListener('input', function() {
                                        this.blur();
                                        document.body.focus();
                                        event.preventDefault();
                                    });
                                });
                        
                            });
                            </script>
                            """)

        ui.page_title('Interactive Segmentation')
        with ui.column(align_items='center').classes('w-full justify-center'):

            ui.markdown("#### **Interactive Segmentation Tool**")

            with ui.row().classes('w-full justify-center'):

                with ui.column().classes('w-1/4'):

                    with ui.scroll_area().classes('w-full h-[calc(100vh-8rem)] justify-center no-wrap'):
                        
                        with ui.row(align_items='center').classes('bg-gray-100 w-full justify-center no-wrap'):
                            with ui.element('div').classes('justify-center'):
                                self.ui_volume_count = ui.markdown(f'Volumes: {len(self.dataset)}')

                            with ui.element('div').classes('justify-center'):
                                self.ui_sample_count = ui.markdown(f'Samples: {len(self.train_samples)}')
                        
                        with ui.row(align_items='center').classes('w-full justify-center no-wrap'):

                            self.ui_select_input_size = ui.select([128,256,384,512],
                                                            value=self.input_size, label='Input Size',
                                                            on_change=self.update_input_size).props('filled').classes('w-1/2')
                            self.ui_select_input_size.on('click', self.update_input_size)
                            self.ui_select_num_classes = ui.select([2,3,4,5,6,7,8,9,10],
                                                            value=self.num_classes, label='Num Classes',
                                                            on_change=self.update_num_classes).props('filled').classes('w-1/2')
                            self.ui_select_num_classes.on('click', self.update_num_classes)

                            if len(self.train_samples) > 0:
                                self.ui_select_input_size.disable()
                                self.ui_select_num_classes.disable()

                        with ui.expansion(text='Model settings').props('dense filled').classes('w-full'):

                            self.ui_select_architecture = ui.select(['U-Net', 'U-Net++', 'FPN',
                                                                'PSPNet', 'DeepLabV3', 'DeepLabV3+',
                                                                'LinkNet', 'MA-Net', 'PAN',
                                                                'UPerNet', 'Segformer'],
                                                            value='U-Net++',
                                                            label='Architecture').props('filled').classes('w-full')

                            with ui.row(align_items='center').classes('w-full justify-center no-wrap'):

                                self.ui_select_encoder = ui.select(smp.encoders.get_encoder_names(), 
                                                            value='timm-tf_efficientnet_lite0',
                                                            label='U-Net Encoder').props('filled').classes('w-3/4')
                                self.ui_checkbox_pretrained = ui.checkbox('Pretrained',
                                                                    value=True).classes('w-1/4')

                                if os.path.isfile('model/model.ckpt'):
                                    
                                    with open('model/model_details.pkl', 'rb') as f:
                                        model_details = pickle.load(f)
                                        self.ui_select_architecture.value = model_details['architecture']
                                        self.ui_select_encoder.value = model_details['encoder_name']
                                        self.ui_checkbox_pretrained.value = model_details['pretrained']

                                    self.ui_select_architecture.disable()
                                    self.ui_select_encoder.disable()
                                    self.ui_checkbox_pretrained.disable()
                                
                        with ui.expansion(text='Training settings').props('dense').classes('w-full'):

                            self.ui_select_lr = ui.select([0.00001,0.0001,0.001,0.01], 
                                                    value=0.001, 
                                                    label='Learning rate').props('filled').classes('w-full')
                            self.ui_select_batch_size = ui.select([2,4,8,16,32],
                                                            value=8,
                                                            label='Batch size').props('filled').classes('w-full')
                            self.ui_select_num_epochs = ui.select([50,100,150,200,250,300],
                                                            value=100,
                                                            label='Num epochs').props('filled').classes('w-full')
                            self.ui_select_loss_function = ui.select(['Crossentropy (CE)', 'Dice',
                                                                'Intersection over Union (IoU)', 
                                                                'Matthews correlation coefficient (MCC)',
                                                                'Dice + CE', 'IoU + CE', 'MCC + CE'],
                                                                value='MCC + CE',
                                                                label='Loss function').props('filled').classes('w-full')

                        with ui.expansion(text='Sampler settings').props('dense').classes('w-full'):

                            self.ui_select_sampling_mode = ui.select(['Random', 'Axially-aligned'],
                                                            value='Random',
                                                            label='Sampling mode',
                                                            on_change=self.update_sampling_mode).props('filled').classes('w-full')
                            self.ui_select_sampling_axis = ui.select(['Random', 'X', 'Y', 'Z'],
                                                            value='Random',
                                                            label='Sampling axis',
                                                            on_change=self.update_sampling_axis).props('filled').classes('w-full')
                            self.ui_select_sampling_axis.bind_visibility_from(self.ui_select_sampling_mode, 'value', backward=lambda v: v == "Axially-aligned")

                        with ui.expansion(text='Display settings').props('dense').classes('w-full'):

                            with ui.row(align_items='center').classes('w-full justify-center no-wrap'):
                                
                                ui.markdown('Cursor opacity').classes('w-1/2')
                                self.ui_slider_cursor_opacity = ui.slider(min=0, max=100, value=25, on_change=self.update_cursor_opacity).classes('w-1/2')

                            with ui.row(align_items='center').classes('w-full justify-center no-wrap'):
                                
                                ui.markdown('Annotation opacity').classes('w-1/2')
                                self.ui_slider_annotation_opacity = ui.slider(min=0, max=100, value=25, on_change=self.update_annotation_opacity).classes('w-1/2')

                            with ui.row(align_items='center').classes('w-full justify-center no-wrap'):
                                
                                ui.markdown('Overlay opacity').classes('w-1/2')
                                self.ui_slider_overlay_opacity = ui.slider(min=0, max=100, value=25, on_change=self.update_overlay_opacity).classes('w-1/2')

                        with ui.expansion(text='Project settings').props('dense').classes('w-full'):

                            with ui.row(align_items='center').classes('w-full justify-center no-wrap'):

                                self.ui_button_clear_model = ui.button('Reset model weights', on_click=self.clear_model).props('filled').classes('w-1/2')
                                self.ui_button_clear_annotations = ui.button('Clear annotations', on_click=self.clear_annotations).props('filled').classes('w-1/2')
                                
                            self.ui_button_reset_all = ui.button('Reset and clear all', on_click=self.reset_all).props('filled').classes('w-full')

                with ui.column():

                    with ui.row(align_items='center').classes('bg-gray-100 w-full justify-center no-wrap'):
                        with ui.element('div').classes('justify-center'):
                            self.ui_display_info = ui.markdown(f'No overlay displayed')

                    self.ii = ui.interactive_image(on_mouse=self.mouse_handler, size=(self.canvas_size, self.canvas_size),
                                            events=['mousedown', 'mousemove', 'mouseup'])
                    self.ii.on('wheel', self.mouse_wheel_handler)

                    self.ui_button_predict = ui.button('Predict', on_click=self.predict_slice).props('filled').classes('w-full')

                with ui.column().classes('w-1/4'):

                    self.ui_select_plot_metric = ui.select(['Loss', 'Dice', 'IoU', 'MCC'],
                                                    value='Loss',
                                                    label='Plot metric',
                                                    on_change=self.update_training_plot).props('filled').classes('w-full')

                    self.ui_plotly_training_plot = ui.plotly(self.fig).classes('w-full h-96')

                    self.ui_button_train = ui.button('Train', on_click=self.train_model).props('filled').classes('w-full')

                    self.ui_button_predict_volumes = ui.button('Predict volumes', on_click=self.predict_volumes).props('filled').classes('w-full')
                    
                    # with ui.expansion(text='Other tools').props('dense').classes('w-full'):
                    #     self.ui_button_build_annotation_volumes = ui.button('Rebuild annotation volumes', on_click=self.build_annotation_volumes).props('filled').classes('w-full')


        self.volume_timer = ui.timer(2.0, callback=self.check_volume_folder)
        self.plot_timer = ui.timer(2.0, callback=self.update_training_plot)
        self.redraw_timer = ui.timer(0.2, callback=self.redraw_check)
        self.defocus_timer = ui.timer(1.0, callback=self.defocus)

        self.keyboard = ui.keyboard(on_key=self.key_handler)

        # Confirmation dialog
        with ui.dialog() as self.dialog, ui.card():
            self.confirmation_label = ui.label('Are you sure?')
            with ui.row():
                ui.button('Yes', on_click=lambda: self.dialog.submit('Yes'))
                ui.button('No', on_click=lambda: self.dialog.submit('No'))

        self.randomize()

    def randomize(self):

        if len(self.dataset) == 0:
            # Create blank slice if no volumes are provided
            self.image_slice = np.zeros((self.input_size,self.input_size), dtype='uint8')
        else:
            # Choose random volume and get random slice
            self.volume_index = np.random.randint(len(self.dataset))
            self.dataset[self.volume_index].randomize(sampling_mode=self.sampling_mode, sampling_axis=self.sampling_axis)
            self.image_slice = self.dataset[self.volume_index].get_slice(slice_width=self.input_size, order=1).astype('uint8')
            self.ii.image_features = (self.image_slice / 255).astype('float32')[None,None,:,:]

        self.annotator.set_image(np.repeat(self.image_slice[:,:,None], 3, axis=2))
        
        self.clear()

    def redraw_check(self): 
        
        if self.interacting is False and self.updated is False:
            self.updated = True
            self.redraw()

    def redraw(self):
        
        self.annotator.update_display(self.ii.annotation_opacity, self.ii.overlay_opacity, overlay=self.ii.overlay)

        if self.interacting: 
            # Fast redraw
            self.ii.source = Image.fromarray(cv2.resize(self.annotator.get_roi_image(size=60), (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_NEAREST))
        else:
            self.ii.source = Image.fromarray(self.annotator.get_roi_image())

        self.redraw_overlay()

    def redraw_overlay(self):

        mask = ''

        if self.ii.is_drawing:
            mask = self.annotator.get_current_path_overlay()

        cursor = f'<circle cx="{self.ii.x}" cy="{self.ii.y}" r="{self.ii.brush_size/2}" fill="{self.colors[self.color_idx]}" stroke="{self.colors[self.color_idx]}" opacity="{self.ii.cursor_opacity}" />'
        self.ii.content = f'<g opacity="{self.ii.annotation_opacity}"> {mask} </g> {cursor}'
        
    def clear(self):
        global annotator, color_idx, interacting, updated

        self.annotator.reset()
        self.color_idx = 1
        self.interacting = False
        self.updated = False

        self.ii.x = 0
        self.ii.y = 0
        self.ii.is_drawing = False
        self.ii.mode = 'paint'
        self.ii.overlay = None
        self.ii.suggestor_model = None
        self.ii.brush_size = 40

        self.ii.cursor_opacity = 0.25
        self.ii.annotation_opacity = 0.25
        self.ii.overlay_opacity = 0.25

        self.ui_slider_cursor_opacity.value = int(self.ii.cursor_opacity * 100)
        self.ui_slider_annotation_opacity.value = int(self.ii.annotation_opacity * 100)
        self.ui_slider_overlay_opacity.value = int(self.ii.overlay_opacity * 100)

        self.ui_button_predict.text = 'Predict'

        self.redraw()

    def key_handler(self, e: KeyEventArguments):
        
        if e.action.keydown and not e.action.repeat:

            # Random slice
            if e.key == 'Space':
                self.randomize()

            # Next slice in stack
            if e.key == 'q':
                self.dataset[self.volume_index].shift_origin(shift_amount=[1,0,0])
                self.image_slice = self.dataset[self.volume_index].get_slice(slice_width=self.input_size, order=1).astype('uint8')
                self.ii.image_features = (self.image_slice / 255).astype('float32')[None,None,:,:]
                self.annotator.set_image(np.repeat(self.image_slice[:,:,None], 3, axis=2))    
                self.ii.suggestor_model = None
                self.redraw()

            # Previous slice in stack
            if e.key == 'a':
                self.dataset[self.volume_index].shift_origin(shift_amount=[-1,0,0])
                self.image_slice = self.dataset[self.volume_index].get_slice(slice_width=self.input_size, order=1).astype('uint8')
                self.ii.image_features = (self.image_slice / 255).astype('float32')[None,None,:,:]
                self.annotator.set_image(np.repeat(self.image_slice[:,:,None], 3, axis=2))
                self.ii.suggestor_model = None   
                self.redraw()

            # Next class/color
            if e.key == 'c':
                self.color_idx += 1
                if self.color_idx == self.num_classes:
                    self.color_idx = 1
                self.redraw_overlay()

            # Previous class/color
            if e.key == 'v':
                self.color_idx -= 1
                if self.color_idx == 0:
                    self.color_idx = self.num_classes - 1
                self.redraw_overlay()

            # Toggle overlay
            if e.key == 'd':
                self.toggle_overlay()
            
            if e.key == 'f':
                self.cycle_overlay()

        if e.modifiers.ctrl and e.action.keydown and not e.action.repeat:

            if e.key == 'z':
                self.annotator.undo_annotation()
                self.redraw()

            if e.key == 'y':
                self.annotator.redo_annotation()
                self.redraw()

            if e.key == 's':
                if (len(self.train_samples) == 0) and (self.annotator.get_num_unique_colors() != self.num_classes):
                    ui.notify(f'The first image in the dataset must contain at least one annotation for each class.'
                            f'The number of classes is set to {self.num_classes} and only {self.annotator.get_num_unique_colors()} classes annotated.')
                else:
                    mask_slice = self.annotator.mask
                    slice_data = {'volume' : self.dataset[self.volume_index].filename,
                                'slicer' : self.dataset[self.volume_index].slicer.to_dict()}
                    utils.save_sample(self.image_slice, mask_slice, slice_data, self.num_classes)

                    self.train_samples = glob.glob('data/train/images/*.tiff')

                    self.ui_select_input_size.disable()
                    self.ui_select_num_classes.disable()

                    self.randomize()
                    self.clear()


    def mouse_handler(self, e: events.MouseEventArguments):

        if e.type == 'mousedown' and e.button != 1:
            # 0 is left, 2 is right

            if e.button == 0 and e.shift:
                self.interacting = True

            if not e.alt and not e.ctrl and not e.shift:

                # If right click, set to background color
                if e.button == 2:
                    self.color_idx_prev = self.color_idx
                    self.color_idx = 0

                self.ii.is_drawing = True
                self.ii.mode = 'paint'
                self.annotator.new_path(e.image_x, e.image_y, self.ii.brush_size, self.colors[self.color_idx], mode=self.ii.mode, overlay=self.ii.overlay)

            if not e.alt and e.ctrl and not e.shift:

                if len(self.annotator.overlays) > 0:

                    self.ii.is_drawing = True
                    self.ii.mode = 'capture_overlay'
                    self.annotator.new_path(e.image_x, e.image_y, self.ii.brush_size, self.colors[self.color_idx], mode=self.ii.mode, overlay=self.ii.overlay)
            
        if e.type == 'mousemove':

            # Translate viewer
            if self.interacting and e.shift:
                self.annotator.translate(self.ii.x, self.ii.y, e.image_x, e.image_y)
                self.redraw()

            # Continue current path
            if self.ii.is_drawing:
                self.annotator.continue_path(self.ii.x, self.ii.y, e.image_x, e.image_y, self.ii.brush_size, self.colors[self.color_idx], mode=self.ii.mode, overlay=self.ii.overlay)
            
        if e.type == 'mouseup':

            if e.button == 0:
                self.interacting = False
                self.redraw()
            
            if e.button == 2:
                self.color_idx = self.color_idx_prev

            if self.ii.is_drawing:
                self.ii.is_drawing = False
                self.annotator.apply_current_path()
                self.redraw()
                self.run_suggestor()

        self.ii.x = e.image_x
        self.ii.y = e.image_y
        
        self.redraw_overlay()
        
    def mouse_wheel_handler(self, e: KeyEventArguments):
        
        # Adjust brush size
        if not e.args['shiftKey'] and not e.args['ctrlKey'] and not e.args['altKey']:

            delta_y = e.args['deltaY']

            if delta_y < 0:
                self.ii.brush_size = self.ii.brush_size * 1.1
                self.redraw_overlay()

            elif delta_y > 0:
                self.ii.brush_size = self.ii.brush_size * (1 / 1.1)
                self.redraw_overlay()

        # Zoom in and out
        if e.args['shiftKey'] and not e.args['ctrlKey'] and not e.args['altKey']:

            delta_y = e.args['deltaY']
            mouse_x = e.args['offsetX']
            mouse_y = e.args['offsetY']

            self.interacting = True
            self.updated = False

            if delta_y < 0:
                self.annotator.zoom_in(mouse_x, mouse_y)
                self.redraw()
            
            elif delta_y > 0:
                self.annotator.zoom_out(mouse_x, mouse_y)
                self.redraw()

            self.interacting = False

    def update_cursor_opacity(self, e):
        self.ii.cursor_opacity = e.value / 100
        self.redraw_overlay()

    def update_annotation_opacity(self, e):
        self.ii.annotation_opacity = e.value / 100
        self.redraw()

    def update_overlay_opacity(self, e):
        self.ii.overlay_opacity = e.value / 100
        self.redraw()

    def update_display_info(self):
        if self.ii.overlay_opacity == 0:
            self.ui_display_info.set_content('No overlay displayed')
        else:
            if self.ii.overlay == 'live_suggestions':
                self.ui_display_info.set_content('Displaying live suggestions')
            elif self.ii.overlay == 'model_predictions':
                self.ui_display_info.set_content('Displaying model predictions')

    def cycle_overlay(self):

        keys = np.array(list(self.annotator.overlays.keys())) 
        next_idx = np.argwhere(keys == self.ii.overlay)[0,0] + 1 
        self.ii.overlay = keys[next_idx % len(keys)]

        self.update_display_info()
        self.redraw()

    def toggle_overlay(self):
        if self.ii.overlay_opacity > 0:
            self.ii.overlay_opacity = 0
            self.ui_slider_overlay_opacity.value = int(self.ii.overlay_opacity * 100)
        elif self.ii.overlay_opacity == 0:
            self.ii.overlay_opacity = 0.25
            self.ui_slider_overlay_opacity.value = int(self.ii.overlay_opacity * 100)
        self.update_display_info()
        self.redraw()

    def update_num_classes(self, e):
        self.num_classes = self.ui_select_num_classes.value
        self.color_idx = 1
        self.clear()

    def update_input_size(self, e):
        self.input_size = self.ui_select_input_size.value
        self.image_slice = self.dataset[self.volume_index].get_slice(slice_width=self.input_size, order=1).astype('uint8')
        self.ii.image_features = (self.image_slice / 255).astype('float32')[None,None,:,:]
        self.annotator.set_image(np.repeat(self.image_slice[:,:,None], 3, axis=2))
        self.clear()

    def update_sampling_mode(self, e):
        if self.ui_select_sampling_mode.value == 'Random':
            self.sampling_mode = 'random'
        elif self.ui_select_sampling_mode.value == 'Axially-aligned':
            self.sampling_mode = 'grid'
        self.randomize()

    def update_sampling_axis(self, e):
        if self.ui_select_sampling_axis.value == 'Random':
            self.sampling_axis = 'random'
        elif self.ui_select_sampling_axis.value == 'X':
            self.sampling_axis = 'x'
        elif self.ui_select_sampling_axis.value == 'Y':
            self.sampling_axis = 'y'
        elif self.ui_select_sampling_axis.value == 'Z':
            self.sampling_axis = 'z'
        self.randomize()

    def update_training_plot(self):
        self.metric = self.ui_select_plot_metric.value
        self.fig = utils.get_training_history_figure(self.metric)
        self.ui_plotly_training_plot.figure = self.fig
        self.ui_plotly_training_plot.update()

    def defocus(self):
        self.ui_select_input_size.run_method('blur')
        self.ui_select_num_classes.run_method('blur')
        
        self.ui_select_architecture.run_method('blur')
        self.ui_select_encoder.run_method('blur')
        self.ui_checkbox_pretrained.run_method('blur')

        self.ui_select_lr.run_method('blur')
        self.ui_select_batch_size.run_method('blur')
        self.ui_select_num_epochs.run_method('blur')
        self.ui_select_loss_function.run_method('blur')

        self.ui_select_sampling_mode.run_method('blur')
        self.ui_select_sampling_axis.run_method('blur')

        self.ui_slider_cursor_opacity.run_method('blur')
        self.ui_slider_annotation_opacity.run_method('blur')
        self.ui_slider_overlay_opacity.run_method('blur')

        self.ui_button_clear_model.run_method('blur')
        self.ui_button_clear_annotations.run_method('blur')
        self.ui_button_reset_all.run_method('blur')

        self.ui_button_predict.run_method('blur')

        # self.ui_button_build_annotation_volumes.run_method('blur')
        self.ui_button_train.run_method('blur')
        self.ui_select_plot_metric.run_method('blur')
        self.ui_plotly_training_plot.run_method('blur')

    async def clear_annotations(self):

        self.confirmation_label.text = "This will remove all saved annotations. Are you sure you want to do this?"

        result = await self.dialog
        if result == 'Yes':
            utils.clear_annotations()

        self.train_samples = glob.glob('data/train/images/*.tiff')
        self.ui_select_input_size.enable()
        self.ui_select_num_classes.enable()

    async def clear_model(self):

        self.confirmation_label.text = "This will reset the model weights and erase all training progress. Are you sure you want to do this?"
        
        result = await self.dialog
        if result == 'Yes':
            utils.clear_model()

        self.ui_select_architecture.enable()
        self.ui_select_encoder.enable()
        self.ui_checkbox_pretrained.enable()

    async def reset_all(self):

        self.confirmation_label.text = "This will erase all training progress and delete all saved annotations. Are you sure you want to do this?"
        
        result = await self.dialog
        if result == 'Yes':
            utils.reset_all()

        self.train_samples = glob.glob('data/train/images/*.tiff')
        self.ui_select_input_size.enable()
        self.ui_select_num_classes.enable()
        self.ui_select_architecture.enable()
        self.ui_select_encoder.enable()
        self.ui_checkbox_pretrained.enable()

    def build_annotation_volumes(self):
        utils.build_annotation_volumes(dataset)
        ui.notify("Finished rebuilding annotation volumes.")

    async def train_model(self):

        # Not necessary, 3D U-Net models not implemented yet
        # utils.build_annotation_volumes(dataset)

        kwargs = {'lr' : self.ui_select_lr.value,
                'batch_size' : self.ui_select_batch_size.value,
                'epochs' : self.ui_select_num_epochs.value,
                'num_channels' : 1,
                'num_classes' : self.num_classes,
                'loss_function_name' : self.ui_select_loss_function.value,
                'architecture' : self.ui_select_architecture.value,
                'encoder_name' : self.ui_select_encoder.value,
                'pretrained' : self.ui_checkbox_pretrained.value}

        with open('model/model_details.pkl', 'wb') as f:
            pickle.dump(kwargs, f)

        self.ui_select_architecture.disable()
        self.ui_select_encoder.disable()
        self.ui_checkbox_pretrained.disable()

        self.training = True

        self.ui_button_train.disable()
        self.ui_button_predict_volumes.disable()

        result = await run.cpu_bound(trainer.train_model, *list(kwargs.values()))

        self.ui_button_train.enable()
        self.ui_button_predict_volumes.enable()

        self.training = False

    def predict_slice_function(self):

        self.annotator.overlays['model_predictions'] = predict.predict_slice(self.image_slice, num_classes=self.num_classes)
        self.ii.overlay = 'model_predictions'
        self.update_display_info()

        self.ii.overlay_opacity = 0.25
        self.ui_slider_overlay_opacity.value = int(self.ii.overlay_opacity * 100)

        self.redraw()

    def predict_slice(self):
        predict_slice_thread = threading.Thread(target=self.predict_slice_function)
        predict_slice_thread.start()

    def predict_volumes(self):

        self.ui_button_train.disable()
        self.ui_button_predict_volumes.disable()

        predict.predict_volumes(input_size=self.input_size, num_classes=self.num_classes)

        self.ui_button_train.enable()
        self.ui_button_predict_volumes.enable()

        self.predicting = False

    def suggestor_function(self):

        self.suggesting = True

        if self.ii.suggestor_model is None:
            suggestions, suggestor_model = suggestor.make_suggestions(self.ii.image_features, self.annotator.mask)
        else:
            suggestions, suggestor_model = suggestor.make_suggestions(self.ii.image_features, self.annotator.mask, model=self.ii.suggestor_model)

        if suggestions is not None:

            self.annotator.overlays['live_suggestions'] = suggestions
            self.ii.overlay = 'live_suggestions'
            self.update_display_info()

            self.ii.suggestor_model = suggestor_model

            self.redraw()

        self.suggesting = False

    def run_suggestor(self):

        if not self.suggesting:        
            suggestor_thread = threading.Thread(target=self.suggestor_function)
            suggestor_thread.start()

    def check_volume_folder(self):

        volume_files = np.sort(glob.glob('data/image_volumes/*.zarr'))

        self.ui_volume_count.content = f'Volumes: {len(volume_files)}'
        self.ui_sample_count.content = f'Samples: {len(self.train_samples)}'

        if len(self.dataset) != len(volume_files):
            self.dataset = utils.load_dataset()
            self.randomize()