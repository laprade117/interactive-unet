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

import torch

from interactive_unet.slicer import Slicer
from interactive_unet.annotator import Annotator
from interactive_unet import utils, trainer, predict, suggestor

# Creates initial directory structure if note already created
utils.create_directories()

# Load data
dataset = utils.load_dataset()
if len(dataset) > 0:
    volume_index = np.random.randint(len(dataset))

train_samples = glob.glob('data/train/images/*.tiff')

# Data parameters
num_classes = utils.get_num_classes()
input_size = utils.get_input_size()

# Color parameters
colors = ['rgba(230, 25, 75, 1)', 'rgba(60, 180, 75, 1)', 'rgba(255, 225, 25, 1)', 'rgba(0, 130, 200, 1)', 'rgba(245, 130, 48, 1)',
        'rgba(145, 30, 180, 1)', 'rgba(70, 240, 240, 1)', 'rgba(240, 50, 230, 1)', 'rgba(210, 245, 60, 1)', 'rgba(170, 255, 195, 1)']
color_idx = 1
color_idx_prev = 1

# Plotly figure parameters
metric = 'Loss'
fig = utils.get_training_history_figure(metric)

# Sampling parameters
sampling_mode = 'random'
sampling_axis = 'x'

# UI/Canvas parameters
canvas_size = 700
annotator = Annotator(canvas_size)

training = False
predicting = False
extracting = False
suggesting = False
interacting = False
updated = True
last_interaction = time.time()

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

def randomize():
    global annotator, dataset, volume_index, image_slice

    if len(dataset) == 0:
        # Create blank slice if no volumes are provided
        image_slice = np.zeros((input_size,input_size), dtype='uint8')
    else:
        # Choose random volume and get random slice
        volume_index = np.random.randint(len(dataset))
        dataset[volume_index].randomize(sampling_mode=sampling_mode, sampling_axis=sampling_axis)
        image_slice = dataset[volume_index].get_slice(slice_width=input_size, order=1).astype('uint8')

    annotator.set_image(np.repeat(image_slice[:,:,None], 3, axis=2))
    
    clear()

def redraw_check():
    global updated  
    
    if interacting is False and updated is False:
        updated = True
        redraw()

def redraw():
    
    annotator.update_display(ii.annotation_opacity, ii.overlay_opacity, overlay=ii.overlay)

    if interacting: 
        # Fast redraw
        ii.source = Image.fromarray(cv2.resize(annotator.get_roi_image(size=60), (canvas_size, canvas_size), interpolation=cv2.INTER_NEAREST))
    else:
        ii.source = Image.fromarray(annotator.get_roi_image())

    redraw_overlay()

def redraw_overlay():

    mask = ''

    if ii.is_drawing:
        mask = annotator.get_current_path_overlay()

    cursor = f'<circle cx="{ii.x}" cy="{ii.y}" r="{ii.brush_size/2}" fill="{colors[color_idx]}" stroke="{colors[color_idx]}" opacity="{ii.cursor_opacity}" />'
    ii.content = f'<g opacity="{ii.annotation_opacity}"> {mask} </g> {cursor}'
    
def clear():
    global annotator, color_idx, interacting, updated

    annotator.reset()
    color_idx = 1
    interacting = False
    updated = False

    ii.x = 0
    ii.y = 0
    ii.is_drawing = False
    ii.mode = 'paint'
    ii.overlay = None
    ii.image_features = {'features': None, # Average features used for training.
                         'features_list': []} # Contains features from multiple scales
    ii.suggestor_model = None
    ii.brush_size = 40

    ii.cursor_opacity = 0.25
    ii.annotation_opacity = 0.25
    ii.overlay_opacity = 0.25

    ui_slider_cursor_opacity.value = int(ii.cursor_opacity * 100)
    ui_slider_annotation_opacity.value = int(ii.annotation_opacity * 100)
    ui_slider_overlay_opacity.value = int(ii.overlay_opacity * 100)

    ui_button_predict.text = 'Predict'

    redraw()

def key_handler(e: KeyEventArguments):
    global annotator, dataset, train_samples, color_idx, image_slice
    
    if e.action.keydown and not e.action.repeat:

        # Random slice
        if e.key == 'Space':
            randomize()

        # Next slice in stack
        if e.key == 'q':
            dataset[volume_index].shift_origin(shift_amount=[1,0,0])
            image_slice = dataset[volume_index].get_slice(slice_width=input_size, order=1).astype('uint8')
            annotator.set_image(np.repeat(image_slice[:,:,None], 3, axis=2))    
            ii.image_features = {'features': None,
                                 'features_list': []}
            ii.suggestor_model = None
            redraw()

        # Previous slice in stack
        if e.key == 'a':
            dataset[volume_index].shift_origin(shift_amount=[-1,0,0])
            image_slice = dataset[volume_index].get_slice(slice_width=input_size, order=1).astype('uint8')
            annotator.set_image(np.repeat(image_slice[:,:,None], 3, axis=2))
            ii.image_features = {'features': None,
                                 'features_list': []}
            ii.suggestor_model = None   
            redraw()

        # Next class/color
        if e.key == 'c':
            color_idx += 1
            if color_idx == num_classes:
                color_idx = 1
            redraw_overlay()

        # Previous class/color
        if e.key == 'v':
            color_idx -= 1
            if color_idx == 0:
                color_idx = num_classes - 1
            redraw_overlay()

        # Toggle overlay
        if e.key == 'd':
            toggle_overlay()
        
        if e.key == 'f':
            cycle_overlay()

    if e.modifiers.ctrl and e.action.keydown and not e.action.repeat:

        if e.key == 'z':
            annotator.undo_annotation()
            redraw()

        if e.key == 'y':
            annotator.redo_annotation()
            redraw()

        if e.key == 's':
            if (len(train_samples) == 0) and (annotator.get_num_unique_colors() != num_classes):
                ui.notify(f'The first image in the dataset must contain at least one annotation for each class.'
                          f'The number of classes is set to {num_classes} and only {annotator.get_num_unique_colors()} classes annotated.')
            else:
                mask_slice = annotator.mask
                slice_data = {'volume' : dataset[volume_index].filename,
                              'slicer' : dataset[volume_index].slicer.to_dict()}
                utils.save_sample(image_slice, mask_slice, slice_data, num_classes)

                train_samples = glob.glob('data/train/images/*.tiff')

                ui_select_input_size.disable()
                ui_select_num_classes.disable()

                randomize()
                clear()


def mouse_handler(e: events.MouseEventArguments):
    global annotator, interacting, color_idx, color_idx_prev

    if e.type == 'mousedown' and e.button != 1:
        # 0 is left, 2 is right

        if e.button == 0 and e.shift:
            interacting = True

        if not e.alt and not e.ctrl and not e.shift:

            # If right click, set to background color
            if e.button == 2:
                color_idx_prev = color_idx
                color_idx = 0

            ii.is_drawing = True
            ii.mode = 'paint'
            annotator.new_path(e.image_x, e.image_y, ii.brush_size, colors[color_idx], mode=ii.mode, overlay=ii.overlay)

        if not e.alt and e.ctrl and not e.shift:

            if len(annotator.overlays) > 0:

                ii.is_drawing = True
                ii.mode = 'capture_overlay'
                annotator.new_path(e.image_x, e.image_y, ii.brush_size, colors[color_idx], mode=ii.mode, overlay=ii.overlay)
        
    if e.type == 'mousemove':

        # Translate viewer
        if interacting and e.shift:
            annotator.translate(ii.x, ii.y, e.image_x, e.image_y)
            redraw()

        # Continue current path
        if ii.is_drawing:
            annotator.continue_path(ii.x, ii.y, e.image_x, e.image_y, ii.brush_size, colors[color_idx], mode=ii.mode, overlay=ii.overlay)
        
    if e.type == 'mouseup':

        if e.button == 0:
            interacting = False
            redraw()
        
        if e.button == 2:
            color_idx = color_idx_prev

        if ii.is_drawing:
            ii.is_drawing = False
            annotator.apply_current_path()
            redraw()
            run_suggestor()

    ii.x = e.image_x
    ii.y = e.image_y
    
    redraw_overlay()
    
def mouse_wheel_handler(e: KeyEventArguments):
    global annotator, interacting, updated
    
    # Adjust brush size
    if not e.args['shiftKey'] and not e.args['ctrlKey'] and not e.args['altKey']:

        delta_y = e.args['deltaY']

        if delta_y < 0:
            ii.brush_size = ii.brush_size * 1.1
            redraw_overlay()

        elif delta_y > 0:
            ii.brush_size = ii.brush_size * (1 / 1.1)
            redraw_overlay()

    # Zoom in and out
    if e.args['shiftKey'] and not e.args['ctrlKey'] and not e.args['altKey']:

        delta_y = e.args['deltaY']
        mouse_x = e.args['offsetX']
        mouse_y = e.args['offsetY']

        interacting = True
        updated = False

        if delta_y < 0:
            annotator.zoom_in(mouse_x, mouse_y)
            redraw()
        
        elif delta_y > 0:
            annotator.zoom_out(mouse_x, mouse_y)
            redraw()

        interacting = False

def update_cursor_opacity(e):
    global annotator
    ii.cursor_opacity = e.value / 100
    redraw_overlay()

def update_annotation_opacity(e):
    global annotator
    ii.annotation_opacity = e.value / 100
    redraw()

def update_overlay_opacity(e):
    global annotator
    ii.overlay_opacity = e.value / 100
    redraw()

def update_display_info():
    if ii.overlay_opacity == 0:
        ui_display_info.set_content('No overlay displayed')
    else:
        if ii.overlay == 'live_suggestions':
            ui_display_info.set_content('Displaying live suggestions')
        elif ii.overlay == 'model_predictions':
            ui_display_info.set_content('Displaying model predictions')

def cycle_overlay():
    global annotator

    keys = np.array(list(annotator.overlays.keys())) 
    next_idx = np.argwhere(keys == ii.overlay)[0,0] + 1 
    ii.overlay = keys[next_idx % len(keys)]

    update_display_info()
    redraw()

def toggle_overlay():
    global annotator
    if ii.overlay_opacity > 0:
        ii.overlay_opacity = 0
        ui_slider_overlay_opacity.value = int(ii.overlay_opacity * 100)
    elif ii.overlay_opacity == 0:
        ii.overlay_opacity = 0.25
        ui_slider_overlay_opacity.value = int(ii.overlay_opacity * 100)
    update_display_info()
    redraw()

def update_num_classes(e):
    global num_classes, color_idx
    num_classes = ui_select_num_classes.value
    color_idx = 1
    clear()

def update_input_size(e):
    global annotator, input_size, image_slice
    input_size = ui_select_input_size.value
    image_slice = dataset[volume_index].get_slice(slice_width=input_size, order=1).astype('uint8')
    annotator.set_image(np.repeat(image_slice[:,:,None], 3, axis=2))
    clear()

def update_sampling_mode(e):
    global sampling_mode
    if ui_select_sampling_mode.value == 'Random':
        sampling_mode = 'random'
    elif ui_select_sampling_mode.value == 'Axially-aligned':
        sampling_mode = 'grid'
    randomize()

def update_sampling_axis(e):
    global sampling_axis
    if ui_select_sampling_axis.value == 'Random':
        sampling_axis = 'random'
    elif ui_select_sampling_axis.value == 'X':
        sampling_axis = 'x'
    elif ui_select_sampling_axis.value == 'Y':
        sampling_axis = 'y'
    elif ui_select_sampling_axis.value == 'Z':
        sampling_axis = 'z'
    randomize()

def update_training_plot():
    global metric, fig
    metric = ui_select_plot_metric.value
    fig = utils.get_training_history_figure(metric)
    ui_plotly_training_plot.figure = fig
    ui_plotly_training_plot.update()

def defocus():
    ui_select_input_size.run_method('blur')
    ui_select_num_classes.run_method('blur')
    
    ui_select_architecture.run_method('blur')
    ui_select_encoder.run_method('blur')
    ui_checkbox_pretrained.run_method('blur')

    ui_select_lr.run_method('blur')
    ui_select_batch_size.run_method('blur')
    ui_select_num_epochs.run_method('blur')
    ui_select_loss_function.run_method('blur')

    ui_select_sampling_mode.run_method('blur')
    ui_select_sampling_axis.run_method('blur')

    ui_slider_cursor_opacity.run_method('blur')
    ui_slider_annotation_opacity.run_method('blur')
    ui_slider_overlay_opacity.run_method('blur')

    ui_button_clear_model.run_method('blur')
    ui_button_clear_annotations.run_method('blur')
    ui_button_reset_all.run_method('blur')

    ui_button_predict.run_method('blur')

    ui_button_build_annotation_volumes.run_method('blur')
    ui_button_train.run_method('blur')
    ui_select_plot_metric.run_method('blur')
    ui_plotly_training_plot.run_method('blur')

async def clear_annotations(e):
    global train_samples

    confirmation_label.text = "This will remove all saved annotations. Are you sure you want to do this?"

    result = await dialog
    if result == 'Yes':
        utils.clear_annotations()

    train_samples = glob.glob('data/train/images/*.tiff')
    ui_select_input_size.enable()
    ui_select_num_classes.enable()

async def clear_model():

    confirmation_label.text = "This will reset the model weights and erase all training progress. Are you sure you want to do this?"
    
    result = await dialog
    if result == 'Yes':
        utils.clear_model()

    ui_select_architecture.enable()
    ui_select_encoder.enable()
    ui_checkbox_pretrained.enable()

async def reset_all():
    global train_samples

    confirmation_label.text = "This will erase all training progress and delete all saved annotations. Are you sure you want to do this?"
    
    result = await dialog
    if result == 'Yes':
        utils.reset_all()

    train_samples = glob.glob('data/train/images/*.tiff')
    ui_select_input_size.enable()
    ui_select_num_classes.enable()
    ui_select_architecture.enable()
    ui_select_encoder.enable()
    ui_checkbox_pretrained.enable()

def build_annotation_volumes():
    utils.build_annotation_volumes(dataset)
    ui.notify("Finished rebuilding annotation volumes.")

# def train_model():

#     # Not necessary, 3D U-Net models not implemented yet
#     # utils.build_annotation_volumes(dataset)

#     kwargs = {'lr' : ui_select_lr.value,
#               'batch_size' : ui_select_batch_size.value,
#               'epochs' : ui_select_num_epochs.value,
#               'num_channels' : 1,
#               'num_classes' : num_classes,
#               'loss_function_name' : ui_select_loss_function.value,
#               'architecture' : ui_select_architecture.value,
#               'encoder_name' : ui_select_encoder.value,
#               'pretrained' : ui_checkbox_pretrained.value}

#     with open('model/model_details.pkl', 'wb') as f:
#         pickle.dump(kwargs, f)

#     ui_select_architecture.disable()
#     ui_select_encoder.disable()
#     ui_checkbox_pretrained.disable()

#     training_thread = threading.Thread(target=trainer.train_model, args=(), kwargs=kwargs)
#     training_thread.start()

async def train_model():
    global training

    # Not necessary, 3D U-Net models not implemented yet
    # utils.build_annotation_volumes(dataset)

    kwargs = {'lr' : ui_select_lr.value,
              'batch_size' : ui_select_batch_size.value,
              'epochs' : ui_select_num_epochs.value,
              'num_channels' : 1,
              'num_classes' : num_classes,
              'loss_function_name' : ui_select_loss_function.value,
              'architecture' : ui_select_architecture.value,
              'encoder_name' : ui_select_encoder.value,
              'pretrained' : ui_checkbox_pretrained.value}

    with open('model/model_details.pkl', 'wb') as f:
        pickle.dump(kwargs, f)

    ui_select_architecture.disable()
    ui_select_encoder.disable()
    ui_checkbox_pretrained.disable()

    training = True

    ui_button_train.disable()
    ui_button_predict_volumes.disable()

    result = await run.cpu_bound(trainer.train_model, *list(kwargs.values()))

    ui_button_train.enable()
    ui_button_predict_volumes.enable()

    training = False

def predict_slice():

    def predict_slice_function():
        global annotator, predicting

        annotator.overlays['model_predictions'] = predict.predict_slice(image_slice, num_classes=num_classes)
        ii.overlay = 'model_predictions'
        update_display_info()

        ii.overlay_opacity = 0.25
        ui_slider_overlay_opacity.value = int(ii.overlay_opacity * 100)

        redraw()

    predict_slice_thread = threading.Thread(target=predict_slice_function)
    predict_slice_thread.start()

# def predict_volumes():

#     kwargs = {'input_size': input_size,
#               'num_classes' : num_classes}

#     training_thread = threading.Thread(target=predict.predict_volumes, args=(), kwargs=kwargs)
#     training_thread.start()

async def predict_volumes():
    global predicting

    ui_button_train.disable()
    ui_button_predict_volumes.disable()

    result = await run.cpu_bound(predict.predict_volumes, input_size=input_size, num_classes=num_classes)

    ui_button_train.enable()
    ui_button_predict_volumes.enable()

    predicting = False

def extract_features():
    global extracting

    extracting = True

    image = (image_slice / 255.0).astype('float32')

    # torch.set_float32_matmul_precision('medium')
    # extractor = suggestor.FeatureExtractor('dinov2', layers=[8], attn_vector='k', stride=7).eval().cuda()

    # scales = [512,256]
    # feature_dim = 2

    # import time

    # start_time = time.time()

    # for scale in scales:
    #     ii.image_features['features_list'].append(suggestor.get_dense_features(image, extractor, scale=scale))
    #     print(f'Finished scale: {scale}')

    # features = np.mean(ii.image_features['features_list'], 0)
    # features = suggestor.feature_pca(features, feature_dim=feature_dim)
    # features = suggestor.remove_banding(features)

    # # Add original image to feature list
    # features = np.concatenate([image[None,None], features], axis=1)

    # ii.image_features['features'] = features
    
    # print(f'Finished feature extraction in {time.time() - start_time}')

    ii.image_features['features'] = image[None,None]

    extracting = False

    
def run_suggestor():
    global suggesting, extracting

    if ii.image_features['features'] is None:
        if not extracting:
            feature_extractor_thread = threading.Thread(target=extract_features)
            feature_extractor_thread.start()

    if ii.image_features['features'] is not None:
        if not extracting:
            def suggestor_function():
                global annotator, suggesting

                suggesting = True

                if ii.suggestor_model is None:
                    suggestions, suggestor_model = suggestor.make_suggestions(ii.image_features['features'], annotator.mask)
                else:
                    suggestions, suggestor_model = suggestor.make_suggestions(ii.image_features['features'], annotator.mask, model=ii.suggestor_model)

                if suggestions is not None:

                    annotator.overlays['live_suggestions'] = suggestions
                    ii.overlay = 'live_suggestions'
                    update_display_info()

                    ii.suggestor_model = suggestor_model

                    redraw()

                suggesting = False
            
            # suggestor_function()
            if not suggesting:        
                suggestor_thread = threading.Thread(target=suggestor_function)
                suggestor_thread.start()

def check_volume_folder():
    global dataset

    volume_files = np.sort(glob.glob('data/image_volumes/*.npy'))

    ui_volume_count.content = f'Volumes: {len(volume_files)}'
    ui_sample_count.content = f'Samples: {len(train_samples)}'

    if len(dataset) != len(volume_files):
        dataset = utils.load_dataset()
        randomize()

ui.page_title('Interactive Segmentation')
with ui.column(align_items='center').classes('w-full justify-center'):

    ui.markdown("#### **Interactive Segmentation Tool**")

    with ui.row().classes('w-full justify-center'):

        with ui.column().classes('w-1/4'):

            with ui.scroll_area().classes('w-full h-[calc(100vh-8rem)] justify-center no-wrap'):
                
                with ui.row(align_items='center').classes('bg-gray-100 w-full justify-center no-wrap'):
                    with ui.element('div').classes('justify-center'):
                        ui_volume_count = ui.markdown(f'Volumes: {len(dataset)}')

                    with ui.element('div').classes('justify-center'):
                        ui_sample_count = ui.markdown(f'Samples: {len(train_samples)}')
                
                with ui.row(align_items='center').classes('w-full justify-center no-wrap'):

                    ui_select_input_size = ui.select([128,256,384,512],
                                                    value=input_size, label='Input Size',
                                                    on_change=update_input_size).props('filled').classes('w-1/2')
                    ui_select_input_size.on('click', update_input_size)
                    ui_select_num_classes = ui.select([2,3,4,5,6,7,8,9,10],
                                                    value=num_classes, label='Num Classes',
                                                    on_change=update_num_classes).props('filled').classes('w-1/2')
                    ui_select_num_classes.on('click', update_num_classes)

                    if len(train_samples) > 0:
                        ui_select_input_size.disable()
                        ui_select_num_classes.disable()

                with ui.expansion(text='Model settings').props('dense filled').classes('w-full'):

                    ui_select_architecture = ui.select(['U-Net', 'U-Net++', 'FPN',
                                                        'PSPNet', 'DeepLabV3', 'DeepLabV3+',
                                                        'LinkNet', 'MA-Net', 'PAN',
                                                        'UPerNet', 'Segformer'],
                                                       value='U-Net',
                                                       label='Architecture').props('filled').classes('w-full')

                    with ui.row(align_items='center').classes('w-full justify-center no-wrap'):

                        ui_select_encoder = ui.select(smp.encoders.get_encoder_names(), 
                                                    value='mit_b0',
                                                    label='U-Net Encoder').props('filled').classes('w-3/4')
                        ui_checkbox_pretrained = ui.checkbox('Pretrained',
                                                            value=True).classes('w-1/4')

                        if os.path.isfile('model/model.ckpt'):
                            
                            with open('model/model_details.pkl', 'rb') as f:
                                model_details = pickle.load(f)
                                ui_select_architecture.value = model_details['architecture']
                                ui_select_encoder.value = model_details['encoder_name']
                                ui_checkbox_pretrained.value = model_details['pretrained']

                            ui_select_architecture.disable()
                            ui_select_encoder.disable()
                            ui_checkbox_pretrained.disable()
                        
                with ui.expansion(text='Training settings').props('dense').classes('w-full'):

                    ui_select_lr = ui.select([0.00001,0.0001,0.001,0.01], 
                                            value=0.001, 
                                            label='Learning rate').props('filled').classes('w-full')
                    ui_select_batch_size = ui.select([2,4,8,16,32],
                                                    value=2,
                                                    label='Batch size').props('filled').classes('w-full')
                    ui_select_num_epochs = ui.select([10,20,30,40,50,60,70,80,90,100],
                                                    value=100,
                                                    label='Num epochs').props('filled').classes('w-full')
                    ui_select_loss_function = ui.select(['Crossentropy (CE)', 'Dice',
                                                        'Intersection over Union (IoU)', 
                                                        'Matthews correlation coefficient (MCC)',
                                                        'Dice + CE', 'IoU + CE', 'MCC + CE'],
                                                        value='MCC + CE',
                                                        label='Loss function').props('filled').classes('w-full')

                with ui.expansion(text='Sampler settings').props('dense').classes('w-full'):

                    ui_select_sampling_mode = ui.select(['Random', 'Axially-aligned'],
                                                    value='Random',
                                                    label='Sampling mode',
                                                    on_change=update_sampling_mode).props('filled').classes('w-full')
                    ui_select_sampling_axis = ui.select(['Random', 'X', 'Y', 'Z'],
                                                    value='Random',
                                                    label='Sampling axis',
                                                    on_change=update_sampling_axis).props('filled').classes('w-full')
                    ui_select_sampling_axis.bind_visibility_from(ui_select_sampling_mode, 'value', backward=lambda v: v == "Axially-aligned")

                with ui.expansion(text='Display settings').props('dense').classes('w-full'):

                    with ui.row(align_items='center').classes('w-full justify-center no-wrap'):
                        
                        ui.markdown('Cursor opacity').classes('w-1/2')
                        ui_slider_cursor_opacity = ui.slider(min=0, max=100, value=25, on_change=update_cursor_opacity).classes('w-1/2')

                    with ui.row(align_items='center').classes('w-full justify-center no-wrap'):
                        
                        ui.markdown('Annotation opacity').classes('w-1/2')
                        ui_slider_annotation_opacity = ui.slider(min=0, max=100, value=25, on_change=update_annotation_opacity).classes('w-1/2')

                    with ui.row(align_items='center').classes('w-full justify-center no-wrap'):
                        
                        ui.markdown('Overlay opacity').classes('w-1/2')
                        ui_slider_overlay_opacity = ui.slider(min=0, max=100, value=25, on_change=update_overlay_opacity).classes('w-1/2')

                with ui.expansion(text='Project settings').props('dense').classes('w-full'):

                    with ui.row(align_items='center').classes('w-full justify-center no-wrap'):

                        ui_button_clear_model = ui.button('Reset model weights', on_click=clear_model).props('filled').classes('w-1/2')
                        ui_button_clear_annotations = ui.button('Clear annotations', on_click=clear_annotations).props('filled').classes('w-1/2')
                        
                    ui_button_reset_all = ui.button('Reset and clear all', on_click=reset_all).props('filled').classes('w-full')

        with ui.column():

            with ui.row(align_items='center').classes('bg-gray-100 w-full justify-center no-wrap'):
                with ui.element('div').classes('justify-center'):
                    ui_display_info = ui.markdown(f'No overlay displayed')

            ii = ui.interactive_image(on_mouse=mouse_handler, size=(canvas_size, canvas_size),
                                    events=['mousedown', 'mousemove', 'mouseup'])
            ii.on('wheel', mouse_wheel_handler)

            ui_button_predict = ui.button('Predict', on_click=predict_slice).props('filled').classes('w-full')

        with ui.column().classes('w-1/4'):

            ui_select_plot_metric = ui.select(['Loss', 'Dice', 'IoU', 'MCC'],
                                            value='Loss',
                                            label='Plot metric',
                                            on_change=update_training_plot).props('filled').classes('w-full')

            ui_plotly_training_plot = ui.plotly(fig).classes('w-full h-96')

            ui_button_train = ui.button('Train', on_click=train_model).props('filled').classes('w-full')

            ui_button_predict_volumes = ui.button('Predict volumes', on_click=predict_volumes).props('filled').classes('w-full')
            
            with ui.expansion(text='Advanced options').props('dense').classes('w-full'):
                ui_button_build_annotation_volumes = ui.button('Rebuild annotation volumes', on_click=build_annotation_volumes).props('filled').classes('w-full')


volume_timer = ui.timer(2.0, callback=check_volume_folder)
plot_timer = ui.timer(2.0, callback=update_training_plot)
redraw_timer = ui.timer(0.2, callback=redraw_check)
defocus_timer = ui.timer(1.0, callback=defocus)

keyboard = ui.keyboard(on_key=key_handler)

# Confirmation dialog
with ui.dialog() as dialog, ui.card():
    confirmation_label = ui.label('Are you sure?')
    with ui.row():
        ui.button('Yes', on_click=lambda: dialog.submit('Yes'))
        ui.button('No', on_click=lambda: dialog.submit('No'))

randomize()

# ui.run(host='0.0.0.0', port=9546, show=False, reload=False)
ui.run(reload=True)
