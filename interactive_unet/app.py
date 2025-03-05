import os
import cv2
import glob
import time
import pickle
import threading
import numpy as np
from PIL import Image
from skimage import io

import plotly.graph_objects as go

from nicegui import ui, events
from nicegui.events import KeyEventArguments

import segmentation_models_pytorch as smp

from interactive_unet.slicer import Slicer
from interactive_unet.annotator import Annotator
from interactive_unet import utils, trainer, predict

utils.create_directories()

volumes = []
slicers = []

volume_files = np.sort(glob.glob('data/image_volumes/*.npy'))
volume_index = 0

if len(volume_files) > 0:
    for f in volume_files:
        volumes.append(np.load(f))
        slicers.append(Slicer(volumes[-1].shape))
    volume_index = np.random.randint(len(volumes))

train_samples = glob.glob('data/train/masks/*.tiff')  
val_samples = glob.glob('data/val/masks/*.tiff')


colors = ['rgba(230, 25, 75, 1)', 'rgba(60, 180, 75, 1)', 'rgba(255, 225, 25, 1)', 'rgba(0, 130, 200, 1)', 'rgba(245, 130, 48, 1)',
        'rgba(145, 30, 180, 1)', 'rgba(70, 240, 240, 1)', 'rgba(240, 50, 230, 1)', 'rgba(210, 245, 60, 1)', 'rgba(170, 255, 195, 1)']
color_idx = 1
color_idx_prev = 1

canvas_size = 700

metric = 'Loss'
fig = utils.get_training_history_figure(metric)

sampling_mode = 'random'
sampling_axis = 'x'

num_classes = 2
input_size = 512
if len(train_samples) > 0:
    mask = io.imread(train_samples[0])
    input_size = mask.shape[0]
    num_classes = np.unique(mask.reshape(-1, mask.shape[-1]), axis=0).shape[0] - 1

annotator = Annotator(canvas_size)

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
    global volume_index, slicers, image_slice, image_slice_with_overlay

    if len(volumes) == 0:
        image_slice = np.zeros((input_size,input_size), dtype='uint8')
    else:
        volume_index = np.random.randint(len(volumes))
        slicers[volume_index].randomize(sampling_mode=sampling_mode, sampling_axis=sampling_axis)

        image_slice = slicers[volume_index].get_slice(volumes[volume_index], slice_width=input_size, order=1).astype('uint8')
    image_slice_with_overlay = np.repeat(image_slice[:,:,None], 3, axis=2)
    
    clear()

def redraw_check():
    global updated  
    
    if interacting is False and updated is False:
        updated = True
        redraw()

def redraw():
    if interacting: 
        # Fast redraw
        ii.source = Image.fromarray(cv2.resize(annotator.get_roi_image(image_slice_with_overlay, size=100), (700, 700), interpolation=cv2.INTER_NEAREST))
    else:
        ii.source = Image.fromarray(annotator.get_roi_image(image_slice_with_overlay))

    redraw_overlay()

def redraw_overlay():
    mask = annotator.get_mask_overlay()
    ii.cursor = f'<circle cx="{ii.x}" cy="{ii.y}" r="{ii.brush_size/2}" fill="{colors[color_idx]}" stroke="{colors[color_idx]}" opacity="{ii.cursor_opacity}" />'
    ii.content = f'<g opacity="{ii.annotation_opacity}"> {mask} </g> {ii.cursor}'

def redraw_prediction():
    global image_slice_with_overlay
    if prediction is not None:
        image_slice_with_overlay = np.repeat(image_slice[:,:,None], 3, axis=2) / 255
        image_slice_with_overlay = (image_slice_with_overlay * (1 - ii.prediction_opacity) + prediction * ii.prediction_opacity)
        image_slice_with_overlay = np.round(255 * image_slice_with_overlay).astype('uint8')
    else:
        image_slice_with_overlay = np.repeat(image_slice[:,:,None], 3, axis=2)
    
def clear():
    global annotator, color_idx, interacting, updated, prediction

    annotator.reset()
    color_idx = 1
    interacting = False
    updated = False

    ii.x = 0
    ii.y = 0
    ii.is_drawing = False
    ii.brush_size = 40
    ii.cursor_opacity = 0.25
    ii.annotation_opacity = 0.25
    ii.prediction_opacity = 0.25

    ui_slider_cursor_opacity.value = int(ii.cursor_opacity * 100)
    ui_slider_annotation_opacity.value = int(ii.annotation_opacity * 100)
    ui_slider_prediction_opacity.value = int(ii.prediction_opacity * 100)

    prediction = None
    ui_button_predict.text = 'Predict'

    redraw_prediction()
    redraw()

def key_handler(e: KeyEventArguments):
    global color_idx, train_samples, val_samples
    
    if e.action.keydown and not e.action.repeat:

        if e.key == 'Space':
            randomize()

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

        # Toggle prediction overlay
        if e.key == 'd':
            toggle_prediction_overlay()

    if e.modifiers.ctrl and e.action.keydown and not e.action.repeat:

        if e.key == 'z':
            annotator.undo_annotation()
            redraw_overlay()

        if e.key == 'y':
            annotator.redo_annotation()
            redraw_overlay()

        if e.key == 's':
            if (len(train_samples) == 0) and (annotator.get_num_unique_colors() != num_classes):
                ui.notify(f'The first image in the dataset must contain at least one annotation for each class.'
                        f'The number of classes is set to {num_classes} and only {annotator.get_num_unique_colors()} classes annotated.')
            else:

                mask_slice = utils.get_mask(annotator.annotations, input_size)
                volume_name = str(volume_files[volume_index].split('image_volumes/')[-1])
                slice_data = np.array([volume_name, slicers[volume_index].to_dict()])
                utils.save_sample(image_slice, mask_slice, slice_data)

                train_samples = glob.glob('data/train/images/*.tiff')  
                val_samples = glob.glob('data/val/images/*.tiff')

                ui_select_input_size.disable()
                ui_select_num_classes.disable()

                randomize()


def mouse_handler(e: events.MouseEventArguments):
    global annotator, interacting, color_idx, color_idx_prev

    if e.type == 'mousedown':
        # 0 is left, 2 is right

        if e.button == 0 and e.shift:
            interacting = True

        if not e.alt and not e.ctrl and not e.shift:

            # If right click, set to background color
            if e.button == 2:
                color_idx_prev = color_idx
                color_idx = 0

            ii.is_drawing = True
            annotator.new_path(e.image_x, e.image_y, ii.brush_size, colors[color_idx])
        
    if e.type == 'mousemove':

        # Translate viewer
        if interacting and e.shift:
            annotator.translate(ii.x, ii.y, e.image_x, e.image_y)
            redraw()

        # Continue current path
        if ii.is_drawing:
            annotator.continue_path(ii.x, ii.y, e.image_x, e.image_y, ii.brush_size, colors[color_idx])
        
    if e.type == 'mouseup':

        if e.button == 0:
            interacting = False
            redraw()
        
        if e.button == 2:
            color_idx = color_idx_prev

        ii.is_drawing = False

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
    ii.cursor_opacity = e.value / 100
    redraw_overlay()

def update_annotation_opacity(e):
    ii.annotation_opacity = e.value / 100
    redraw_overlay()

def update_prediction_opacity(e):
    ii.prediction_opacity = e.value / 100
    redraw_prediction()
    redraw()

def toggle_prediction_overlay():
    if ii.prediction_opacity > 0:
        ii.prediction_opacity = 0
        ui_slider_prediction_opacity.value = int(ii.prediction_opacity * 100)
    elif ii.prediction_opacity == 0:
        ii.prediction_opacity = 0.25
        ui_slider_prediction_opacity.value = int(ii.prediction_opacity * 100)

def update_num_classes(e):
    global num_classes, color_idx
    num_classes = ui_select_num_classes.value
    color_idx = 1
    clear()

def update_input_size(e):
    global input_size, image_slice, image_slice_with_overlay
    input_size = ui_select_input_size.value
    image_slice = slicers[volume_index].get_slice(volumes[volume_index], slice_width=input_size, order=1).astype('uint8')
    image_slice_with_overlay = np.repeat(image_slice[:,:,None], 3, axis=2)
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
    ui_slider_prediction_opacity.run_method('blur')

    ui_button_clear_model.run_method('blur')
    ui_button_clear_annotations.run_method('blur')
    ui_button_reset_all.run_method('blur')

    ui_button_predict.run_method('blur')

    ui_button_train.run_method('blur')
    ui_select_plot_metric.run_method('blur')
    ui_plotly_training_plot.run_method('blur')

async def clear_annotations(e):
    global train_samples, val_samples

    confirmation_label.text = "This will remove all saved annotations. Are you sure you want to do this?"

    result = await dialog
    if result == 'Yes':
        utils.clear_annotations()

    train_samples = glob.glob('data/train/images/*.tiff')  
    val_samples = glob.glob('data/val/images/*.tiff')
    ui_select_input_size.enable()
    ui_select_num_classes.enable()

async def clear_model():

    confirmation_label.text = "This will reset the model weights and erase all training progress. Are you sure you want to do this?"
    
    result = await dialog
    if result == 'Yes':
        utils.clear_model()

    ui_select_encoder.enable()
    ui_checkbox_pretrained.enable()

async def reset_all():
    global train_samples, val_samples

    confirmation_label.text = "This will erase all training progress and delete all saved annotations. Are you sure you want to do this?"
    
    result = await dialog
    if result == 'Yes':
        utils.reset_all()

    train_samples = glob.glob('data/train/images/*.tiff')  
    val_samples = glob.glob('data/val/images/*.tiff')
    ui_select_input_size.enable()
    ui_select_num_classes.enable()
    ui_select_encoder.enable()
    ui_checkbox_pretrained.enable()

def train_model():
    kwargs = {'lr' : ui_select_lr.value,
            'batch_size' : ui_select_batch_size.value,
            'epochs' : ui_select_num_epochs.value,
            'num_classes' : num_classes,
            'loss_function_name' : ui_select_loss_function.value,
            'encoder_name' : ui_select_encoder.value,
            'pretrained' : ui_checkbox_pretrained.value}

    with open('model/model_details.pkl', 'wb') as f:
        pickle.dump(kwargs, f)

    ui_select_encoder.disable()
    ui_checkbox_pretrained.disable()

    training_thread = threading.Thread(target=trainer.train_model, args=(), kwargs=kwargs)
    training_thread.start()


def predict_slice():
    global prediction
    prediction = predict.predict_slice(image_slice, num_classes=num_classes) / 255

    ii.prediction_opacity = 0.25
    ui_slider_prediction_opacity.value = int(ii.prediction_opacity * 100)

    redraw_prediction()
    redraw()

def predict_volumes():
    predict.predict_volumes(input_size=input_size, num_classes=num_classes)

def check_volume_folder():
    global volume_files, volumes, slicers

    old_file_count = len(volume_files)
    volume_files = np.sort(glob.glob('data/image_volumes/*.npy'))
    new_file_count = len(volume_files)

    ui_volume_count.content = f'Volumes: {len(volume_files)}'
    ui_sample_count.content = f'Samples: {len(train_samples)}'

    if new_file_count != old_file_count:
        volumes = []
        slicers = []
        
        if new_file_count > 0:
            volumes = []
            slicers = []
            for f in volume_files:
                volumes.append(np.load(f))
                slicers.append(Slicer(volumes[-1].shape))
            volume_index = np.random.randint(len(volumes))
            randomize()
        if new_file_count == 0:
            randomize()

ui.page_title('Interactive Segmentation')
with ui.column(align_items='center').classes('w-full justify-center'):

    ui.markdown("#### **Interactive Segmentation Tool**")

    with ui.row().classes('w-full justify-center'):

        with ui.column().classes('w-1/4'):

            with ui.scroll_area().classes('w-full h-[calc(100vh-8rem)] justify-center no-wrap'):
                
                with ui.row(align_items='center').classes('bg-gray-100 w-full justify-center no-wrap'):
                    with ui.element('div').classes('justify-center'):
                        ui_volume_count = ui.markdown(f'Volumes: {len(volume_files)}')

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

                    with ui.row(align_items='center').classes('w-full justify-center no-wrap'):

                        ui_select_encoder = ui.select(smp.encoders.get_encoder_names(), 
                                                    value='timm-mobilenetv3_large_100', 
                                                    label='U-Net Encoder').props('filled').classes('w-3/4')
                        ui_checkbox_pretrained = ui.checkbox('Pretrained',
                                                            value=True).classes('w-1/4')

                        if os.path.isfile('model/model.ckpt'):
                            
                            with open('model/model_details.pkl', 'rb') as f:
                                model_details = pickle.load(f)
                                ui_select_encoder.value = model_details['encoder_name']
                                ui_checkbox_pretrained.value = model_details['pretrained']

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
                                                    value=50,
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
                        
                        ui.markdown('Prediction opacity').classes('w-1/2')
                        ui_slider_prediction_opacity = ui.slider(min=0, max=100, value=25, on_change=update_prediction_opacity).classes('w-1/2')

                with ui.expansion(text='Project settings').props('dense').classes('w-full'):

                    with ui.row(align_items='center').classes('w-full justify-center no-wrap'):

                        ui_button_clear_model = ui.button('Reset model weights', on_click=clear_model).props('filled').classes('w-1/2')
                        ui_button_clear_annotations = ui.button('Clear annotations', on_click=clear_annotations).props('filled').classes('w-1/2')
                        
                    ui_button_reset_all = ui.button('Reset and clear all', on_click=reset_all).props('filled').classes('w-full')

        with ui.column():

            ii = ui.interactive_image(on_mouse=mouse_handler, size=(canvas_size, canvas_size),
                                    events=['mousedown', 'mousemove', 'mouseup'])
            ii.on('wheel', mouse_wheel_handler)

            ui_button_predict = ui.button('Predict', on_click=predict_slice).props('filled').classes('w-full')

        with ui.column().classes('w-1/4'):

            ui_button_train = ui.button('Train', on_click=train_model).props('filled').classes('w-full')

            ui_select_plot_metric = ui.select(['Loss', 'Dice', 'IoU', 'MCC'],
                                            value='Loss',
                                            label='Plot metric',
                                            on_change=update_training_plot).props('filled').classes('w-full')

            ui_plotly_training_plot = ui.plotly(fig).classes('w-full h-96')

            ui_button_predict_volumes = ui.button('Predict volumes', on_click=predict_volumes).props('filled').classes('w-full')


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

ui.run(reload=False, port=45321)
