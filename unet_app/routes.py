from unet_app import app
from flask import render_template, request, jsonify, session
import os
import glob
import threading
import numpy as np
from skimage import io
from pathlib import Path
from skimage.transform import resize

from unet_app.unet.slicer import Slicer
from unet_app.unet import utils, trainer, predict, plot

# Only variables which are never modified (only viewed) should be used a global variables
# All other variables should be stored in the flask session

# Load data and create volumes
# files = np.sort(glob.glob('./data/sperm/*'))
# image_vol = []
# for i in range(len(files)):
#     image_vol.append(io.imread(files[i]))
# image_vol = np.array(image_vol)
# bounds_vol = np.ones(image_vol.shape)

# vol_index = np.random.randint(len(images))
# session['VolumeIndex'] = np.random.randint(len(images))
# image_vol = images[vol_index]
# bounds_vol = np.ones(image_vol.shape)


# Create necessary directories
utils.create_directories()

# Set up necessary volumes and files
images, image_volume_files, mask_volume_files, weight_volume_files = utils.setup_volumes()

training_thread = None
prediction_thread = None

@app.route('/')
def initialize():
    """
    Initializes basic web interface from index.html
    """
    global images

    # Define some default parameters
    session['InputSize'] = 256
    session['NumChannels'] = 1
    if len(images[0].shape) > 3:
        session['NumChannels'] = images[0].shape[-1]
    session['NumClasses'] = 2
    session['LearningRate'] = 0.0001
    session['BatchSize'] = 2
    session['NumEpochs'] = 20
    session['SamplingMode'] = 'random'
    session["NumTrainSamples"] = 0
    session["NumValSamples"] = 0
    session["CurrentSample"] = "Validation"

    # Load previous parameters if available
    mask_files = glob.glob('data/val/masks/*')
    if len(mask_files) > 0:
        mask = io.imread(mask_files[0])
        session['InputSize'] = mask.shape[0]
        session['NumClasses'] = mask.shape[-1]
        session["NumTrainSamples"] = len(glob.glob('data/train/images/*'))
        session["NumValSamples"]= len(glob.glob('data/val/images/*'))

    if session["NumValSamples"] <= 0.2 * session["NumTrainSamples"]:
        session["CurrentSample"] = 'Validation'
    else:
        session["CurrentSample"] = 'Training'

    return render_template('index.html')

@app.route('/check_parameters', methods=["POST"])
def check_parameters():
    """
    Checks certain parameters and locks them if necessary in the interface.
    """

    mask_files = glob.glob('data/val/masks/*')

    if len(mask_files) > 0:
        mask = io.imread(mask_files[0])
        session['InputSize'] = mask.shape[0]
        session['NumClasses'] = mask.shape[-1]
        lock_parameters = 'true'
    else:
        lock_parameters = 'false'

    return jsonify({'lockParameters': lock_parameters,
                    'inputSize': str(session['InputSize']),
                    'numClasses': str(session['NumClasses']),
                    'numTrainSamples': str(session["NumTrainSamples"]),
                    'numValSamples': str(session["NumValSamples"]),
                    'currentSample': str(session["CurrentSample"])})

@app.route('/shift_origin', methods=["POST"])
def shift_origin():
    global images

    # Get current image volume
    image_vol = images[session['VolumeIndex']]

    # Load slicer state from session
    slicer = Slicer(image_vol.shape)
    slicer.from_dict(session['Slicer'])
    
    # Get shift amounts
    data = request.get_json(force=True)
    x = float(data['x'])
    y = float(data['y'])
    z = float(data['z'])
    
    # Shift the origin position
    slicer.shift_origin([x,y,z])

    # Retrieve slice
    image_slice = slicer.get_slice(image_vol, slice_width=session['InputSize'], order=0).astype('uint8')

    # Store slicer state in session
    session['Slicer'] = slicer.to_dict()

    return jsonify({'imageData': utils.encode_to_base64(image_slice)})

# Sampler settings --------------------------------------------------------------------------------------
@app.route('/update_input_size', methods=["POST"])
def update_input_size():
    """
    Updates the session variable InputSize and also resizes the displayed slice.
    """
    global images

    # Get current image volume
    image_vol = images[session['VolumeIndex']]

    # Load slicer state from session
    slicer = Slicer(image_vol.shape)
    slicer.from_dict(session['Slicer'])

    # Retrieve JSON data
    json_data = request.get_json(force=True)

    # Store InputSize in session variable
    session['InputSize'] = int(json_data['inputSize'])

    # Retrieve slice
    image_slice = slicer.get_slice(image_vol, slice_width=session['InputSize'], order=0).astype('uint8')

    return jsonify({'imageData': utils.encode_to_base64(image_slice)})
    
@app.route('/update_sampling_mode', methods=["POST"])
def update_sampling_mode():
    """
    Updates the session variable Sampling and also refreshes the displayed slice.
    """
    global images

    # Get current image volume
    image_vol = images[session['VolumeIndex']]

    # Load slicer state from session
    slicer = Slicer(image_vol.shape)
    slicer.from_dict(session['Slicer'])

    # Retrieve JSON data
    json_data = request.get_json(force=True)

    # Store AlongGrid in session variable
    session['SamplingMode'] = str(json_data['samplingMode'])

    # Retrieve slice
    slicer.randomize(sampling_mode=session['SamplingMode'])
    image_slice = slicer.get_slice(image_vol, slice_width=session['InputSize'], order=0).astype('uint8')

    # Store slicer state in session
    session['Slicer'] = slicer.to_dict()

    return jsonify({'imageData': utils.encode_to_base64(image_slice)})

@app.route('/clear_data', methods=["POST"])
def clear_data():
    """
    Clears all saved training and validation samples as well as models and model history.
    """

    utils.clear_data()

    images, image_volume_files, mask_volume_files, weight_volume_files = utils.setup_volumes()

    # Update dataset and sample info
    session["NumTrainSamples"] = len(glob.glob('data/train/images/*'))
    session["NumValSamples"]= len(glob.glob('data/val/images/*'))
    if session["NumValSamples"] <= 0.2 * session["NumTrainSamples"]:
        session["CurrentSample"] = 'Validation'
    else:
        session["CurrentSample"] = 'Training'
    
    return jsonify({'numTrainSamples': str(session["NumTrainSamples"]),
                    'numValSamples': str(session["NumValSamples"]),
                    'currentSample': str(session["CurrentSample"])})

@app.route('/predict_volumes', methods=["POST"])
def predict_volumes():
    """
    Predicts the volumes using the 3 grid views.
    """

    global prediction_thread

    session['InputSize']

    kwargs = {'input_size' : session['InputSize'],
              'n_channels' : session['NumChannels'],
              'n_classes' : session['NumClasses']}

    prediction_thread = threading.Thread(target=predict.predict_volumes, args=(), kwargs=kwargs)
    prediction_thread.start()
    
    return jsonify({})

@app.route('/check_prediction_status', methods=["POST"])
def check_prediction_status():
    global prediction_thread

    if prediction_thread.is_alive():
        return jsonify({'predicting': 'true'})
    else:
        prediction_thread = None
        return jsonify({'predicting': 'false'})

# Annotator settings ------------------------------------------------------------------------------------
@app.route('/randomize', methods=["POST"])
def randomize():
    """
    Generate a random sample.
    """
    global images

    # Select a random volume
    session['VolumeIndex'] = utils.get_random_volume_index(fewest_first=False)
    image_vol = images[session['VolumeIndex']]

    # Create slicer object
    slicer = Slicer(image_vol.shape)

    # Extract randomly oriented and positioned slice
    slicer.randomize(sampling_mode=session['SamplingMode'])
    image_slice = slicer.get_slice(image_vol, slice_width=session['InputSize'], order=0).astype('uint8')

    # Store slicer state in session
    session['Slicer'] = slicer.to_dict()

    return jsonify({'imageData': utils.encode_to_base64(image_slice)})

@app.route('/save_sample', methods=["POST"])
def save_sample():
    """
    Saves current sample and annotations.
    """
    global images

    # Load current volume
    image_vol = images[session['VolumeIndex']]

    # Define bounds volume with identical shape as the image_vol
    bounds_vol = np.ones(image_vol.shape)

    # Load slicer information from session
    slicer = Slicer(image_vol.shape)
    slicer.from_dict(session['Slicer'])

    # Retrieve annotation from interface
    base64_image = str(request.values['imageData']).split(',')[1]
    canvas_image = utils.decode_to_numpy(base64_image)[...,:3]
    canvas_image = resize(canvas_image, (session['InputSize'], session['InputSize']), order=0)

    # Convert colored annotation to categorical
    mask_slice, weight_slice = utils.colored_to_categorical(canvas_image, num_classes=session['NumClasses'])

    # Extract the necessary sample information
    image_slice = slicer.get_slice(image_vol, slice_width=session['InputSize'], order=1).astype('uint8')
    bounds_slice = slicer.get_slice(bounds_vol, slice_width=session['InputSize'], order=0).astype('uint8')
    mask_slice = ((mask_slice * bounds_slice[:,:,None]).astype('uint8') * 255).astype('uint8')
    weight_slice = ((weight_slice * bounds_slice).astype('uint8') * 255).astype('uint8')
    slice_data = np.array([session['VolumeIndex'], session['Slicer']])

    # Compute current number of training and validation samples
    n_train_samples = len(glob.glob('data/train/images/*'))
    n_val_samples = len(glob.glob('data/val/images/*'))
    n_samples = n_train_samples + n_val_samples

    # Save the sample
    if n_val_samples <= 0.2*n_train_samples:
        io.imsave(f'data/val/images/{n_samples}.tiff', image_slice)
        io.imsave(f'data/val/masks/{n_samples}.tiff', mask_slice)
        io.imsave(f'data/val/weights/{n_samples}.tiff', weight_slice)
        np.save(f'data/val/slices/{n_samples}.npy', slice_data)
    else:
        io.imsave(f'data/train/images/{n_samples}.tiff', image_slice)
        io.imsave(f'data/train/masks/{n_samples}.tiff', mask_slice)
        io.imsave(f'data/train/weights/{n_samples}.tiff', weight_slice)
        np.save(f'data/train/slices/{n_samples}.npy', slice_data)

    # Update dataset and sample info
    session["NumTrainSamples"] = len(glob.glob('data/train/images/*'))
    session["NumValSamples"]= len(glob.glob('data/val/images/*'))
    if session["NumValSamples"] <= 0.2 * session["NumTrainSamples"]:
        session["CurrentSample"] = 'Validation'
    else:
        session["CurrentSample"] = 'Training'
    
    # Update mask volume
    # mask_vol = np.load(mask_volume_files[session['VolumeIndex']])
    # if mask_vol.shape[-1] != mask_slice.shape[-1]:
    #     mask_vol = np.zeros((image_vol.shape[0],
    #                          image_vol.shape[1],
    #                          image_vol.shape[2],
    #                          mask_slice.shape[-1]), dtype='uint8')
    # mask_vol = slicer.update_volume(mask_slice, mask_vol)
    # np.save(mask_volume_files[session['VolumeIndex']], mask_vol)

    # Update weight volume
    # weight_vol = np.load(weight_volume_files[session['VolumeIndex']])
    # weight_vol = slicer.update_volume(weight_slice, weight_vol)
    # np.save(weight_volume_files[session['VolumeIndex']], weight_vol)

    # Select a random volume
    session['VolumeIndex'] = utils.get_random_volume_index(fewest_first=True)
    image_vol = images[session['VolumeIndex']]

    # Create slicer object
    slicer = Slicer(image_vol.shape)

    # Extract randomly oriented and positioned slice
    slicer.randomize(sampling_mode=session['SamplingMode'])
    image_slice = slicer.get_slice(image_vol, slice_width=session['InputSize'], order=0).astype('uint8')

    # Store slicer state in session
    session['Slicer'] = slicer.to_dict()

    return jsonify({'imageData': utils.encode_to_base64(image_slice),
                    'numTrainSamples': str(session["NumTrainSamples"]),
                    'numValSamples': str(session["NumValSamples"]),
                    'currentSample': str(session["CurrentSample"])})

@app.route('/save_training_sample', methods=["POST"])
def save_training_sample():
    """
    Saves current sample and annotations to the training set.
    """
    global images

    # Load current volume
    image_vol = images[session['VolumeIndex']]

    # Define bounds volume with identical shape as the image_vol
    bounds_vol = np.ones(image_vol.shape)

    # Load slicer information from session
    slicer = Slicer(image_vol.shape)
    slicer.from_dict(session['Slicer'])

    # Retrieve annotation from interface
    base64_image = str(request.values['imageData']).split(',')[1]
    canvas_image = utils.decode_to_numpy(base64_image)[...,:3]
    canvas_image = resize(canvas_image, (session['InputSize'], session['InputSize']), order=0)

    # Convert colored annotation to categorical
    mask_slice, weight_slice = utils.colored_to_categorical(canvas_image, num_classes=session['NumClasses'])

    # Extract the necessary sample information
    image_slice = slicer.get_slice(image_vol, slice_width=session['InputSize'], order=1).astype('uint8')
    bounds_slice = slicer.get_slice(bounds_vol, slice_width=session['InputSize'], order=0).astype('uint8')
    mask_slice = ((mask_slice * bounds_slice[:,:,None]).astype('uint8') * 255).astype('uint8')
    weight_slice = ((weight_slice * bounds_slice).astype('uint8') * 255).astype('uint8')
    slice_data = np.array([session['VolumeIndex'], session['Slicer']])

    # Compute current number of training and validation samples
    n_train_samples = len(glob.glob('data/train/images/*'))
    n_val_samples = len(glob.glob('data/val/images/*'))
    n_samples = n_train_samples + n_val_samples

    # Save the sample
    io.imsave(f'data/train/images/{n_samples}.tiff', image_slice)
    io.imsave(f'data/train/masks/{n_samples}.tiff', mask_slice)
    io.imsave(f'data/train/weights/{n_samples}.tiff', weight_slice)
    np.save(f'data/train/slices/{n_samples}.npy', slice_data)

    # Update dataset and sample info
    session["NumTrainSamples"] = len(glob.glob('data/train/images/*'))
    session["NumValSamples"]= len(glob.glob('data/val/images/*'))
    if session["NumValSamples"] <= 0.2 * session["NumTrainSamples"]:
        session["CurrentSample"] = 'Validation'
    else:
        session["CurrentSample"] = 'Training'
    
    # Update mask volume
    # mask_vol = np.load(mask_volume_files[session['VolumeIndex']])
    # if mask_vol.shape[-1] != mask_slice.shape[-1]:
    #     mask_vol = np.zeros((image_vol.shape[0],
    #                          image_vol.shape[1],
    #                          image_vol.shape[2],
    #                          mask_slice.shape[-1]), dtype='uint8')
    # mask_vol = slicer.update_volume(mask_slice, mask_vol)
    # np.save(mask_volume_files[session['VolumeIndex']], mask_vol)

    # Update weight volume
    # weight_vol = np.load(weight_volume_files[session['VolumeIndex']])
    # weight_vol = slicer.update_volume(weight_slice, weight_vol)
    # np.save(weight_volume_files[session['VolumeIndex']], weight_vol)

    # Select a random volume
    session['VolumeIndex'] = utils.get_random_volume_index(fewest_first=True)
    image_vol = images[session['VolumeIndex']]

    # Create slicer object
    slicer = Slicer(image_vol.shape)

    # Extract randomly oriented and positioned slice
    slicer.randomize(sampling_mode=session['SamplingMode'])
    image_slice = slicer.get_slice(image_vol, slice_width=session['InputSize'], order=0).astype('uint8')

    # Store slicer state in session
    session['Slicer'] = slicer.to_dict()

    return jsonify({'imageData': utils.encode_to_base64(image_slice),
                    'numTrainSamples': str(session["NumTrainSamples"]),
                    'numValSamples': str(session["NumValSamples"]),
                    'currentSample': str(session["CurrentSample"])})

@app.route('/save_validation_sample', methods=["POST"])
def save_validation_sample():
    """
    Saves current sample and annotations to the validation set.
    """
    global images

    # Load current volume
    image_vol = images[session['VolumeIndex']]

    # Define bounds volume with identical shape as the image_vol
    bounds_vol = np.ones(image_vol.shape)

    # Load slicer information from session
    slicer = Slicer(image_vol.shape)
    slicer.from_dict(session['Slicer'])

    # Retrieve annotation from interface
    base64_image = str(request.values['imageData']).split(',')[1]
    canvas_image = utils.decode_to_numpy(base64_image)[...,:3]
    canvas_image = resize(canvas_image, (session['InputSize'], session['InputSize']), order=0)

    # Convert colored annotation to categorical
    mask_slice, weight_slice = utils.colored_to_categorical(canvas_image, num_classes=session['NumClasses'])

    # Extract the necessary sample information
    image_slice = slicer.get_slice(image_vol, slice_width=session['InputSize'], order=1).astype('uint8')
    bounds_slice = slicer.get_slice(bounds_vol, slice_width=session['InputSize'], order=0).astype('uint8')
    mask_slice = ((mask_slice * bounds_slice[:,:,None]).astype('uint8') * 255).astype('uint8')
    weight_slice = ((weight_slice * bounds_slice).astype('uint8') * 255).astype('uint8')
    slice_data = np.array([session['VolumeIndex'], session['Slicer']])

    # Compute current number of training and validation samples
    n_train_samples = len(glob.glob('data/train/images/*'))
    n_val_samples = len(glob.glob('data/val/images/*'))
    n_samples = n_train_samples + n_val_samples

    # Save the sample
    io.imsave(f'data/val/images/{n_samples}.tiff', image_slice)
    io.imsave(f'data/val/masks/{n_samples}.tiff', mask_slice)
    io.imsave(f'data/val/weights/{n_samples}.tiff', weight_slice)
    np.save(f'data/val/slices/{n_samples}.npy', slice_data)

    # Update dataset and sample info
    session["NumTrainSamples"] = len(glob.glob('data/train/images/*'))
    session["NumValSamples"]= len(glob.glob('data/val/images/*'))
    if session["NumValSamples"] <= 0.2 * session["NumTrainSamples"]:
        session["CurrentSample"] = 'Validation'
    else:
        session["CurrentSample"] = 'Training'
    
    # Update mask volume
    # mask_vol = np.load(mask_volume_files[session['VolumeIndex']])
    # if mask_vol.shape[-1] != mask_slice.shape[-1]:
    #     mask_vol = np.zeros((image_vol.shape[0],
    #                          image_vol.shape[1],
    #                          image_vol.shape[2],
    #                          mask_slice.shape[-1]), dtype='uint8')
    # mask_vol = slicer.update_volume(mask_slice, mask_vol)
    # np.save(mask_volume_files[session['VolumeIndex']], mask_vol)

    # Update weight volume
    # weight_vol = np.load(weight_volume_files[session['VolumeIndex']])
    # weight_vol = slicer.update_volume(weight_slice, weight_vol)
    # np.save(weight_volume_files[session['VolumeIndex']], weight_vol)

    # Select a random volume
    session['VolumeIndex'] = utils.get_random_volume_index(fewest_first=True)
    image_vol = images[session['VolumeIndex']]

    # Create slicer object
    slicer = Slicer(image_vol.shape)

    # Extract randomly oriented and positioned slice
    slicer.randomize(sampling_mode=session['SamplingMode'])
    image_slice = slicer.get_slice(image_vol, slice_width=session['InputSize'], order=0).astype('uint8')

    # Store slicer state in session
    session['Slicer'] = slicer.to_dict()

    return jsonify({'imageData': utils.encode_to_base64(image_slice),
                    'numTrainSamples': str(session["NumTrainSamples"]),
                    'numValSamples': str(session["NumValSamples"]),
                    'currentSample': str(session["CurrentSample"])})

@app.route('/predict_sample', methods=["POST"])
def predict_sample():
    """
    Predicts and displays currently viewed sample.
    """
    global images

    # Load current volume
    image_vol = images[session['VolumeIndex']]

    # Create bounds volume with identical shape
    bounds_vol = np.ones(image_vol.shape)

    # Load slicer information from session
    slicer = Slicer(image_vol.shape)
    slicer.from_dict(session['Slicer'])

    # Make prediction
    image_slice = slicer.get_slice(image_vol, slice_width=session['InputSize'], order=1).astype('uint8') / 255
    bounds_slice = slicer.get_slice(bounds_vol, slice_width=session['InputSize'], order=0).astype('uint8')
    pred_slice = (predict.predict(image_slice,
                                  n_channels=session['NumChannels'],
                                  n_classes=session['NumClasses'])).astype('uint8') * bounds_slice[:,:,None]

    return jsonify({'imageData': utils.encode_to_base64(pred_slice)})

# Training settings -------------------------------------------------------------------------------------
@app.route('/update_num_classes', methods=["POST"])
def update_num_classes():
    """
    Updates the session variable NumClasses.
    """
    global images

    # Retrieve JSON data
    json_data = request.get_json(force=True)

    # Store NumClasses in session variable
    session['NumClasses'] = int(json_data['numClasses'])

    return jsonify({})

@app.route('/update_learning_rate', methods=["POST"])
def update_learning_rate():
    """
    Updates the session variable LearningRate.
    """
    global images

    # Retrieve JSON data
    json_data = request.get_json(force=True)

    # Store LearningRate in session variable
    session['LearningRate'] = float(json_data['learningRate'])

    return jsonify({})

@app.route('/update_batch_size', methods=["POST"])
def update_batch_size():
    """
    Updates the session variable BatchSize.
    """
    global images

    # Retrieve JSON data
    json_data = request.get_json(force=True)

    # Store BatchSize in session variable
    session['BatchSize'] = int(json_data['batchSize'])

    return jsonify({})

@app.route('/update_num_epochs', methods=["POST"])
def update_num_epochs():
    """
    Updates the session variable NumEpochs.
    """
    global images

    # Retrieve JSON data
    json_data = request.get_json(force=True)

    # Store NumEpochs in session variable
    session['NumEpochs'] = int(json_data['numEpochs'])

    return jsonify({})

# @app.route('/update_plot', methods=["POST"])
# def update_plot():
#     """
#     Updates training history plot.
#     """
    
#     # Retrieve JSON data
#     json_data = request.get_json(force=True)


#     if str(json_data['includeTraining']) == 'true':
#         include_training = True
#     if str(json_data['includeTraining']) == 'false':
#         include_training = False

#     plot_image = plot.get_training_plot(metric=str(json_data['metric']), include_training=include_training)
    
#     return jsonify({'imageData' : utils.encode_to_base64(plot_image)})

@app.route('/train_model', methods=["POST"])
def train_model():
    """
    Trains or continues training a U-Net model.
    """
    global training_thread

    # Retrieve json data
    json_data = request.get_json(force=True)

    # Check if continue
    if str(json_data['continue']) == 'true':
        continue_training = True
    if str(json_data['continue']) == 'false':
        continue_training = False

    kwargs = {'initial_lr' : session['LearningRate'],
              'batch_size' : session['BatchSize'],
              'epochs' : session['NumEpochs'],
              'n_classes' : session['NumClasses'],
              'continue_training' : continue_training}

    training_thread = threading.Thread(target=trainer.train_model, args=(), kwargs=kwargs)
    training_thread.start()
    
    return jsonify({})

@app.route('/check_train_status', methods=["POST"])
def check_train_status():
    global training_thread

    if training_thread.is_alive():
        return jsonify({'training': 'true',
                        'imageData' : "None"})
    else:
        training_thread = None
        return jsonify({'training': 'false',
                        'imageData' : "None"})
    # else:
    #     training_thread = None
    #     return jsonify({'training': 'false',
    #                     'imageData' : utils.encode_to_base64(plot.get_training_plot())})
