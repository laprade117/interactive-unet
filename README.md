
## Interactive U-Net

A segmentation tool that utilizes the U-Net deep learning architecture to quickly and efficiently segment 3D volumetric images.

### Installation:

`pip install git+https://github.com/laprade117/interactive-unet`


### Usage

Run the tool in a designated project folder using:

`interactive-unet`

This will create the necessary folder structure at the current working directory and then provide a link that can be opened in any web browser to access the interface.

Next, copy any 3D volumetric images that you want to segment into the `data/image_volumes` folder. Ensure they are stored as numpy (.npy) files in `uint8`. Shape and size doesn't matter as long as all the volumes that you place in this folder can all be loaded into RAM simultaneously.

Using the paint tool annotate at least one image and then train a model. Repeat until desired segmentation result.

### Keyboard Shortcuts

- **Left click**: Paint displayed color
- **Right click**: Paint background color (red)
- **Mouse Wheel**: Adjust brush size
- **Ctrl + S**: Save sample
- **Spacebar**: Next random slice
  
- **C**: Cycles through additional colors if using more than two classes.
- **D**: Toggle prediction overlay
  
- **Ctrl + Z**: Undo last paint stroke
- **Ctrl + Y**: Redo last paint stroke
  
- **Shift + Left Click + Drag**: Drag image
- **Shift + Mouse Wheel**: Zoom in and out

### DTU Thinlinc Setup
1. Activate an interactive GPU session:
`sxm2sh -X`

2. Create a virtual environment:
`python3 -m venv env`

3. Activate the environment:
`source env/bin/activate`

4. Install the tool:
`pip install git+https://github.com/laprade117/interactive-unet`

Currently, you can only access the tool from the browser within the the ThinLinc Client. External usage is not working yet.
