
## Interactive U-Net

A segmentation tool that utilizes the U-Net deep learning architecture to quickly and efficiently segment 3D volumetric images.

### Installation:

`pip install git+https://github.com/laprade117/interactive-unet`


### Usage

Run the tool in a designated project folder using:

`interactive-unet`

This will create the necessary folder structure at the current working directory and then provide a link that can be opened in any web browser to access the interface. On first run, the tool will automatically download a sample volume to get started.

To work with your own data, remove the sample volume from `data/image_volumes` folder and copy any 3D volumetric images that you want to segment into the same folder. Ensure they are stored as multi-scale Zarr v3 files in `uint8` with a chunk size of 128 and a shard size of 256.

Using the paint tool annotate at least one image and then train a model. Repeat until desired segmentation result.

### Keyboard Shortcuts

- **Left click**: Paint displayed color
- **Right click**: Paint background color (red)
- **Ctrl + Left Click**: Push displayed overlay onto annotation map

- **Mouse Wheel**: Adjust brush size
- **Ctrl + S**: Save sample

- **Q**: Next slice in stack
- **A**: Previous slice in stack
- **Spacebar**: Random slice
  
- **C**: Cycles through additional colors if using more than two classes.
- **D**: Toggle prediction overlay
- **F**: Cycle between overlay types
  
- **Ctrl + Z**: Undo last paint stroke
- **Ctrl + Y**: Redo last paint stroke
  
- **Shift + Left Click + Drag**: Drag image
- **Shift + Mouse Wheel**: Zoom in and out

### DTU Thinlinc Setup

1. Activate an interactive GPU session:
`sxm2sh -X`

2. Create a conda environment:
`conda create --name unet python=3.12`

3. Activate the environment:
`conda activate unet`

4. Install the tool:
`pip install git+https://github.com/laprade117/interactive-unet`

5. Launch the tool:
`interactive-unet`

Currently, you can only access the tool from the browser within the the ThinLinc Client. External usage is not working yet.
