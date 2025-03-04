from setuptools import setup

setup(
    name='interactive-segmentation',
    version='0.1.0',    
    description='An interactive segmentation tool for 3D volumetric data.',
    license='BSD 2-clause',
    packages=['intseg'],
    entry_points = {
        'console_scripts': ['interactive-unet=interactive_unet:app'],
    },
    install_requires=['torch',
                      'torchvision',
                      'nicegui',
                      'scikit-image',
                      'opencv-python',
                      'lightning',
                      'segmentation-models-pytorch',
                      'scipy',
                      'monai',
                      'plotly',
                      'pandas'],
)
