from setuptools import setup, find_packages

setup(
    name='interactive-unet',
    version='0.1.0',    
    description='An interactive segmentation tool for 3D volumetric data.',
    license='BSD 2-clause',
    # packages=['interactive_unet'],
    packages=find_packages(),
    entry_points = {
        'console_scripts': ['interactive-unet=interactive_unet:app'],
    },
    install_requires=['nicegui',
                      'torch',
                      'torchvision',
                      'scikit-image',
                      'opencv-python',
                      'lightning',
                      'segmentation-models-pytorch',
                      'scipy',
                      'plotly',
                      'pandas',
                      'zarr',
                      'numba',
                      'joblib',
                      'tqdm'],
)
