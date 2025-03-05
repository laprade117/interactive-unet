import os 
import time

import torch

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from interactive_unet import utils, loader, unet
		            
def train_model(lr=0.0001, batch_size=1, epochs=10,
                 num_channels=1, num_classes=2,
                 loss_function_name='MCC + CE',
                 architecture='U-Net',
                 encoder_name='mobileone_s4',
                 pretrained=True):

    torch.set_float32_matmul_precision('medium')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_folder = f'data/train'
    val_folder = f'data/val'

    train_loader = loader.get_data_loader(train_folder, batch_size, augment=True, shuffle=True)
    val_loader = loader.get_data_loader(val_folder, batch_size, augment=False, shuffle=False)

    loss_function = utils.loss_name_to_function(loss_function_name)

    # If model exists - continue training
    if os.path.isfile('model/model.ckpt'):
        model = unet.UNet.load_from_checkpoint(checkpoint_path='model/model.ckpt')
        model.lr = lr
        model.loss_function = loss_function
    else:
        model = unet.UNet(lr=lr, num_channels=num_channels, num_classes=num_classes, 
                          loss_function=loss_function, architecture=architecture,
                          encoder_name=encoder_name, pretrained=pretrained)

    # Remove old checkpoint
    if os.path.isfile('model/model.ckpt'):
        os.remove('model/model.ckpt')
    
    # Save best model callback
    checkpoint_callback = ModelCheckpoint(dirpath='model/',
                                          filename='model',
                                          monitor="val/Loss",
                                          mode="min")
    
    # Training logger
    logger = CSVLogger("model/history", name=time.strftime("%Y-%m-%d_%H-%M-%S"))

    # Train model
    model.train()
    trainer = L.Trainer(max_epochs=epochs,
                        log_every_n_steps=1,
                        callbacks=[checkpoint_callback], 
                        precision='16-mixed',
                        logger=logger)
    trainer.fit(model, train_loader, val_loader)
