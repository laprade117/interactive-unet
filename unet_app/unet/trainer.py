import os 

import pytorch_lightning as pl

from unet_app.unet import loader, unet2d

		            
def train_model(initial_lr=0.0001, batch_size=8, epochs=20, n_channels=1, n_classes=2, continue_training=False):

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_folder = f'data/train'
    val_folder = f'data/val'

    train_loader = loader.get_data_loader(train_folder, batch_size, augment=True)
    val_loader = loader.get_data_loader(val_folder, batch_size, augment=False)

    if continue_training and os.path.isfile('model/model.ckpt'):
        model = unet2d.UNet.load_from_checkpoint(checkpoint_path='model/model.ckpt')
        model.set_learning_rate(initial_lr)
    else:
        model = unet2d.UNet(lr=initial_lr, num_channels=n_channels, num_classes=n_classes)

    # Remove old checkpoint
    if os.path.isfile('model/model.ckpt'):
        os.remove('model/model.ckpt')
    
    # Save best model callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='model/',
                                                       filename='model',
                                                       monitor="val/mcc_ce_loss",
                                                       mode="min")

    # Train model
    model.train()
    trainer = pl.Trainer(max_epochs=epochs,
                         log_every_n_steps=1,
                         callbacks=[checkpoint_callback],
                         gpus=1,
                         accelerator="gpu")
    trainer.fit(model, train_loader, val_loader)