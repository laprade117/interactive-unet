import torch
import torch.nn as nn

import lightning as L

import segmentation_models_pytorch as smp

from . import metrics

class UNet(L.LightningModule):
    """
    The UNet model.
    """
    
    def __init__(self, lr=0.0001,
                 num_channels=1, num_classes=2,
                 loss_function=metrics.mcc_ce_loss,
                 architecture='U-Net',
                 encoder_name='mit_b0',
                 pretrained=True):
        super().__init__()
        
        self.save_hyperparameters()

        self.lr = lr
        self.loss_function = loss_function

        if pretrained:
            encoder_weights = 'imagenet'
        else:
            encoder_weights = None

        if architecture == 'U-Net':
            model_builder = smp.Unet
        elif architecture == 'U-Net++':
            model_builder = smp.UnetPlusPlus
        elif architecture == 'FPN':
            model_builder = smp.FPN
        elif architecture == 'PSPNet':
            model_builder = smp.PSPNetc
        elif architecture == 'DeepLabV3':
            model_builder = smp.DeepLabV3
        elif architecture == 'DeepLabV3+':
            model_builder = smp.DeepLabV3Plus
        elif architecture == 'LinkNet':
            model_builder = smp.Linknet
        elif architecture == 'MA-Net':
            model_builder = smp.MAnet
        elif architecture == 'PAN':
            model_builder = smp.PAN
        elif architecture == 'UPerNet':
            model_builder = smp.UPerNet
        elif architecture == 'Segformer':
            model_builder = smp.Segformer

        self.model = model_builder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=num_channels,
            classes=num_classes,
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        output = self.softmax(self.model(x))
        
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def _log_metrics(self, set_name, loss, y_hat, y, w):
        """
        Logs requested metrics to wandb.
        """
        
        y = torch.round(y)
        y_hat = torch.round(y_hat)
        
        self.log(f"{set_name}/Loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{set_name}/Dice", metrics.dice(y_hat, y, w, axes=[0,2,3]), on_step=False, on_epoch=True)
        self.log(f"{set_name}/IoU", metrics.iou(y_hat, y, w, axes=[0,2,3]), on_step=False, on_epoch=True)
        self.log(f"{set_name}/MCC", metrics.mcc(y_hat, y, w, axes=[0,2,3]), on_step=False, on_epoch=True)
        
    def training_step(self, batch):
        
        # Split batch
        X, y, w = batch


        # Forward pass
        y_hat = self(X)

        # Compute loss
        loss = self.loss_function(y_hat, y, w, axes=[0,2,3])
        
        # Log metrics to wandb
        self._log_metrics('train', loss, y_hat, y, w)
        return loss
    
    def validation_step(self, batch):
        
        # Split batch
        X, y, w = batch
        
        # Forward pass
        y_hat = self(X)
        
        # Compute loss
        loss = self.loss_function(y_hat, y, w, axes=[0,2,3])
        
        # Log metrics to wandb
        self._log_metrics('val', loss, y_hat, y, w)