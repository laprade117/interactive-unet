import torch
import torch.nn as nn

import lightning as L

import segmentation_models_pytorch as smp

from interactive_unet import metrics

class UNet(L.LightningModule):
    """
    The UNet model.
    """
    
    def __init__(self, lr=0.0001,
                 num_channels=1, num_classes=2,
                 loss_function=metrics.mcc_ce_loss,
                 encoder_name='mobileone_s4',
                 pretrained=True):
        super().__init__()
        
        self.save_hyperparameters()

        self.lr = lr
        self.loss_function = loss_function

        if pretrained:
            encoder_weights = 'imagenet'
        else:
            encoder_weights = None

        self.model = smp.Unet(
            encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,         # use `imagenet` pre-trained weights for encoder initialization
            in_channels=num_channels,           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                # model output channels (number of classes in your dataset)
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
        self.log(f"{set_name}/Dice", metrics.dice(y_hat, y, w, axes=[2,3]), on_step=False, on_epoch=True)
        self.log(f"{set_name}/IoU", metrics.iou(y_hat, y, w, axes=[2,3]), on_step=False, on_epoch=True)
        self.log(f"{set_name}/MCC", metrics.mcc(y_hat, y, w, axes=[2,3]), on_step=False, on_epoch=True)
        
    def training_step(self, batch):
        
        # Split batch
        X, y, w = batch


        # Forward pass
        y_hat = self(X)

        # Compute loss
        loss = self.loss_function(y_hat, y, w, axes=[2,3])
        
        # Log metrics to wandb
        self._log_metrics('train', loss, y_hat, y, w)
        return loss
    
    def validation_step(self, batch):
        
        # Split batch
        X, y, w = batch
        
        # Forward pass
        y_hat = self(X)
        
        # Compute loss
        loss = self.loss_function(y_hat, y, w, axes=[2,3])
        
        # Log metrics to wandb
        self._log_metrics('val', loss, y_hat, y, w)