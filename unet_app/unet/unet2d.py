from click import progressbar
import torch
import torch.nn as nn
import pytorch_lightning as pl

from unet_app.unet import metrics

class UNetConvBlock(nn.Module):
    """
    A two layer convolution block with ReLU activation and batch normalization.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        
        super(UNetConvBlock, self).__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
                  nn.ReLU(),
                  nn.BatchNorm2d(out_channels),
                  nn.Conv2d(out_channels, out_channels, kernel_size, padding='same'),
                  nn.ReLU(),
                  nn.BatchNorm2d(out_channels)]
        self.convblock = nn.Sequential(*layers)

    def forward(self, x):
    
        x = self.convblock(x)
        
        return x

class UNetDownBlock(nn.Module):
    """
    A downscaling block for UNet located in the contracting path. Applies
    max pooling prior to a standard UNet convolutional block.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, use_max_pool=True):
        
        super(UNetDownBlock, self).__init__()
        
        self.use_max_pool = use_max_pool
        self.maxpool = nn.MaxPool2d(2,2)
        self.convblock = UNetConvBlock(in_channels, out_channels, kernel_size)

    def forward(self, x):
        
        if self.use_max_pool:
            x = self.maxpool(x)
        x = self.convblock(x)
        
        return x
    
    
class UNetUpBlock(nn.Module):
    """
    An upscaling block for UNet located in the expanding path. Upsampling followed by 
    a convolutional layer with kernal size of 2, then concatenation with the skip 
    connections from the contracting path, followed a by a standard UNet convolutional 
    block.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, mode='nearest'):
        
        super(UNetUpBlock, self).__init__()
        
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode=mode),
                                      nn.Conv2d(in_channels, out_channels, 2, padding='same'),
                                      nn.ReLU(),
                                      nn.BatchNorm2d(out_channels))
        
        self.convblock = UNetConvBlock(in_channels, out_channels, kernel_size)

    def forward(self, x, skip):
        
        x = self.upsample(x)
        x = torch.cat([skip, x], 1)
        x = self.convblock(x)
        
        return x
    
class UNet(pl.LightningModule):
    """
    The UNet model.
    """
    
    def __init__(self, lr=0.001, num_channels=1, num_classes=2, num_filters=64, kernel_size=3):
        super().__init__()
        
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.lr = lr
        
        self.d1 = UNetDownBlock(num_channels, num_filters, kernel_size, use_max_pool=False)
        self.d2 = UNetDownBlock(num_filters, num_filters*2, kernel_size)
        self.d3 = UNetDownBlock(num_filters*2, num_filters*4, kernel_size)
        self.d4 = UNetDownBlock(num_filters*4, num_filters*8, kernel_size)
        
        self.d5 = UNetDownBlock(num_filters*8, num_filters*16, kernel_size)
        
        self.u1 = UNetUpBlock(num_filters*16, num_filters*8, kernel_size)
        self.u2 = UNetUpBlock(num_filters*8, num_filters*4, kernel_size)
        self.u3 = UNetUpBlock(num_filters*4, num_filters*2, kernel_size)
        self.u4 = UNetUpBlock(num_filters*2, num_filters, kernel_size)
        
        self.conv = nn.Conv2d(num_filters, num_classes, kernel_size, padding='same')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        
        d5 = self.d5(d4)
        
        u1 = self.u1(d5, d4)
        u2 = self.u2(u1, d3)
        u3 = self.u3(u2, d2)
        u4 = self.u4(u3, d1)
        
        output = self.conv(u4)
        output = self.softmax(output)
        
        return output

    def set_learning_rate(self, lr):
        self.lr = lr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def _log_metrics(self, set_name, loss, y_hat, y, w):
        """
        Logs requested metrics to wandb.
        """
        
        y = torch.round(y)
        y_hat = torch.round(y_hat)
        
        self.log(f"{set_name}/mcc_ce_loss", loss, prog_bar=True)
        self.log(f"{set_name}/dice", metrics.dice(y_hat, y, w, axes=[1,2,3]))
        self.log(f"{set_name}/iou", metrics.iou(y_hat, y, w, axes=[1,2,3]))
        self.log(f"{set_name}/mcc", metrics.mcc(y_hat, y, w, axes=[1,2,3]))
        self.log(f"{set_name}/accuracy", metrics.mcc(y_hat, y, w, axes=[1,2,3]))
        
    def training_step(self, batch, batch_idx):
        
        # Split batch
        X, y, w = batch
        
        # Get optimizers
        opt = self.optimizers(use_pl_optimizer=True)
        
        # Zero the parameter gradients
        opt.zero_grad()

        # Forward pass
        y_hat = self(X)

        # Compute loss
        loss = metrics.mcc_ce_loss(y_hat, y, w, axes=[1,2,3])
        
        # Backwards pass
        self.manual_backward(loss)
        
        # Update weights
        opt.step()
        
        # Log metrics to wandb
        self._log_metrics('train', loss, y_hat, y, w)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        # Split batch
        X, y, w = batch
        
        # Forward pass
        y_hat = self(X)
        
        # Compute loss
        loss = metrics.mcc_ce_loss(y_hat, y, w, axes=[1,2,3])
        
        # Log metrics to wandb
        self._log_metrics('val', loss, y_hat, y, w)