# import math
# import types
import numpy as np

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
# import torch.nn.modules.utils as nn_utils

from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
# import torchvision.transforms.functional as tvf

# from torch_pca import PCA as tPCA

# from einops import rearrange
# from monai.inferers import SlidingWindowInferer

from interactive_unet import utils, metrics

# class FeatureExtractor(nn.Module):

#     def __init__(self, model_type='dinov2', layers=[8], attn_vector='k', stride=7):
#         super().__init__()

#         self.model_type = model_type
#         self.layers = layers
#         self.stride = stride
#         self.attn_vector = attn_vector

#         if self.model_type == 'dino':
#             self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
#         elif self.model_type == 'dinov2':
#             self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            
#         self.model = self._patch_vit_resolution(self.model, stride=self.stride)
        
#         self.features = []
        
#     def _fix_pos_enc(self, patch_size, stride_hw):
#         """
#         Creates a method for position encoding interpolation.
#         :param patch_size: patch size of the model.
#         :param stride_hw: A tuple containing the new height and width stride respectively.
#         :return: the interpolation method
#         """
#         def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
#             npatch = x.shape[1] - 1
#             N = self.pos_embed.shape[1] - 1
#             if npatch == N and w == h:
#                 return self.pos_embed
#             class_pos_embed = self.pos_embed[:, 0]
#             patch_pos_embed = self.pos_embed[:, 1:]
#             dim = x.shape[-1]
#             # compute number of tokens taking stride into account
#             w0 = 1 + (w - patch_size) // stride_hw[1]
#             h0 = 1 + (h - patch_size) // stride_hw[0]
#             assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
#                                             stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
#             # we add a small number to avoid floating point error in the interpolation
#             # see discussion at https://github.com/facebookresearch/dino/issues/8
#             w0, h0 = w0 + 0.1, h0 + 0.1
#             patch_pos_embed = nn.functional.interpolate(
#                 patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
#                 scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
#                 mode='bicubic',
#                 align_corners=False, recompute_scale_factor=False
#             )
#             assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
#             patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
#             return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
#         return interpolate_pos_encoding
    
#     def _patch_vit_resolution(self, model, stride):
#         """
#         change resolution of model output by changing the stride of the patch extraction.
#         :param model: the model to change resolution for.
#         :param stride: the new stride parameter.
#         :return: the adjusted model
#         """

#         if self.model_type == 'dino':
#             patch_size = model.patch_embed.patch_size
#         elif self.model_type == 'dinov2':
#             patch_size = model.patch_embed.patch_size[0]
        
#         if stride == patch_size:  # nothing to do
#             return model
    
#         stride = nn_utils._pair(stride)
#         # assert all([(patch_size // s_) * s_ == patch_size for s_ in
#         #             stride]), f'stride {stride} should divide patch_size {patch_size}'
    
#         # fix the stride
#         model.patch_embed.proj.stride = stride
#         # fix the positional encoding code
#         model.interpolate_pos_encoding = types.MethodType(self._fix_pos_enc(patch_size, stride), model)
#         return model

#     def _hook(self, module, input, output):
        
#         input = input[0]
#         B, N, C = input.shape
#         qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)

#         # qkv[i] 0 is q, 1 is k, 2 is v
#         if self.attn_vector == 'q':
#             self.features.append(qkv[0])
#         elif self.attn_vector == 'k':
#             self.features.append(qkv[1])
#         elif self.attn_vector == 'v':
#             self.features.append(qkv[2])

#     def _translate(self, x, translation=[0,0]):
#         return tvf.affine(x, angle=0, translate=translation, scale=1, shear=0, interpolation=tvf.InterpolationMode.NEAREST)
        
#     def _rotate(self, x, angle=0):
#         return tvf.affine(x, angle=angle, translate=[0,0], scale=1, shear=0, interpolation=tvf.InterpolationMode.NEAREST)
        
#     def _get_feature_maps(self, x):
        
#         outputs = self.model(x)

#         # Concatenate the collected features
#         feature_maps = torch.concatenate(self.features, 1)

#         # Drop class token and reshape into feature map image
#         feature_map_size = int((feature_maps.shape[2] - 1) ** 0.5)
#         feature_maps = rearrange(feature_maps[:,:,1:,:],
#                                  'b h (hp wp) f -> b (h f) hp wp',
#                                  hp=feature_map_size)
        
#         # Ensure feature vector is reset
#         self.features = []
        
#         return feature_maps

#     def _get_corrected_feature_maps(self, x):

#         feature_maps = self._get_feature_maps(x)

#         for i in range(3):
#             angle = (i + 1) * 90
#             x_r = self._rotate(x, angle)
#             feature_maps += self._rotate(self._get_feature_maps(x_r), -angle)
            
#         feature_maps = feature_maps / 4
        
#         return feature_maps   
            
#     def forward(self, x):

#         batch_size = x.shape[0]
#         image_size = x.shape[-1]
        
#         # Ensure feature vector is reset
#         self.features = []

#         # Set hooks to grab features
#         handles = []
#         for l in self.layers:
#             handle = self.model.blocks[l].attn.register_forward_hook(self._hook)
#             handles.append(handle)

#         with torch.no_grad():
            
#             x = (x - 0.456) / 0.224
    
#             # Adjust image resolution
#             x = tvf.resize(x, size=(518, 518))
    
#             # Ensure 3 channels
#             if x.shape[1] == 1:
#                 x = torch.concatenate([x,x,x], 1)

#             feature_maps = self._get_feature_maps(x)

#             feature_maps = tvf.resize(feature_maps, size=(image_size, image_size))

#             feature_maps = feature_maps - torch.mean(feature_maps, 1)[:,None,:]
#             feature_maps = feature_maps / torch.std(feature_maps, 1)[:,None,:]

#         for handle in handles:
#             handle.remove()
        
#         return feature_maps

# def feature_pca(feature_maps, feature_dim=128):
    
#     feature_maps = torch.tensor(feature_maps).cuda()
    
#     # Apply PCA to reduce feature dimensionality
#     feature_maps = rearrange(feature_maps, 'b c h w -> b (h w) c')
#     feature_maps = feature_maps - torch.mean(feature_maps, 1)
#     feature_maps = feature_maps / torch.std(feature_maps, 1)
    
#     pca = tPCA(n_components=feature_dim).fit(feature_maps[0])
#     print(f'Feature PCA Explained Variance: {torch.sum(pca.explained_variance_ratio_)}')
#     feature_maps = pca.transform(feature_maps[0])[None,...]
    
#     feature_maps = rearrange(feature_maps, 'b (h w) c -> b c h w', w=512)
    
#     feature_maps = feature_maps.detach().cpu().numpy()

#     return feature_maps
    
# def get_dense_features(image, extractor, scale=64):
    
#     image = torch.tensor(image.astype('float32')[None,None,...]).cuda()

#     inferer = SlidingWindowInferer(scale, 
#                                     sw_batch_size=1, 
#                                     overlap=0.05,
#                                     mode='gaussian',
#                                     padding_mode='reflect',
#                                     sw_device=torch.device('cuda'),
#                                     device=torch.device('cpu'))
        
#     features = inferer(image, extractor).detach().cpu().numpy()

#     return features

# def remove_banding(image):
#     '''
#     Removes horizontal and vertical banding that is caused
#     by patch-wise dense ViT feature extraction.

#     Can take multi-dimensional arrays. Assumes height and width are in the last two axes.
#     '''
    
#     vertical_banding = np.repeat(np.mean(image,-2)[...,None,:],image.shape[-2],-2)
#     horizontal_banding = np.repeat(np.mean(image,-1)[...,:,None],image.shape[-1],-1)

#     fixed_image = image - (vertical_banding + horizontal_banding)

#     return fixed_image

class Suggestor(nn.Module):

    def __init__(self, num_channels, num_classes):
        super().__init__()


        import segmentation_models_pytorch as smp

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.unet = smp.Unet(encoder_name='mobilenet_v2',
                             encoder_weights='imagenet',
                             in_channels=self.num_channels,
                             classes=self.num_classes,)
                            #  encoder_depth=4,
                            #  decoder_channels=(128, 64, 32, 16))

        self.model = nn.Sequential(self.unet,
                                   nn.Softmax(dim=1))     

        # self.model = nn.Sequential(nn.Conv2d(self.num_channels, self.num_classes, kernel_size=1, padding='same'),
        #                            nn.Softmax(dim=1)) 

        # self.model = nn.Sequential(nn.Conv2d(self.num_channels, self.num_channels // 2, kernel_size=3, padding='same'),
        #                            nn.ReLU(),
        #                            nn.Conv2d(self.num_channels // 2, self.num_channels // 4, kernel_size=3, padding='same'),
        #                            nn.ReLU(),
        #                            nn.Conv2d(self.num_channels // 4, self.num_classes, kernel_size=3, padding='same'),
        #                            nn.Softmax(dim=1))   

        # self.model = nn.Sequential(nn.Linear(self.num_channels, self.num_channels // 2),
        #                            nn.BatchNorm1d(self.num_channels // 2),
        #                            nn.ReLU(),
        #                            nn.Linear(self.num_channels // 2, self.num_channels // 4),
        #                            nn.BatchNorm1d(self.num_channels // 4),
        #                            nn.ReLU(),
        #                            nn.Linear(self.num_channels // 4, self.num_classes),
        #                            nn.Softmax(dim=1))

            
    def forward(self, x):

        b, c, h, w = x.shape

        pred = self.model(x) 

        # x = rearrange(x, 'b c h w -> (b h w) c')
        # pred = rearrange(self.model(x), '(b h w) c -> b c h w', h=h, w=w)

        return pred

# class RandomBrightnessContrast(torch.nn.Module):

#     def __init__(self, brightness=(0.75, 1.25), contrast=(0.5, 2.0)):
#         super().__init__()

#         self.brightness = brightness
#         self.contrast = contrast

#     def forward(self, image, mask, weight):

#         brightess_factor = self.brightness[0] + torch.rand(1) * (self.brightness[1] - self.brightness[0])
#         contrast_factor = self.contrast[0] + torch.rand(1) * (self.contrast[1] - self.contrast[0])

#         if np.random.rand() > 0.5:
#             for i in range(image.shape[1]):

#                 image_channel = image[:,i,:,:][:,None,:,:]

#                 image[:,i,:,:] = tvf.adjust_brightness(image_channel, brightess_factor)
#                 image[:,i,:,:] = tvf.adjust_contrast(image_channel, contrast_factor)
#         else:
#             for i in range(image.shape[1]):

#                 image_channel = image[:,i,:,:][:,None,:,:]

#                 image[:,i,:,:] = tvf.adjust_contrast(image_channel, contrast_factor)
#                 image[:,i,:,:] = tvf.adjust_brightness(image_channel, brightess_factor)

#         return image, mask, weight

def make_suggestions(image_features, mask, lr=0.0005, steps=30, model=None):


    torch.set_float32_matmul_precision('medium')

    image_size = mask.shape[0]
    
    unique_colors = utils.get_unique_colors(mask)[1:]
    num_classes = len(unique_colors)

    if num_classes == 1:
        # Return all same class
        suggestions = (np.ones((image_size, image_size, 3)) * unique_colors[0][None,None,:]).astype('uint8')
    else:

        mask, _ = utils.colored_to_categorical(mask)
        mask = (mask > 127)

        x = torch.tensor(image_features).to(torch.float32).cuda()
        y = torch.tensor(np.moveaxis(mask,-1,0)[None,...]).to(torch.float32).cuda()

        mask = np.sum([mask[:,:,i] * (i + 1) for i in range(num_classes)], 0)
        w = torch.tensor(np.repeat((mask > 0)[None,None,...], num_classes, 1)).to(torch.float32).cuda()

        if model is None:
            model = Suggestor(x.shape[1], num_classes).cuda()
        elif model.num_classes != num_classes:
            model = Suggestor(x.shape[1], num_classes).cuda()

        best_model = model.state_dict()
        best_loss = 100
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        transforms = v2.Compose([v2.RandomHorizontalFlip(p=0.5),
                                 v2.RandomVerticalFlip(p=0.5),
                                 v2.RandomRotation(degrees=(-360,360), interpolation=InterpolationMode.NEAREST),
                                #  v2.RandomAffine(degrees=(-360,360), translate=(0.25,0.25), interpolation=InterpolationMode.NEAREST)
                                 ])

        x = tv_tensors.Image(x)
        y = tv_tensors.Mask(y)
        w = tv_tensors.Mask(w)

        import time
        t1 = time.time()
        for t in range(steps):

            xt, yt, wt = transforms(x, y, w)
            
            y_pred = model(xt)
            loss = metrics.mcc_ce_loss(y_pred, yt, wt)

            if loss.isnan().any():
                model = Suggestor(x.shape[1], num_classes).cuda()
                best_model = model.state_dict()
                best_loss = 100

            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('Training time: ', time.time() - t1)

        model.load_state_dict(best_model)
        model.eval()

        predictions = model(x).detach().cpu().numpy()
        predictions = np.argmax(predictions[0],0).reshape((image_size,image_size))
        
        suggestions = np.zeros((image_size,image_size,3)).astype('uint8')
        for i in range(len(unique_colors)):
            suggestions[predictions == i,:] = unique_colors[i]

    return suggestions, model
