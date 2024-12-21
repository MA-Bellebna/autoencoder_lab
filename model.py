"""
Stitches submodels together.
"""
import numpy as np
import time, os
import itertools

from functools import partial
from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Custom modules
import hyperprior
import losses
import utils, metrics
import encoder, generator

from skimage.metrics import structural_similarity as ssim

Intermediates = namedtuple("Intermediates",
    ["input_image",             # [0, 1] (after scaling from [0, 255])
     "reconstruction",          # [0, 1]
     "latents_quantized",       # Latents post-quantization.
     "n_bpp",                   # Differential entropy estimate.
     "q_bpp"])                  # Shannon entropy estimate.

class Model(nn.Module):

    def __init__(self, image_dims, checkpoint):
        super(Model, self).__init__()

        """
        Builds hific model from submodels in network.
        """
        self.image_dims = image_dims
        self.checkpoint = checkpoint

        self.latent_channels = 220
        self.model_mode = 'evaluation'
        self.model_type = 'compression'
        self.n_residual_blocks = 9
        self.use_channel_norm = True
        self.noise_dim = 32
        self.sample_noise = False
        self.normalize_input_image = True
        self.use_latent_mixture_model = False
        self.likelihood_type = 'gaussian'
        self.entropy_code = False

        #Trainable
        self.Encoder = encoder.Encoder(self.image_dims, C=self.latent_channels)
        
        self.Encoder = self.load_submodel(self.Encoder, self.checkpoint, freeze = False, sub_model = 'Encoder')

        #Non Trainable
        self.Decoder = generator.Generator(self.image_dims, 1, C=self.latent_channels,
            n_residual_blocks=self.n_residual_blocks, channel_norm=self.use_channel_norm, sample_noise=
            self.sample_noise, noise_dim=self.noise_dim)
        
        self.Decoder = self.load_submodel(self.Decoder, self.checkpoint, False)   # Load pretrained HiFI weights 

        if self.use_latent_mixture_model is True:
            self.Hyperprior = hyperprior.HyperpriorDLMM(bottleneck_capacity=self.latent_channels,
                likelihood_type=self.likelihood_type, mixture_components=self.mixture_components, entropy_code=self.entropy_code)
        else:
            self.Hyperprior = hyperprior.Hyperprior(bottleneck_capacity=self.latent_channels,
                likelihood_type=self.likelihood_type, entropy_code=self.entropy_code)


        self.Hyperprior = self.load_submodel(self.Hyperprior, self.checkpoint, freeze=False, sub_model = "Hyperprior")

        self.squared_difference = torch.nn.MSELoss(reduction='none')
        # Expects [-1,1] images or [0,1] with normalize=True flag
        self.perceptual_ssim_loss

                

    def load_checkpoint(self, checkpoint):
        self.logger.info('Loading checkpoint {}'.format(checkpoint))

        self.logger.info("Loading Encoder + Hyperprior pipelines")
        self.Encoder = self.load_submodel(self.Encoder, checkpoint, False, sub_model="Encoder")
        self.Hyperprior = self.load_submodel(self.Hyperprior, checkpoint, False, sub_model="Hyperprior")

        self.Decoder = generator.Generator(self.image_dims, self.batch_size, C=self.latent_channels,
        n_residual_blocks=self.n_residual_blocks, channel_norm=self.use_channel_norm, sample_noise=
        self.sample_noise, noise_dim=self.noise_dim)
        self.Decoder = self.load_submodel(self.Decoder, checkpoint, False, sub_model="Generator")



    def load_submodel(self, model, path, freeze=True, sub_model='Generator'):
        """ Loading the pretrained weights from the HIFI ckpt to the Generator 
        
        path : path of HIFI ckpt
        model : Generator 

        """
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load = torch.load(path, map_location = device)

        new_state_dict = {}
        for name, weight in load['model_state_dict'].items():
            if sub_model in name:
                new_state_dict[name.replace(sub_model + ".", "")] = weight

        if freeze == True :  
            model.eval()
            for param in model.parameters():
                param.requires_grad = False 

        model.load_state_dict(new_state_dict, strict = False)

        return model


    def compression_forward(self, x):
        """
        Forward pass through encoder, hyperprior, and decoder.

        Inputs
        x:  Input image. Format (N,C,H,W), range [0,1],
            or [-1,1] if args.normalize_image is True
            torch.Tensor
        
        Outputs
        intermediates: NamedTuple of intermediate values
        """

        # x = torch.clone(input_x)
        image_dims = tuple(x.size()[1:])  # (C,H,W)

    
        n_encoder_downsamples = self.Encoder.n_downsampling_layers
        factor = 2 ** n_encoder_downsamples
        x = utils.pad_factor(x, x.size()[2:], factor)

        # Encoder forward pass
        y = self.Encoder(x)

    
        n_hyperencoder_downsamples = self.Hyperprior.analysis_net.n_downsampling_layers
        factor = 2 ** n_hyperencoder_downsamples
        y = utils.pad_factor(y, y.size()[2:], factor)

        hyperinfo = self.Hyperprior(y, spatial_shape=x.size()[2:])

        latents_quantized = hyperinfo.decoded
        total_nbpp = hyperinfo.total_nbpp
        total_qbpp = hyperinfo.total_qbpp

        # Use quantized latents as input to G
        self.Decoder.eval()
        reconstruction = self.Decoder(latents_quantized)
        
        if self.normalize_input_image is True:
            reconstruction = torch.tanh(reconstruction)

        # Undo padding
        reconstruction = reconstruction[:, :, :image_dims[1], :image_dims[2]]

        intermediates = Intermediates(x, reconstruction, latents_quantized, 
            total_nbpp, total_qbpp)

        return intermediates, hyperinfo


    def distortion_loss(self, x_gen, x_real):
        # loss in [0,255] space but normalized by 255 to not be too big
        # - Delegate scaling to weighting
        sq_err = self.squared_difference(x_gen* 255., x_real* 255.) / 255.
        return torch.mean(sq_err)

    def perceptual_ssim_loss(self, x_gen, x_real):
        # Convert PyTorch tensors to NumPy arrays
        x_real = x_real.detach().cpu().numpy()
        x_gen = x_gen.detach().cpu().numpy()

        # Ensure the images are in the correct format (height, width, channels)
        x_real = np.moveaxis(x_real, 1, -1)  # Move the channel dimension to the last axis
        x_gen = np.moveaxis(x_gen, 1, -1)  # Move the channel dimension to the last axis

        # Compute SSIM for the batch with a smaller window size and correct channel_axis
        return np.mean([ssim(x_real[i], x_gen[i], data_range=1, win_size=3, channel_axis=-1) for i in range(x_real.shape[0])])  # Average SSIM for batch



    
    def perceptual_loss_wrapper(self, x_gen, x_real, normalize=True):
        """ Assumes inputs are in [0, 1] if normalize=True, else [-1, 1] """
        LPIPS_loss = self.perceptual_loss.forward(x_gen, x_real, normalize=normalize)
        return torch.mean(LPIPS_loss)
    

    def compression_loss(self, x_real, intermediates, hyperinfo):

        x_gen = intermediates.reconstruction

        if self.normalize_input_image is True:
            # [-1.,1.] -> [0.,1.]
            x_real = (x_real + 1.) / 2.
            x_gen = (x_gen + 1.) / 2.


        distortion_loss = self.distortion_loss(x_gen, x_real)
        perceptual_loss = self.perceptual_ssim_loss(x_gen, x_real) 
        
        # self.store_loss('distortion', distortion_loss.item())
        # self.store_loss('rate_penalty', rate_penalty)
        # self.store_loss('perceptual', perceptual_loss.item())
        # self.store_loss('n_rate', intermediates.n_bpp.item())
        # self.store_loss('q_rate', intermediates.q_bpp.item())
        # self.store_loss('n_rate_latent', hyperinfo.latent_nbpp.item())
        # self.store_loss('q_rate_latent', hyperinfo.latent_qbpp.item())
        # self.store_loss('n_rate_hyperlatent', hyperinfo.hyperlatent_nbpp.item())
        # self.store_loss('q_rate_hyperlatent', hyperinfo.hyperlatent_qbpp.item())

        # self.store_loss('weighted_rate', weighted_rate.item())
        # self.store_loss('compression_loss_sans_G', rec_compression_loss.item())

        return distortion_loss, perceptual_loss


    def forward(self, x):

        intermediates, hyperinfo = self.compression_forward(x)

        reconstruction = intermediates.reconstruction

        mse, ssim_rec, psnr = self.hific_metric(x, intermediates, hyperinfo)

        if self.normalize_input_image is True:
            # [-1.,1.] -> [0.,1.]
            reconstruction = (reconstruction + 1.) / 2.

        reconstruction = torch.clamp(reconstruction, min=0., max=1.)

        return reconstruction, mse, ssim_rec, psnr
        

    def hific_metric(self, x, intermediates, hyperinfo):
        mse, ssim_rec = self.compression_loss(x, intermediates, hyperinfo)

        reconstruction = intermediates.reconstruction
        psnr = metrics.psnr((reconstruction + 1) / 2, (x + 1) / 2, 1, lib = "torch")
        
        # if (self.step_counter % self.log_interval == 1):
        # self.store_loss('perceptual rec', ssim_rec.item())
        # self.store_loss('mse rec', mse.item())
        # self.store_loss('psnr rec', psnr.item())
        return mse, ssim_rec, psnr

