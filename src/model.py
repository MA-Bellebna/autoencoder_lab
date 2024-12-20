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
from src import hyperprior
from src.loss import losses
from src.helpers import maths, datasets, utils, metrics
from src.network import encoder, generator, discriminator, hyper
from src.loss.perceptual_similarity import perceptual_loss as ps 

from default_config import ModelModes, ModelTypes, hific_args, directories, args

from pytorch_msssim import ssim, ms_ssim

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

        #Trainable
        self.Encoder = encoder.Encoder(self.image_dims, C=self.latent_channels)
        
        self.Encoder = self.load_submodel(self.Encoder, self.checkpoint, freeze = self.optimal_latent, sub_model = 'Encoder')

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


        self.Hyperprior = self.load_submodel(self.Hyperprior, self.checkpoint, freeze=self.optimal_latent, sub_model = "Hyperprior")

        
        self.init_optimizer()

        self.squared_difference = torch.nn.MSELoss(reduction='none')
        # Expects [-1,1] images or [0,1] with normalize=True flag
        self.perceptual_loss = ps.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available(), gpu_ids=[args.gpu])

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
            
        load = torch.load(path)

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

        if self.model_mode == ModelModes.EVALUATION and (self.training is False):
            n_encoder_downsamples = self.Encoder.n_downsampling_layers
            factor = 2 ** n_encoder_downsamples
            x = utils.pad_factor(x, x.size()[2:], factor)

        # Encoder forward pass
        y = self.Encoder(x)

        if self.model_mode == ModelModes.EVALUATION and (self.training is False):
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
        if self.model_mode == ModelModes.EVALUATION and (self.training is False):
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
        """ Assumes inputs are in [0, 1] if normalize=True, else [-1, 1] """
        SSIM_loss = 1 - ssim(x_gen, x_real, data_range=1, size_average=True)
        return SSIM_loss
    
    def perceptual_msssim_loss(self, x_gen, x_real):
        """ Assumes inputs are in [0, 1] if normalize=True, else [-1, 1] """
        SSIM_loss = 1 - ms_ssim(x_gen, x_real, data_range=1, size_average=True)
        return SSIM_loss

    
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


    def forward(self, x, train_generator=False, return_intermediates=False, writeout=True):

        self.writeout = writeout

        losses = dict()
        if train_generator is True:
            # Define a 'step' as one cycle of G-D training
            self.step_counter += 1

        intermediates, hyperinfo = self.compression_forward(x)

        if self.model_mode == ModelModes.EVALUATION:
            reconstruction = intermediates.reconstruction

            rec_compression_loss, mse, ssim_rec, weighted_rate = self.hific_metric(x, intermediates, hyperinfo)

            if self.normalize_input_image is True:
                # [-1.,1.] -> [0.,1.]
                reconstruction = (reconstruction + 1.) / 2.

            reconstruction = torch.clamp(reconstruction, min=0., max=1.)

            return reconstruction, intermediates.q_bpp
        
        mse, ssim_rec, psnr = self.hific_metric(x, intermediates, hyperinfo)


        losses['mse'] = mse
        losses['ssim'] = 1 - ssim_rec
        losses['psnr'] = psnr

        if return_intermediates is True:
            return losses, intermediates
        else:
            return losses

    def hific_metric(self, x, intermediates, hyperinfo):
        mse, ssim_rec = self.compression_loss(x, intermediates, hyperinfo)

        reconstruction = intermediates.reconstruction
        psnr = metrics.psnr((reconstruction + 1) / 2, (x + 1) / 2, 1, lib = "torch")
        
        # if (self.step_counter % self.log_interval == 1):
        self.store_loss('perceptual rec', ssim_rec.item())
        self.store_loss('mse rec', mse.item())
        self.store_loss('psnr rec', psnr.item())
        return mse, ssim_rec, psnr


if __name__ == '__main__':

    compress_test = True

    if compress_test is True:
        model_mode = ModelModes.EVALUATION
    else:
        model_mode = ModelModes.TRAINING

    logger = utils.logger_setup(logpath=os.path.join(directories.experiments, 'logs'), filepath=os.path.abspath(__file__))
    device = utils.get_device()
    logger.info(f'Using device {device}')
    storage_train = defaultdict(list)
    storage_val = defaultdict(list)

    model = Model(hific_args, logger, storage_train, storage_val, model_mode=model_mode, model_type=ModelTypes.COMPRESSION_GAN)
    model.to(device)

    logger.info(model)

    transform_param_names = list()
    transform_params = list()
    logger.info('ALL PARAMETERS')
    for n, p in model.named_parameters():
        if ('Encoder' in n) or ('Generator' in n):
            transform_param_names.append(n)
            transform_params.append(p)
        if ('analysis' in n) or ('synthesis' in n):
            transform_param_names.append(n)
            transform_params.append(p)      
        logger.info(f'{n} - {p.shape}')

    logger.info('AMORTIZATION PARAMETERS')
    amortization_named_parameters = itertools.chain.from_iterable(
            [am.named_parameters() for am in model.amortization_models])
    for n, p in amortization_named_parameters:
        logger.info(f'{n} - {p.shape}')

    logger.info('AMORTIZATION PARAMETERS')
    for n, p in zip(transform_param_names, transform_params):
        logger.info(f'{n} - {p.shape}')

    logger.info('HYPERPRIOR PARAMETERS')
    for n, p in model.Hyperprior.hyperlatent_likelihood.named_parameters():
        logger.info(f'{n} - {p.shape}')

    if compress_test is False:
        logger.info('DISCRIMINATOR PARAMETERS')
        for n, p in model.Discriminator.named_parameters():
            logger.info(f'{n} - {p.shape}')

    logger.info("Number of trainable parameters: {}".format(utils.count_parameters(model)))
    logger.info("Estimated size: {} MB".format(utils.count_parameters(model) * 4. / 10**6))

    B = 10
    shape = [B, 3, 256, 256]
    x = torch.randn(shape).to(device)

    start_time = time.time()

    if compress_test is True:
        model.eval()
        logger.info('Starting compression with input shape {}'.format(shape))
        compression_output = model.compress(x)
        reconstruction = model.decompress(compression_output)

        logger.info(f"n_bits: {compression_output.total_bits}")
        logger.info(f"bpp: {compression_output.total_bpp}")
        logger.info(f"MSE: {torch.mean(torch.square(reconstruction - x)).item()}")
    else:
        logger.info('Starting forward pass with input shape {}'.format(shape))
        losses = model(x)
        compression_loss, disc_loss = losses['compression'], losses['disc']

    logger.info('Delta t {:.3f}s'.format(time.time() - start_time))

