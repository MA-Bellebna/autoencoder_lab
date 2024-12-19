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

Disc_out= namedtuple("disc_out",
    ["D_real", "D_gen", "D_real_logits", "D_gen_logits"])


class Model(nn.Module):

    def __init__(self, args, logger, storage_train=defaultdict(list), storage_val=defaultdict(list), storage_test=defaultdict(list), storage_kodak=defaultdict(list), model_mode=ModelModes.TRAINING, 
            model_type=ModelTypes.COMPRESSION):
        super(Model, self).__init__()

        """
        Builds hific model from submodels in network.
        """
        self.args = args
        self.model_mode = model_mode
        self.model_type = model_type
        self.logger = logger
        self.log_interval = args.log_interval
        self.storage_train = storage_train
        self.storage_val = storage_val
        self.storage_test = storage_test
        self.storage_kodak = storage_kodak
        self.step_counter = 0
        self.val_data = "selfie"
        
        # if self.args.original_w == True:  
        self.args.checkpoint = r"/home/bellelbn/DL/datapart/models/hific_{}.pt".format('hi' if self.args.regime == "high" else 'med' if self.args.regime == "med" else "low")
        self.args.checkpoint_high = r"/home/bellelbn/DL/datapart/models/hific_hi.pt"
        self.args.checkpoint_med = r"/home/bellelbn/DL/datapart/models/hific_med.pt"
        self.args.checkpoint_low = r"/home/bellelbn/DL/datapart/models/hific_low.pt"


        self.logger.info('Loading HiFiC regime {} from : {}'.format(self.args.regime, self.args.checkpoint))

        self.optimal_latent = False if args.default_task in self.args.tasks else True

    
        if self.args.use_latent_mixture_model is True:
            self.args.latent_channels = self.args.latent_channels_DLMM

        if not hasattr(ModelTypes, self.model_type.upper()):
            raise ValueError("Invalid model_type: [{}]".format(self.model_type))
        if not hasattr(ModelModes, self.model_mode.upper()):
            raise ValueError("Invalid model_mode: [{}]".format(self.model_mode))

        self.image_dims = self.args.image_dims  # Assign from dataloader
        self.batch_size = self.args.batch_size

        self.entropy_code = False
        if model_mode == ModelModes.EVALUATION:
            self.entropy_code = True

        #Trainable
        self.Encoder = encoder.Encoder(self.image_dims, self.batch_size, C=self.args.latent_channels,
            channel_norm=self.args.use_channel_norm)
        
        self.Encoder = self.load_submodel(self.Encoder, self.args.checkpoint, freeze = self.optimal_latent, sub_model = 'Encoder')


        #Non Trainable
        self.Decoder = generator.Generator(self.image_dims, self.batch_size, C=self.args.latent_channels,
            n_residual_blocks=self.args.n_residual_blocks, channel_norm=self.args.use_channel_norm, sample_noise=
            self.args.sample_noise, noise_dim=self.args.noise_dim)
        
        self.Decoder = self.load_submodel(self.Decoder, self.args.checkpoint, False)   # Load pretrained HiFI weights 

        if self.args.use_latent_mixture_model is True:
            self.Hyperprior = hyperprior.HyperpriorDLMM(bottleneck_capacity=self.args.latent_channels,
                likelihood_type=self.args.likelihood_type, mixture_components=self.args.mixture_components, entropy_code=self.entropy_code)
        else:
            self.Hyperprior = hyperprior.Hyperprior(bottleneck_capacity=self.args.latent_channels,
                likelihood_type=self.args.likelihood_type, entropy_code=self.entropy_code)


        self.Hyperprior = self.load_submodel(self.Hyperprior, self.args.checkpoint, freeze=self.optimal_latent, sub_model = "Hyperprior")

        
        self.init_optimizer()


        # Use discriminator if GAN mode enabled and in training/validation
        self.use_discriminator = (
            self.model_type == ModelTypes.COMPRESSION_GAN
            and (self.model_mode != ModelModes.EVALUATION)
        )

        if self.use_discriminator is True:
            assert self.args.discriminator_steps > 0, 'Must specify nonzero training steps for D!'
            self.discriminator_steps = self.args.discriminator_steps
            self.logger.info('GAN mode enabled. Training discriminator for {} steps.'.format(
                self.discriminator_steps))
            self.Discriminator = discriminator.Discriminator(image_dims=self.image_dims,
                context_dims=self.args.latent_dims, C=self.args.latent_channels)
            self.gan_loss = partial(losses.gan_loss, args.gan_loss_type)
        else:
            self.discriminator_steps = 0
            self.Discriminator = None

        self.squared_difference = torch.nn.MSELoss(reduction='none')
        # Expects [-1,1] images or [0,1] with normalize=True flag
        self.perceptual_loss = ps.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available(), gpu_ids=[args.gpu])

        self.perceptual_ssim_loss

    def init_optimizer(self):
        self.amortization_models = []
        if (self.args.default_task in self.args.tasks):
            self.logger.info("Add Encoder + Hyperprior to the optimizer")
            self.amortization_models.append(self.Encoder)
            self.amortization_models.append(self.Decoder)
            self.amortization_models.extend(self.Hyperprior.amortization_models)
                

    def load_checkpoint(self, checkpoint):
        self.logger.info('Loading checkpoint {}'.format(checkpoint))

        if self.args.default_task in self.args.tasks:
            self.logger.info("Loading Encoder + Hyperprior pipelines")
            self.Encoder = self.load_submodel(self.Encoder, checkpoint, False, sub_model="Encoder")
            self.Hyperprior = self.load_submodel(self.Hyperprior, checkpoint, False, sub_model="Hyperprior")

            self.Decoder = generator.Generator(self.image_dims, self.batch_size, C=self.args.latent_channels,
            n_residual_blocks=self.args.n_residual_blocks, channel_norm=self.args.use_channel_norm, sample_noise=
            self.args.sample_noise, noise_dim=self.args.noise_dim)
            self.Decoder = self.load_submodel(self.Decoder, checkpoint, False, sub_model="Decoder")



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

    def store_loss(self, key, loss):
        assert type(loss) == float, 'Call .item() on loss before storage'

        storage = self.storage_val
        
        if self.writeout is True:
            storage[key].append(loss)


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
        
        if self.args.normalize_input_image is True:
            reconstruction = torch.tanh(reconstruction)

        # Undo padding
        if self.model_mode == ModelModes.EVALUATION and (self.training is False):
            reconstruction = reconstruction[:, :, :image_dims[1], :image_dims[2]]

        intermediates = Intermediates(x, reconstruction, latents_quantized, 
            total_nbpp, total_qbpp)

        return intermediates, hyperinfo




    def discriminator_forward(self, intermediates, train_generator):
        """ Train on gen/real batches simultaneously. """
        x_gen = intermediates.reconstruction
        x_real = intermediates.input_image

        # Alternate between training discriminator and compression models
        if train_generator is False:
            x_gen = x_gen.detach()

        D_in = torch.cat([x_real, x_gen], dim=0)

        latents = intermediates.latents_quantized.detach()
        latents = torch.repeat_interleave(latents, 2, dim=0)

        D_out, D_out_logits = self.Discriminator(D_in, latents)
        D_out = torch.squeeze(D_out)
        D_out_logits = torch.squeeze(D_out_logits)

        D_real, D_gen = torch.chunk(D_out, 2, dim=0)
        D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)

        return Disc_out(D_real, D_gen, D_real_logits, D_gen_logits)

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

        if self.args.normalize_input_image is True:
            # [-1.,1.] -> [0.,1.]
            x_real = (x_real + 1.) / 2.
            x_gen = (x_gen + 1.) / 2.


        distortion_loss = self.distortion_loss(x_gen, x_real)
        perceptual_loss = self.perceptual_ssim_loss(x_gen, x_real) 
        #self.perceptual_loss_wrapper(x_gen, x_real, normalize=True)

        
        weighted_rate, rate_penalty = losses.weighted_rate_loss(self.args, total_nbpp=intermediates.n_bpp,
            total_qbpp=intermediates.q_bpp, step_counter=self.step_counter, ignore_schedule=self.args.ignore_schedule)

    
        rec_compression_loss = distortion_loss + weighted_rate

        
        self.store_loss('distortion', distortion_loss.item())
        self.store_loss('rate_penalty', rate_penalty)
        self.store_loss('perceptual', perceptual_loss.item())
        self.store_loss('n_rate', intermediates.n_bpp.item())
        self.store_loss('q_rate', intermediates.q_bpp.item())
        self.store_loss('n_rate_latent', hyperinfo.latent_nbpp.item())
        self.store_loss('q_rate_latent', hyperinfo.latent_qbpp.item())
        self.store_loss('n_rate_hyperlatent', hyperinfo.hyperlatent_nbpp.item())
        self.store_loss('q_rate_hyperlatent', hyperinfo.hyperlatent_qbpp.item())

        self.store_loss('weighted_rate', weighted_rate.item())
        self.store_loss('compression_loss_sans_G', rec_compression_loss.item())

        return rec_compression_loss, distortion_loss, perceptual_loss, weighted_rate


    
    def GAN_loss(self, intermediates, train_generator=False):
        """
        train_generator: Flag to send gradients to generator
        """
        disc_out = self.discriminator_forward(intermediates, train_generator)
        D_loss = self.gan_loss(disc_out, mode='discriminator_loss')
        G_loss = self.gan_loss(disc_out, mode='generator_loss')

        # Bookkeeping 
        if (self.step_counter % self.log_interval == 1):
            self.store_loss('D_gen', torch.mean(disc_out.D_gen).item())
            self.store_loss('D_real', torch.mean(disc_out.D_real).item())
            self.store_loss('disc_loss', D_loss.item())
            self.store_loss('gen_loss', G_loss.item())
            self.store_loss('weighted_gen_loss', (self.args.beta * G_loss).item())

        return D_loss, G_loss

    def compress(self, x, silent=False):

        """
        * Pass image through encoder to obtain latents: x -> Encoder() -> y 
        * Pass latents through hyperprior encoder to obtain hyperlatents:
          y -> hyperencoder() -> z
        * Encode hyperlatents via nonparametric entropy model. 
        * Pass hyperlatents through mean-scale hyperprior decoder to obtain mean,
          scale over latents: z -> hyperdecoder() -> (mu, sigma).
        * Encode latents via entropy model derived from (mean, scale) hyperprior output.
        """

        assert self.model_mode == ModelModes.EVALUATION and (self.training is False), (
            f'Set model mode to {ModelModes.EVALUATION} for compression.')
        
        spatial_shape = tuple(x.size()[2:])

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

        compression_output = self.Hyperprior.compress_forward(y, spatial_shape)
        attained_hbpp = 32 * len(compression_output.hyperlatents_encoded) / np.prod(spatial_shape)
        attained_lbpp = 32 * len(compression_output.latents_encoded) / np.prod(spatial_shape)
        attained_bpp = 32 * ((len(compression_output.hyperlatents_encoded) +  
            len(compression_output.latents_encoded)) / np.prod(spatial_shape))

        if silent is False:
            self.logger.info('[ESTIMATED]')
            self.logger.info(f'BPP: {compression_output.total_bpp:.3f}')
            self.logger.info(f'HL BPP: {compression_output.hyperlatent_bpp:.3f}')
            self.logger.info(f'L BPP: {compression_output.latent_bpp:.3f}')

            self.logger.info('[ATTAINED]')
            self.logger.info(f'BPP: {attained_bpp:.3f}')
            self.logger.info(f'HL BPP: {attained_hbpp:.3f}')
            self.logger.info(f'L BPP: {attained_lbpp:.3f}')

        return compression_output


    def decompress(self, compression_output):

        """
        * Recover z* from compressed message.
        * Pass recovered hyperlatents through mean-scale hyperprior decoder obtain mean,
          scale over latents: z -> hyperdecoder() -> (mu, sigma).
        * Use latent entropy model to recover y* from compressed image.
        * Pass quantized latent through generator to obtain the reconstructed image.
          y* -> Generator() -> x*.
        """

        assert self.model_mode == ModelModes.EVALUATION and (self.training is False), (
            f'Set model mode to {ModelModes.EVALUATION} for decompression.')

        latents_decoded = self.Hyperprior.decompress_forward(compression_output, device=utils.get_device())

        # Use quantized latents as input to G
        reconstruction = self.Decoder(latents_decoded)

        if self.args.normalize_input_image is True:
            reconstruction = torch.tanh(reconstruction)

        # Undo padding
        image_dims = compression_output.spatial_shape
        reconstruction = reconstruction[:, :, :image_dims[0], :image_dims[1]]

        if self.args.normalize_input_image is True:
            # [-1.,1.] -> [0.,1.]
            reconstruction = (reconstruction + 1.) / 2.

        reconstruction = torch.clamp(reconstruction, min=0., max=1.)

        return reconstruction

    def forward(self, x, train_generator=False, return_intermediates=False, writeout=True):

        self.writeout = writeout

        losses = dict()
        if train_generator is True:
            # Define a 'step' as one cycle of G-D training
            self.step_counter += 1


        intermediates, hyperinfo = self.compression_forward(x)


        if self.model_mode == ModelModes.EVALUATION:
            reconstruction = intermediates.reconstruction

            # compression_model_loss, _ = self.compression_loss(intermediates, hyperinfo)
            rec_compression_loss, mse, ssim_rec, weighted_rate = self.hific_metric(x, intermediates, hyperinfo)

            if self.args.normalize_input_image is True:
                # [-1.,1.] -> [0.,1.]
                reconstruction = (reconstruction + 1.) / 2.

            reconstruction = torch.clamp(reconstruction, min=0., max=1.)

            return reconstruction, intermediates.q_bpp
        
        rec_compression_loss, mse, msssim_loss, ssim_rec, weighted_rate = self.hific_metric(x, intermediates, hyperinfo)
    
        if not(self.optimal_latent):
            compression_model_loss = mse
        else : 
            compression_model_loss = 0


        if self.args.default_task in self.args.tasks:
            compression_model_loss += weighted_rate

        if self.use_discriminator is True:
            # Only send gradients to generator when training generator via
            # `train_generator` flag
            D_loss, G_loss = self.GAN_loss(intermediates, train_generator)
            weighted_G_loss = self.args.beta * G_loss
            compression_model_loss += weighted_G_loss
            losses['disc'] = D_loss
        
        losses['compression'] = compression_model_loss

        # Bookkeeping 
        # if (self.step_counter % self.log_interval == 1):
        self.store_loss('weighted_compression_loss', compression_model_loss.item())        
        if return_intermediates is True:
            return losses, intermediates
        else:
            return losses

    def hific_metric(self, x, intermediates, hyperinfo):
        rec_compression_loss, mse, ssim_rec, weighted_rate = self.compression_loss(x, intermediates, hyperinfo)

        reconstruction = intermediates.reconstruction
        psnr = metrics.psnr((reconstruction + 1) / 2, (x + 1) / 2, 1, lib = "torch")
        
        # if (self.step_counter % self.log_interval == 1):
        self.store_loss('perceptual rec', ssim_rec.item())
        self.store_loss('mse rec', mse.item())
        self.store_loss('psnr rec', psnr.item())
        return rec_compression_loss, mse, ssim_rec, weighted_rate


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

