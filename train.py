"""
Learnable generative compression model modified from [1], 
implemented in Pytorch.

Example usage:
python3 train.py -h

[1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
    arXiv:2006.09965 (2020).
"""
import numpy as np
from PIL import Image
import os, glob, time, datetime
import logging, pickle, argparse
import functools, itertools

from tqdm import tqdm, trange
from collections import defaultdict
import copy

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms

# Custom modules
from src.model import Model
from src.helpers import utils, datasets
from default_config import hific_args, mse_lpips_args, directories, ModelModes, ModelTypes

from sklearn.decomposition import PCA
from tqdm import tqdm

# go fast boi!!
torch.backends.cudnn.benchmark = True

def create_model(args, device, logger, storage, storage_val, storage_test):

    start_time = time.time()
    model = Model(args, logger, storage, storage_val, storage_test, model_type=args.model_type)
    # logger.info(model)
    logger.info('Trainable parameters:')
    for n, p in model.named_parameters():
        logger.info('{} - {}'.format(n, p.shape))

    # Freeze the Generator : 
    model.Decoder.eval()
    for param in model.Decoder.parameters():
        param.requires_grad = False 

    logger.info("Number of trainable parameters: {}".format(utils.count_parameters(model)))
    logger.info("Estimated size (under fp32): {:.3f} MB".format(utils.count_parameters(model) * 4. / 10**6))
    logger.info('Model init {:.3f}s'.format(time.time() - start_time))

    return model

def optimize_loss(loss, opt, retain_graph=False):
    loss.backward(retain_graph=retain_graph)
    opt.step()
    opt.zero_grad()

def optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt, gammas_opt = None):
    compression_loss.backward()
    amortization_opt.step()
    hyperlatent_likelihood_opt.step()
    amortization_opt.zero_grad()
    hyperlatent_likelihood_opt.zero_grad()

    if gammas_opt is not None:
        gammas_opt.step()
        gammas_opt.zero_grad()
        


def test(args, model, epoch, idx, data, test_data, test_bpp, device, epoch_test_loss, storage, best_val_loss, 
         start_time, epoch_start_time, logger, train_writer, val_writer):

    model.eval()  
    with torch.no_grad():
        data = data.to(device, dtype=torch.float)

        losses, intermediates = model(data, return_intermediates=True, writeout=False)
        utils.save_images(train_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
            fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TRAIN_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))

        test_data = test_data.to(device, dtype=torch.float)
        losses, intermediates = model(test_data, return_intermediates=True, writeout=True)
        utils.save_images(val_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
            fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TEST_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))
    
        compression_loss = losses['compression'] 
        epoch_test_loss.append(compression_loss.item())
        mean_test_loss = np.mean(epoch_test_loss)
        
        best_val_loss = utils.log(model, storage, epoch, idx, mean_test_loss, compression_loss.item(), 
                                     best_val_loss, start_time, epoch_start_time, 
                                     batch_size=data.shape[0], bpp=test_bpp.mean().item(),header='[TEST]', 
                                     logger=logger, writer=val_writer)
        
    return best_val_loss, epoch_test_loss

def load_generator(model,path):
    """ Loading the pretrained weights from the HIFI ckpt to the Generator 
    
    path : path of HIFI ckpt
    model : Generator 

    """
    load = torch.load(path)

    new_state_dict = {}
    for name, weight in load['model_state_dict'].items():
        if 'Generator' in name:
            new_state_dict[name] = weight

    model.eval()
    for param in model.parameters():
        param.requires_grad = False 

    return model.load_state_dict(new_state_dict,strict=False)


def eval_hific(args, model, test_loader, device):

    val_loss = utils.AverageMeter()
    ssim_rec_val = utils.AverageMeter()    
    psnr_rec_val = utils.AverageMeter()

    metrics = {
    'loss': val_loss,
    'ssim_rec': ssim_rec_val,
    'psnr_rec': psnr_rec_val,
    }
    
    model.eval()

    idx = 0
    with torch.no_grad():
        model.model_mode = ModelModes.EVALUATION

        for idx, (data, labels) in enumerate(tqdm(test_loader, desc='Val'), 0):

            data = data.to(device, dtype=torch.float)
            model(data, return_intermediates=True, writeout=True)
            update_performance(args, metrics, model.storage_val)
            print_performance(args, len(test_loader), metrics, idx)


def save_image(filename, reconst):
    transform = transforms.ToPILImage()
    image = transform(reconst.squeeze())
    # Save the image
    image.save(filename)


def compare_params(initial_params, current_params):
    for key in initial_params.keys():
        if not torch.equal(initial_params[key], current_params[key]):
            return False
    return True


def update_performance(args, metrics, store):
    loss = metrics['loss']
    ssim_rec = metrics['ssim_rec']
    psnr_rec = metrics['psnr_rec']

    batch_size = args.batch_size

    if args.default_task in args.tasks:
        ssim = store['perceptual rec'][-1]
        ssim_rec.update(ssim, batch_size)

        psnr = store['psnr rec'][-1]
        psnr_rec.update(psnr, batch_size)


def print_performance(args, data_size, metrics, idx):
    
    loss = metrics['loss']
    ssim_rec = metrics['ssim_rec']
    psnr_rec = metrics['psnr_rec']
 
    display = '\nValidation: [{0}/{1}]'.format(idx, data_size)

    if loss is not None:
        display += '\tloss {loss.val:.3f} ({loss.avg:.3f})'.format(loss = loss)

    if args.default_task in args.tasks:
        display += '\tpsnr_rec {psnr_rec.val:.3f} ({psnr_rec.avg:.3f})\t ssim_rec {ssim_rec.val:.3f} ({ssim_rec.avg:.3f})'.format(psnr_rec = psnr_rec, ssim_rec = ssim_rec)
    

    display += '\n'
    
    logger.info(display)


if __name__ == '__main__':

    description = "Learnable generative compression."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options - see `default_config.py` for full options
    general = parser.add_argument_group('General options')
    general.add_argument("-n", "--name", default=None, help="Identifier for checkpoints and metrics.")
    general.add_argument("-mt", "--model_type", required=True, default=ModelTypes.COMPRESSION, choices=(ModelTypes.COMPRESSION, ModelTypes.COMPRESSION_GAN), 
        help="Type of model - with or without GAN component")
    general.add_argument("-regime", "--regime", choices=('low','med','high'), default='high', help="Set target bit rate - Low (0.14), Med (0.30), High (0.45)")
    general.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU ID.")
    general.add_argument("-os_gpu", "--os_gpu", type=int, default=0, help="GPU ID.")
    general.add_argument("-log_intv", "--log_interval", type=int, default=hific_args.log_interval, help="Number of steps between logs.")
    general.add_argument("-save_intv", "--save_interval", type=int, default=hific_args.save_interval, help="Number of steps between checkpoints.")
    general.add_argument("-multigpu", "--multigpu", help="Toggle data parallel capability using torch DataParallel", action="store_true")
    general.add_argument("-norm", "--normalize_input_image", help="Normalize input images to [-1,1]", action="store_true")
    general.add_argument("-eval", "--evaluate", help="Evaluate the framework before training", action = "store_true")
    general.add_argument('-bs', '--batch_size', type=int, default=hific_args.batch_size, help='input batch size for training')
    general.add_argument('--save', type=str, default='experiments', help='Parent directory for stored information (checkpoints, logs, etc.)')
    general.add_argument("-lt", "--likelihood_type", choices=('gaussian', 'logistic'), default='gaussian', help="Likelihood model for latents.")
    general.add_argument("-force_gpu", "--force_set_gpu", help="Set GPU to given ID", action="store_true")
    general.add_argument("-LMM", "--use_latent_mixture_model", help="Use latent mixture model as latent entropy model.", action="store_true")


    # Optimization-related options
    optim_args = parser.add_argument_group("Optimization-related options")
    optim_args.add_argument('-steps', '--n_steps', type=float, default=hific_args.n_steps, 
        help="Number of gradient steps. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument('-epochs', '--n_epochs', type=int, default=hific_args.n_epochs, 
        help="Number of passes over training dataset. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument("-lr", "--learning_rate", type=float, default=hific_args.learning_rate, help="Optimizer learning rate.")
    optim_args.add_argument("-wd", "--weight_decay", type=float, default=hific_args.weight_decay, help="Coefficient of L2 regularization.")

    # Architecture-related options
    arch_args = parser.add_argument_group("Architecture-related options")
    arch_args.add_argument('-lc', '--latent_channels', type=int, default=hific_args.latent_channels,
        help="Latent channels of bottleneck nominally compressible representation.")
    arch_args.add_argument('-nrb', '--n_residual_blocks', type=int, default=hific_args.n_residual_blocks,
        help="Number of residual blocks to use in Generator.")
    
    arch_args.add_argument('-t', '--tasks', choices=['HiFiC'], nargs='+', default=[hific_args.default_task], help="Choose which task to add into the MTL framework")
    # Warmstart adversarial training from autoencoder/hyperprior
    warmstart_args = parser.add_argument_group("Warmstart options")
    warmstart_args.add_argument("-warmstart", "--warmstart", help="Warmstart adversarial training from autoencoder + hyperprior ckpt.", action="store_true")

    cmd_args = parser.parse_args()

    if (cmd_args.gpu != 0) or (cmd_args.force_set_gpu is True):
        torch.cuda.set_device(cmd_args.gpu)

    if cmd_args.model_type == ModelTypes.COMPRESSION:
        args = mse_lpips_args
    elif cmd_args.model_type == ModelTypes.COMPRESSION_GAN:
        args = hific_args

    start_time = time.time()
    device = utils.get_device()

    # Override default arguments from config file with provided command line arguments
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d, cmd_args_d = dictify(args), vars(cmd_args)
    args_d.update(cmd_args_d)
    args = utils.Struct(**args_d)
    args = utils.setup_generic_signature(args, special_info=args.model_type)
    args.target_rate = args.target_rate_map[args.regime]
    args.lambda_A = args.lambda_A_map[args.regime]
    args.n_steps = int(args.n_steps)

    storage = defaultdict(list)
    storage_val = defaultdict(list)
    storage_test = defaultdict(list)
    logger = utils.logger_setup(logpath=os.path.join(args.snapshot, 'logs.txt'), filepath=os.path.abspath(__file__))

    args.checkpoint = "/home/bellelbn/DL/datapart/models/hific_hi.pt"
    model = create_model(args, device, logger, storage, storage_val, storage_test)
    model = model.to(device)
    
    multi_gpu = torch.cuda.device_count() > 1 if torch.cuda.is_available() else False
    if multi_gpu:
        model = nn.DataParallel(model)


    amortization_parameters = itertools.chain.from_iterable(
        [am.parameters() for am in model.amortization_models])

    hyperlatent_likelihood_parameters = model.Hyperprior.hyperlatent_likelihood.parameters()

    amortization_opt = torch.optim.Adam(amortization_parameters,
        lr=args.learning_rate)
    
    hyperlatent_likelihood_opt = torch.optim.Adam(hyperlatent_likelihood_parameters, 
        lr=args.learning_rate)
    
    optimizers = dict(amort=amortization_opt, hyper=hyperlatent_likelihood_opt)


    if model.use_discriminator is True:
        discriminator_parameters = model.Discriminator.parameters()
        disc_opt = torch.optim.Adam(discriminator_parameters, lr=args.learning_rate)
        optimizers['disc'] = disc_opt

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and args.multigpu is True:
        # Not supported at this time
        raise NotImplementedError('MultiGPU not supported yet.')
        logger.info('Using {} GPUs.'.format(n_gpus))
        model = nn.DataParallel(model)

    logger.info('MODEL TYPE: {}'.format(args.model_type))
    logger.info('MODEL MODE: {}'.format(args.model_mode))
    logger.info('TASKS: {}'.format(args.tasks))
    logger.info('BITRATE REGIME: {}'.format(args.regime))
    logger.info('SAVING LOGS/CHECKPOINTS/RECORDS TO {}'.format(args.snapshot))
    logger.info('USING DEVICE {}'.format(device))
    logger.info('USING GPU ID {}'.format(args.gpu))


    eval_hific(args, model, test_loader, device)