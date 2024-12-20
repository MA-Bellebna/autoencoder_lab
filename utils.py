import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import json
import os, time, datetime
import logging
import itertools
from default_config import directories
from collections import OrderedDict
from torchvision.utils import save_image

META_FILENAME = "metadata.json"

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def get_device(is_gpu=True):
    """Return the correct device"""

    return torch.device("cuda" if torch.cuda.is_available() and is_gpu
                        else "cpu")


def use_multi_gpu():
    use_gpu = True if get_device().type == "cuda" else False
    multi_gpu = torch.cuda.device_count() > 1 if use_gpu else False
    
    return multi_gpu

def get_model_device(model):
    """Return the device where the model sits."""
    return next(model.parameters()).device

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def quick_restore_model(model, filename):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["state_dict"])
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pad_factor(input_image, spatial_dims, factor):
    """Pad `input_image` (N,C,H,W) such that H and W are divisible by `factor`."""

    if isinstance(factor, int) is True:
        factor_H = factor
        factor_W = factor_H
    else:
        factor_H, factor_W = factor

    H, W = spatial_dims[0], spatial_dims[1]
    pad_H = (factor_H - (H % factor_H)) % factor_H
    pad_W = (factor_W - (W % factor_W)) % factor_W
    return F.pad(input_image, pad=(0, pad_W, 0, pad_H), mode='reflect')

def get_scheduled_params(param, param_schedule, step_counter, ignore_schedule=False):
    # e.g. schedule = dict(vals=[1., 0.1], steps=[N])
    # reduces param value by a factor of 0.1 after N steps
    if ignore_schedule is False:
        vals, steps = param_schedule['vals'], param_schedule['steps']
        assert(len(vals) == len(steps)+1), f'Mispecified schedule! - {param_schedule}'
        idx = np.where(step_counter < np.array(steps + [step_counter+1]))[0][0]
        param *= vals[idx]
    return param

def update_lr(args, optimizer, itr, logger):
    lr = get_scheduled_params(args.learning_rate, args.lr_schedule, itr)
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
        if old_lr != lr:
            logger.info('=============================')
            logger.info(f'Changing learning rate {old_lr} -> {lr}')
            param_group['lr'] = lr



def setup_generic_signature(args, special_info):

    time_signature = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()).replace(':', '_')
    if args.name is not None:
        args.name = '{}_{}_{}_{}'.format(args.name, args.dataset, special_info, time_signature)
    else:
        name = [special_info]


        name.append(args.regime)


        if args.evaluate is not None:
            name.append("Eval")

        name.append(time_signature)

        
        name = "_".join(name)
        args.name = name #'{}_{}_{}'.format(args.dataset, special_info, time_signature)


    checkpoint_folder = []
    # Append the relevant strings to the list if their corresponding tasks are present
    if args.default_task in args.tasks:
        checkpoint_folder.append(args.default_task)


    # Join the components with underscores to form the final folder name
    checkpoint_folder = "_".join(checkpoint_folder)

    if args.default_task in args.tasks:
        print(args.name)
        args.snapshot = os.path.join(directories.experiments, args.name)
    else:
        args.snapshot = directories.baseline_experiments

        args.snapshot += "_{}".format(args.regime)
            

        if os.path.exists(args.snapshot):
            raise FileExistsError(f"Baseline already exists: {args.snapshot}")

    args.checkpoints_save = os.path.join(args.snapshot, checkpoint_folder)
    args.figures_save = os.path.join(args.snapshot, 'figures')
    args.storage_save = os.path.join(args.snapshot, 'storage')
    args.tensorboard_runs = os.path.join(args.snapshot, 'tensorboard')

    makedirs(args.snapshot)
    makedirs(args.checkpoints_save)
    makedirs(args.figures_save)
    makedirs(args.storage_save)
    makedirs(os.path.join(args.tensorboard_runs, 'train'))
    makedirs(os.path.join(args.tensorboard_runs, 'val'))
    makedirs(os.path.join(args.tensorboard_runs, 'jpegai'))

    return args

def save_metadata(metadata, directory='results', filename=META_FILENAME, **kwargs):
    """ Save the metadata of a training directory.
    Parameters
    ----------
    metadata:
        Object to save
    directory: string
        Path to folder where to save model. For example './experiments/runX'.
    kwargs:
        Additional arguments to `json.dump`
    """

    makedirs(directory)
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True)  #, **kwargs)

def save_model(model, optimizers, mean_epoch_loss, epoch, device, args, logger, multigpu=False):

    directory = args.checkpoints_save
    makedirs(directory)
    model.cpu()  # Move model parameters to CPU for consistency when restoring

    metadata = dict(image_dims=args.image_dims, epoch=epoch, steps=model.step_counter)
    args_d = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('_') or 'logger' in n))
    metadata.update(args_d)
    timestamp = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now())
    args_d['timestamp'] = timestamp
    
    model_name = args.name
    metadata_path = os.path.join(directory, 'metadata/model_{}_metadata.json'.format(model_name))
    makedirs(os.path.join(directory, 'metadata'))
    
    if not os.path.isfile(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True)
            
    model_path = os.path.join(directory, 'best_checkpoint.pt'.format(model_name, epoch, model.step_counter, timestamp))

    save_dict = {   'model_state_dict': model.module.state_dict() if args.multigpu is True else model.state_dict(),
                    'compression_optimizer_state_dict': optimizers['amort'].state_dict(),
                    'hyperprior_optimizer_state_dict': optimizers['hyper'].state_dict(),
                    'epoch': epoch,
                    'steps': model.step_counter,
                    'args': args_d,
                }

    if model.use_discriminator is True:
        save_dict['discriminator_state_dict'] = model.module.Discriminator.state_dict() \
            if args.multigpu is True else model.Discriminator.state_dict()
        save_dict['discriminator_optimizer_state_dict'] = optimizers['disc'].state_dict()

    torch.save(save_dict, f=model_path)
    logger.info('Saved model at Epoch {} to {}'.format(epoch, model_path))
    
    model.to(device)  # Move back to device
    return model_path
   

def load_model(save_path, logger, device, model_type=None, model_mode=None, current_args_d=None, prediction=True, 
    strict=False, silent=False):

    start_time = time.time()
    from src.model import Model
    checkpoint = torch.load(save_path)
    loaded_args_d = checkpoint['args']

    args = Struct(**loaded_args_d)

    if current_args_d is not None:
        if silent is False:
            for k,v in current_args_d.items():
                try:
                    loaded_v = loaded_args_d[k]
                except KeyError:
                    logger.warning('Argument {} (value {}) not present in recorded arguments. Using current argument.'.format(k,v))
                    continue

                if loaded_v !=v:
                    logger.warning('Current argument {} (value {}) does not match recorded argument (value {}). Recorded argument will be overriden.'.format(k, v, loaded_v))

        # HACK
        loaded_args_d.update(current_args_d)
        args = Struct(**loaded_args_d)

    if model_type is None:
        model_type = args.model_type

    if model_mode is None:
        model_mode = args.model_mode

    args.ignore_schedule = True
    
    # Backward compatibililty
    if hasattr(args, 'use_latent_mixture_model') is False:
        args.use_latent_mixture_model = False
    if hasattr(args, 'sample_noise') is False:
        args.sample_noise = False
        args.noise_dim = 0

    model = Model(args, logger, model_type=model_type, model_mode=model_mode)


    # `strict` False if warmstarting
    # model.load_state_dict(checkpoint['model_state_dict'], strict=strict)



    # sd = {}
    # for name, value in torch.load("/home/bellelbn/DL/jpegai/Experiment_II/Outputs/out_Adapted_LV_MTL_ALL_20231216_011335/saved_models/Encoder_FFX_lr_0.00010 momentum_0.85 epochs_80 bs_32_an 0.083_ bn 0.246_NormalizedLoss_best.ckpt").items():
    # for name, value in torch.load("/home/bellelbn/DL/jpegai/Experiment_II/Outputs/out_Global_framework_ALL_20231216_011335/saved_models/Encoder_lr_0.00010 momentum_0.85 epochs_80 bs_32 Rec_0.083 HR_0.101 FFX_0.228_NormalizedLoss_best.ckpt").items():
    #     if "module" in name:
    #         sd[name.replace("module.", "Encoder.")] = value
    
    # print("=" * 100)
    # print("loading state dict")
    # print("=" * 100)

    # model.load_state_dict(sd, strict=False)

    logger.info('Loading model ...')
    if silent is False:
        logger.info('MODEL TYPE: {}'.format(model_type))
        logger.info('MODEL MODE: {}'.format(model_mode))
        logger.info(model)
        logger.info('Trainable parameters:')
        for n, p in model.named_parameters():
            logger.info('{} - {}'.format(n, p.shape))

        logger.info("Number of trainable parameters: {}".format(count_parameters(model)))
    logger.info("Estimated model size (under fp32): {:.3f} MB".format(count_parameters(model) * 4. / 10**6))
    logger.info('Model init {:.3f}s'.format(time.time() - start_time))

    model = model.to(device)

    if prediction is True:
        model.eval()
        optimizers = None
    else:
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
            
        if args.sample_noise is True:
            optimizers['amort'].add_param_group({'params': list(model.Generator.latent_noise_map.parameters())})
        
        optimizers['amort'].load_state_dict(checkpoint['compression_optimizer_state_dict'])
        optimizers['hyper'].load_state_dict(checkpoint['hyperprior_optimizer_state_dict'])
        if (model.use_discriminator is True) and ('disc' in optimizers.keys()):
            try:
                optimizers['disc'].load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
            except KeyError:
                pass

        model.train()

    return args, model, optimizers


def logger_setup(logpath, filepath, package_files=[]):
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s', 
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO'.upper())

    stream = logging.StreamHandler()
    stream.setLevel('INFO'.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    info_file_handler = logging.FileHandler(logpath, mode="a")
    info_file_handler.setLevel('INFO'.upper())
    info_file_handler.setFormatter(formatter)
    logger.addHandler(info_file_handler)

    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())
    return logger

def log_summaries(args, writer, metrics, step, mode, use_discriminator=False):
# def log_summaries(args, writer, storage, loss, ssim_rec, ssim_zoom, psnr_rec, psnr_zoom, cosine_ffx, step, mode, use_discriminator=False):

    
    loss = metrics['loss']
    ssim_rec = metrics['ssim_rec']
    psnr_rec = metrics['psnr_rec']


    writer.add_scalar('{}/loss'.format(mode), loss.avg, step)

    if args.default_task in args.tasks:
        weighted_compression_scalars = ['compression_loss_sans_G',
                                        'weighted_rate',
                                        'perceptual']

        weighted_compression_scalars.append('rate_penalty')

        compression_scalars = ['n_rate', 'q_rate', 'n_rate_latent' ,'q_rate_latent', 
            'n_rate_hyperlatent', 'q_rate_hyperlatent', 'perceptual']
        gan_scalars = ['disc_loss', 'gen_loss', 'weighted_gen_loss', 'D_gen', 'D_real']

        compression_loss_breakdown = dict(total_comp=metrics['compression_loss_sans_G'].avg,
                                        weighted_rate=metrics['weighted_rate'].avg,
                                        #   weighted_distortion=metrics['weighted_distortion'].avg,
                                        weighted_perceptual=metrics['perceptual'].avg)

        for scalar in weighted_compression_scalars:
            writer.add_scalar('weighted_compression/{}'.format(scalar), metrics[scalar].avg, step)
        
        for scalar in compression_scalars:
            writer.add_scalar('compression/{}'.format(scalar), metrics[scalar].avg, step)

        # if use_discriminator is True:
        #     compression_loss_breakdown['weighted_gen_loss'] = metrics['weighted_gen_loss'].avg
        #     for scalar in gan_scalars:
        #         writer.add_scalar('GAN/{}'.format(scalar), metrics[scalar].avg, step)

        # Breakdown overall loss
        writer.add_scalars('compression_loss_breakdown', compression_loss_breakdown, step)

        writer.add_scalar('{}/ssim_rec_avg'.format(mode), ssim_rec.avg, step)
        writer.add_scalar('{}/psnr_rec_avg'.format(mode), psnr_rec.avg, step)


def log(model, storage, epoch, idx, mean_epoch_loss, current_loss, best_loss, start_time, epoch_start_time, 
        batch_size, avg_bpp, header='[TRAIN]', logger=None, writer=None, **kwargs):
    
    improved = ''
    t0 = epoch_start_time
    
    if current_loss < best_loss:
        best_loss = current_loss
        improved = '[*]'  
    
    storage['epoch'].append(epoch)
    storage['mean_compression_loss'].append(mean_epoch_loss)
    storage['time'].append(time.time())

    # Tensorboard
    if writer is not None:
        log_summaries(writer, storage, model.step_counter, use_discriminator=model.use_discriminator)

    if logger is not None:
        report_f = logger.info   
    else:
        report_f = print
    
    report_f('================>>>')
    report_f(header)
    report_f('================>>>')
    if header == '[TRAIN]':
        report_f(model.args.snapshot)
        report_f("Epoch {} | Mean epoch comp. loss: {:.3f} | Current comp. loss: {:.3f} | "
                 "Rate: {} examples/s | Time: {:.1f} s | Improved: {}".format(epoch, mean_epoch_loss, current_loss,
                 int(batch_size*idx / ((time.time()-t0))), time.time()-start_time, improved))
    else:
        report_f("Epoch {} | Mean epoch comp. loss: {:.3f} | Current comp. loss: {:.3f} | Improved: {}".format(epoch, 
                 mean_epoch_loss, current_loss, improved))
    report_f('========>')
    report_f("Rate-Distortion:")
    report_f("Weighted Rate: {:.3f} | Perceptual: {:.3f} | "
             "Rate Penalty: {:.3f}".format(storage['weighted_rate'][-1], 
                                           storage['perceptual'][-1], storage['rate_penalty'][-1]))
    report_f('========>')
    report_f("Rate Breakdown")
    report_f("avg. original bpp: {:.3f} | n_bpp (total): {:.3f} | q_bpp (total): {:.3f} | n_bpp (latent): {:.3f} | q_bpp (latent): {:.3f} | "
             "n_bpp (hyp-latent): {:.3f} | q_bpp (hyp-latent): {:.3f}".format(avg_bpp, storage['n_rate'][-1], storage['q_rate'][-1], 
             storage['n_rate_latent'][-1], storage['q_rate_latent'][-1], storage['n_rate_hyperlatent'][-1], storage['q_rate_hyperlatent'][-1]))
    if model.use_discriminator is True:
        report_f('========>')
        report_f("Generator-Discriminator:")
        report_f("G Loss: {:.3f} | D Loss: {:.3f} | D(gen): {:.3f} | D(real): {:.3f}".format(storage['gen_loss'][-1],
                storage['disc_loss'][-1], storage['D_gen'][-1], storage['D_real'][-1]))
    return best_loss


def save_images(writer, step, real, decoded, fname):

    imgs = torch.cat((real,decoded), dim=0)
    save_image(imgs, fname, nrow=4, normalize=True, scale_each=True)
    writer.add_images('gen_recon', imgs, step)

def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    return x