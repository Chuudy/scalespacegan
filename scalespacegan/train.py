# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

import numpy as np

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run, opts):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    if(opts.iter_dir):
        c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)
    else:
        c.run_dir = outdir

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir,exist_ok=True)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data, training_mode):
    try:
        dataset_class_name = 'training.dataset.MultiscaleImageFolderDataset' if 'multiscale' in training_mode else 'training.dataset.ImageFolderDataset'
        dataset_kwargs = dnnlib.EasyDict(class_name=dataset_class_name, path=data, use_labels=True, max_size=None, xflip=False)
        
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        try:
            dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        except AssertionError:
            print("Cannot determine default dataset resolution, will try to use specified arguments")
            dataset_kwargs.resolution = None
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['stylegan3-r', 'stylegan3-ms']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='ada', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--auto-resume',  help='Auto resume from given directory', metavar='[PATH|URL]',  type=str)
@click.option('--tick_stop',    help='Number of ticks in current session of training',          type=click.IntRange(min=-1), default=-1)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid1k_full_multiscale_mix', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.FloatRange(min=0.01), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)
@click.option('--iter-dir',     help='Create iterative directory if dir exists', metavar='BOOL',type=bool, default=True, show_default=False)
@click.option('--progfreqs',            help='Enable progressive frequencies',                          is_flag=True)
@click.option('--centerzoom',           help='Zoom into image center only',                             is_flag=True)

# additional options for scale sampling
@click.option('--uniform_sampling',     help='Sample all magnitudes in a uniform manner instead of sampling bound to input data',               is_flag=True)
@click.option('--skewed_sampling',      help='Sample magnitudes in a skewed manner towards the high magnitudes, number means how many kimgs are needed to reach the final distribution',   type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--warmup_kimgs',         help='Describes how many kimgs need to be processed before sampling starts to skew',   type=click.IntRange(min=0), default=0, show_default=True)

# additional base options
@click.option('--training_mode',        help='generator training mode', type=click.Choice(['global', 'multiscale']), required=True)

# additional options for scale consistency loss
@click.option('--consistency_lambda',           help='scale consistency regularization weight', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--consistency_mode',             help='scale consistency loss mode', type=click.Choice(['multiscale_forward', 'multiscale_inverse']), default='multiscale_inverse', show_default=True)
@click.option('--consistency_loss_select',      help='selection of losses used for scale consistency: L1 or (L1 and LPIPS)', type=click.Choice(['both', 'l1']), default='both', show_default=True)
@click.option('--consistency_scale_min',        help='minimum scale consistency reference scale delta', metavar='FLOAT', type=click.FloatRange(min=1), default=1.0, show_default=True)
@click.option('--consistency_scale_max',        help='maximum scale consistency reference scale delta', metavar='FLOAT', type=click.FloatRange(min=1), default=2.0, show_default=True)
@click.option('--consistency_scale_sampling',   help='specifies the sampling/clamping method for scale consistency reference', type=click.Choice(['unbounded', 'uniform', 'beta']), default='beta', show_default=True)
@click.option('--random_gradient_off',          help='disables gradient backprop randomly through one or the other branch in scale consistency loss', is_flag=True)

# additional options for scale normalization
@click.option('--scale_mapping_norm',       help='normalization type for scale mapping branch', type=click.Choice(['positive', 'zerocentered']), default='positive')

# additional misc options
@click.option('--debug',                    help='does not save pickles', is_flag=True)
@click.option('--n_redist_layers',          help='number of layers frequencies are redistributed across', type=click.IntRange(min=1), default=1)
@click.option('--last_redist_layer',        help='last layer frequencies are redistributed across', type=click.IntRange(min=1), default=10)
@click.option('--auto_last_redist_layer',   help='sets first max resolution layer as last frequency redistribution layer', is_flag=True)
@click.option('--num_layers',               help='Number of generator layers (does not include synthesis input and final toRGB layer)', type=click.IntRange(min=10), default=14)
@click.option('--scale_count_fixed',        help='Decreases scale count in SynthesisInput', type=click.IntRange(min=-1), default=5)
@click.option('--boost_first',              help='multiplies the chance of last magnitude being sampled during the training', type=click.IntRange(min=1), default=1)
@click.option('--boost_last',               help='multiplies the chance of last magnitude being sampled during the training', type=click.IntRange(min=1), default=1)
@click.option('--distribution_threshold',   help='Sets distribution threshold factor', metavar='FLOAT', type=click.FloatRange(min=1), default=2*np.sqrt(2), show_default=True)

# options for binned frequency distribution
@click.option('--n_bins',                   help='Sets number of bins for binned frequencies', type=click.IntRange(min=1), default=1)
@click.option('--bin_transform_offset',     help='Sets transform offset for binned frequencies', type=click.FloatRange(min=0), default=3)
@click.option('--bin_blend_width',          help='Sets blending range for binned frequencies', metavar='FLOAT', type=click.FloatRange(min=0.001), default=1, show_default=True)
@click.option('--bin_blend_offset',         help='Sets blending offset for binned frequencie', metavar='FLOAT', type=click.FloatRange(min=-2), default=-1, show_default=True)



def main(**kwargs):

    print("\nTraining starts\n", flush=True)

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Set training mode
    training_mode = c.G_kwargs.training_mode = opts.training_mode
    c.G_kwargs.progressive_freqs = opts.progfreqs
    
    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data, training_mode=training_mode)
    if not 'multiscale' in training_mode:
        if opts.cond and not c.training_set_kwargs.use_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        c.training_set_kwargs.use_labels = opts.cond
    else:
        c.training_set_kwargs.use_labels = True # use scale labels
    c.training_set_kwargs.xflip = opts.mirror

    # Modify parameters for scalespacegan training
    if 'multiscale' in training_mode:
        # added G_kwargs
        c.G_kwargs.scale_mapping_kwargs = dnnlib.EasyDict(
            scale_mapping_norm = opts.scale_mapping_norm
        )
        # added training options
        c.added_kwargs = dnnlib.EasyDict(
            consistency_lambda=opts.consistency_lambda,
            consistency_mode=opts.consistency_mode,
            consistency_scale_min=opts.consistency_scale_min,
            consistency_scale_max=opts.consistency_scale_max,
            consistency_loss_select=opts.consistency_loss_select,
            consistency_scale_sampling=opts.consistency_scale_sampling,            
            random_gradient_off = opts.random_gradient_off
        )

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = (8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    c.center_zoom = opts.centerzoom

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        c.loss_kwargs.style_mixing_prob = 0.9 # Enable style mixing regularization.
        c.loss_kwargs.pl_weight = 2 # Enable path length regularization.
        c.G_reg_interval = 4 # Enable lazy regularization for G.
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    else:
        c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
        c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
        c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
        c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32 # Fade out the blur during the first N kimg.
        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
        if opts.cfg == 'stylegan3-ms':            
            c.G_kwargs.channel_base *= 4 # Quadruple the number of feature maps.
            c.G_kwargs.channel_max *= 4
        
    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        # c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Auto resume.
    if opts.auto_resume:

        pickle_dir = os.path.join(opts.outdir, f'{opts.auto_resume}-{desc}')
        last_pickle_dummy_filepath = os.path.join(pickle_dir, 'last.pkl')
        if(os.path.exists(last_pickle_dummy_filepath)):
            with open(last_pickle_dummy_filepath) as f:
                last_pickle_filename = f.readline()
            last_pickle_filepath = os.path.join(pickle_dir, last_pickle_filename)
            c.resume_pkl = last_pickle_filepath
            c.resume_kimg = int(re.search(r"\d{6}", last_pickle_filename).group())
            c.auto_resume_p = True
        opts.iter_dir = False
        opts.outdir = pickle_dir

    if opts.uniform_sampling:
        c.uniform_sampling = True

    if opts.debug:
        c.debug = True

    c.G_kwargs.n_redist_layers = opts.n_redist_layers
    c.G_kwargs.last_redist_layer = opts.last_redist_layer
    c.G_kwargs.num_layers = opts.num_layers
    c.G_kwargs.scale_count_fixed = opts.scale_count_fixed

    c.boost_first = opts.boost_first
    c.boost_last = opts.boost_last
    c.max_skew_kimgs = opts.skewed_sampling
    c.warmup_kimgs = opts.warmup_kimgs

    c.G_kwargs.auto_last_redist_layer = True if opts.auto_last_redist_layer else False
    c.G_kwargs.distribution_threshold = opts.distribution_threshold

    c.G_kwargs.n_bins = opts.n_bins
    c.G_kwargs.bin_transform_offset = opts.bin_transform_offset
    c.G_kwargs.bin_blend_width = opts.bin_blend_width
    c.G_kwargs.bin_blend_offset = opts.bin_blend_offset

    c.cond = opts.cond
    c.tick_stop = opts.tick_stop

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run, opts=opts)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
