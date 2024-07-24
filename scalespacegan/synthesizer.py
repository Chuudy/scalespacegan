import pickle
import torch
import numpy as np
from util import patch_util, renormalize
import click
import os
import dnnlib
from util.multiscale_util import tensor_to_image
from util.misc import save_image

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

#--------------------------------------------------------------------


@click.command()

# Required.
@click.option('--model_dir',            help='Where the model pikle is located',    metavar='DIR',    required=True)
@click.option('--model_file',           help='Filename of the model pickle',        metavar='DIR',    required=True)
@click.option('--output_dir',           help='Where the model pikle is located',    metavar='DIR',    required=True)

# Optional
@click.option('--resolution',           help='Filename of the model pickle',        metavar='INT',    type=click.IntRange(min=256),   default=256)
@click.option('--seed',                 help='Filename of the model pickle',        metavar='INT',                                    default=0)


#--------------------------------------------------------------------


def main(**kwargs):

    opts = dnnlib.EasyDict(kwargs)

    pkl_filepath = os.path.join(opts.model_dir, opts.model_file)
    full_size = opts.resolution
    seed = opts.seed

    output_dir = opts.output_dir
    output_filepath = os.path.join(output_dir, f"{full_size}.png")

    with open(pkl_filepath, 'rb') as f:
        G_base = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

    rng = np.random.RandomState(seed)
    z = torch.from_numpy(rng.standard_normal(G_base.z_dim)).float()
    z = z[None].cuda()
    c = None

    ws = G_base.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
    full = torch.zeros([1, 3, full_size, full_size])
    patches = patch_util.generate_full_from_patches(full_size, G_base.img_resolution)
    for bbox, transform in patches:
        transform[0:2,2] /= 2.25
        transform = transform.unsqueeze(0).cuda()
        g_kwargs = {"transform": transform, "noise_mode": "const"}
        img = G_base.synthesis(ws, **g_kwargs)
        full[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]] = img
    img_normalized = tensor_to_image(full, denormalize=True)

    os.makedirs(output_dir, exist_ok=True)
    save_image(img_normalized, output_filepath)


#--------------------------------------------------------------------


if __name__ == "__main__":    
    main()



