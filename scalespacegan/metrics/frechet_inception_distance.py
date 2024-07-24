# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import sys
import numpy as np
import torch
import scipy.linalg
from . import metric_utils

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------

def compute_fid_multiscale_continuous_mix(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator_multiscale_continuous_mix(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------

def compute_fid_multiscale_continuous(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    feature_stats_real = metric_utils.compute_feature_stats_for_dataset_multiscale_continuous(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real)

    if opts.num_gpus > 1:
        torch.distributed.barrier()

    mu_sigma_real = {}
    for scale, stats in feature_stats_real.items():
        if stats.num_items > 0:
            mu_sigma_real[scale] = stats.get_mean_cov()
    data_scales = [float(s) for s in mu_sigma_real]
    
    feature_stats_gen = metric_utils.compute_feature_stats_for_generator_multiscale_continuous(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        scales=data_scales,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen)

    if opts.num_gpus > 1:
        torch.distributed.barrier()
    
    mu_sigma_gen = {}
    for scale, stats in feature_stats_gen.items():
        if stats.num_items > 0:
            mu_sigma_gen[scale] = stats.get_mean_cov()

    assert mu_sigma_real.keys() == mu_sigma_gen.keys(), "Scales of real and generated data don't match"
    
    fids = {}
    fid_mean = 0
    metric_base_string = f'fid_multiscale_{int(round(num_gen/1000))}k_'
    for scale in mu_sigma_real:
        fid = float('nan')
        if opts.rank == 0:
            mu_real, sigma_real = mu_sigma_real[scale]
            mu_gen, sigma_gen = mu_sigma_gen[scale]
            m = np.square(mu_gen - mu_real).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
            fid = float(np.real(m + np.trace(sigma_gen + sigma_real - s * 2)))
        fids[metric_base_string + str(scale)] = fid
        fid_mean += fid
    
    fid_mean /= len(fids)
    fids[metric_base_string + 'mean'] = fid_mean

    return fids

#----------------------------------------------------------------------------

def compute_fid_full(opts, resolution, max_real, num_gen, mode='full'):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    assert(opts.dataset_kwargs.resolution == resolution)
    opts.target_resolution = resolution

    # real image features
    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset_full(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    # gen features
    if mode == 'full':
        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator_full(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()
    elif mode == 'up':
        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator_up(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()
    else:
        raise NotImplementedError

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------

def compute_fid_patch(opts, resolution, max_real, num_gen, mode='patch'):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    assert(opts.dataset_kwargs.resolution == resolution)
    opts.target_resolution = resolution
    assert(max_real is not None)

    # real image features
    if mode.split('-')[0] == 'patch':
        stats = metric_utils.compute_feature_stats_for_dataset_patch(
           opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
           rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real)
        mu_real, sigma_real = stats.get_mean_cov()
        transformations = stats.get_all_transforms_torch()
    elif mode.split('-')[0] == 'subpatch':
        stats = metric_utils.compute_feature_stats_for_dataset_patch(
           opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
           is_subpatch=True, rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real)
        mu_real, sigma_real = stats.get_mean_cov()
        transformations = stats.get_all_transforms_torch()

    # gen features
    if mode == 'patch':
        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator_patch(
            opts=opts, transformations=transformations,
            detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()
    elif mode == 'subpatch':
        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator_patch(
            opts=opts, transformations=transformations, is_subpatch=True,
            detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()
    else:
        raise NotImplementedError

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)
