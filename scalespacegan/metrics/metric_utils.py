# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Miscellaneous utilities used internally by the quality metrics."""

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib

# added
from tqdm import tqdm
import math
import random
import sys
from util import renormalize
from util import patch_util
from util.multiscale_util import random_transform_from_scale, transform_from_scale

#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=True, center_zoom=False, cond=False):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache
        self.center_zoom    = center_zoom
        self.cond           = cond

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = pickle.load(f).to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

def iterate_random_labels(opts, batch_size):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            yield c

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None
        self.all_transforms = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def append_transform(self, x):
        # x = transformation matrix
        # note: run this funcation after append, as it does not keep track of num_items
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 3
        assert(x.shape[1] == 3)
        assert(x.shape[2] == 3)
        if self.all_transforms is None:
            self.all_transforms = []
        self.all_transforms.append(x)

    def get_all_transforms_torch(self):
        transforms = np.concatenate(self.all_transforms, axis=0)
        transforms = torch.from_numpy(transforms)
        assert(transforms.shape[0] >= self.max_items)
        return transforms[:self.max_items]

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset_multiscale_continuous(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    
    assert opts.dataset_kwargs.class_name == "training.dataset.MultiscaleImageFolderDataset", "Multiscale-FID requires MultiscaleImageFolderDataset"
    
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    
    # Binning scales 
    dataset.sort_dataset_by_label()
    dataset.floor_labels()

    scale_counts = dataset.count_scale_labels()
    
    stats_multiscale = {}
    for scale, count in scale_counts.items():
        stats_multiscale[scale] = FeatureStats(max_items=count, **stats_kwargs)
        
    progress = opts.progress.sub(tag='dataset features (all scales)', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = []
    for i in range(num_items // opts.num_gpus + 1):
        item = i * opts.num_gpus + opts.rank
        if item < num_items:
            item_subset.append(item)

    for images, labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
       
        # the scales need to be floored again for binning
        labels = torch.Tensor(np.floor(np.clip(labels, 0, max(labels)-0.001)))

        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        
        # if all labels are the same, put the features into the respective bin
        if torch.numel(torch.unique(labels, sorted=False)) == 1:
            label = float(labels[0].cpu())
            stats_multiscale[label].append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        else:
            # scales only change sporadically in the dataset, so we split the batch manually
            labels = torch.cat((labels, torch.tensor([-1]))) # add dummy label to signify end of batch, simplifying the code below
            start_label = labels[0]
            start_idx = 0
            for sample_idx, label in enumerate(labels):
                if label != start_label: # a jump from one scale to another occurred
                    l = float(start_label.cpu())
                    stats_multiscale[l].append_torch(features[start_idx:sample_idx], num_gpus=opts.num_gpus, rank=opts.rank)
                    start_label = label
                    start_idx = sample_idx

        full_num_items = 0
        for s in stats_multiscale.values():
            full_num_items += s.num_items
        progress.update(full_num_items)
        
    return stats_multiscale



#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    while not stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            img = G(z=z, c=next(c_iter), **opts.G_kwargs)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_generator_multiscale_continuous_mix(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # create distribution of scales from data
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    min_scale = np.floor(dataset._get_raw_labels().min())
    scale_counts = dataset.count_scale_labels()
    scale_values_tensor = torch.from_numpy(np.array(list(scale_counts.keys()))) - min_scale
    pmf = torch.zeros((len(scale_counts),))
    for idx, count in enumerate(scale_counts.values()):
        pmf[idx] = count
    pmf /= len(dataset)
    categorical = torch.distributions.categorical.Categorical(pmf)
    
    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    while not stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            scale_batch = categorical.sample((batch_gen,))
            scale_batch = scale_values_tensor[scale_batch]
            if opts.center_zoom:
                transform = transform_from_scale(scale_batch).to(opts.device)
            else:
                transform = random_transform_from_scale(scale_batch).to(opts.device)
            if opts.cond:
                img = G(z=z, c=scale_batch.unsqueeze(1).to(opts.device), transform=transform, **opts.G_kwargs)
            else:
                img = G(z=z, c=next(c_iter), transform=transform, **opts.G_kwargs)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_generator_multiscale_continuous(opts, detector_url, detector_kwargs, scales, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Set up generator.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

    # Initialize.
    max_items_per_scale = stats_kwargs['max_items']
    assert max_items_per_scale is not None
    stats_multiscale = {}
    for scale in scales:
        stats_multiscale[scale] = FeatureStats(**stats_kwargs)
    full_sample_count = len(stats_multiscale) * max_items_per_scale

    progress = opts.progress.sub(tag='generator features (all scales)', num_items=full_sample_count, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)
    
    min_scale = min(scales)

    # Main loop.
    for scale, stats in stats_multiscale.items():
    
        while not stats.is_full():
            images = []
            for _i in range(batch_size // batch_gen):

                z = torch.randn([batch_gen, G.z_dim], device=opts.device)
        
                scale_batch = torch.tensor(scale - min_scale).repeat(batch_gen)
                scale_batch = torch.rand(batch_gen) + scale - min_scale
                
                if opts.center_zoom:
                    transform = transform_from_scale(scale_batch).to(opts.device)
                else:
                    transform = random_transform_from_scale(scale_batch).to(opts.device)
            
                img = G(z=z, c=None, transform=transform, **opts.G_kwargs)
            
                img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                images.append(img)
            images = torch.cat(images)
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])

            features = detector(images, **detector_kwargs)
            stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
            
            full_num_items = 0
            for s in stats_multiscale.values():
                full_num_items += s.num_items
            progress.update(full_num_items)
    
    return stats_multiscale

#----------------------------------------------------------------------------

## additional functions added

# computes feature stats and transformation matrix for patches
def compute_feature_stats_for_dataset_patch(opts, detector_url, detector_kwargs, is_subpatch=False,
                                            rel_lo=0, rel_hi=1, batch_size=64,
                                            data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    assert(opts.dataset_kwargs.class_name == "training.dataset.ImagePatchDataset")
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=8, prefetch_factor=2)

    patch_size = opts.dataset_kwargs.resolution
    scale_anneal = opts.dataset_kwargs.scale_anneal
    scale_min = opts.dataset_kwargs.scale_min
    scale_max = opts.dataset_kwargs.scale_max
    assert(scale_anneal == -1)
    max_size = 0 if scale_min is None else int(patch_size / scale_min)
    min_size = int(patch_size / scale_max)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        if is_subpatch:
            cache_tag = f'patch{patch_size}-subpatch299-min{min_size}max{max_size}-{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        else:
            cache_tag = f'patch{patch_size}-min{min_size}max{max_size}-{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            print("loaded from cache: %s" % cache_tag)
            return FeatureStats.load(cache_file)

    # Initialize.
    assert(max_items is not None)
    assert(opts.num_gpus == 1)
    assert(opts.rank == 0)
    num_items = max_items
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    from torch_utils import misc
    random_seed = opts.dataset_kwargs.random_seed
    sampler = misc.InfiniteSampler(dataset=dataset, rank=opts.rank, num_replicas=opts.num_gpus, seed=random_seed)
    iterator = iter(torch.utils.data.DataLoader(
        dataset=dataset, sampler=sampler, batch_size=batch_size, **data_loader_kwargs))

    num_batches = math.ceil(max_items / batch_size)
    for _ in tqdm(range(num_batches)):
        data, _labels = next(iterator)
        images = data['image']
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        if is_subpatch:
            # crop the image, and save the x,y coords in the transformation matrix
            crops = []
            transform = data['params']['transform'].clone()
            for i, im in enumerate(images):
                crop_width = 299
                x = random.randint(0, im.shape[-1]-crop_width)
                y = random.randint(0, im.shape[-2]-crop_width)
                crops.append(im[:, y:y+crop_width, x:x+crop_width])
                transform[i, 2, 0] = y
                transform[i, 2, 1] = x
            crops = torch.stack(crops)
            features = detector(crops.to(opts.device), **detector_kwargs)
            stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
            stats.append_transform(transform)
        else:
            features = detector(images.to(opts.device), **detector_kwargs)
            stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
            stats.append_transform(data['params']['transform'])
        progress.update(stats.num_items)
        if stats.is_full():
            break
    assert(stats.is_full())

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

# compute features on the full dataset; similar to the above one 
# but changes naming to cache desired resolution
def compute_feature_stats_for_dataset_full(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64,
                                           data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    assert(opts.dataset_kwargs.class_name == "training.dataset.ImageFolderDataset")
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=8, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'full{dataset.resolution}-{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)
        # Load.
        if flag:
            print("loaded from cache: %s" % cache_tag)
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, _labels in tqdm(torch.utils.data.DataLoader(
        dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs)):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

# pieces together a generated full image from multiple base transformations
def compute_feature_stats_for_generator_full(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, **stats_kwargs):
    target_resolution = opts.target_resolution # need to define a target resolution
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # get patch bounding boxes
    from util.patch_util import generate_full_from_patches
    patches = generate_full_from_patches(target_resolution, G.img_resolution)
    assert(opts.G_kwargs == {}) # sanity check

    # Main loop.
    num_batches = math.ceil(stats.max_items / batch_size)
    for _ in tqdm(range(num_batches)):
        images = []
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            ws = G.mapping(z=z, c=next(c_iter), **opts.G_kwargs)
            full = torch.zeros((batch_gen, G.img_channels, target_resolution, target_resolution), device=opts.device)
            for bbox, transform in patches:
                transform = transform.repeat(batch_gen, 1, 1).cuda()
                img = patch_util.scale_condition_wrapper(G, ws, transform, **opts.G_kwargs)
                full[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]] = img
            img = full
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
        if stats.is_full():
            break
    assert(stats.is_full())
    return stats


# upsamples the full image using patches
def compute_feature_stats_for_generator_up(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, **stats_kwargs):
    target_resolution = opts.target_resolution # need to define a target resolution
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # get patch bounding boxes
    from util.patch_util import generate_full_from_patches
    from metrics import equivariance
    patches = generate_full_from_patches(target_resolution, G.img_resolution)

    # Main loop.
    num_batches = math.ceil(stats.max_items / batch_size)
    assert(opts.G_kwargs == {}) # sanity check
    for _ in tqdm(range(num_batches)):
        images = []
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            img = G(z=z, c=next(c_iter), **opts.G_kwargs)
            full = torch.zeros((batch_gen, G.img_channels, target_resolution, target_resolution), device=opts.device)
            for bbox, transform in patches:
                up_patch, mask = equivariance.apply_affine_transformation(img, transform.inverse()[0])
                full[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]] = up_patch
            img = full
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
        if stats.is_full():
            break
    assert(stats.is_full())
    return stats

# generates individual patches for pFID
def compute_feature_stats_for_generator_patch(opts, transformations, detector_url, detector_kwargs,
                                              is_subpatch=False, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, **stats_kwargs):
    assert(opts.num_gpus == 1)
    assert(opts.rank == 0)
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)
    assert(opts.G_kwargs == {}) # sanity check

    counter = 0

    # Main loop.
    num_batches = math.ceil(stats.max_items / batch_size)
    assert(transformations.shape[0] % batch_gen == 0)
    for _ in tqdm(range(num_batches)):
        images = []
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            ws = G.mapping(z=z, c=next(c_iter), **opts.G_kwargs)
            if counter + batch_gen > transformations.shape[0]:
                assert(counter == transformations.shape[0])
                # no more transformations to use
                break
            transform = transformations[counter:counter+batch_gen]
            transform = transform.to(opts.device)
            counter += batch_gen
            if is_subpatch:
                # extract crop locations from transform, reset the transform,
                # and crop the image
                y_start = transform[:, 2, 0].long()
                x_start = transform[:, 2, 1].long()
                crop_width = 299
                transform_new = transform.clone()
                transform_new[:, 2, :2] = 0.
                img = patch_util.scale_condition_wrapper(G, ws, transform_new, **opts.G_kwargs)
                crops = []
                for im, x, y in zip(img, x_start, y_start):
                    crops.append(im[:, y:y+crop_width, x:x+crop_width])
                crops = torch.stack(crops)
                img = crops
            else:
                img = patch_util.scale_condition_wrapper(G, ws, transform, **opts.G_kwargs)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
        if stats.is_full():
            break
    assert(stats.is_full())
    assert(counter == stats.max_items)
    return stats

