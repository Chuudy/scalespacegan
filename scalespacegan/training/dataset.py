# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

pyspng = None # disable pyspng for image resizing on load
# https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from util import patch_util
import random


#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 1], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels
    
    def _set_raw_labels(self, raw_labels):
        self._raw_labels = raw_labels

    def _set_raw_idx(self, raw_idx):
        self._raw_idx = raw_idx

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        if not self.is_patch: # full image loader 
            image = self._load_raw_image(self._raw_idx[idx])
            assert isinstance(image, np.ndarray)
            assert list(image.shape) == self.image_shape
            assert image.dtype == np.uint8
            if self._xflip[idx]:
                assert image.ndim == 3 # CHW
                image = image[:, :, ::-1]
            return image.copy(), self.get_label(idx)
        else: # image patch loader
            # handle xflips when loading the image
            data = self._load_raw_image(self._raw_idx[idx], self._xflip[idx])
            assert isinstance(data, dict)
            assert list(data['image'].shape) == self.image_shape
            assert data['image'].dtype == np.uint8
            data['image'] = data['image'].copy()
            return data, self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        else:
            label = np.array([label])
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class BaseImageDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution = None,      # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):

        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            if os.path.isfile(self._path + '_cache.txt'):
                # use cache file if it exists
                with open(self._path + '_cache.txt') as cache:
                    self._all_fnames = set([line.strip() for line in cache])
            else:
                print("Walking dataset...")
                self._all_fnames = [os.path.relpath(os.path.join(root, fname), start=self._path)
                                    for root, _dirs, files in os.walk(self._path, followlinks=True) for fname in files]
                with open(self._path + '_cache.txt', 'w') as cache:
                    [cache.write("%s\n" % fname) for fname in self._all_fnames]
                self._all_fnames = set(self._all_fnames)
                print("Done walking")
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        if resolution is not None:
            raw_shape = [len(self._image_fnames)] + [3, resolution, resolution]
        else:
            # do not resize it to determine initial shape (will fail if images not square)
            raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0, resize=False).shape)

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


class ImageFolderDataset(BaseImageDataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution = None,      # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.is_patch = False
        super().__init__(path=path, resolution=resolution,  **super_kwargs)

    def _load_raw_image(self, raw_idx, resize=True):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = PIL.Image.open(f).convert('RGB')
                w, h = image.size
                if resize:
                    # at this point it should be square
                    assert(image.size[0] == image.size[1])
                    target_size = tuple(self.image_shape[1:])
                    if image.size != target_size:
                        # it should only downsize, but there are a small number
                        # of images in the datasets that are a few pixels
                        # smaller than 256, so allow a small leeway
                        assert(target_size[-1] < image.size[-1] + 10)
                        image = image.resize(target_size, PIL.Image.ANTIALIAS)
                image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

class ImagePatchDataset(BaseImageDataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution,             # patch size
        scale_min,              # minimum scale of the patches  (largest image size)
        scale_max,              # maximum scale of the patches (smallest image size)
        scale_anneal=-1,        # annealing rate
        random_crop=True,       # add random crop for non-square images
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        assert(resolution is not None) # patch resolution must be specified

        # annealing not implemented, need to update iteration counter and
        # adjust counter when resuming training
        assert(scale_anneal == -1)

        # crop sampler
        self.patch_size = resolution
        self.random_crop = random_crop
        self.sampler = patch_util.PatchSampler(
            patch_size=self.patch_size, scale_anneal=scale_anneal,
            min_scale=scale_min, max_scale=scale_max)
        self.is_patch = True

        super().__init__(path=path, resolution=resolution,  **super_kwargs)

    def _load_raw_image(self, raw_idx, is_flipped):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = PIL.Image.open(f).convert('RGB')

        # first, flip image if necessary
        if is_flipped:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        # add random crop if necessary
        if self.random_crop:
            w, h = image.size
            min_size = min(w, h)
            x_start = random.randint(0, max(0, w - min_size))
            y_start = random.randint(0, max(0, h - min_size))
            image = image.crop((x_start, y_start, x_start+min_size, y_start+min_size))
        else:
            # otherwise, center crop
            w, h = image.size
            min_size = min(w, h)
            if w != h:
                if w == min_size:
                    x_start = 0
                    y_start = (h - min_size) // 2
                else:
                    x_start = (w - min_size) // 2
                    y_start = 0
                image = image.crop((x_start, y_start, x_start+min_size, y_start+min_size))

        # sample the resize and crop parameters
        crop, params = self.sampler.sample_patch(image)
        image = np.asarray(crop)
        image = image.transpose(2, 0, 1) # HWC => CHW
        data = {
            'image': image,
            'params': params,
        }
        return data



class MultiscaleImageFolderDataset(ImageFolderDataset):
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)

    # make sure that labels are not converted into one-hot vectors
    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        
        assert(labels), "Scale labels not found."
        
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype(np.float32)
        return labels

    @property
    def label_shape(self):
        if self._label_shape is None:
            self._label_shape = (1,)
        return list(self._label_shape)

    # get min and max scales in dataset
    def scale_range(self):
        raw_labels = self._get_raw_labels()
        min_scale = np.min(raw_labels)
        max_scale = np.max(raw_labels)
        return min_scale, max_scale
        
    # count images per scale
    def count_scale_labels(self):
        flat_labels = self._get_raw_labels().flatten()
        distinct_scales = set(flat_labels)
        scale_count = {}
        for s in distinct_scales:
            scale_count[s] = np.count_nonzero(flat_labels == s)
        return scale_count
    
    def sort_dataset_by_label(self):
        raw_labels = self._get_raw_labels()
        sorted_indices = sorted(range(len(raw_labels)), key=raw_labels.__getitem__)
        sorted_raw_labels = np.array([raw_labels[i] for i in sorted_indices])
        sorted_raw_idx = np.array([self._raw_idx[i] for i in sorted_indices])
        self._set_raw_labels(sorted_raw_labels)
        self._set_raw_idx(sorted_raw_idx)

    def floor_labels(self):
        raw_labels = self._get_raw_labels()
        raw_labels_floored = np.floor(np.clip(raw_labels, 0, max(raw_labels)-0.001))
        self._set_raw_labels(raw_labels_floored)

