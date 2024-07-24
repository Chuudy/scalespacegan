# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import re
import contextlib
import numpy as np
import torch
import warnings
import dnnlib
import copy

#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

#----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.

try:
    nan_to_num = torch.nan_to_num # 1.8.0a0
except AttributeError:
    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)

#----------------------------------------------------------------------------
# Symbolic assert.

try:
    symbolic_assert = torch._assert # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert # 1.7.0

#----------------------------------------------------------------------------
# Context manager to temporarily suppress known warnings in torch.jit.trace().
# Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672

@contextlib.contextmanager
def suppress_tracer_warnings():
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)

#----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().

def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

#----------------------------------------------------------------------------
# Function decorator that calls torch.autograd.profiler.record_function().

def profiled_function(fn):
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator

#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.

class InfiniteBalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5, boost_last=1):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size
        self.boost_last = boost_last-1
        self.create_scale_index_dictionary()

    def __iter__(self):
        # order = np.arange(len(self.dataset))
        # rnd = None
        # window = 0

        if self.shuffle:
            for i in range(self.n_scales):
                rnd = np.random.RandomState(self.seed)
                order = self.scale_index_dictionary[i]
                rnd.shuffle(order)
                window = int(np.rint(order.size * self.window_size))
                self.scale_index_dictionary[i] = order
                self.window[i] = window

        while True:
            # select random scale
            curr_scale_index = int(np.clip(rnd.randint(self.n_scales + self.boost_last), a_min=None, a_max=self.n_scales-1))

            # read all necessary variables for current scale
            idx = self.scale_iterator[curr_scale_index]
            order = self.scale_index_dictionary[curr_scale_index]
            window = self.window[curr_scale_index]

            # get sample and shuffle
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

            # write modifed values back to object fields
            self.scale_iterator[curr_scale_index] = idx
            self.scale_index_dictionary[curr_scale_index] = order


    def create_scale_index_dictionary(self):
        tmp_dataset = copy.deepcopy(self.dataset)
        indices = tmp_dataset._raw_idx
        scales = tmp_dataset._get_raw_labels()
        raw_scales_floored = np.floor(np.clip(scales, 0, max(scales)-0.001))
        distinct_scales = set(raw_scales_floored)
        scale_index_dictionary = {}
        scale_iterator = {}
        scale_n_elements = {}        
        window = {}
        for scale in distinct_scales:
            scale_indices = np.where(raw_scales_floored == scale)[0]
            scale_index_dictionary[scale] = scale_indices
            scale_iterator[scale] = 0
            window[scale] = 0
            scale_n_elements[scale] = len(scale_indices)
        self.scale_index_dictionary = scale_index_dictionary
        self.scale_iterator = scale_iterator
        self.scale_n_elements = scale_n_elements
        self.window = window
        self.n_scales = len(distinct_scales)
        pass



#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset skewing distribution towards high magnitudes over time
# indefinitely, shuffling items as it goes.

class InfiniteSkewedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5, boost_first=1, boost_last=1, max_skew_kimgs=5000, warmup_kimgs=0, curr_kimgs=0):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size
        self.boost_first = boost_first
        self.boost_last = boost_last
        self.warmup_imgs = warmup_kimgs*1000
        self.max_skew_imgs = max_skew_kimgs*1000 - self.warmup_imgs
        self.curr_skew_imgs = curr_kimgs * 1000
        self.curr_max_prob = 1
        self.create_scale_index_dictionary()

    def __iter__(self):
        # order = np.arange(len(self.dataset))
        # rnd = None
        # window = 0

        if self.shuffle:
            for i in range(self.n_scales):
                rnd = np.random.RandomState(self.seed)
                order = self.scale_index_dictionary[i]
                rnd.shuffle(order)
                window = int(np.rint(order.size * self.window_size))
                self.scale_index_dictionary[i] = order
                self.window[i] = window

        while True:
            # select random scale

            tmp_skew_imgs = max([self.curr_skew_imgs - self.warmup_imgs, 0])
            signed_skew_imgs = self.curr_skew_imgs - self.warmup_imgs
            self.curr_skew_imgs += 1

            if signed_skew_imgs < 0:
                tmp_skew_imgs = np.abs(signed_skew_imgs)
                alpha = tmp_skew_imgs / self.warmup_imgs
                max_prob = alpha * self.boost_first + (1-alpha)
                self.curr_max_prob = -max_prob
                probs = 2**np.linspace(max_prob, 1, self.n_scales)
            else:
                alpha = tmp_skew_imgs / self.max_skew_imgs      
                max_prob = alpha * self.boost_last + (1-alpha)
                self.curr_max_prob = max_prob
                probs = np.linspace(1, max_prob, self.n_scales)
            m = torch.distributions.categorical.Categorical(torch.tensor(probs))
            curr_scale_index = int(m.sample())

            # curr_scale_index = int(np.clip(rnd.randint(self.n_scales + self.boost_last), a_min=None, a_max=self.n_scales-1))

            # read all necessary variables for current scale
            idx = self.scale_iterator[curr_scale_index]
            order = self.scale_index_dictionary[curr_scale_index]
            window = self.window[curr_scale_index]

            # get sample and shuffle
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

            # write modifed values back to object fields
            self.scale_iterator[curr_scale_index] = idx
            self.scale_index_dictionary[curr_scale_index] = order


    def create_scale_index_dictionary(self):
        tmp_dataset = copy.deepcopy(self.dataset)
        indices = tmp_dataset._raw_idx
        scales = tmp_dataset._get_raw_labels()
        raw_scales_floored = np.floor(np.clip(scales, 0, max(scales)-0.001))
        distinct_scales = set(raw_scales_floored)
        scale_index_dictionary = {}
        scale_iterator = {}
        scale_n_elements = {}        
        window = {}        
        min_scale = np.min(list(distinct_scales))
        for scale in distinct_scales:
            scale_index = int(scale - min_scale)
            scale_indices = np.where(raw_scales_floored == scale)[0]
            scale_index_dictionary[scale_index] = scale_indices
            scale_iterator[scale_index] = 0
            window[scale_index] = 0
            scale_n_elements[scale_index] = len(scale_indices)
        self.scale_index_dictionary = scale_index_dictionary
        self.scale_iterator = scale_iterator
        self.scale_n_elements = scale_n_elements
        self.window = window
        self.n_scales = len(distinct_scales)
        pass
            


#----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)

#----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.

@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield

#----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.

def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (tensor == other).all(), fullname

#----------------------------------------------------------------------------
# Print summary table of module hierarchy.

def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(dnnlib.EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs

#----------------------------------------------------------------------------
