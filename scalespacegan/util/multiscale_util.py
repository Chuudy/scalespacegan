import torch
from util.misc import isotropic_scaling_matrix_2D_torch, translation_matrix_2D_torch
import numpy as np
from pkg_resources import parse_version

_use_pytorch_1_11_api = parse_version(torch.__version__) >= parse_version('1.11.0a') # Allow prerelease builds of 1.11
_use_pytorch_1_12_api = parse_version(torch.__version__) >= parse_version('1.12.0a') # Allow prerelease builds of 1.12

# a batch of 2D affine transformation matrices which, given a scale, produce random crops of the unit square
def random_transform_from_scale(scale):
    batch_size = scale.shape[0]
    scale_factor = torch.exp2(scale)
    s = isotropic_scaling_matrix_2D_torch(1/scale_factor)
    translation = 2 * torch.rand(batch_size, 2) - 1
    translation *= 0.5 * (1 - 1 / scale_factor.unsqueeze(-1))
    t = translation_matrix_2D_torch(translation)
    return torch.bmm(t, s)


# a batch of 2D affine transformation matrices which, given a scale, produce a center crop of the unit square
def transform_from_scale(scale):
    scale_factor = torch.exp2(scale)
    return isotropic_scaling_matrix_2D_torch(1/scale_factor)


# create a 2D isotropic scaling matrix
def isotropic_scaling_matrix_2D(s):
    m = np.eye(3, dtype=np.float32)
    m[0:2, 0:2] *= s
    return m


# create a 2D translation matrix
def translation_matrix_2D(translation):
    m = np.eye(3, dtype=np.float32)
    m[0:2, 2] = translation
    return m
    

# print a string to the console into the same line
def print_same_line(v):
    print("\r"+str(v), end="")



# the next power of 2
def next_power_of_2(x):
    assert x > 0
    logx = np.log2(x)
    if logx == int(logx):
        logx += 1
    return np.exp2(np.ceil(logx))


# the previous power of 2
def prev_power_of_2(x):
    assert x > 0
    logx = np.log2(x)
    if logx == int(logx):
        logx -= 1
    return np.exp2(np.floor(logx))


def tensor_to_image(tensor, *, denormalize=True):
    """
    Converts a torch tensor image to a numpy array image.

    Parameters
    ----------
    tensor : torch.Tensor
        The image as a torch tensor of shape (channels, height, width) or (1, channels, height, width).
    denormalize : bool
        If true, transform the data range from [-1, 1] to [0, 1].

    Returns
    -------
    img : np.ndarray
        The image as a numpy array of shape (height, width, channels).
    """

    if tensor.ndim == 4:
        if tensor.size(0) != 1:
            raise ValueError("If the image tensor has a batch dimension, it must have length 1.")
        tensor = tensor[0]
    if denormalize:
        tensor = (tensor + 1) * 0.5
        if _use_pytorch_1_12_api:
            img = tensor.numpy(force=True)
        else:
            img = tensor.cpu().numpy()
    return np.transpose(img, (1, 2, 0))
