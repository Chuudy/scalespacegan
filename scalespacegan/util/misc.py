import os
import cv2
import numpy as np
import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
ldr_extensions = [".jpg", ".png"]
hdr_extensions = [".exr", ".hdr"]

#==============================================

# create a new directory if it does not exist
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

#==============================================

# load an image
def load_image(path, normalize=True, append_alpha=False):
    
    assert os.path.isfile(path), "Image file does not exist"
    is_hdr = is_hdr_from_file_extension(path)
    flags = (cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) if is_hdr else cv2.IMREAD_UNCHANGED

    img = cv2.imread(path, flags)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize and not is_hdr:
        img = img.astype(np.float32) / 255.
    if append_alpha and img.shape[2] == 3:
        alpha = np.ones_like(img[..., 0:1])
        img = np.concatenate([img, alpha], axis=-1)
    return img

#==============================================

# save an image
def save_image(img, path, channels=3, jpeg_quality=95):
    is_hdr = is_hdr_from_file_extension(path)

    if img.ndim == 2:
        out_img = img[..., None]
    if img.ndim == 3 and img.shape[2] >= 2:
        if channels == 2:
            out_img = np.zeros((*img.shape[0:2], 3))
            out_img[..., 1:3] = img[..., 2::-1]
        if channels == 3:
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if channels == 4:
            out_img = cv2.cv2Color(img, cv2.COLOR_RGBA2BGRA)
    if (out_img.dtype == np.float32 or out_img.dtype == np.float64) and not is_hdr:
        out_img = np.clip(out_img, 0, 1) * 255
        out_img = out_img.astype(np.uint8)
    if is_hdr:
        out_img = out_img.astype(np.float32)
        
    cv2.imwrite(path, out_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    
#==============================================

# get spatial resolution of an image
def get_image_resolution(img):
    return img.shape[:2][::-1]
    
#==============================================

# sample an image patch at coord (x,y) with size (sx, sy)
def sample_image(img, x, y, sx=1, sy=1):
    assert sx>=1 and sy>=1, "Need a patch of at least one pixel size"
    res = get_image_resolution(img)
    assert 0<=x<=res[0]-sx and 0<=y<=res[1]-sy, f"Sample coordinate {x=},{y=},{sx=},{sy=} outside of image with resolution {res}"
    return img[y:y+sy, x:x+sx, :]

#==============================================

# Check if image format should be one of hdr_extensions
def is_hdr_from_file_extension(file_path):
    extension = os.path.splitext(file_path)[1]
    return extension in hdr_extensions

#==============================================

# create a 2D isotropic scaling matrix
def isotropic_scaling_matrix_2D_torch(s, to_cuda=False):
    s, batch_size = torch_aux_params_boilerplate(s, to_cuda)
    m = torch_identity_matrix_boilerplate(3, batch_size, to_cuda)
    m[:, :2, :2] *= s.view(-1, 1, 1)
    return m

#==============================================

# pytorch version of the above, including batch dimension
def translation_matrix_2D_torch(translation, to_cuda=False):
    translation, batch_size = torch_aux_params_boilerplate(translation, to_cuda)
    m = torch_identity_matrix_boilerplate(3, batch_size, to_cuda)
    m[:, :2, 2] = translation
    return m

#==============================================

# prepare auxiliary matrix parameters for pytorch
def torch_aux_params_boilerplate(x, to_cuda=False):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    batch_size = x.shape[0]
    if to_cuda:
        x = x.cuda()
    return x, batch_size

#==============================================

# create a batch of dim x dim identity matrices for pytorch
def torch_identity_matrix_boilerplate(dim, batch_size, to_cuda=False):
    m = torch.eye(dim, dtype=torch.float32)
    if to_cuda:
        m = m.cuda()
    return m.repeat(batch_size, 1, 1)

#==============================================

def create_full_path(model_dir, model_file):
    filepath = os.path.join(model_dir, model_file)
    if "best.pkl" in model_file or "last.pkl" in model_file:
        with open(filepath) as f:
            model_file = f.read()
            filepath = os.path.join(model_dir, model_file)
    print(f"Model to be loaded: {filepath}")
    return filepath

#==============================================

# print a string to the console into the same line
def print_same_line(v):
    print("\r"+str(v), end="", flush=True)

#==============================================

def parse_comma_separated_number_list(s):
    if isinstance(s, list):
        return list(map(int, s))
    if s is None or s.lower() == 'none' or s == '':
        return []
    return list(map(int, s.split(',')))