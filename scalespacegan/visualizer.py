import pickle
import torch
import numpy as np
import glfw
import logging
import click
import dnnlib

from torch_utils import misc
from util.misc import create_full_path

from util.multiscale_util import tensor_to_image
from util.openGL.window import OpenGLWindow
from util.openGL.texture import Texture2D
from util.openGL.display import ImageDisplayOP
from util.openGL.text import TextOP

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

#=======================================================

@click.command()

# Required.
@click.option('--model_dir',           help='Directory where models are stored',         metavar='DIR',            required=True    )
@click.option('--model_file',          help='Filename of the model',                     metavar='DIR',            required=True    )

#=======================================================

def main(**kwargs):

    # Parse arguments
    opts = dnnlib.EasyDict(kwargs)
    model_dir = opts.model_dir
    model_file = opts.model_file

    window = MultiscaleGANVisualizer(create_full_path(model_dir, model_file))
    window.run()

#=======================================================

class MultiscaleGANVisualizer(OpenGLWindow):

    def __init__(self, gan_file):
        
        display_res = (512, 512)
        super().__init__(display_res, "Multiscale GAN")    

        with open(gan_file, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda()
        
        logging.info(f"Generator resolution: {self.G.img_resolution}")
        logging.info(f"Training mode: {self.G.training_mode}")

        z = torch.empty([1, self.G.z_dim]).cuda()
        c = torch.empty([1, self.G.c_dim]).cuda()
        misc.print_module_summary(self.G, [z, c])

        self.seed = 0
        self.reset_latent_code()

        self.gan_output_tex = Texture2D()
        self.gan_output_tex.allocate_memory(2*(self.G.img_resolution,), channels=3)

        self.init_shaders()
        
    #===================================================

    def init_shaders(self):
        super().init_shaders()
        self.text_op = TextOP((200, 37), font_size=17)
        self.overlay_op = ImageDisplayOP(2*(self.display_res[1],), rendertarget_count=1)
    
    #===================================================

    def render(self):

        # produce latent code
        z = torch.from_numpy(self.current_z).float()[None].cuda()

        # set 2D transform
        transform = torch.from_numpy(self.transform).unsqueeze(0).cuda()
            
        img = self.G(z=z, c=None, transform=transform, noise_mode='const')
        img = tensor_to_image(img, denormalize=True)
        self.gan_output_tex.upload_image(img)
 
        # overlay
        info_text_tex = self.text_op.render(
            f"Scale: {1./self.scale:.2f} | {np.log2(1./self.scale):.1f}", 
            position=(15, 12),
            background_color=(0, 0, 0, 0.7)
        )
        display_tex = self.overlay_op.render(
            self.gan_output_tex, 
            overlay_tex=info_text_tex, 
            overlay_pos=(self.overlay_op.rendertargets[0].color.resolution[0] - self.text_op.rendertargets[0].color.resolution[0], 0)
        )

        return display_tex

    #===================================================

    def reset_latent_code(self):
        self.seed += 1
        self.rng = np.random.RandomState(self.seed)
        self.base_z = self.rng.standard_normal(self.G.z_dim)
        self.current_z = self.base_z

    #===================================================

    # some very simple motion in latent space
    def drag_latent_code(self, move_position):
        offset = (self.click_position - move_position) * 0.0005
        self.current_z[:int(self.G.z_dim/2)] = self.base_z[:int(self.G.z_dim/2)] + offset[0]
        self.current_z[int(self.G.z_dim/2):] = self.base_z[int(self.G.z_dim/2):] + offset[1]

    #===================================================

    def mouse_move(self, move_position):
        super().mouse_move(move_position)
        if self.right_mouse_down:
            self.drag_latent_code(move_position)

    #===================================================

    def right_mouse_release(self):
        self.base_z = self.current_z
        
    #===================================================

    def key_press(self, key):
        super().key_press(key)
        if key == glfw.KEY_S:
            self.reset_latent_code()

#=======================================================

if __name__ == "__main__":    
    main()