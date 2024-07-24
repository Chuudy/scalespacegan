from util.openGL import *
import math
import numpy as np
from abc import ABC, abstractmethod

# an abstract class for an OpenGL texture
class Texture(ABC):

    @abstractmethod
    def __init__(self):
        self.handle = glGenTextures(1)
        self._resolution = (0, 0)
        self._channels = 0
        self._gl_format = None
        self.target = None
        self.need_allocation = True

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        if self._resolution != value:
            self.need_allocation = True
        self._resolution = value

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, value):
        if self._channels != value:
            self.need_allocation = True
        self._channels = value

    @property
    def gl_format(self):
        return self._gl_format

    @gl_format.setter
    def gl_format(self, value):
        if self._gl_format != value:
            self.need_allocation = True
        self._gl_format = value

    # bind the texture
    def bind(self):
        glBindTexture(self.target, self.handle)

    # unbind the texture
    def unbind(self):
        glBindTexture(self.target, 0)

    # set sampling parameters
    def set_params(self, min_filter=GL_NEAREST, mag_filter=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER):
        self.bind()
        glTexParameteri(self.target, GL_TEXTURE_MIN_FILTER, min_filter)
        glTexParameteri(self.target, GL_TEXTURE_MAG_FILTER, mag_filter)
        glTexParameterfv(self.target, GL_TEXTURE_BORDER_COLOR, [0, 0, 0, 1])
        glTexParameteri(self.target, GL_TEXTURE_WRAP_S, wrap)
        glTexParameteri(self.target, GL_TEXTURE_WRAP_T, wrap)

    # infer format and internal format from channel count
    @staticmethod
    def gl_format_from_channel_count(c):
        assert c > 0 and c < 5, "Channel count can only be in [1-4]"
        formats = [
            (GL_R32F, GL_RED),
            (GL_RG32F, GL_RG),
            (GL_RGB32F, GL_RGB),
            (GL_RGBA32F, GL_RGBA)
        ]
        return formats[c-1]


#=====================================================================

# an OpenGL 2D texture
class Texture2D(Texture):

    def __init__(self, image=None, flip_h=True):
        super().__init__()
        self.target = GL_TEXTURE_2D
        if image is not None:
            self.upload_image(image, flip_h=flip_h)

    # allocate GPU memory for the texture
    def allocate_memory(self, resolution, channels=4, depth_texture=False, force_allocation=False):
        self.resolution = resolution
        if depth_texture:
            self.channels = 1
            self.gl_format = GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT
        else:
            self.channels = channels
            self.gl_format = self.gl_format_from_channel_count(channels)
        if self.need_allocation or force_allocation:
            self.bind()
            glTexImage2D(self.target, 0, self.gl_format[0], resolution[0], resolution[1], 0, self.gl_format[1], GL_FLOAT, None)
            self.need_allocation = False

    # upload an image into the texture
    def upload_image(self, image, flip_h=True):
        if image.ndim == 2:
            image = image[..., None]
        w, h, self.channels = image.shape
        self.resolution = (w, h)
        if flip_h:
            image = np.flip(image, 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        self.gl_format = self.gl_format_from_channel_count(self.channels)
        self.bind()
        glTexImage2D(self.target, 0, self.gl_format[0], self.resolution[1], self.resolution[0], 0, self.gl_format[1], GL_FLOAT, image)
        self.need_allocation = False

    # download the texture as an image
    def download_image(self, flip_h=True):
        self.bind()

        # PyOpenGL does not support downloading GL_RG textures
        # therefore: make it GL_RGB temporarily
        temp_channels = 3 if self.channels == 2 else self.channels

        _, gl_format = self.gl_format_from_channel_count(temp_channels)
        image = glGetTexImage(self.target, 0, gl_format, GL_FLOAT)
        self.unbind()
        if self.channels == 1:
            image = image[..., None]
        w, h, c = image.shape
        image = np.reshape(image, (h, w, c))

        # if temp channel was necessary, throw it away
        if not temp_channels == self.channels:
            image = image[..., 0:2]

        if flip_h:
            image = np.flip(image, 0)
        return image