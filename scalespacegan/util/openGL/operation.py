from util.openGL import *
from util.openGL.shader import create_shader
from util.openGL.texture import Texture2D

#===============================================

# a simple OpenGL rendertarget structure
class Rendertarget:
    def __init__(self):
        self.color = None
        self.depth = None

    # def __del__(self):
    #     if self.color is not None:
    #         delete_texture(self.color)
    #     if self.depth is not None:
    #         delete_texture(self.depth)

#===============================================

# base class for OpenGL operations
class OpenGLOP:

    def __init__(
        self,
        vertex_shaders, 
        geometry_shaders,
        fragment_shaders, 
        rendertarget_count, 
        rendertarget_resolution=None,
        create_depth_rendertargets=False):
        
        # init shader(s)
        self.shaders = []
        if not isinstance(vertex_shaders, list):
            vertex_shaders = [vertex_shaders]
        if not isinstance(geometry_shaders, list):
            geometry_shaders = [geometry_shaders]
        if not isinstance(fragment_shaders, list):
            fragment_shaders = [fragment_shaders]
        assert len(vertex_shaders) == len(geometry_shaders) == len(fragment_shaders), "Shader list lengths must match."
        for v, g, f in zip(vertex_shaders, geometry_shaders, fragment_shaders):
            self.shaders.append(create_shader(v, g, f))
        if len(self.shaders) == 1:
            self.shader = self.shaders[0]

        # init uniforms
        self.uniforms = {}
        
        # init render target(s)
        self.rendertarget_count = rendertarget_count
        if rendertarget_count > 0:
            assert rendertarget_resolution is not None, "Need resolution to create render target."
            if isinstance(rendertarget_resolution, int):
                rendertarget_resolution = (rendertarget_resolution, rendertarget_resolution)
            self.rendertargets = []
            for _ in range(rendertarget_count):
                self.rendertargets.append(self.create_rendertargets(rendertarget_resolution, create_depth_rendertargets))

    #----------------------------

    # initialize uniforms of a shader
    def init_uniform(self, name, shader_idx=0):
        self.uniforms[name] = glGetUniformLocation(self.shaders[shader_idx], name)

    #----------------------------

    # create render targets for rendering to a texture
    def create_rendertargets(self, res, create_depthbuffer, channels=4):        
        rt = Rendertarget()
        rt.color = Texture2D()
        rt.color.allocate_memory(res, channels)
        if create_depthbuffer:
            rt.depth = Texture2D()
            rt.depth.allocate_memory(res, depth_texture=True)
        return rt