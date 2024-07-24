import os
from OpenGL.GL import *
from util.openGL.render import render_screen_quad
from util.openGL.texture import Texture2D
from util.openGL.operation import OpenGLOP

# Render a textured screen quad. Allows slicing of array textures and MIP levels and rendering directly to the screen
class ImageDisplayOP(OpenGLOP):

    def __init__(self, res, rendertarget_count=0):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(
            os.path.join(this_dir, "shaders", "vertex2D_uv"),
            None,
            os.path.join(this_dir, "shaders", "textured_quad"), 
            rendertarget_count,
            rendertarget_resolution=res)
        self.init_uniform("outputRes")
        self.init_uniform("showArray")
        self.init_uniform("level")
        self.init_uniform("layer")        
        self.init_uniform("showOverlay")
        self.init_uniform("overlayPosition")
        self.res = res

    def render(self, tex, to_screen=False, overlay_tex=None, overlay_pos=(0, 0), level=0, layer=0, rendertarget_id=0):
    
        if not to_screen:
            glFramebufferTexture2D(
                GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 
                self.rendertargets[rendertarget_id].color.handle, 0)

        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, self.res[0], self.res[1])
        glUseProgram(self.shader)

        if type(tex) == Texture2D:
            glActiveTexture(GL_TEXTURE0)
            glProgramUniform1i(self.shader, self.uniforms["showArray"], False)
        else:
            glActiveTexture(GL_TEXTURE1)
            glProgramUniform1i(self.shader, self.uniforms["showArray"], True)
        
        tex.bind()
        min_filter = GL_NEAREST if level == 0 else GL_NEAREST_MIPMAP_NEAREST
        tex.set_params(min_filter=min_filter)

        glProgramUniform2i(self.shader, self.uniforms["outputRes"], *self.res)
        glProgramUniform1i(self.shader, self.uniforms["level"], level)
        glProgramUniform1i(self.shader, self.uniforms["layer"], layer)
        
        if overlay_tex is not None:
            glProgramUniform1i(self.shader, self.uniforms["showOverlay"], True)
            glActiveTexture(GL_TEXTURE2)
            overlay_tex.bind()
            overlay_tex.set_params()
            glProgramUniform2i(self.shader, self.uniforms["overlayPosition"], *overlay_pos)
        else:
            glProgramUniform1i(self.shader, self.uniforms["showOverlay"], False)

        glClear(GL_COLOR_BUFFER_BIT)
        render_screen_quad()
        
        tex.unbind()
        glUseProgram(0)
        glActiveTexture(GL_TEXTURE0)

        if not to_screen:
            return self.rendertargets[rendertarget_id].color