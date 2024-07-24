import os
import freetype
from OpenGL.GL import *
from util.openGL.texture import Texture2D
from util.openGL.render import render_screen_quad
from util.openGL.operation import OpenGLOP
from util.openGL import matrix

class TextOP(OpenGLOP):

    def __init__(self, resolution, font_size=50, rendertarget_count=1):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(
            os.path.join(this_dir, "shaders", "text"),
            None,
            os.path.join(this_dir, "shaders", "text"), 
            rendertarget_count, 
            resolution)
        self.init_uniform("modelMatrix")
        self.init_uniform("textColor")

        self.characters = []

        self.make_font(os.path.join(this_dir, "data", 'arial.ttf'), font_size)

    #--------------------

    def make_font(self, filename, font_size):
        
        face = freetype.Face(filename)
        face.set_pixel_sizes(0, font_size)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glActiveTexture(GL_TEXTURE0)

        for c in range(128):
            face.load_char(chr(c), freetype.FT_LOAD_RENDER)
            glyph   = face.glyph
            bitmap  = glyph.bitmap
            size    = bitmap.width, bitmap.rows
            bearing = glyph.bitmap_left, glyph.bitmap_top 
            advance = glyph.advance.x

            # create glyph texture
            tex = Texture2D()
            tex.bind()
            tex.set_params(mag_filter=GL_LINEAR, min_filter=GL_LINEAR)
            glTexImage2D(tex.target, 0, GL_R8, *size, 0, GL_RED, GL_UNSIGNED_BYTE, bitmap.buffer)
            self.characters.append((tex, size, bearing, advance))

        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        tex.unbind()

    #--------------------

    def render(self, text, position=[15, 15], scale=1., color=[1, 1, 1], background_color=[0, 0, 0, 0], rendertarget_id=0):
        assert rendertarget_id < self.rendertarget_count
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 
            self.rendertargets[rendertarget_id].color.handle, 0)
        glViewport(0, 0, *self.rendertargets[rendertarget_id].color.resolution)
        
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glUseProgram(self.shader)
                
        global_shift_matrix = matrix.translation_matrix_3D((position[0], -position[1], 0))
        proj_matrix = matrix.ortho_projection_matrix(
            (self.rendertargets[rendertarget_id].color.resolution[0], 
            -self.rendertargets[rendertarget_id].color.resolution[1]))

        glProgramUniform3f(self.shader, self.uniforms["textColor"], *color)
        
        glClearColor(*background_color)
        glClear(GL_COLOR_BUFFER_BIT)

        glActiveTexture(GL_TEXTURE0)

        char_x = 0
        for c in text:
                    
            c = ord(c)
            ch          = self.characters[c]
            w, h        = ch[1][0] * scale, ch[1][1] * scale
            xrel, yrel  = char_x + ch[2][0] * scale, (ch[1][1] - ch[2][1]) * scale
            char_x     += (ch[3] >> 6) * scale

            scale_matrix = matrix.anisotropic_scaling_matrix_3D((w, h, 1))
            rel_shift_matrix = matrix.translation_matrix_3D((xrel, yrel, 0))
            model_matrix = proj_matrix @ global_shift_matrix @ rel_shift_matrix @ scale_matrix
            glProgramUniformMatrix4fv(self.shader, self.uniforms["modelMatrix"], 1, True, *model_matrix)
            
            ch[0].bind()    
            render_screen_quad()
            
        glUseProgram(0)
        glDisable(GL_BLEND)

        return self.rendertargets[rendertarget_id].color