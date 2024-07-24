from util.openGL import *

# an OpenGL framebuffer
class Framebuffer:

    def __init__(self):
        self.handle = glGenFramebuffers(1)

    def bind(self, target=GL_FRAMEBUFFER):
        glBindFramebuffer(target, self.handle)

    def unbind(self, target=GL_FRAMEBUFFER):
        glBindFramebuffer(target, 0)
