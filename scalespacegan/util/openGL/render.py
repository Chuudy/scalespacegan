from util.openGL import *

# create a screen-filling quad
def render_screen_quad():
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex2f(-1, -1)
    glTexCoord2f(0, 1)
    glVertex2f(1, -1)
    glTexCoord2f(1, 1)
    glVertex2f(1, 1)
    glTexCoord2f(1, 0)
    glVertex2f(-1, 1)
    glEnd()

