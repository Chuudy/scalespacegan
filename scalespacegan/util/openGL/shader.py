from util.openGL import *
from OpenGL.GL.shaders import compileProgram, compileShader


# create a shader program, consisting of a vertex, fragment, and (optionally) geometry shader
def create_shader(path_vertex, path_geometry, path_fragment):
    vertex_src = open(path_vertex + ".vert", "r").read()
    fragment_src = open(path_fragment + ".frag", "r").read()
    arg_list = [
        compileShader(vertex_src, GL_VERTEX_SHADER), 
        compileShader(fragment_src, GL_FRAGMENT_SHADER)]
    if path_geometry:
        geometry_src = open(path_geometry + ".geom", "r").read()
        arg_list.append(compileShader(geometry_src, GL_GEOMETRY_SHADER))
    return compileProgram(*arg_list)
