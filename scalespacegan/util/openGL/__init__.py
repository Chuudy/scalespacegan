import os
import enum
import glfw

class OpenGLFramework(enum.Enum):
    none = 0,
    glfw = 1,
    egl = 2

OGL_FRAMEWORK = OpenGLFramework.none

# default: GLFW
glfw.ERROR_REPORTING = False
if glfw.init():
    OGL_FRAMEWORK = OpenGLFramework.glfw
    print("[Using OpenGL with GLFW]")

# fallback: EGL for headless rendering
else:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'  # must be set before importing OpenGL
    OGL_FRAMEWORK = OpenGLFramework.egl
    print("[Using OpenGL with EGL]")

glfw.ERROR_REPORTING = True

from OpenGL.GL import *
# Creating a dictionary of possible keys for usage later on
key_mappings = {
    glfw.MOD_CONTROL: "Control",
    glfw.MOD_NUM_LOCK: "Num Lock",
    glfw.MOD_CAPS_LOCK: "Caps Lock",
    glfw.MOD_SUPER: "Super",
    glfw.MOD_ALT: "Alt",
    glfw.MOD_SHIFT: "Shift",
    glfw.KEY_UNKNOWN: "Unknown key",
    glfw.KEY_SPACE: "SpaceBar",
    glfw.KEY_APOSTROPHE: "Apostrophe",
    glfw.KEY_COMMA: "Comma",
    glfw.KEY_MINUS: "Minus",
    glfw.KEY_PERIOD: "Period",
    glfw.KEY_SLASH: "Slash",

    **{k: chr(ord('0') + (k - glfw.KEY_0)) for k in range(glfw.KEY_0, glfw.KEY_9 + 1)},  # 0-9

    glfw.KEY_SEMICOLON: "SemiColon",
    glfw.KEY_EQUAL: "Equal",

    **{k: chr(ord('A') + (k - glfw.KEY_A)) for k in range(glfw.KEY_A, glfw.KEY_Z + 1)},  # A-Z

    glfw.KEY_LEFT_BRACKET: "Left Bracket",
    glfw.KEY_BACKSLASH: "BackSlash",
    glfw.KEY_RIGHT_BRACKET: "Right Bracket",
    glfw.KEY_GRAVE_ACCENT: "Grave Accent",
    glfw.KEY_WORLD_1: "World 1",
    glfw.KEY_WORLD_2: "World 2",
    glfw.KEY_ESCAPE: "ESC",
    glfw.KEY_ENTER: "Enter",
    glfw.KEY_TAB: "Tab",
    glfw.KEY_BACKSPACE: "BackSpace",
    glfw.KEY_INSERT: "Insert",
    glfw.KEY_DELETE: "Delete",
    glfw.KEY_RIGHT: "Right Arrow",
    glfw.KEY_LEFT: "Left Arrow",
    glfw.KEY_DOWN: "Down Arrow",
    glfw.KEY_UP: "Up Arrow",
    glfw.KEY_PAGE_UP: "Page Up",
    glfw.KEY_PAGE_DOWN: "Page Down",
    glfw.KEY_HOME: "Home",
    glfw.KEY_END: "End",
    glfw.KEY_CAPS_LOCK: "Caps Lock",
    glfw.KEY_SCROLL_LOCK: "Scroll Lock",
    glfw.KEY_NUM_LOCK: "Num Lock",
    glfw.KEY_PRINT_SCREEN: "Print Screen",
    glfw.KEY_PAUSE: "Pause",

    **{k: f"F{k - glfw.KEY_F1 + 1}" for k in range(glfw.KEY_F1, glfw.KEY_F25 + 1)},  # F1-F25
    **{k: f"Keypad {k - glfw.KEY_KP_0}" for k in range(glfw.KEY_KP_0, glfw.KEY_KP_9 + 1)}, ## Keypad numbers (0-9)

    glfw.KEY_KP_DECIMAL: "Keypad Decimal",
    glfw.KEY_KP_DIVIDE: "Keypad Divide",
    glfw.KEY_KP_MULTIPLY: "Keypad Multiply",
    glfw.KEY_KP_SUBTRACT: "Keypad Subtract",
    glfw.KEY_KP_ADD: "Keypad Add",
    glfw.KEY_KP_ENTER: "Keypad Enter",
    glfw.KEY_KP_EQUAL: "Keypad Equal",
    glfw.KEY_LEFT_SHIFT: "Shift",
    glfw.KEY_LEFT_CONTROL: "Control",
    glfw.KEY_LEFT_ALT: "Alt",
    glfw.KEY_RIGHT_SHIFT: "Shift",
    glfw.KEY_RIGHT_CONTROL: "Control",
    glfw.KEY_RIGHT_ALT: "Alt",  
    glfw.KEY_MENU: "Menu",
    glfw.KEY_LAST: "Last Key"   
}


def init_egl():
    import OpenGL.EGL as egl
    import ctypes
    display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
    assert display != egl.EGL_NO_DISPLAY, "Cannot access display"

    major = ctypes.c_int32()
    minor = ctypes.c_int32()
    ok = egl.eglInitialize(display, major, minor)
    assert ok, "Cannot initialize EGL"

    # Choose config.
    config_attribs = [
        egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT,
        egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT,
        egl.EGL_NONE]
    configs = (ctypes.c_int32 * 1)()
    num_configs = ctypes.c_int32()
    ok = egl.eglChooseConfig(display, config_attribs, configs, 1, num_configs) 
    assert ok
    assert num_configs.value == 1
    config = configs[0]

    # Create dummy pbuffer surface.
    surface_attribs = [
        egl.EGL_WIDTH, 1,
        egl.EGL_HEIGHT, 1,
        egl.EGL_NONE
    ]
    surface = egl.eglCreatePbufferSurface(display, config, surface_attribs)
    assert surface != egl.EGL_NO_SURFACE

    # Setup GL context.
    ok = egl.eglBindAPI(egl.EGL_OPENGL_API) 
    assert ok
    context = egl.eglCreateContext(display, config, egl.EGL_NO_CONTEXT, None) 
    assert context != egl.EGL_NO_CONTEXT
    ok = egl.eglMakeCurrent(display, surface, surface, context)
    assert ok
    return ok


if OGL_FRAMEWORK == OpenGLFramework.egl:
    if not init_egl():
        OGL_FRAMEWORK = OpenGLFramework.none

assert OGL_FRAMEWORK != OpenGLFramework.none, "Could not initialize OpenGL"