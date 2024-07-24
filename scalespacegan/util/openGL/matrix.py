import numpy as np

# create a 3D translation matrix
def translation_matrix_3D(translation):
    m = np.eye(4, dtype=np.float32)
    m[0:3, 3] = translation
    return m

# create a 3D anisotropic scaling matrix
def anisotropic_scaling_matrix_3D(s):
    m = np.eye(4, dtype=np.float32)
    m[0,0] = s[0]
    m[1,1] = s[1]
    m[2,2] = s[2]
    return m

# map [0, w[0]]*[0, w[1]] to [-1, 1]^2
def ortho_projection_matrix(w):
    s = anisotropic_scaling_matrix_3D((2/w[0], 2/w[1], 1))
    t = translation_matrix_3D((-1, -1, 0))
    return t @ s



