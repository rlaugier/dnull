
import jax.numpy as jp



def singular_values(matrix):
    s = jp.linalg.svd(matrix,
                        compute_uv=False,
                        full_matrices=False)
    return s

def reg_singular(M):
    s = singular_values(M)
    myreg = jp.sum((jp.abs(s) - 1)**2)
    return myreg

def reg_smoothness(m):
    s = jp.gradient(m, axis=0)
    myreg = jp.std(s)
    return myreg
