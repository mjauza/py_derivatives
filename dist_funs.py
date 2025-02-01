import numpy as np

def norm_char_fun_np(mu, sigma, u):
    return np.exp( mu*u*1j - sigma**2*u**2/2  )
