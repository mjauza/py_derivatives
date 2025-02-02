import numpy as np

def norm_char_fun_np(mu, sigma, u):
    return np.exp( mu*u*1j - sigma**2*u**2/2  )


def var_gamma_char_fun_np_1(u, sigma, theta, nu, t = 1):
    """
    u = function input
    sigma : parameter
    theta : parameter
    nu : parameter
    t : time
    """
    phi = (1 - 1j*u*theta*nu + 0.5*sigma**2*nu*u**2)**(-1/nu)
    return phi**t


def var_gamma_char_fun_np_2(u, C, G, M, t = 1, m = 0):
    """
    Difference of two Gamma processes parametrization
    u = function input
    C > 0: parameter
    G > 0: parameter
    M > 0: parameter
    t : time
    m : drift parameter
    """
    phi = (G*M/(G*M+(M-G)*1j*u+u**2))**C
    phi = phi * np.exp(1j*u*m)
    return phi**t


def get_var_gamma_char_fun_np_2(C, G, M, t = 1, m = 0):
    assert C > 0
    assert G > 0
    assert M > 0 
    c_fun = lambda u : var_gamma_char_fun_np_2(u, C, G, M, t, m)
    return c_fun


