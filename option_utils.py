import numpy as np

def lewis_integrand(z, k , psi):
    """
    k = log(S0/K) + r*T
    psi is charactersitic function of XT, where ST = S0*exp(r*T + XT)
    """
    a1 = np.real( np.exp(1j*z*k)*psi(z - 1j/2) )
    a2 = z**2 +1/4
    return a1/a2

def get_lewis_integrand(k, psi):
    return lambda z: lewis_integrand(z, k, psi)