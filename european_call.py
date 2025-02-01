import torch
import mc_simulation
import numpy as np
import dist_funs
import option_utils
import scipy.integrate as integrate


class BlackScholes:

    def __init__(self, r, sigma, K, S0):
        self.r = r
        self.sigma = sigma
        self.device = "cuda"
        self.K = K
        self.S0 = S0
        torch.set_default_device(self.device)

    def payoff(self, S_t, K_t = None):
        if K_t is None:
            K_t = self.K
        return torch.maximum(S_t - K_t, torch.zeros(1))
    
    def mc_sim_price(self, T, num_paths, K = None):
        time_points_t = T*torch.ones( (1, num_paths), device=self.device)
        S0_t = self.S0*torch.ones((1, num_paths), device=self.device)
        ST_t = mc_simulation.gbm_sim( self.r, self.sigma, time_points_t , S0_t )
        payoff_t = self.payoff(ST_t, K)
        mean_payoff_t = torch.mean(payoff_t, 1)
        d_factor = np.exp( -self.r*T  )
        price = d_factor * mean_payoff_t
        return price

    def exact_price(self, T, K = None):
        if K is None:
            K = self.K
        d1 = (1/(self.sigma*np.sqrt(T)))*(np.log(self.S0/K) + (self.r+self.sigma**2/2)*T )
        d2 = d1 - self.sigma*np.sqrt(T)
        d1_t = torch.tensor([d1])
        d2_t = torch.tensor([d2])
        n_dist = torch.distributions.normal.Normal(0,1)
        price = n_dist.cdf(d1_t)*self.S0 - n_dist.cdf(d2_t)*np.exp(-self.r*T)*K
        return price
    

    def fourier_trans_price(self, T, K = None, z_min=0, z_max=np.inf):
        if K is None:
            K = self.K
        # Char Func of XT = -sigma**2/2*T + sigma*T**0.5*Z
        psi = lambda u: dist_funs.norm_char_fun_np(mu = -self.sigma**2/2*T , sigma=self.sigma*(T**0.5), u=u)
        #
        k = np.log(self.S0/K) + self.r*T
        # get lewis integrand
        lewis_fun = option_utils.get_lewis_integrand(k, psi)
        # integrate the lewwis integrand
        lew_res = integrate.quad(lewis_fun, z_min, z_max)
        a1 = lew_res[0]
        a2 = (np.sqrt(self.S0*K) * np.exp(-self.r*T/2))/np.pi
        price = self.S0 -  a2*a1
        return price


    

