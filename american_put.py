import torch
import numpy as np



class BlackSholes:
    def __init__(self, r, sigma, K, S0):
        self.r = r
        self.sigma = sigma
        self.device = "cuda"
        self.K = K
        self.S0 = S0
        torch.set_default_device(self.device)
        

    def payoff_t(self, S_t, K_t = None):
        if K_t is None:
            K_t = self.K
        return torch.maximum(K_t - S_t, torch.zeros(1))
    
    def payoff_np(self, S_np, K_np = None):
        if K_np is None:
            K_np = self.K
        return np.maximum(K_np - S_np, 0)
    
    
    
    def binom_price(self, T, num_time_points, return_whole_tree = False):
        delta_t, u, d, q = self.get_binom_model_params(T, num_time_points)
        St_arr = self.build_binom_tree_fast( T, num_time_points)
        Vt_arr = np.empty_like(St_arr)
        Vt_arr[:] = np.nan
        # set terminal value
        Vt_arr[:, -1] = self.payoff_np(St_arr[:, -1] , self.K)
        # iterate backwards
        for time_index in range(num_time_points-1 , -1, -1):
            St = St_arr[:(time_index+1) , time_index]
            Vt_up = Vt_arr[:(time_index+1) , time_index+1]
            Vt_down = Vt_arr[1:(time_index+2) , time_index+1]
            exp_value = np.exp(-self.r*delta_t)*(q*Vt_up+ (1-q)*Vt_down)
            intr_value = self.payoff_np(St , self.K)
            value_t = np.maximum(exp_value , intr_value)
            Vt_arr[:(time_index+1),time_index] = value_t
        
        if return_whole_tree:
            return Vt_arr , St_arr
        else:
            return Vt_arr[0,0]


    def get_binom_model_params(self, T, num_time_points):
        delta_t = T/num_time_points
        u = np.exp(self.sigma*np.sqrt(delta_t))
        d = 1/u
        assert d < np.exp(self.r*delta_t)
        assert np.exp(self.r*delta_t) < u

        # prob of up move
        q = (np.exp(self.r*delta_t) - d)/(u-d)

        return delta_t, u, d, q


    
    def build_binom_tree_fast(self, T, num_time_points):
        delta_t, u, d, q = self.get_binom_model_params(T, num_time_points)
        
        # St-arr , columns = time, rows = possible states
        St_arr = np.empty((num_time_points+1 , num_time_points+1))
        St_arr[:] = np.nan
        # Set S0
        St_arr[0,0] = self.S0
        # iterate forwards
        for time_index in range(0 , num_time_points):
            St = St_arr[:(time_index+1), time_index]
            St_up = St*u
            St_down = St[-1]*d
            St_arr[:(time_index+1), time_index+1] = St_up
            St_arr[time_index+1, time_index+1] = St_down
        return St_arr
                  
        

class LS_MC:
    """Least Squares Monte Carlo"""
    pass