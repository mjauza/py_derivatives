import torch


def bm_sim( time_points_t , device = "cuda"):
    """
    time_points_t : torch.Tenosor, 1D or 2D, 0 dim is time and 1 dim is paths
    """
    if len(time_points_t.shape) == 1:
        zeros = torch.zeros(1, device=device)
    else:
        zeros = torch.zeros(1, time_points_t.shape[1], device=device)
    time_points_diff = torch.diff(time_points_t, dim=0, prepend=zeros)
    bm_inc =torch.normal(0 , torch.sqrt(time_points_diff))
    bm = torch.cumsum(bm_inc, dim=0)
    return bm


def gbm_sim( mu, sigma, time_points_t , S0_t ):
    """
    mu : float
    sigma : float
    time_points_t   : torch.Tenosor, 1D or 2D, 0 dim is time and 1 dim is paths
    S0_t : 1D or 2D torch.Tensor, if 2D, then shape = (1 , num paths)
    """
    if len(time_points_t.shape) > 1:
        assert time_points_t.shape[1] == S0_t.shape[1]

    bm = bm_sim( time_points_t )

    gbm = S0_t * torch.exp( (mu-sigma**2/2)*time_points_t + sigma*bm )
    return gbm
