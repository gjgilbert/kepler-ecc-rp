import numpy as np
import pymc3 as pm
from   scipy.interpolate import interp1d

__all__ = ['weighted_percentile',
           'draw_random_samples',
           'gelman_rubin'
           ]

def weighted_percentile(a, q, w=None):
    """
    Compute the q-th percentile of data array a
    Similar to np.percentile, but allows for weights (but not axis-wise computation)
    """
    a = np.array(a)
    q = np.array(q)
    
    if w is None:
        return np.percentile(a,q)
    
    else:
        w = np.array(w)
        
        assert np.all(q >= 0) and np.all(q <= 100), "quantiles should be in [0, 100]"

        order = np.argsort(a)
        a = a[order]
        w = w[order]

        weighted_quantiles = np.cumsum(w) - 0.5 * w
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    
        return np.interp(q/100., weighted_quantiles, a)


def draw_random_samples(pdf, domain, N, *args, **kwargs):
    """
    Draw random samples from a given pdf using inverse transform sampling
    
    Paramters
    --------
    pdf : univarite function
        outputs pdf as a function of input value
    domain : tuple 
        domain over which pdf is defined
    N : int
        number of random samples to draw
        
    Returns
    -------
    samples : ndarray
        vector of random samples drawn
    """
    x = np.linspace(domain[0], domain[1], int(1e5))
    y = pdf(x, *args, **kwargs)
    cdf_y = np.cumsum(y)
    cdf_y = cdf_y/cdf_y.max()
    inverse_cdf = interp1d(cdf_y,x)
        
    return inverse_cdf(np.random.uniform(cdf_y.min(), cdf_y.max(), N))


def gelman_rubin(c):
    J, L = c.shape
    
    mean_of_chain = np.mean(c, axis=1)
    mean_of_means = np.mean(mean_of_chain)

    B = L/(J-1) * np.sum((mean_of_chain - mean_of_means)**2)
    W = (1/J) * np.sum(1/(L-1) * np.sum((c.T - mean_of_chain)**2))

    Rhat = ((L-1)/L * W + B/L) / W

    return Rhat