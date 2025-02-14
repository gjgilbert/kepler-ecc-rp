import numpy as np
import pymc3 as pm
import aesara_theano_fallback.tensor as T
from   scipy.interpolate import interp1d

__all__ = ['NormDistLogPDF',
           'ExponDistLogPDF',
           'RayleighDistLogPDF',
           'InvParetoDistLogPDF',
           'BetaDistLogPDF',
          ]

def NormDistLogPDF(mu, sd, x, bounds=None, eps=None):
    """
    Parameters
    ----------
        mu : mean
        sd : standard deviation
        eps : (optional) noise floor 
        bounds : (optional) bounds on domains; tuple of size 2 (lower, upper)
    """
    # set noise floor
    if eps is None:
        if bounds is None:
            raise ValueError("Bounds must be specified if noise parameter eps is given")
        else:
            eps = 1e-100

    # base distribution
    ln_f = pm.logaddexp(-T.log(sd) - 0.5*np.log(2*np.pi) - 0.5*(x-mu)**2/sd**2, T.log(eps))

    # normalization correction
    if bounds is None:
        ln_normalization = 0.0
    else:
        F0 = 0.5*(1 + T.erf((bounds[0]-mu)/(sd*np.sqrt(2)))) + eps*bounds[0]
        F1 = 0.5*(1 + T.erf((bounds[1]-mu)/(sd*np.sqrt(2)))) + eps*bounds[1]
        ln_normalization = T.log(F1-F0)
   
    return ln_f - ln_normalization
    

def ExponDistLogPDF(c, x, bounds=None, eps=None):
    """
    Parameters
    ----------
        c : rate parameter, c > 0
        eps : (optional) noise floor 
        bounds : (optional) bounds on domains; tuple of size 2 (lower, upper)
    """
    # set noise floor
    if eps is None:
        if bounds is None:
            raise ValueError("Bounds must be specified if noise parameter eps is given")
        else:
            eps = 1e-100

    # base distribution
    ln_f = pm.logaddexp(T.log(c)-c*x, T.log(eps))
    
    # normalization correction
    if bounds is None:
        ln_normalization = 0.0
    else:
        F0 = 1. - T.exp(-c*bounds[0]) + eps*bounds[0]
        F1 = 1. - T.exp(-c*bounds[1]) + eps*bounds[1]
        ln_normalization = T.log(F1-F0)

    return ln_f - ln_normalization


def RayleighDistLogPDF(sd, x, bounds=None, eps=None):
    """
    Parameters
    ----------
        sd : scale parameter
        eps : (optional) noise floor 
        bounds : (optional) bounds on domains; tuple of size 2 (lower, upper)
    """
    # set floor
    if eps is None:
        if bounds is None:
            raise ValueError("Bounds must be specified if noise parameter eps is given")
        else:
            eps = 1e-100

    # base distribution
    ln_f = pm.logaddexp(T.log(x) - 2*T.log(sd) - x**2/(2*sd**2), T.log(eps))

    # normalization correction
    if bounds is None:
        ln_normalization = 0.0
    else:
        F0 = 1. - T.exp(-bounds[0]**2/(2*sd**2)) + eps*bounds[0]
        F1 = 1. - T.exp(-bounds[1]**2/(2*sd**2)) + eps*bounds[1]
        ln_normalization = T.log(F1-F0)
    
    return ln_f - ln_normalization
    
    
def InvParetoDistLogPDF(c, x, eps):
    '''
    Probability density f(x) = (c+1)*(1-x)**c; support over [0,1]
    Derived assuming 1/(1-x) is Pareto distributed with xm = 1
    I'm calling this the "Inverse Pareto", but there's probably a better name

    Parameters
    ----------
        c : rate parameter, c > 0
        eps : (optional) noise floor 
    '''
    # noise floor
    if eps is None:
        eps = 1e-100

    # base distribution
    ln_f = pm.logaddexp(T.log(c+1) + c*T.log(1-x), T.log(eps))

    # normalization correction
    ln_normalization = T.log(1. + eps)
    
    return ln_f - ln_normalization


def BetaDistLogPDF(a, b, x, ln_B=None):
    '''
    The ln(beta) function ln_B can be precomputed to improve performance
    This is necessary if looping over multiple realizations w/in a PyMC model
    '''
    if ln_B is None:
        ln_B = T.gammaln(a) + T.gammaln(b) - T.gammaln(a+b)
        
    return (a-1)*np.log(x) + (b-1)*np.log(1-x) - ln_B