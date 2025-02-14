import astropy.constants as apc
import numpy as np
from   scipy.special import hyp2f1

__all__ = ['calc_sma',
           'calc_aRs',
           'calc_T14_circ',
           'jacobian',
           'jacobian_integral',
           'detection_prior',
           'duration_ratio'
          ]

pi = np.pi

RSAU = (apc.R_sun/apc.au).value                                 # solar radius [AU]
RSRE = (apc.R_sun/apc.R_earth).value                            # R_sun/R_earth
RHOSUN_GCM3 = (3*apc.M_sun/(4*pi*apc.R_sun**3)).value/1000      # solar density [g/cm^3]


def calc_sma(P, Ms):
    """
    Calculate semi-major axis in units of [Solar radii] from Kepler's law
    
    Parameters
    ----------
    P : period [days]
    Ms : stellar mass [Solar masses]
    
    Returns
    -------
    sma : semimajor axis [Solar radii]
    """
    return Ms**(1./3)*(P/365.24)**(2./3)/RSAU


def calc_aRs(P, rho):
    """
    P : period [days]
    rho : stellar density [g/cm3]
    """
    P_   = P*86400.       # [seconds]
    rho_ = rho*1000.      # [kg/m3]
    G    = apc.G.value    # Newton's constant

    return ((G*P_**2*rho_)/(3*pi))**(1./3)
    
    
def calc_T14_circ(P, ror, b, rho):
    """
    P : [days]
    rho: [g/cm3]
    dur: [days]
    """
    aRs = calc_aRs(P, rho)
    num = (1+ror)**2 - b**2
    den = aRs**2 - b**2
    
    return (P/pi) * np.arcsin(np.sqrt(num/den))    # T14 transit duration [days]


def jacobian(P, ror, b, T14):
    """
    P   : [days]
    T14 : [days]

    jac:
    """
    P_   = P*86400.       # [seconds]
    dur_ = T14*86400.     # [seconds]
    G    = apc.G.value    # Newton's constant

    return (12*pi**3)/(P_**3*G) * ((1+ror)**2 - b**2)**1.5 * (pi*dur_/P_)**-4


def jacobian_integral(P, ror, b, T14):
    """
    P   : [days]
    T14 : [days]

    jac:
    """
    P_   = P*86400.       # [seconds]
    dur_ = T14*86400.     # [seconds]
    G    = apc.G.value    # Newton's constant

    const = (12*pi**3)/(P_**3*G) * (pi*dur_/P_)**-4
    dep = (1+ror)**3 * b * hyp2f1(-1.5,0.5,1.5, b**2/(1+ror)**2)

    return const*dep


def detection_prior(ecc, omega):
    return (1+ecc*np.sin(omega))/(1-ecc**2)


def duration_ratio(ecc, omega):
    return np.sqrt(1-ecc**2) / (1+ecc*np.sin(omega))