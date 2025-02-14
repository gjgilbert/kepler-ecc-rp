#!/usr/bin/env python
# coding: utf-8


import os
import sys
import glob
import shutil
import warnings
from datetime import datetime
from timeit import default_timer as timer

print("")
print("+"*shutil.get_terminal_size().columns)
print("Hierarchical Bayesian Analysis of Kepler Eccentricities")
print("Initialized {0}".format(datetime.now().strftime("%d-%b-%Y at %H:%M:%S")))
print("+"*shutil.get_terminal_size().columns)
print("")

# track date
YYYYMMDD = datetime.now().strftime("%Y%m%d")

# start program timer
global_start_time = timer()


# ## Parse arguments


import argparse
import random
import string


try:
    # read the arguments
    parser = argparse.ArgumentParser(description="Inputs for Kepler hierarchical eccentricities analysis")
    parser.add_argument("--project_dir", default='/data/user/gjgilbert/projects/kepler-ecc-rp/', type=str, required=False, \
                        help="project directory for storing inputs and outputs")
    parser.add_argument("--data_dir", default=None, type=str, required=True, \
                        help="data directory for accessing DR25 posterior chains")
    parser.add_argument("--data_source", default=None, type=str, required=True, \
                        help="can be 'DR25' or 'ALDERAAN'")
    parser.add_argument("--run_id", default=None, type=str, required=True, \
                        help="unique run identifier")
    parser.add_argument("--distribution", default='empirical', type=str, required=False, \
                        help="probability density function to use; can be 'histogram', 'empirical', 'beta', 'halfnormal', 'expon', 'rayleigh', 'invpareto'")
    parser.add_argument("--nsamp", default=1000, type=int, required=False, \
                        help="number of (re)samples to feed into hbayes model")
    parser.add_argument("--nbin", default=100, type=int, required=False, \
                        help="number of bins for probability density function'")
    parser.add_argument("--bootstrap", default='none', type=str, required=False, \
                        help="bootrap method to use, can be 'none', 'wlb', or an integer")

    parser.add_argument("--multiplicity", default=(1,99), nargs=2, type=int, required=False, \
                        help="(lower,upper) limits on multiplicity")
    parser.add_argument("--per_lim", default=(1,300), nargs=2, type=float, required=False, \
                        help="(lower,upper) limits on period")
    parser.add_argument("--rad_type", default=None, type=str, required=True, \
                        help="radius type to use' can by 'rp', 'rp10', or 'rpadj'")
    parser.add_argument("--rad_lim", default=None, nargs=2, type=float, required=True, \
                        help="(lower,upper) limits on radius; set lower=upper to use Gaussian binning")
    parser.add_argument("--rad_fwhm", default=None, type=float, required=False, \
                        help="fractional FWHM on radius bins if using Gaussian binning'")

    parser.add_argument("--mstar_lim", default=(0.,10.), nargs=2, type=float, required=False, \
                        help="(lower,upper) limits on stellar mass")
    parser.add_argument("--rstar_lim", default=(0.7,1.4), nargs=2, type=float, required=False, \
                        help="(lower,upper) limits on stellar radius")
    parser.add_argument("--feh_lim", default=(-0.6,0.6), nargs=2, type=float, required=False, \
                        help="(lower,upper) limits on stellar metallicity")
    parser.add_argument("--teff_lim", default=(4700,6500), nargs=2, type=float, required=False, \
                        help="(lower,upper) limits on stellar effective temperature")
    parser.add_argument("--age_lim", default=(0,14), nargs=2, type=float, required=False, \
                        help="(lower,upper) limits on stellar age")

    parser.add_argument("--e_detprior", default=1, type=int, required=False, \
                        help="flag to use geometric eccentricity detection prior (1) or not (0)")
    parser.add_argument("--b_detprior", default=0, type=int, required=False, \
                        help="flag to use impact parameter detection prior (1) or not (0)")
    parser.add_argument("--b_cut", default=0.01, type=float, required=False, \
                        help="fraction of impact parameter samples which are allowed to be grazing")

    # parse the arguments
    args = parser.parse_args()
    PROJECT_DIR  = args.project_dir
    DATA_DIR     = args.data_dir
    DATA_SOURCE  = args.data_source
    RUN_ID       = args.run_id
    DISTRIBUTION = args.distribution
    NSAMP        = args.nsamp
    NBIN         = args.nbin
    BOOTSTRAP    = args.bootstrap
    MULTIPLICITY = args.multiplicity
    PER_LIM      = args.per_lim
    RAD_TYPE     = args.rad_type
    RAD_LIM      = args.rad_lim
    RAD_FWHM     = args.rad_fwhm
    MSTAR_LIM    = args.mstar_lim
    RSTAR_LIM    = args.rstar_lim
    FEH_LIM      = args.feh_lim
    TEFF_LIM     = args.teff_lim
    AGE_LIM      = args.age_lim
    E_DETPRIOR   = bool(args.e_detprior)
    B_DETPRIOR   = bool(args.b_detprior)
    B_CUT        = args.b_cut
    DO_PLOT      = False

except:
    pass

try:
    BOOTSTRAP = int(BOOTSTRAP)
except ValueError:
    pass
    
RESULTS_DIR = os.path.join(PROJECT_DIR, 'Results', YYYYMMDD)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(" Distribution: {0}".format(DISTRIBUTION))
print("")
print("   npl = {0}-{1}".format(MULTIPLICITY[0], MULTIPLICITY[1]))
print("   P   = {0}-{1}".format(PER_LIM[0], PER_LIM[1]))
print("   {0} = {1}-{2}".format(RAD_TYPE, RAD_LIM[0], RAD_LIM[1]))
print("")


# ## Import packages and define constants


import astropy.constants as apc
from   astropy.io import fits
from   copy import deepcopy
import diptest
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from   matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from   scipy.interpolate import interp1d, UnivariateSpline, CubicSpline
from   scipy import stats
from   scipy.special import erf, erfinv, logsumexp
import seaborn as sns
import warnings

import aesara_theano_fallback.tensor as T
from   aesara_theano_fallback import aesara as theano
from   celerite2.theano import GaussianProcess
from   celerite2.theano import terms as GPterms
import pymc3 as pm
import pymc3_ext as pmx

sys.path.append(PROJECT_DIR)
from utils.astro import calc_T14_circ, calc_sma, calc_aRs, jacobian, detection_prior, duration_ratio
from utils.distributions import BetaDistLogPDF, NormDistLogPDF, ExponDistLogPDF, RayleighDistLogPDF, InvParetoDistLogPDF
from utils.eccsamp import imp_sample_rhostar
from utils.io import load_dr25_data_from_hdf5
from utils.models import build_simple_model, build_multilevel_model
from utils.stats import weighted_percentile, draw_random_samples, gelman_rubin

sys.path.append('/Users/research/projects/alderaan/')
sys.path.append('/data/user/gjgilbert/projects/alderaan/')
from alderaan.Results import Results

pi = np.pi

RSAU = (apc.R_sun/apc.au).value                                 # solar radius [AU]
RSRE = (apc.R_sun/apc.R_earth).value                            # R_sun/R_earth
RHOSUN_GCM3 = (3*apc.M_sun/(4*pi*apc.R_sun**3)).value/1000      # solar density [g/cm^3]


# MAIN SCRIPT BEGINS HERE
def main():    
    
    # ## Load data
    
    
    print("Loading data...")
    
    
    if DATA_SOURCE == 'Kepler':
        CATALOG = os.path.join(PROJECT_DIR, 'Catalogs/kepler_dr25_gaia_dr2_crossmatch.csv')
    elif DATA_SOURCE == 'ALDERAAN':    
        CATALOG = os.path.join(PROJECT_DIR, 'Catalogs/kepler_dr25_gaia_dr2_crossmatch.csv')
    elif DATA_SOURCE == 'ALDERAAN-INJECTION':
        CATALOG = os.path.join(DATA_DIR, '{0}.csv'.format(DATA_DIR[DATA_DIR.find('Results')+8:-1]))
    else:
        raise ValueError("Unsuported data source")
    
    catalog = pd.read_csv(CATALOG, index_col=0)
    
    # make injection catalog look like real catalog
    if DATA_SOURCE == 'ALDERAAN-INJECTION':
        n = len(catalog.koi_id)
        
        catalog.columns = map(str.lower, catalog.columns)
        
        # track ground-truth injected values
        catalog['true_rp'] = np.copy(catalog.ror * catalog.rstar * RSRE)
        catalog['true_rstar'] = np.copy(catalog.rstar)
        catalog['true_rhostar'] = np.copy(10**catalog.logrho)
        
        # add measurement error to stellar properties
        catalog['rstar'] = catalog.rstar + 0.04*stats.truncnorm(a=-2, b=2, loc=0,scale=1).rvs(n)*catalog.rstar
        catalog['rstar_err1'] = 0.04*catalog.rstar
        catalog['rstar_err2'] = -0.04*catalog.rstar
        catalog['rhostar'] = 10**catalog.logrho + 0.13*stats.truncnorm(a=-2, b=2, loc=0,scale=1).rvs(n)*(10**catalog.logrho)
        catalog['rhostar_err1'] = 0.13*catalog.rhostar
        catalog['rhostar_err2'] = -0.13*catalog.rhostar
         
        catalog['mstar'] = catalog.rhostar * catalog.rstar**3
        catalog['age'] = 5.0*np.ones(n)
    
        catalog['fpp'] = np.zeros(n)
        catalog['rcf'] = np.ones(n)
        catalog['ruwe'] = np.ones(n)
        
        catalog['disposition'] = ['CONFIRMED']*len(catalog['koi_id'])
        catalog['rp'] = catalog.ror * catalog.rstar * RSRE
        catalog['rp_err'] = 0.1*catalog.rp
    
        planet_name = np.array(catalog.koi_id.values, dtype='U9')
        for i, p in enumerate(planet_name):
            planet_name[i] = planet_name[i] + '.01'
        catalog['planet_name'] = planet_name
            
    
    # hard-code period and radius limits
    use  = (catalog.period > 1) * (catalog.period < 300)
    use *= (catalog.rp > 0) * (catalog.rp < 16)
    
    # remove likely false positives
    use *= (catalog.fpp < 0.1) + (catalog.disposition == 'CONFIRMED')
    
    # clean up stellar sample
    use *= catalog.logg > 4.0                                   # surface gravity as proxy for main sequence
    use *= ((catalog.rcf - 1) < 0.05) + np.isnan(catalog.rcf)   # radius correction factor (Furlan+ 2017)
    use *= catalog.ruwe < 1.4                                   # Gaia RUWE
    
    Rstar = catalog.rstar
    Rstar_err = np.sqrt(catalog.rstar_err1**2 + catalog.rstar_err2**2)/np.sqrt(2)
    
    use *= Rstar_err/Rstar < 0.2
    
    # slice subpopulation
    use *= ((catalog.npl >= MULTIPLICITY[0]) &
            (catalog.npl <= MULTIPLICITY[1]) &
            (catalog.period >= PER_LIM[0]) &
            (catalog.period <= PER_LIM[1]) &
            (catalog.mstar >= MSTAR_LIM[0]) &
            (catalog.mstar <= MSTAR_LIM[1]) &
            (catalog.rstar >= RSTAR_LIM[0]) &
            (catalog.rstar <= RSTAR_LIM[1]) &
            (catalog.feh >= FEH_LIM[0]) &
            (catalog.feh <= FEH_LIM[1]) &
            (catalog.teff >= TEFF_LIM[0]) &
            (catalog.teff <= TEFF_LIM[1]) &
            (catalog.age >= AGE_LIM[0]) &
            (catalog.age <= AGE_LIM[1])
           )
    
    
    # tophat binning (with 30% buffer)
    if RAD_LIM[0] != RAD_LIM[1]:
        use *= ((catalog[RAD_TYPE] >= 0.3*RAD_LIM[0]) & (catalog[RAD_TYPE] <= 1.3*RAD_LIM[1]))
        
    
    # Gaussian binning (5-sigma)
    else:
        # grab objects within 3-sigma of bin center
        rad_center = RAD_LIM[0]
        rad_sigma  = np.sqrt(catalog[RAD_TYPE+'_err']**2 + (RAD_FWHM/2.355*rad_center)**2)
        rad_low    = np.max([0.1, np.min(rad_center-5*rad_sigma)])
        rad_high   = np.min([20., np.max(rad_center+5*rad_sigma)])
    
        use *= np.abs(catalog[RAD_TYPE] - rad_center)/rad_sigma < 5.0
        
        
    # update targets and catalog
    catalog = catalog.loc[use].reset_index(drop=True)
    targets = np.array(catalog.planet_name)
    
    
    # a hacky bit of code to simulate a pristine radius gap
    GAP_WIDTH = None
    
    if GAP_WIDTH is not None:
        use = (catalog.true_rp < 1.84/(1+GAP_WIDTH/2)) + (catalog.true_rp > 1.84*(1+GAP_WIDTH/2))
    
        catalog = catalog.loc[use].reset_index(drop=True)
        targets = np.array(catalog.planet_name)
    
    
    # #### Load posterior chains
    
    
    def infer_planet_koi_from_period(star_koi, P_samp, catalog):
        P_all = catalog.loc[catalog.koi_id==star_koi, 'period'].values
        P_cat = P_all[np.argmin(np.abs(P_all - P_samp))]
            
        return catalog.loc[catalog.period==P_cat, 'planet_name'].values[0]
    
    
    def infer_index_from_planet_koi(planet_koi, results, catalog):
        periods = np.zeros(results.npl)
    
        for n in range(results.npl):
            periods[n] = np.median(results.samples(n).PERIOD)
        
        return np.argmin(np.abs(periods - catalog.loc[catalog.planet_name==planet_koi, 'period'].values))    
    
    
    chains  = {}
    failure = []
    
    # read in the data
    if DATA_SOURCE == 'DR25':
        CHAINS = os.path.join(DATA_DIR, 'dr25-chains_trimmed-thinned.hdf')
    
        for i, t in enumerate(targets):
            try:
                chains[t] = pd.DataFrame(load_dr25_data_from_hdf5(CHAINS, t))
                chains[t]['DUR14'] = calc_T14_circ(chains[t].PERIOD, chains[t].ROR, chains[t].IMPACT, chains[t].RHOTILDE)
    
                if np.any(chains[t].values < 0):
                    raise ValueError("Negative values in posterior chain")
                if np.sum(np.isnan(chains[t].values)) > 0:
                    raise ValueError("NaN values in posterior chain")
    
            except:
                warnings.warn("{0} failed to load".format(t))
                failure.append(t)
    
            
    # Alderaan analysis only configured for single-transiting systems
    elif DATA_SOURCE == 'ALDERAAN':
        files = np.sort(glob.glob(os.path.join(DATA_DIR, '*/*results.fits')))
        
        for i, t in enumerate(targets):
            try:
                results = Results(t[:-3], DATA_DIR)
                n = infer_index_from_planet_koi(t, results, catalog)
    
                chains[t] = results.samples(n).sample(n=8000, replace=True, weights=results.posteriors.weights(), ignore_index=True)
                chains[t] = chains[t].drop(columns='LN_WT')
                
            except FileNotFoundError:
                warnings.warn("{0} failed to load".format(t))
                failure.append(t)
    
    
    elif DATA_SOURCE == 'ALDERAAN-INJECTION':
        files = np.sort(glob.glob(os.path.join(DATA_DIR, '*/*results.fits')))
        
        for i, t in enumerate(targets):
            try:
                
                results = Results('S'+t[1:-3], DATA_DIR)
                n = infer_index_from_planet_koi(t, results, catalog)
    
                chains[t] = results.samples(n).sample(n=8000, replace=True, weights=results.posteriors.weights(), ignore_index=True)
                chains[t] = chains[t].drop(columns='LN_WT')
                
            except FileNotFoundError:
                warnings.warn("{0} failed to load".format(t))
                failure.append(t)            
                
    else:
        raise ValueError("Data source must be either 'DR25'or 'ALDERAAN' or 'ALDERAAN-INJECTION'")
        
    
    # update targets and catalog
    targets = list(np.array(targets)[~np.isin(targets,failure)])
    catalog = catalog.loc[np.isin(catalog.planet_name, targets)].reset_index(drop=True)
    
    print("{0} targets loaded".format(len(targets)))
    
    
    print("{0} planets".format(len(targets)))
    print("{0} stars".format(len(np.unique(catalog.koi_id))))
    print("{0} singles".format(np.sum(catalog.npl == 1)))
    print("{0} multis".format(np.sum(catalog.npl > 1)))
    
    
    # #### Update transit parameters
    
    
    catalog_keys = 'period epoch ror duration impact'.split()
    samples_keys = 'PERIOD T0 ROR DUR14 IMPACT'.split()
    
    for i, t in enumerate(targets):
        use = catalog.planet_name == t
        samples = chains[t]
        
        for j, ck in enumerate(catalog_keys):
            sk = samples_keys[j]
    
            catalog.loc[use, ck] = np.median(samples[sk])
            catalog.loc[use, ck+'_err1'] = np.percentile(samples[sk], 84) - catalog.loc[use, ck]
            catalog.loc[use, ck+'_err2'] = np.percentile(samples[sk], 16) - catalog.loc[use, ck]
    
    
    # #### Calculate self-consistent planet radii
    
    
    ror = catalog.ror
    ror_err = np.sqrt(catalog.ror_err1**2 + catalog.ror_err2**2)/np.sqrt(2)
    
    Rstar = catalog.rstar
    Rstar_err = np.sqrt(catalog.rstar_err1**2 + catalog.rstar_err2**2)/np.sqrt(2)
    
    # radius gap location from Petigura+2022; R = R0*(P/10)^y, R0 = 1.84 +/- 0.03, y = 0.11 +/- 0.02
    catalog['rgap'] = np.array(1.84*(catalog.period/10)**-0.11)
    catalog['rgap_err'] = catalog.rgap * np.sqrt( 0.02**2*np.log(catalog.period/10)**2 + (0.03/1.84)**2)
    
    # physical planet radius
    catalog['rp'] = np.array(ror*Rstar*RSRE)
    catalog['rp_err'] = np.array(catalog.rp * np.sqrt((ror_err/ror)**2 + (Rstar_err/Rstar)**2))
    
    # radius corrected to P=10 days (see Ho & Van Eylen 2023); equivalent to using diagonal bins
    catalog['rp10'] = np.exp(np.log(catalog.rp) - np.log(catalog.rgap) + np.log(1.84))
    catalog['rp10_err'] = np.sqrt(catalog.rp_err**2 + catalog.rgap_err**2)
    
    # radius adjusted for super-Earths and sub-Neptunes only
    rp = catalog.rp
    rgap = catalog.rgap
    rp_adj = np.array(rp)
    
    rp_lower_lim = 1.0
    rp_gap10_loc = 1.84
    rp_giant_lim = 4.0
    
    SE = (rp >= rp_lower_lim)*(rp < rgap)
    SN = (rp >= rgap)*(rp < rp_giant_lim)
    GP = (rp >= rp_giant_lim)
    
    rp_adj[SE] = ((rp - rp_lower_lim)/(rgap - rp_lower_lim) * (rp_gap10_loc - rp_lower_lim) + rp_lower_lim)[SE]
    rp_adj[SN] = ((rp - rgap)/(rp_giant_lim - rgap) * (rp_giant_lim - rp_gap10_loc) + rp_gap10_loc)[SN]
    rp_adj[GP] = rp[GP]
    
    rp_adj_err = np.copy(catalog['rp_err'])
    rp_adj_err[SE+SN] = np.array(catalog['rp10_err'])[SE+SN]
    
    catalog['rpadj'] = rp_adj
    catalog['rpadj_err'] = rp_adj_err
    
    
    # require better than 20% precision on radius
    use = catalog.rp_err/catalog.rp < 0.2
    
    catalog = catalog.loc[use].reset_index(drop=True)
    targets = np.array(catalog.planet_name)
    
    print("{0} targets found with uncertain planetary radii (> 20%)".format(np.sum(~use)))
    
    
    # #### Flag unreliable posterior chains
    
    
    failure = []
    
    if DATA_SOURCE == 'DR25':
        for i, t in enumerate(targets):
            # remove grazing transits
            if np.any(chains[t].IMPACT.values > 1 - chains[t].ROR.values):
                failure.append(t)
    
            # eliminate NaN and zero-valued chains
            if np.any(chains[t].values < 0):
                failure.append(t)
            if np.sum(np.isnan(chains[t].values)) > 0:
                failure.append(t)
    
            # check Gelman-Rubin convergence statistic
            for k in chains[t].keys():
                Rhat = gelman_rubin(chains[t][k].values.reshape(4,-1))
                if Rhat > 1.05:
                    failure.append(t)
    
            # check Hartigan dip test for multimodality
            for k in chains[t].keys():
                dip, pval = diptest.diptest(chains[t][k].values)
                if pval < 0.05:            
                    failure.append(t)
    
                    
    grazing_fraction = []                
                    
    if DATA_SOURCE == 'ALDERAAN' or 'ALDERAAN-INJECTION':
        for i, t in enumerate(targets):
            # remove grazing transits
            grazing = chains[t].IMPACT.values > 1 - chains[t].ROR.values
            
            grazing_fraction.append(np.sum(grazing)/len(grazing))
    
            if grazing_fraction[i] > B_CUT:
                failure.append(t)
    
    # update targets and catalog
    targets = list(np.array(targets)[~np.isin(targets,failure)])
    catalog = catalog.loc[np.isin(catalog.planet_name, targets)].reset_index(drop=True)
    
    print("{0} targets found with unreliable chains".format(len(np.unique(failure))))
    
    
    gf = np.array(grazing_fraction)
    
    if DO_PLOT:
        plt.figure()
        plt.hist(gf[gf > 0], color='lightgrey', bins=np.linspace(0,0.2,41))
        plt.hist(gf[gf > 0], color='k', histtype='step', bins=np.linspace(0,0.2,41))
        plt.axvline(0.01, color='r', ls='--')
        plt.axvline(0.05, color='r', ls=':')
        plt.xticks(np.linspace(0,0.2,11))
        plt.xlabel("fraction of samples with $b > 1 - R_p/R_s$", fontsize=14)
        plt.ylabel("number of planets", fontsize=14)
        plt.show()
    
    
    # #### Grab stellar densities
    
    
    density = {}
    failure = []
    
    for i, t in enumerate(targets):
        try:
            use = catalog.planet_name == t
            
            rho_mu = catalog.loc[use, 'rhostar'].iloc[0]
            rho_err1 = np.abs(catalog.loc[use, 'rhostar_err1'].iloc[0])
            rho_err2 = np.abs(catalog.loc[use, 'rhostar_err2'].iloc[0])
    
            # don't use highly asymmetric density constraints
            if np.abs(rho_err1-rho_err2)/(0.5*(rho_err1+rho_err2)) > 0.30:
                failure.append(t)
            else:
                density[t] = rho_mu, np.sqrt(rho_err1**2 + rho_err2**2)/np.sqrt(2)
        
        except:
            warnings.warn("{0} has mising or unreliable density".format(t))
            failure.append(t)
    
            
    # update targets and catalog
    targets = list(np.array(targets)[~np.isin(targets,failure)])
    catalog = catalog.loc[np.isin(catalog.planet_name, targets)].reset_index(drop=True)
    
    print("{0} targets found with missing or unreliable densities".format(len(np.unique(failure))))
    
    
    # ## Slice subpopulation
    
    
    # Tophat binning
    if RAD_LIM[0] != RAD_LIM[1]:
        use = (catalog[RAD_TYPE] >= RAD_LIM[0]) & (catalog[RAD_TYPE] <= RAD_LIM[1])
        
        catalog = catalog.loc[use].reset_index(drop=True)
        targets = np.array(catalog.planet_name)
    
    
    # Gaussian binning (3 sigma)
    else:
        rad_center = RAD_LIM[0]
        rad_sigma  = np.sqrt(catalog[RAD_TYPE+'_err']**2 + (RAD_FWHM/2.355*rad_center)**2)
        rad_low    = np.max([0.1, np.min(rad_center-3*rad_sigma)])
        rad_high   = np.min([20., np.max(rad_center+3*rad_sigma)])
    
        use = np.abs(catalog[RAD_TYPE] - rad_center)/rad_sigma < 3.0
        
        catalog = catalog.loc[use].reset_index(drop=True)
        targets = np.array(catalog.planet_name)
        
    
    # ensure radius adjustment is valid
    if RAD_TYPE == 'rp10':
        use = (catalog.rp < 4.0) * (catalog.rp > 1.0)
        
        if np.sum(use) == 0:
            raise ValueError("Rp10 correction is only valid for planets between 1-4 Earth-radii")
            
        catalog = catalog.loc[use].reset_index(drop=True)
        targets = np.array(catalog.planet_name)
    
    
    # ## Importance sample $\{e,\omega,\rho_\star\}$
    
    
    print("Importance sampling...")
    
    
    data = {}
    failure = []
    
    for i, t in enumerate(targets):
        try:
            # true stellar density (tuple) in g/cm3
            rho_true = 1.41*density[t][0], 1.41*density[t][1]
    
            w, d = imp_sample_rhostar(chains[t], rho_true, ew_obs_prior=False, upsample=500)
            d = d.sample(n=NSAMP, replace=True, weights=w, ignore_index=True)
    
            if DATA_SOURCE == 'DR25':
                J = jacobian(d.PERIOD, d.ROR, d.IMPACT, d.DUR14)
                d = d.sample(n=NSAMP, replace=True, weights=1/np.abs(J), ignore_index=True)
    
            d = d.drop(columns = 'WEIGHTS')
            data[t] = d
    
        except:
            warnings.warn("{0} failed during sampling and will not be included in the analysis".format(t))
            failure.append(t)
    
            
        
    # update targets and catalog
    targets = list(np.array(targets)[~np.isin(targets,failure)])
    catalog = catalog.loc[np.isin(catalog.planet_name, targets)].reset_index(drop=True)
    
    print("{0} targets failed importance sampling routine".format(len(np.unique(failure))))        
    
    
    if DO_PLOT:
        sns.set_context('paper', font_scale=1.2)
        
        if len(targets) > 50:
            targets_to_plot = np.random.choice(targets, size=50, replace=False)
        else:
            targets_to_plot = np.copy(targets)
        
    
        plt.figure(figsize=(5,5))
        for i, t in enumerate(targets_to_plot):
            e  = data[t].ECC.values
            e_ = np.linspace(0,1,1000)
        
            kde_e = stats.gaussian_kde(np.hstack([-e,e]), bw_method=0.1)
            pdf_e_ = 2*kde_e(e_)
        
            mode_e = e_[np.argmax(pdf_e_)]
    
            my_color = "grey"
            my_cmap  = LinearSegmentedColormap.from_list("my_cmap", [my_color, "k"], N=50)
        
            plt.plot(e_, pdf_e_, c=my_cmap(mode_e), zorder=100*mode_e, lw=2)
            
        plt.xlim(0,1)
        plt.xlabel("$e$", fontsize=16)
        plt.ylabel("samples density", fontsize=16)
        plt.ylim(-0.1,5)
        plt.yticks([])
        
        #plt.text(0.98, 4.85, "sub-Earths", fontsize=16, color=my_color, ha='right', va='top')
        #plt.savefig(os.path.join(PROJECT_DIR, "Figures/ecc-posterior-kde-subearths.pdf"), bbox_inches='tight')
        #plt.text(0.98, 4.85, "$R_p$ = {0:.2f}-{1:.2f}".format(RAD_LIM[0],RAD_LIM[1]), fontsize=16, color=my_color, ha='right', va='top')
        #plt.savefig(os.path.join(PROJECT_DIR, "Figures/ecc-posterior-kde-{0:.2f}-{1:.2f}.pdf".format(RAD_LIM[0],RAD_LIM[1])), bbox_inches='tight')
        plt.show()
    
    
    # ## Calculate bin weights
    
    
    sample_weights = {}
    
    for i, t in enumerate(targets):
        d = data[t]
        
        rgap = catalog.loc[catalog.planet_name == t, 'rgap'].iloc[0]
        rstar = catalog.loc[catalog.planet_name == t, 'rstar'].iloc[0]
        
        rp_samples = d.ROR.values*rstar*RSRE
        rp10_samples = np.exp(np.log(rp_samples) - np.log(rgap) + np.log(1.84))
        rpadj_samples = np.array(rp_samples)
    
        rp_lower_lim = 1.0
        rp_gap10_loc = 1.84
        rp_giant_lim = 4.0
    
        SE = (rp_samples >= rp_lower_lim)*(rp_samples < rgap)
        SN = (rp_samples >= rgap)*(rp_samples < rp_giant_lim)
        GP = (rp_samples >= rp_giant_lim)
    
        rpadj_samples[SE] = ((rp_samples - rp_lower_lim)/(rgap - rp_lower_lim) * (rp_gap10_loc - rp_lower_lim) + rp_lower_lim)[SE]
        rpadj_samples[SN] = ((rp_samples - rgap)/(rp_giant_lim - rgap) * (rp_giant_lim - rp_gap10_loc) + rp_gap10_loc)[SN]
        rpadj_samples[GP] = rp_samples[GP]
            
        
        if RAD_LIM[0] != RAD_LIM[1]:
            sample_weights[t] = np.ones(NSAMP, dtype='float')/NSAMP
    
        else:
            rad_center = RAD_LIM[0]
            rad_sigma  = RAD_FWHM/2.355*rad_center
    
            if RAD_TYPE == 'rp':
                w_ = stats.norm(rad_center, rad_sigma).pdf(rp_samples)/(NSAMP*stats.norm(0,rad_sigma).pdf(0))
            
            elif RAD_TYPE == 'rp10':
                w_ = stats.norm(rad_center, rad_sigma).pdf(rp10_samples)/(NSAMP*stats.norm(0,rad_sigma).pdf(0))
            
            elif RAD_TYPE == 'rpadj':
                w_ = stats.norm(rad_center, rad_sigma).pdf(rpadj_samples)/(NSAMP*stats.norm(0,rad_sigma).pdf(0))
    
            else:
                raise ValueError("RAD_TYPE not implemented")
            
            sample_weights[t] = w_
    
    
    # #### Vectorize samples
    
    
    samples_array = {}
    
    samples_array['ecc']    = np.zeros((len(targets),NSAMP))
    samples_array['omega']  = np.zeros((len(targets),NSAMP))
    samples_array['impact'] = np.zeros((len(targets),NSAMP))
    
    for i, t in enumerate(targets):
        samples_array['ecc'][i]    = np.array(data[t].ECC)
        samples_array['omega'][i]  = np.array(data[t].OMEGA)
        samples_array['impact'][i] = np.array(data[t].IMPACT)
    
    weights = np.zeros(len(targets))
    
    for i, t in enumerate(targets):
        weights[i] = np.sum(sample_weights[t])
    
    
    # #### Select high weight objects
    
    
    if RAD_LIM[0] != RAD_LIM[1]:
        weights = None
    
    else:
        w = weights/np.sum(weights)
        
        order = np.argsort(w)
        keep = order[np.cumsum(w[order]) > 0.01]
        
        for k in samples_array.keys():
            samples_array[k] = samples_array[k][keep]
    
        weights = weights[keep]
        
        print("{0} targets have negligible weight within radius bin".format(len(order)-len(keep)))
    
    
    # ## Run hierarchical model
    
    
    print("Running hierarchical MCMC using {0} planets".format(len(samples_array['ecc'])))
    
    
    # #### Load empirical distribution template (if needed)
    
    
    if DISTRIBUTION == 'empirical':
        template_values = np.loadtxt(os.path.join(PROJECT_DIR, "template_distribution.txt")).T
        template_spline = CubicSpline(template_values[0], template_values[1], extrapolate=False)
    else:
        template_values = None
        template_spline = None
    
    
    # #### Build model and run MCMC
    
    
    if BOOTSTRAP == 'none':
        use_wlb = False
    elif BOOTSTRAP == 'wlb':
        use_wlb = True
    elif isinstance(BOOTSTRAP, int):
        use_wlb = False
    else:
        raise ValueError("BOOTRAP must be 'none', 'wlb', or an integer")
    
    
    if isinstance(BOOTSTRAP, int):
        ndraw = int(np.ceil(np.sum(weights)))
        
        trace_list = []
        bin_edges_list = []
        
        for i in range(BOOTSTRAP):
            inds = np.random.choice(np.arange(0,len(weights)), size=ndraw, replace=True, p=weights/np.sum(weights))
            
            
            samples = {}
            for k in samples_array.keys():
                samples[k] = samples_array[k][inds]
    
            # build model and sample from posterior
            model, bin_edges = build_simple_model(samples,
                                                  DISTRIBUTION,
                                                  NBIN,
                                                  e_detprior=E_DETPRIOR,
                                                  b_detprior=B_DETPRIOR,
                                                  weights=None,
                                                  use_wlb=False,
                                                  template_spline=template_spline,
                                                  eps=1e-6
                                                 )
    
            with model:
                trace = pmx.sample(tune=5000, draws=1000, chains=2, cores=2, target_accept=0.95, return_inferencedata=True)
                
            trace_list.append(trace)
            bin_edges_list.append(bin_edges)
         
        
    else:
        # build model and sample from posterior
        model, bin_edges = build_simple_model(samples_array,
                                              DISTRIBUTION,
                                              NBIN,
                                              e_detprior=E_DETPRIOR,
                                              b_detprior=B_DETPRIOR,
                                              weights=weights,
                                              use_wlb=use_wlb,
                                              template_spline=template_spline,
                                              eps=1e-6
                                             )
    
        with model:
            trace = pmx.sample(tune=5000, draws=1000, chains=2, cores=2, target_accept=0.95, return_inferencedata=True)
    
        trace_list = [trace]
        bin_edges_list = [bin_edges]
    
    
    if DO_PLOT:
        ln_pdf = np.percentile(trace.posterior.ln_pdf, [16,50,84], axis=(0,1))
        mean_ecc = np.percentile(trace.posterior.mean_ecc, [16,50,84], axis=(0,1))
    
        pdf = np.exp(ln_pdf)
        bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
    
        plt.figure()
        plt.plot(bin_centers, pdf[1], 'C0', lw=2)
        plt.fill_between(bin_centers, pdf[0], pdf[2], color='C0', alpha=0.3)
        plt.axvline(mean_ecc[0], ls=':', color='C1')
        plt.axvline(mean_ecc[1], ls='-', color='C1')
        plt.axvline(mean_ecc[2], ls=':', color='C1')
        plt.show()
    
    
    # ## Save posterior trace
    
    
    for i, trace in enumerate(trace_list):
        bin_edges = bin_edges_list[i]
    
        # refactor dataframe for export
        df = trace.to_dataframe(groups='posterior', include_coords=False)
    
        column_map = {}
        for k in list(df.keys()):    
            if k.find('[') == -1:
                column_map[k] = k
            else:
                column_map[k] = k[:k.find('[')] + '_{:02d}'.format(int(k[k.find('[')+1:-1]))
    
        df = df.rename(columns=column_map)
        
        # saves as fits HDU
        primary_hdu = fits.PrimaryHDU()
        header = primary_hdu.header
    
        header['YYYYMMDD'] = YYYYMMDD
        header['DIST']     = DISTRIBUTION
        header['NSAMP']    = NSAMP
        header['NBIN']     = NBIN
        header['NOBJ']     = samples_array['ecc'].shape[0]
        header['BOOTSTR']  = BOOTSTRAP
        header['MULT_0']   = MULTIPLICITY[0]
        header['MULT_1']   = MULTIPLICITY[1]
        header['PER_0']    = PER_LIM[0]
        header['PER_1']    = PER_LIM[1]
        header['RAD_TYPE'] = RAD_TYPE
        header['RAD_0']    = RAD_LIM[0]
        header['RAD_1']    = RAD_LIM[1]
        header['RAD_FWHM'] = RAD_FWHM
        header['MSTAR_0']  = MSTAR_LIM[0]
        header['MSTAR_1']  = MSTAR_LIM[1]
        header['RSTAR_0']  = RSTAR_LIM[0]
        header['RSTAR_1']  = RSTAR_LIM[1]
        header['FEH_0']    = FEH_LIM[0]
        header['FEH_1']    = FEH_LIM[1]
        header['TEFF_0']   = TEFF_LIM[0]
        header['TEFF_1']   = TEFF_LIM[1]
        header['AGE_0']    = AGE_LIM[0]
        header['AGE_1']    = AGE_LIM[1]
        header['E_PRIOR']  = E_DETPRIOR
        header['B_PRIOR']  = B_DETPRIOR
        header['B_CUT']    = B_CUT
        
        samples_hdu = fits.BinTableHDU(data=df.to_records(index=False), name='SAMPLES')
        binedges_hdu = fits.ImageHDU(bin_edges, name='BINEDGES')
    
        hduL  = fits.HDUList([primary_hdu, samples_hdu, binedges_hdu])
        fname = os.path.join(RESULTS_DIR, "{0}_{1}_{2}.fits".format(YYYYMMDD, RUN_ID, str(i).zfill(3)))
    
        hduL.writeto(fname, overwrite=True)
    
    
    # this complicated loading routine deals with big-endian vs little-endian mismatch between pandas and FITS
    RELOAD = False
    if RELOAD:
        with fits.open(fname) as hduL:
            data = hduL['SAMPLES'].data
            keys = data.names
    
            _samples = []
            for k in keys:
                _samples.append(data[k])
    
            samples = pd.DataFrame(np.array(_samples).T, columns=keys)
            bin_edges = np.array(hduL['BINEDGES'].data)
    
    
    # ## Exit program
    
    
    print("")
    print("+"*shutil.get_terminal_size().columns)
    print("Inference complete {0}".format(datetime.now().strftime("%d-%b-%Y at %H:%M:%S")))
    print("Total runtime = %.1f min" %((timer()-global_start_time)/60))
    print("+"*shutil.get_terminal_size().columns)
    
    
if __name__ == '__main__':
    main()