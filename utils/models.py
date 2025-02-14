import aesara_theano_fallback.tensor as T
from   aesara_theano_fallback import aesara as theano
from   celerite2.theano import GaussianProcess
from   celerite2.theano import terms as GPterms
import numpy as np
import pymc3 as pm
from   scipy.special import logsumexp

from .astro import detection_prior
from .distributions import BetaDistLogPDF, NormDistLogPDF, ExponDistLogPDF, RayleighDistLogPDF, InvParetoDistLogPDF 

__all__ = ['build_simple_model',
           'build_multilevel_model'
          ]


def build_simple_model(samples_array, distribution, Nbin, e_detprior=True, b_detprior=False, weights=None, use_wlb=False, eps=1e-15, template_spline=None):
    # rename for backwards compatibility with previous version of code
    ecc_array    = samples_array['ecc']
    omega_array  = samples_array['omega']
    impact_array = samples_array['impact']
    
    # number of planets
    N = len(ecc_array)
    
    # determine weight for each object
    if weights is None:
        W = np.ones(N)
    else:
        W = weights
    
    # calculate detection prior
    ln_detprior = np.zeros_like(ecc_array)

    if e_detprior:
        ln_detprior -= np.log(detection_prior(ecc_array, omega_array))
    if b_detprior:
        ln_detprior -= (1-impact_array**2)**0.25

    ln_detprior -= logsumexp(ln_detprior)
    
    # set up histogram bins
    if len(ecc_array) == 0:
        bin_edges = np.linspace(0, 1, Nbin+1)
    else:
        bin_edges = np.percentile(ecc_array.flatten(), np.linspace(0,100,Nbin+1))
        bin_edges[0] = 0.
        bin_edges[-1] = 1.

    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
        
    # indexes of samples in bins
    inds = np.digitize(ecc_array, bin_edges[1:], right=True)
    

    with pm.Model() as model:
        if distribution == 'histogram':        
            # hyperpriors on GP
            log_s = pm.TruncatedNormal("log_s", mu=3, sd=1, lower=0, upper=6)
            log_r = pm.TruncatedNormal("log_r", mu=0, sd=1, lower=-3, upper=3)
            kernel = GPterms.Matern32Term(sigma=T.exp(log_s), rho=T.exp(log_r))

            # calculate bin heights from latent draws
            latent = pm.Normal("latent", mu=0, sd=1, shape=len(bin_centers), testval=np.linspace(1,-1,len(bin_centers)))
            LS = T.exp(log_s)*latent
            diag = T.var(LS[1:]-LS[:-1])/T.sqrt(2)*T.ones(len(bin_centers))
        
            gp = GaussianProcess(kernel, mean=T.mean(LS))
            gp.compute(bin_centers, diag=diag)
            y_ = gp.predict(LS)
            
            # estimate the log-pdf
            ln_pdf = pm.Deterministic("ln_pdf", y_ - pm.logsumexp(y_+T.log(bin_widths)))

            # hierarchical distribution
            F = ln_pdf[inds]
            Z = pm.logsumexp(F + ln_detprior, axis=1).squeeze()
    
    
        if distribution == 'empirical':
            # hyperprior on vertical scale parameter
            log_v = pm.Uniform("log_v", lower=np.log(0.02), upper=np.log(50), testval=np.log(1))
            v = pm.Deterministic("v", T.exp(log_v))

            # hyperprior on horizontal scale parameter
            log_h = pm.Uniform("log_h", lower=np.log(0.02), upper=np.log(50), testval=np.log(1))
            h = pm.Deterministic("h", T.exp(log_h))

            # regularization on scale parameters
            pm.Potential("scale_penalty", -(v-h)**2)

            # horizontal scaling
            x_ = bin_centers*h
            idx = T.TensorVariable.searchsorted(template_spline.x[1:], x_)

            coeffs = theano.shared(template_spline.c)
            coeffs = T.concatenate([coeffs.T, np.array([[0,0,0,template_spline(1)]])]).T
            coeffs = coeffs[:,idx]

            x_ = x_ - theano.shared(template_spline.x)[idx]
            x_ = T.stack([T.ones_like(x_), x_, x_**2, x_**3])[::-1]

            # vertical scaling
            y_ = T.sum(coeffs*x_, axis=0)
            y_ = (y_-y_[0])*v + y_[0]

            # estimate the log-pdf
            ln_pdf = pm.Deterministic("ln_pdf", y_ - pm.logsumexp(y_+T.log(bin_widths)))
            
            # hierarchical distribution
            F = ln_pdf[inds]
            Z = pm.logsumexp(F + ln_detprior, axis=1).squeeze()
    
    
        if distribution == 'beta':
            # hyperpriors (from Gelman Baysian Data Analysis Chapter 5)
            mu  = pm.Uniform("mu", lower=0.01, upper=0.99)                     # mu = a/(a+b) ~ mean
            tau = pm.Uniform("tau", lower=0.01, upper=10)                      # tau = 1/sqrt(a+b) ~ 'inverse concentration'
                
            a = pm.Deterministic("a", mu/tau**2)
            b = pm.Deterministic("b", (1-mu)/tau**2)
            
            # precompute the beta function
            ln_B = T.gammaln(a) + T.gammaln(b) - T.gammaln(a+b)
            
            # track the log-pdf
            ln_pdf = pm.Deterministic("ln_pdf", BetaDistLogPDF(a, b, bin_centers, ln_B=ln_B))

            # hierarchical distribution
            F = BetaDistLogPDF(a, b, ecc_array, ln_B=ln_B)
            Z = pm.logsumexp(F + ln_detprior, axis=1).squeeze()
    
    
        if distribution == 'halfnormal':
            # hyperpriors
            log_sd = pm.Normal("log_sd", mu=0, sd=5, testval=np.log(0.2))
            sd = pm.Deterministic("sd", T.exp(log_sd))
            mu = 0.0

            # track the pdf
            ln_pdf = pm.Deterministic("ln_pdf", NormDistLogPDF(mu, sd, bin_centers, bounds=(0,1), eps=eps))
                        
            # hierarchical distribution
            F = NormDistLogPDF(mu, sd, ecc_array, bounds=(0,1), eps=eps)
            Z = pm.logsumexp(F + ln_detprior, axis=1).squeeze()
    
    
        if distribution == 'expon':
            # hyperpriors
            log_c = pm.Normal("log_c", mu=0, sd=5, testval=np.log(0.2))
            c = pm.Deterministic("c", T.exp(log_c))

            # track the pdf
            ln_pdf = pm.Deterministic("ln_pdf", ExponDistLogPDF(c, bin_centers, bounds=(0,1), eps=eps))

            # hierarchical distribution
            F = ExponDistLogPDF(c, ecc_array, bounds=(0,1), eps=eps)
            Z = pm.logsumexp(F + ln_detprior, axis=1).squeeze()
        
    
        if distribution == 'rayleigh':
            # hyperpriors
            log_sd = pm.Normal("log_sd", mu=0, sd=5, testval=np.log(0.2))
            sd = pm.Deterministic("sd", T.exp(log_sd))

            # track the pdf
            ln_pdf = pm.Deterministic("ln_pdf", RayleighDistLogPDF(sd, bin_centers, bounds=(0,1), eps=eps))
            
            # hierarchical distribution
            F = RayleighDistLogPDF(sd, ecc_array, bounds=(0,1), eps=eps)
            Z = pm.logsumexp(F + ln_detprior, axis=1).squeeze()
    
    
        if distribution == 'invpareto':
            # hyperpriors
            log_c = pm.Normal("log_c", mu=0, sd=5, testval=np.log(0.2))
            c = pm.Deterministic("c", T.exp(log_c))

            # track the pdf
            ln_pdf = pm.Deterministic("ln_pdf", InvParetoDistLogPDF(c, bin_centers, eps=eps))
            
            # hierarchical distribution
            F = InvParetoDistLogPDF(c, ecc_array, eps=eps)
            Z = pm.logsumexp(F + ln_detprior, axis=1).squeeze()
    
    
        # track mean eccentricity
        p_ = T.exp(ln_pdf)
        w_ = (p_*bin_widths)/T.sum(p_*bin_widths)
        mean_ecc = pm.Deterministic("mean_ecc", T.sum(w_*bin_centers))
            
        # penalty against very low/high mean eccentricity (to avoid numerical edge effects)
        pm.Potential("edge_penalty_low", -1000/(1+T.exp(500*mean_ecc)))
        pm.Potential("edge_penalty_high", T.log(0.5-mean_ecc))
        
        # weighted likelihood bootstrap
        if use_wlb:
            Y = pm.Dirichlet("Y", a=np.ones(N))
        else:
            Y = np.ones(N)/N

        # likelihood
        ln_like = T.sum(Z*Y*W*N)
        
        pm.Potential("potential", ln_like)
        pm.Deterministic("ln_like", ln_like)
    
    
    return model, bin_edges    


### MULTILEVEL MODEL HAS NOT BEEN MAINTAINED!!!!!

def build_multilevel_model(ecc_list, omega_list, distribution, Nbin, use_detprior=False, eps=1e-15, template_spline=None):
    # set up histogram bins
    bin_edges = np.linspace(0, 1, Nbin+1)**2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    if eps is None:
        eps = 1e-15

    ngroups = len(ecc_list)

    if distribution == 'halfnormal':
        with pm.Model() as model:
            # hyperpriors
            log_s_mu = pm.Normal("log_s_mu", mu=0, sd=5)
            log_s_sd = pm.HalfCauchy("log_s_sd", beta=2)
            log_s_off = pm.Normal("log_s_off", mu=0, sd=1, shape=ngroups)
            
            log_s = pm.Deterministic("log_s", log_s_mu + log_s_sd*log_s_off)
            s = pm.Deterministic("s", T.exp(log_s))

            # hierarchical model
            F = [None]*ngroups
            Z = [None]*ngroups
            
            for i, ecc in enumerate(ecc_list):
                if len(ecc) > 0:
                    pm.Deterministic("ln_pdf_{0}".format(i), NormDistLogPDF(0, s[i], bin_centers, bounds=(0,1), eps=eps))

                    F[i] = NormDistLogPDF(0, s[i], ecc, bounds=(0,1), eps=eps)
                    Z[i] = pm.logsumexp(F[i].T, axis=0)
            
                    # likelihood
                    pm.Potential("ln_like_{0}".format(i), T.sum(Z[i]))
    
    
    if distribution == 'empirical':
        with pm.Model() as model:
            # hyperpriors on vertical scale parameter
            log_v_mu = pm.Uniform("log_v_mu", lower=np.log(0.02), upper=np.log(50))
            log_v_sd = pm.HalfCauchy("log_v_sd", beta=2)

            lower = (np.log(0.02)-log_v_mu)/log_v_sd
            upper = (np.log(50)-log_v_mu)/log_v_sd
            
            log_v_off = pm.Bound(pm.Normal, lower=lower, upper=upper)("log_v_off", mu=0, sd=1, shape=ngroups)
            log_v = pm.Deterministic("log_v", log_v_mu + log_v_sd*log_v_off)
            v = pm.Deterministic("v", T.exp(log_v))

            # hyperpriors on horizontal scale parameter
            log_h_mu = pm.Uniform("log_h_mu", lower=np.log(0.02), upper=np.log(50))
            log_h_sd = pm.HalfCauchy("log_h_sd", beta=2)

            lower = (np.log(0.02)-log_h_mu)/log_h_sd
            upper = (np.log(50)-log_h_mu)/log_h_sd
            
            log_h_off = pm.Bound(pm.Normal, lower=lower, upper=upper)("log_h_off", mu=0, sd=1, shape=ngroups)
            log_h = pm.Deterministic("log_h", log_h_mu + log_h_sd*log_h_off)
            h = pm.Deterministic("h", T.exp(log_h))

            # regularization on scale parameters
            pm.Potential("scale_penalty", -T.sum((v-h)**2))

            # empty lists for hierarchical model
            ln_pdf = [None]*ngroups
            F = [None]*ngroups
            Z = [None]*ngroups

            # loop through each planet group
            for i, ecc in enumerate(ecc_list):
                if len(ecc) > 0:
                    # horizontal scaling
                    x_ = bin_centers*h[i]
                    inds = T.TensorVariable.searchsorted(template_spline.x[1:], x_)

                    coeffs = theano.shared(template_spline.c)
                    coeffs = T.concatenate([coeffs.T, np.array([[0,0,0,template_spline(1)]])]).T
                    coeffs = coeffs[:,inds]

                    x_ = x_ - theano.shared(template_spline.x)[inds]
                    x_ = T.stack([T.ones_like(x_), x_, x_**2, x_**3])[::-1]
                    y_ = T.sum(coeffs*x_, axis=0)

                    # vertical scaling
                    y_ = (y_-y_[0])*v[i] + y_[0]

                    # final estimate of normalized log-pdf
                    ln_pdf[i] = pm.Deterministic("ln_pdf_{0}".format(i), y_ - pm.logsumexp(y_+T.log(bin_widths)))
                    
                    # penalty against very low/high mean eccentricity (to avoid numerical edge effects)
                    p_ = T.exp(ln_pdf[i])
                    w_ = (p_*bin_widths)/T.sum(p_*bin_widths)
                    emu_ = T.sum(w_*bin_centers)
            
                    pm.Potential("edge_penalty_low_{0}".format(i), -1000/(1+T.exp(500*emu_)))
                    pm.Potential("edge_penalty_high_{0}".format(i), T.log(0.5-emu_))

                    # track mean eccentricity
                    pm.Deterministic("mean_ecc_{0}".format(i), emu_)
           
                    # indexes of samples in bins
                    inds = np.digitize(ecc, bin_edges[1:], right=True)
            
                    # log-pdf for each sample
                    F[i] = ln_pdf[i][inds]
                    Z[i] = pm.logsumexp(F[i].T, axis=0)

                    # likelihood
                    pm.Potential("ln_like_{0}".format(i), T.sum(Z[i]))
                    
    return model, bin_edges