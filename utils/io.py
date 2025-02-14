from   astropy.io import fits
import numpy as np
import h5py
import pandas as pd

__all__ = ['load_dr25_data_from_hdf5',
           'load_posteriors',
           'extract_posteriors'
          ]

def load_dr25_data_from_hdf5(file, koi_id):
    """
    file : path to hdf5 chains file, '{ROOT}/dr25-chains_trimmed-thinned.hdf'
    koi_id : KOI planet id, e.g. 'K00000.01'
    """
    with h5py.File(file, 'r') as h5file:
        group = h5file[koi_id]
        keys  = list(group['block0_items'])
    
        #planet_no = str(int(koi_id[-1])).encode()
        planet_no = str(1).encode()
    
        data = {}
        data['PERIOD']   = group['block0_values'][:,keys.index(b'PE'+planet_no)]    # period [days]
        data['EPOCH']    = group['block0_values'][:,keys.index(b'EP'+planet_no)]    # epoch BJD-2454900
        data['ROR']      = group['block0_values'][:,keys.index(b'RD'+planet_no)]    # Rp/Rstar
        data['IMPACT']   = group['block0_values'][:,keys.index(b'BB'+planet_no)]    # impact parameter
        data['RHOTILDE'] = group['block0_values'][:,keys.index(b'RHO')]             # stellar density [g/cm3]

    return data


def load_posteriors(files):
    """
    Load posterior outputs from infer_hbayes_ecc.py

    Arguments
    ---------
        files : list of fits files containing posterior outputs
        
    Returns
    -------
        samples : list of pandas dataframes containing MCMC samples
        headers : list of fits headers containing metadata
        bin_edges : list of numpy arrays with bin edges of eccentricity distribution
    """
    nfiles = len(files)

    samples = [None]*nfiles
    headers = [None]*nfiles
    bin_edges = [None]*nfiles
    
    for i, fname in enumerate(files):
        with fits.open(fname) as hduL:
            headers[i] = hduL[0].header
            data = hduL['SAMPLES'].data
            keys = data.names
            
            _samples = []
            for k in keys:
                _samples.append(data[k])
        
            samples[i] = pd.DataFrame(np.array(_samples).T, columns=keys)
            bin_edges[i] = np.array(hduL['BINEDGES'].data)


    return samples, headers, bin_edges


def extract_posteriors(samples, headers):
    """
    Extract commonly used values from samples and header (see 'load_posteriors')

    Arguments
    ---------
        samples : list of pandas dataframes containing MCMC samples
        headers : list of fits headers containing metadata

    Returns
    -------
        rp   : numpy array (N x 2) with Rp bins
        ecc  : numpy array (N x L) with ecc samples
        mult : numpy array (N) with multiplicity
        nobj : numpy array (N) with number of objects
    """
    rp = np.zeros((len(samples),2), dtype='float')
    ecc = np.zeros((len(samples),len(samples[0].mean_ecc)), dtype='float')
    mult = np.zeros(len(samples), dtype='int')
    nobj = np.zeros(len(samples), dtype='int')
    
    for i, s in enumerate(samples):
        head = headers[i]
        
        rp[i]   = (head['RAD_0'], head['RAD_1'])
        ecc[i]  = s.mean_ecc.values
        mult[i] = head['MULT_1']
        nobj[i] = head['NOBJ']

    return rp, ecc, mult, nobj