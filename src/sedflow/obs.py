'''

module for reading in observations 


'''



def NSA(): 
    ''' read in NSA catalog in dictionary form with clean photometry along with
    indices for 100 randomly selected galaxies.
    '''
    dat_dir = data_dir()

    # read in NSA data with clean photometry
    nsa = {}
    f = h5py.File(os.path.join(dat_dir, 'nsa.clean_photo.hdf5'), 'r')
    for k in f.keys(): 
        nsa[k] = f[k][...]
    f.close()

    # read in indices of example galaxies 
    igals = np.load(os.path.join(dat_dir, 'nsa.subample.igals.npy'))
    return nsa, igals 


def load_nsa_data(test_set=True): 
    ''' load nsa data
    '''
    nsa, igals = NSA() 

    flux        = nsa['NMGY'][:,2:]
    sigma_flux  = nsa['NMGY_IVAR'][:,2:]**-0.5

    # calculate magnitudes and uncertainties in the magnitudes
    nsa_mags = flux2mag(flux) 
    nsa_sigs = sigma_flux2mag(sigma_flux, flux) 
    nsa_zred = np.atleast_2d(nsa['Z'].flatten()).T
    
    if test_set: 
        condits = [] 
        for igal in igals: 
            _condit = np.concatenate([nsa_mags[igal], nsa_sigs[igal], nsa_zred[igal]])
            condits.append(_condit) 

        return np.array(condits)  
    else: 
        return np.concatenate([nsa_mags, nsa_sigs, nsa_zred], axis=1)


def load_nsa_mcmc(): 
    dat_dir = data_dir() 
    return np.load(os.path.join(dat_dir, 'mcmc.props.nsa.npy'))


