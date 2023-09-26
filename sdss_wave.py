from scipy.special import eval_legendre
import numpy as np
from astropy.io import fits


def get_sdss_wave(hdus_data_, spec_num):
    xmin = hdus_data_[1]
    xmax = hdus_data_[2]
    wavelen_legendre_coeffs = hdus_data_[3]

    def calc_mu(x):
        xmid = (xmax-xmin)/2
        return 2*((x-xmid)/(xmax-xmin))

    log10_wavelen = np.zeros(2048)
    fibers = np.arange(xmin, xmax+0.5)
    for i in range(5):
        log10_wavelen += wavelen_legendre_coeffs[spec_num][i] * eval_legendre(i, calc_mu(fibers))
    wavelen = 10 ** log10_wavelen

    return wavelen

def get_sdss_wave_by_ID(spec_ID, root_path='/rdata/datasets/SDSS/spectra/dr17.sdss.org/sas/dr17/sdss/spectro/redux/'):

    subdir = '/'.join(spec_ID.replace('_','/').split('/')[:-1]) #the last component is the index of the spec _inside_ the fits file!
    infile_index = int(spec_ID.split('_')[-1])
    
    filename = f'{root_path}/{subdir}.fits.gz'
    hdus = fits.open(filename)
    from copy import deepcopy
    hdus3_data0 = deepcopy(hdus[3].data[0]) 
    hdus3_data0_ = [[], hdus3_data0[1], hdus3_data0[2], hdus3_data0[3]]
    
    return get_sdss_wave(hdus3_data0_, infile_index)
