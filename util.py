from astropy.io import fits
import os

def load_harps_spectrum(specID, data_path='/rdata/harps/fits'):
    """
    Load a HARPS spectrum from the FITS file.
    """
    hdu = fits.open(os.path.join(data_path, specID+'.fits'))
    data = hdu[1].data
    wave = hdu[1].data.field('WAVE').T
    flux = hdu[1].data.field('FLUX').T

    return wave, flux
