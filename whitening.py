import numpy as np
from scipy.interpolate import interp1d

def velocity_transform(lambda_orig, v):
    c = 299792.458 # km/s
    lambda_target = lambda_orig*(1-v/c)
    return lambda_target

def apply_rad_vel(wave, flux, v):
    wave_new = velocity_transform(wave,v)
    wave_new_aux = np.insert(wave_new,0,wave[0])
    wave_new_aux = np.append(wave_new_aux,wave[-1])
    flux_aux = flux.T
    flux_aux = np.insert(flux_aux,0,0)
    flux_aux = np.append(flux_aux,0)
    func = interp1d(wave_new_aux,flux_aux)
    flux_new = func(wave)
    flux_new = flux_new.astype(np.float32)

    return wave_new, flux_new
