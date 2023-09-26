import numpy as np

class sdss_spec_info():
    #--- Fixed parameters based on .bin format
    PaddedLength = 2**11 #2048
    item_bytes = 4 #sizeof(np.float32)
    spectrum_bytes = PaddedLength*item_bytes
    eps = 1e-5
    desiredMinW = 5820 #inclusive
    desiredMaxW = 9175 #inclusive
    wave_resolution = 1.66 #(Angstrom)

    desiredTrimmedLength = (desiredMaxW-desiredMinW+eps)//wave_resolution + 1

    padL = int((PaddedLength-desiredTrimmedLength)//2)
    padR = int(PaddedLength-padL-desiredTrimmedLength)

    #- parameters pertaining the 0-region artifacts
    left_last_zero =  padL #last zero - left
    right_first_zero = int((desiredMaxW-desiredMinW+eps)//wave_resolution+padL) #first zero point - right
    mid_first_zero =  -1 #first zero point
    mid_last_zero =  -1 #last zero point

    WAVE = np.arange(desiredMinW-padL*wave_resolution,desiredMaxW+.001+padR*wave_resolution,step=wave_resolution)
