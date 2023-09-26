import numpy as np

class harps_spec_info():
    PaddedLength = 2**18+2**16 #327680
    item_bytes = 4 #sizeof(np.float32)
    spectrum_bytes = PaddedLength*item_bytes
    eps = 1e-5
    desiredMinW = 3785 #inclusive
    desiredMaxW = 6910 #inclusive
    wave_resolution = 0.01
    desiredTrimmedLength = (desiredMaxW-desiredMinW)/wave_resolution+1 #should be 312501
    padL = int((PaddedLength-desiredTrimmedLength)//2)
    padR = int(PaddedLength-padL-desiredTrimmedLength)

    #- parameters pertaining the 0-region artifacts
    left_last_zero = 7588 #last zero - left
    right_first_zero = 320090 #first zero point - right
    mid_first_zero = 159453 #first zero point
    mid_last_zero = 162893 #last zero point

    WAVE = np.arange(desiredMinW-padL*wave_resolution,
                     desiredMaxW+.001+padR*wave_resolution,
                     step=wave_resolution)
