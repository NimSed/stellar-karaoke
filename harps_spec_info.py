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

    @staticmethod
    def get_artifact_mask():
        mask = np.ones((1,harps_spec_info.PaddedLength),dtype=np.float32)
        mask[0,:harps_spec_info.left_last_zero+1] = 0
        mask[0,harps_spec_info.right_first_zero:] = 0
        mask[0,harps_spec_info.mid_first_zero:harps_spec_info.mid_last_zero+1] = 0
        return mask

    @staticmethod
    def median_normalize(spec, med_threshold):
        mask = harps_spec_info.get_artifact_mask()
        masked_spec = np.ma.masked_array(spec,1-mask)
        med = np.ma.median(masked_spec)

        if(med <= med_threshold):
            spec[:] = 0 #this is the case of some degenerate (low-amp, noise-like) spectra
            return spec, 0

        if(np.mean(spec) <= 0):
            spec[:] = 0 #this takes care of some other degenerate spectra
            return spec, 0

        return spec/med, med


    @staticmethod
    def preprocess(wave, flux):
        #--- Trim
        wtrimmed = wave[ np.logical_and(wave >= (harps_spec_info.desiredMinW - harps_spec_info.eps),
                                        wave <= (harps_spec_info.desiredMaxW + harps_spec_info.eps)) ]
        ftrimmed = flux[ np.logical_and(wave >= (harps_spec_info.desiredMinW - harps_spec_info.eps),
                                        wave <= (harps_spec_info.desiredMaxW + harps_spec_info.eps)) ]
        #--- Pad
        fPadded = np.pad(ftrimmed, pad_width=(harps_spec_info.padL, harps_spec_info.padR), mode='constant', constant_values=(0,0))

        #--- Normalize
        fNorm, _ = harps_spec_info.median_normalize(fPadded, med_threshold=50)

        return harps_spec_info.WAVE, fNorm
