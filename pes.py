import numpy as np
import constants as cs

class PotentialEnergySurface:

    def harmonic(self, cord, w, disp):
        return 0.5*w*(cord - disp)**2
    
    ### TODO: Make sure paramters are provided in cm-1 (fs for now).
    def morse(self, cord, wdof, disp, D):
        D *= cs.CM2FS
        alp = np.sqrt(wdof/(2*D))
        return D*( 1 - np.exp(-alp*(cord - disp)) )**2
