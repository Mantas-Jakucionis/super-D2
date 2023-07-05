import numpy as np
import constants as cs

def MEInteraction(Mol, Env, ref=None, id = 1):

    refType = type(np.empty(0))
    isMulti = [type(Mol)==refType, type(Env)==refType]
    
    if not np.any(isMulti):
        return MoleculeEnvironmentInteraction(Mol, Env, id)    
    
    elif np.all(isMulti):
        
        N_multi = len(Mol)    
        if N_multi != len(Env):
            raise("Length of molecule and environment lists have to be the same.")
        
        mei_list = np.empty(N_multi, dtype=np.object)
        
        for n in range(N_multi):
        
            mei_list[n] = MoleculeEnvironmentInteraction(Mol[n], Env[n], n+id)

            if ref != None :
                mei_list[n].el_phon_cpl  = ref.el_phon_cpl 
                mei_list[n].spdName      = ref.spdName     
                mei_list[n].spdArgs      = ref.spdArgs     
                mei_list[n].vp_mtp       = ref.vp_mtp
                mei_list[n].vp_pow_mtp   = ref.vp_pow_mtp
                mei_list[n].reorg_energy = ref.reorg_energy
        
        # print("Created a list of MEIs with mei.id = ", ref.id, " parameters as reference.")            
        if ref != None :
            ref.id = -1
                
        return mei_list
    
class MoleculeEnvironmentInteraction:

    ### Defined with parameters given in cm-1, thus need to be transformed to fs-1.
    def drude_spd(self, gam):
        gam *= cs.CM2FS
        self.spdVal = gam*self.E.w_bt / (self.E.w_bt*self.E.w_bt + gam*gam)    
    def debye_spd(self, gam):
        gam *= cs.CM2FS
        self.spdVal = self.E.w_bt / (self.E.w_bt*self.E.w_bt + gam*gam)
    def whitenoise_spd(self):
        self.spdVal = np.ones(self.E.N_coh)
    def bo_spd(self, w0, gam):
        gam *= cs.CM2FS
        w0 *= cs.CM2FS
        self.spdVal = self.E.w_bt*gam / ( (w0*w0 - self.E.w_bt*self.E.w_bt)**2 + self.E.w_bt*self.E.w_bt*gam*gam) 
    def ohmic_spd(self, wc, n):
        wc *= cs.CM2FS
        self.spdVal = self.E.w_bt*np.power(self.E.w_bt/wc, n)*np.exp(-self.E.w_bt/wc)

    def setSpd(self, spdName, args):
        if spdName == 'whitenoise':
            self.whitenoise_spd()
        elif spdName == 'drude':
            self.drude_spd(args)
        elif spdName == 'debye':
            self.debye_spd(args)
        elif spdName == 'bo':
            self.bo_spd(*args)
        elif spdName == 'ohmic':
            self.ohmic_spd(*args)
        else:
            raise('Invalid spectral density name.')

    def setUnits(self):
        self.reorg_energy = np.array(self.reorg_energy) * cs.CM2FS

    def __init__(self, Mol, Env, id = 1):
        
        self.init_flag = 0
        self.id = id

        self.M = Mol
        self.E = Env

        self.el_phon_cpl = None
        self.spdName     = None
        self.spdArgs     = None # cm-1
        
        self.vp_mtp       = np.empty(self.M.N_dof,     dtype=np.int)
        self.vp_pow_mtp   = np.empty((3, self.M.N_el), dtype=np.float)
        self.reorg_energy = np.empty(self.M.N_el) # cm-1

    def initialize(self):
        
        if self.M.init_flag != 1 or self.E.init_flag != 1:
            raise("Molecule and it's environment parameters have to be initialized before interaction between them.") 

        self.setUnits()
        self.setSpd(self.spdName, self.spdArgs)

        self.bt_disp    = np.empty( self.M.N_el,           dtype=np.ndarray)
        self.vp_coef_LL = np.empty((self.M.N_el, self.M.N_dof), dtype=np.ndarray)
        self.vp_coef_LQ = np.empty((self.M.N_el, self.M.N_dof), dtype=np.ndarray)
        self.vp_coef_QL = np.empty((self.M.N_el, self.M.N_dof), dtype=np.ndarray)
        for n in range(self.M.N_el):
            self.bt_disp[n] = np.sqrt(2*self.reorg_energy[n]*self.spdVal[n]/np.sum(self.spdVal[n]/(self.E.w_bt)))/(self.E.w_bt)*self.el_phon_cpl
            for q in range(self.M.N_dof):
                self.vp_coef_LL[n][q] = self.vp_mtp[q]*self.bt_disp[n]*self.E.w_bt/cs.SQ2 * self.vp_pow_mtp[0][n]
                self.vp_coef_LQ[n][q] = self.vp_mtp[q]*self.bt_disp[n]*self.E.w_bt/cs.SQ2 * self.vp_pow_mtp[1][n]
                self.vp_coef_QL[n][q] = self.vp_mtp[q]*self.bt_disp[n]*self.E.w_bt/cs.SQ2 * self.vp_pow_mtp[2][n]
        self.init_flag = 1
