import numpy as np
from scipy.stats import multivariate_normal

import constants as cs
from pes import PotentialEnergySurface

class Molecule:

    def setInitialWavepacket(self):
        mv_normal = multivariate_normal(mean=np.zeros(self.N_dof), cov=np.ones(self.N_dof))
        self.ndg = mv_normal.pdf(self.pos)
        self.ndg /= np.sum(np.abs(self.ndg)**2)**0.5
        self.ndg_flat = self.ndg.ravel(order='C')

    def setTensorShapes(self):
        self.mtr_sys_shape =          (self.N_el, *np.shape(self.ndg)) 
        self.mtr_sys_flat_shape =     (self.N_el, *np.shape(self.ndg_flat)) 
        self.mtr_gr_sys_shape =       (1, *np.shape(self.ndg))
        self.mtr_gr_sys_flat_shape =  (1, *np.shape(self.ndg_flat))

    def setPotentialSurfaces(self):
        PES = PotentialEnergySurface()
        self.pot_surf = np.empty((self.N_dof, self.N_el, *np.shape(self.ndg)))
        self.psf_energy = np.zeros(self.mtr_sys_shape, dtype=np.float)
        self.pot_surf_ground = np.empty((self.N_dof, 1, *np.shape(self.ndg)))

        for n in range(self.N_el):
            self.psf_energy[n] += self.en_el[n]
            for q in range(self.N_dof):
                if self.PEStypes[q][n] == 'harm':
                    self.pot_surf[q][n] = PES.harmonic(self.cord_vec[q], self.w_dof[q][n], self.disp_dof[q][n])
                elif self.PEStypes[q][n] == 'morse':
                    self.pot_surf[q][n] = PES.morse(self.cord_vec[q], self.w_dof_gr[q][n], self.disp_dof[q][n], *self.PESparams[q][n])
                else:
                    raise('Invalid potential surface function name.')
                self.psf_energy[n] += self.pot_surf[q][n]

        for q in range(self.N_dof):
            self.pot_surf_ground[q][0] = PES.harmonic(self.cord_vec[q], self.w_dof_gr[q][0], 0)

    def diagonalizePotentialSurfaces(self):
        if self.N_el > 1:
            self.diag_pot      = np.zeros((self.N_el, len(self.ndg_flat)), dtype=np.ndarray)
            self.diag_pot_vec  = np.empty(len(self.ndg_flat), dtype=np.ndarray)
            for i in range(len(self.ndg_flat)):
                h = np.zeros((self.N_el, self.N_el), dtype=np.float)
                for n in range(self.N_el):
                    h[n][n] = (self.psf_energy[n]).ravel()[i]
                    for m in range(self.N_el):
                        if n != m:
                            h[n][m] = (self.con_int[n][m]).ravel()[i]   
                val, vec = np.linalg.eigh(h)
                for n in range(self.N_el):
                    self.diag_pot[n][i] = val[n]
                    self.diag_pot_vec[i] = vec

    def setAdiabaticCouplings(self):
        self.con_int     = np.empty((self.N_el, self.N_el), dtype=np.ndarray)
        self.intersects  = np.zeros((self.N_el, self.N_el))
        for n in range(self.N_el):
            for m in range(self.N_el):
                self.con_int[n][m] = np.zeros(self.intersects.shape, dtype=np.float)
                if n!=m and n>m:
                    if self.adia_cpl_type == 'g':   
                        ### Gaussian coupling
                        gs = multivariate_normal(mean=self.adia_cpl_disp, cov=np.ones(self.N_dof))
                        gsform = gs.pdf(self.pos)
                        gsint = np.sum(np.trapz(gsform, self.cord_vec))
                        gsform_norm = gsform * self.con_int_str / gsint
                        self.con_int[n][m] = gsform_norm * self.con_int_mtp
                        self.con_int[m][n] = gsform_norm * self.con_int_mtp
                    elif self.adia_cpl_type == 'v':
                        ### Vibronic coupling 
                        disp_cord_vec = self.cord_vec[0] - self.adia_cpl_disp
                        vibro_norm = disp_cord_vec * self.con_int_str
                        self.con_int[n][m] = vibro_norm * self.con_int_mtp
                        self.con_int[m][n] = vibro_norm * self.con_int_mtp
                    else:
                        raise('Invalid definition of adiabatic coupling function.')

    def setUnits(self):
        self.w_dof    = np.array(self.w_dof) * cs.CM2FS 
        self.w_dof_gr = np.array(self.w_dof_gr) * cs.CM2FS 
        self.en_el = np.array(self.en_el) * cs.CM2FS 
        self.con_int_str = np.array(self.con_int_str) * cs.CM2FS 
    
    def setTensors(self):

        if self.con_int_str == 0:
            self.con_int_mtp = 0
        else:
            self.con_int_mtp = 1

        for q in range(self.N_dof):
            self.xmax[q] += self.dx[q]
        for q in range(self.N_dof):
            self.x[q] = np.arange(self.xmin[q],self.xmax[q],self.dx[q])
            self.N_x[q] = len(self.x[q])
            self.k_space_d2[q] = -4*np.pi**2 * np.fft.fftfreq(self.N_x[q], self.dx[q])**2
        self.N_xpts = np.prod(self.N_x)

        if self.N_dof == 1:
            self.cord_vec = np.ndarray((self.N_dof,self.N_x[0]), dtype=np.float)
            self.k_vec_d2 = np.ndarray((self.N_dof,self.N_x[0]), dtype=np.float)
            self.x1 = np.mgrid[self.xmin[0]:self.xmax[0]:self.dx[0]]
            self.cord_vec[self.N_dof-1] = self.x1
            self.pos = self.x1
            self.k_vec_d2[self.N_dof-1] = self.k_space_d2[0]
        elif self.N_dof == 2:
            self.x1, self.x2 = np.mgrid[self.xmin[0]:self.xmax[0]:self.dx[0], self.xmin[1]:self.xmax[1]:self.dx[1]]
            self.pos = np.stack((self.x1,self.x2), axis=self.N_dof)               
            self.cord_vec = np.asarray((self.x1, self.x2))
            self.x1k_space, self.x2k_space = np.meshgrid(self.k_space_d2[0], self.k_space_d2[1], indexing='ij')
            self.k_vec_d2 = np.asarray((self.x1k_space, self.x2k_space))
        elif self.N_dof == 3:
            self.x1, self.x2, self.x3 = np.mgrid[self.xmin[0]:self.xmax[0]:self.dx[0], self.xmin[1]:self.xmax[1]:self.dx[1], self.xmin[2]:self.xmax[2]:self.dx[2]]
            self.pos = np.stack((self.x1,self.x2,self.x3), axis=self.N_dof)
            self.cord_vec = np.asarray((self.x1, self.x2, self.x3))
            self.x1k_space, self.x2k_space, self.x3k_space = np.meshgrid(self.k_space_d2[0], self.k_space_d2[1], self.k_space_d2[2], indexing='ij')
            self.k_vec_d2 = np.asarray((self.x1k_space, self.x2k_space, self.x3k_space))
        elif self.N_dof == 4:
            self.x1, self.x2, self.x3, self.x4 = np.mgrid[self.xmin[0]:self.xmax[0]:self.dx[0], self.xmin[1]:self.xmax[1]:self.dx[1], self.xmin[2]:self.xmax[2]:self.dx[2], self.xmin[3]:self.xmax[3]:self.dx[3]]
            self.pos = np.stack((self.x1,self.x2,self.x3,self.x4), axis=self.N_dof)
            self.cord_vec = np.asarray((self.x1, self.x2, self.x3, self.x4))
            self.x1k_space, self.x2k_space, self.x3k_space, self.x4k_space = np.meshgrid(self.k_space_d2[0], self.k_space_d2[1], self.k_space_d2[2], self.k_space_d2[3], indexing='ij')
            self.k_vec_d2 = np.asarray((self.x1k_space, self.x2k_space, self.x3k_space, self.x4k_space))
        

    def __init__(self, Nel = 1, Ndof = 1, id = 1):

        self.init_flag = 0
        self.id = id

        self.N_el        = Nel
        self.N_dof       = Ndof

        self.x           = np.empty(self.N_dof, dtype=np.ndarray)
        self.N_x         = np.zeros(self.N_dof, dtype=np.int64)
        self.k_space_d2  = np.empty(self.N_dof, dtype=np.ndarray)
        
        self.PEStypes      = np.empty((self.N_dof, self.N_el), dtype=np.object)
        self.PESparams     = np.empty((self.N_dof, self.N_el), dtype=np.ndarray)

        # These have to be set every time
        self.en_el       = np.empty(self.N_el, dtype=np.float)
        self.init_pop    = np.empty(self.N_el, dtype=np.float)
        self.w_dof_gr    = np.empty((self.N_dof, 1), dtype=np.float) # cm-1
        self.w_dof       = np.empty((self.N_dof, self.N_el), dtype=np.float) # cm-1
        self.disp_dof    = np.empty((self.N_dof, self.N_el), dtype=np.float)
        self.xmin        = np.empty(self.N_dof, dtype=np.float)
        self.xmax        = np.empty(self.N_dof, dtype=np.float)
        self.dx          = np.empty(self.N_dof, dtype=np.float)
        
        self.con_int_mtp   = None
        self.con_int_str   = None
        self.adia_cpl_disp = None
        self.adia_cpl_type = None

    def initialize(self):
        self.setTensors()
        self.setUnits()
        self.setInitialWavepacket()
        self.setTensorShapes()
        self.setAdiabaticCouplings()
        self.setPotentialSurfaces()
        self.diagonalizePotentialSurfaces()
        self.init_flag = 1