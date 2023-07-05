import numpy as np
import constants as cs
import functions as fn

class sD2_Wavefunction():
    
    init_flag = 0
    tp = 'sD2'

    def normalizeWavefunction(self, wf, M, E):
        mtr_sys = np.reshape(wf[:M.N_el*M.N_xpts], M.mtr_sys_shape, order='C')
        vec_mtp = wf[M.N_el*M.N_xpts:M.N_el*M.N_xpts+E.N_bt_mtp]
        vec_bt = fn.getSeperateBaths(wf, M, E, self.cP)
        S = fn.S_overlap(vec_bt, E)
        sys_scale = np.sum(np.abs(mtr_sys)**2)
        mtp_scale = np.einsum( "a,b,ab", np.conj(vec_mtp), vec_mtp, S )
        wf[:M.N_el*M.N_xpts] /= np.sqrt(sys_scale)
        wf[M.N_el*M.N_xpts:M.N_el*M.N_xpts+E.N_bt_mtp] /= np.sqrt(mtp_scale)
        return wf

    def setInitialConditions(self, intcond, M, E):
        initpop = np.array(M.init_pop)/np.sum(M.init_pop[0:M.N_el])
        for rlz in range(self.cP.N_rlz):
            intcond[rlz] = np.empty(0, dtype=np.complex)
        
            for n in range(M.N_el):
                intcond[rlz] = np.append(intcond[rlz], M.ndg_flat*np.sqrt(initpop[n]))
        
            if self.cP.sep_thermal == 'same-same':
                intcond[rlz] = np.append(intcond[rlz], np.ones(E.N_bt_mtp))
                thermal_bath_state = fn.generate_bath(E.w_bt, E.T)
                for b in range(E.N_bt_mtp):
                    intcond[rlz] = np.append(intcond[rlz], thermal_bath_state) 
        
            elif self.cP.sep_thermal == 'same-random':
                intcond[rlz] = np.append(intcond[rlz], np.ones(E.N_bt_mtp))
                for b in range(E.N_bt_mtp):
                    intcond[rlz] = np.append(intcond[rlz], fn.generate_bath(E.w_bt, E.T))
                 
            elif self.cP.sep_thermal == 'first-random':
                camp = np.zeros(E.N_bt_mtp)
                camp[0] = 1
                intcond[rlz] = np.append(intcond[rlz], camp)
                for b in range(E.N_bt_mtp):
                    intcond[rlz] = np.append(intcond[rlz], fn.generate_bath(E.w_bt, E.T)) 
            
            elif self.cP.sep_thermal == 'first-distributed':
                camp = np.zeros(E.N_bt_mtp)
                camp[0] = 1
                intcond[rlz] = np.append(intcond[rlz], camp)
                st1 = fn.generate_bath(E.w_bt, 0)
                for b in range(E.N_bt_mtp):
                    if b == 0:
                        intcond[rlz] = np.append(intcond[rlz], st1) 
                    else:
                        xoff = (-1)**(int(b/2))   * (int((b-1)/4)+1) * (self.cP.doff/cs.SQ2 + 1.j*0.0)
                        poff = (-1)**(int(b/1)-1) * (int((b-1)/4)+1) * (0.0                 + 1.j*self.cP.doff/cs.SQ2)
                        intcond[rlz] = np.append(intcond[rlz], st1 + xoff + poff)
                        # print(xoff, poff) 
            
            elif self.cP.sep_thermal == 'file':
                if self.cP.ens_name == None:
                    raise("Folder for thermal ensemble initial conditions have not been defined.")
                if E.N_bt_mtp == 1:
                    intcond[rlz] = np.append(intcond[rlz], np.ones(E.N_bt_mtp))
                    thermal_bath_state = fn.generate_bath(E.w_bt, E.T)
                    for b in range(E.N_bt_mtp):
                        intcond[rlz] = np.append(intcond[rlz], thermal_bath_state)
                elif E.N_bt_mtp > 1:
                    theta_inp  = np.load('./' + self.cP.ens_name + '/theta.npy')
                    lambda_inp = np.load('./' + self.cP.ens_name + '/lambda.npy')
                    rlzoff = rlz + self.cP.N_rlz*self.cP.roff
                    intcond[rlz] = np.append(intcond[rlz], theta_inp[rlzoff])
                    for b in range(E.N_bt_mtp):
                        intcond[rlz] = np.append(intcond[rlz], lambda_inp[rlzoff][b]) 
        
            intcond[rlz] = self.normalizeWavefunction(intcond[rlz], M, E)

    def __init__(self, Mol, Env, MEI, cP):

        refType = type(np.empty(0))    
        isMulti = [type(Mol)==refType, type(Env)==refType, type(MEI)==refType]

        if not np.any(isMulti):

            self.multi = False
            self.N_multi = 1
            
            self.M  = Mol
            self.E  = Env
            self.MEI = MEI
            self.cP = cP

            self.M.initialize()
            self.E.initialize()
            self.MEI.initialize()
            self.cP.initialize()

            if self.M.init_flag != 1 or self.E.init_flag != 1 or self.MEI.init_flag != 1:
                raise("Molecule, environment or parameters have not been initialized.") 
            
            if self.M.init_flag == -1 or self.E.init_flag == -1 or self.MEI.init_flag == -1:
                raise("List of either a molecule, environment or the interactions have been created, but not used.") 

            self.initial_cond = np.empty(self.cP.N_rlz, dtype=np.ndarray)
            self.setInitialConditions(self.initial_cond, self.M, self.E)

            self.init_flag = 1            
            print("sD2 wavefunction of a SINGLE system have been created.")

        elif np.all(isMulti):

            self.multi = True
            self.N_multi = len(Mol)

            self.M  = Mol
            self.E  = Env
            self.MEI = MEI
            self.cP = cP
            self.cP.initialize()
            
            self.initial_cond = np.empty((self.N_multi, self.cP.N_rlz), dtype=np.ndarray)
            
            for n in range(self.N_multi):
                
                self.M[n].initialize()
                self.E[n].initialize()
                self.MEI[n].initialize()

                if self.M[n].init_flag != 1 or self.E[n].init_flag != 1 or self.MEI[n].init_flag != 1:
                    raise("Molecule, environment or parameters have not been initialized.") 
                
                if self.M[n].init_flag == -1 or self.E[n].init_flag == -1 or self.MEI[n].init_flag == -1:
                    raise("List of either a molecule, environment or the interactions have been created, but not used.") 

                self.setInitialConditions(self.initial_cond[n], self.M[n], self.E[n])

            self.init_flag = 1
            print("sD2 wavefunction of MULTIPLE systems have been created.")

        else:
            raise("Number of system parts provided to wavefunction is invalid.")
            
