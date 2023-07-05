import time
import numpy as np
import multiprocessing as mp
from scipy.integrate import solve_ivp
from propagator import sD2_Propagator 
import functions as fn
import constants as cs

class Calculator():

    def calculateExpectationValues(self, M, E, MEI, cP, exc_st_prop):
        
        prob_distr              = np.zeros((M.N_el, M.N_el, cP.N_t), dtype=np.ndarray)
        diag_prob_distr_flat    = np.zeros((M.N_el, M.N_el, cP.N_t, *np.shape(M.ndg_flat)), dtype=np.ndarray)
        diag_prob_distr         = np.zeros((M.N_el, M.N_el, cP.N_t, *np.shape(M.ndg)), dtype=np.ndarray)
        mean_coord              = np.zeros((M.N_el, M.N_dof, cP.N_t), dtype=np.float)
        diag_populations        = np.zeros((M.N_el, cP.N_t), dtype=np.float)        
        populations             = np.zeros((M.N_el, cP.N_t), dtype=np.float)
        bath_density            = np.zeros((E.N_bt_mtp, E.N_bt_mtp, cP.N_t), dtype=np.complex)
        bath_energy             = np.zeros(cP.N_t, dtype=np.complex)
        bath_energy_distr       = np.zeros((cP.N_t, E.N_coh), dtype=np.complex)
        lin_pol                 = np.zeros(cP.N_t, dtype=np.complex)
        sep_energy              = np.zeros((7, cP.N_t), dtype=np.complex)
        KEav                    = np.zeros((len(cP.t_aver_axis), E.N_coh), dtype=np.complex)
        el_exc                  = np.empty(cP.N_t, dtype=np.ndarray)
        mtp_exc                 = np.empty(cP.N_t, dtype=np.ndarray)
        bt_exc                  = np.empty(cP.N_t, dtype=np.ndarray)
        bwp_x                   = np.zeros((E.N_coh,len(cP.t_out)), dtype=np.complex)
        bwp_p                   = np.zeros((E.N_coh,len(cP.t_out)), dtype=np.complex)
        bwp_2x                  = np.zeros((E.N_coh,len(cP.t_out)), dtype=np.complex)
        bwp_2p                  = np.zeros((E.N_coh,len(cP.t_out)), dtype=np.complex)

        for t in range(cP.N_t):
            
            mtr_sys = np.reshape(exc_st_prop[:M.N_el*M.N_xpts,t], M.mtr_sys_shape)
            vec_mtp = np.reshape(exc_st_prop[M.N_el*M.N_xpts:M.N_el*M.N_xpts+E.N_bt_mtp,t], E.N_bt_mtp, order='C')
            vec_bt = fn.getSeperateBaths(exc_st_prop[:,t], M, E, cP)
            S = fn.S_overlap(vec_bt, E)

            for n in range(M.N_el):
                for m in range(M.N_el):
                    for a in range(E.N_bt_mtp):
                        for b in range(E.N_bt_mtp):
                            prob_distr[n][m][t] += np.conj(mtr_sys[n]) * mtr_sys[m] * np.conj(vec_mtp[a]) * vec_mtp[b] * S[a][b]

            for a in range(E.N_bt_mtp):
                for b in range(E.N_bt_mtp):
                    bath_density[a][b][t] = np.conj(vec_mtp[a]) * vec_mtp[b] * S[a][b]

            for n in range(M.N_el):
                for a in range(E.N_bt_mtp):
                   for b in range(E.N_bt_mtp):                
                        bath_energy_distr[t] += np.sum(np.conj(mtr_sys[n]) * mtr_sys[n]) * np.conj(vec_mtp[a]) * vec_mtp[b] * S[a][b] * (E.w_bt/cs.CM2FS) * np.conj(vec_bt[a])*vec_bt[b]
                        populations[n][t] += np.real(np.sum(np.conj(mtr_sys[n]) * mtr_sys[n]) * np.conj(vec_mtp[a]) * vec_mtp[b] * S[a][b])
                        for q in range(M.N_dof):
                            mean_coord[n][q][t] += np.real(np.sum(M.cord_vec[q] * np.conj(mtr_sys[n]) * mtr_sys[n]) * np.conj(vec_mtp[a]) * vec_mtp[b] * S[a][b])

            bath_energy[t] = np.sum(bath_energy_distr[t])
            
            # Transforming populations to potential surface eigenbasis
            if M.con_int_mtp == 1:
                if M.N_el > 1:
                    h = np.zeros((cP.N_t, len(M.ndg_flat), M.N_el, M.N_el), dtype=np.complex)
                    diag_h = np.empty((cP.N_t, len(M.ndg_flat)), dtype=np.ndarray)
                    for i in range(len(M.ndg_flat)):
                        for n in range(M.N_el):
                            for m in range(M.N_el):
                                h[t][i][n][m] = prob_distr[n][m][t].ravel()[i]
                        diag_h[t][i] = np.dot(np.conj(M.diag_pot_vec[i]), np.dot(h[t][i], M.diag_pot_vec[i]))
                        # for n in range(M.N_el):
                            # diag_populations[n][t] += np.real(diag_h[t][i][n][n])
                    for n in range(M.N_el):
                        for m in range(M.N_el):
                            for i in range(len(M.ndg_flat)):
                                diag_prob_distr_flat[n][m][t][i] += diag_h[t][i][n][m]
                            diag_prob_distr[n][m][t] += np.reshape(diag_prob_distr_flat[n][m][t], np.shape(M.ndg))
                        diag_populations[n][t] += np.real(np.sum(diag_prob_distr[n][n][t]))
                else:
                    diag_populations = populations
            
            el_exc[t]  = np.reshape(exc_st_prop[:M.N_el*M.N_xpts,t], M.mtr_sys_shape)
            mtp_exc[t] = np.reshape(exc_st_prop[M.N_el*M.N_xpts:M.N_el*M.N_xpts+E.N_bt_mtp,t], E.N_bt_mtp, order='C')
            bt_exc[t]  = fn.getSeperateBaths(exc_st_prop[:,t], M, E, cP)
            S = fn.S_overlap(bt_exc[t], E)
            for a in range(E.N_bt_mtp):
                for b in range(E.N_bt_mtp):
                    bwp_x[:,t] +=       np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b] * (np.conj(bt_exc[t][a]) + bt_exc[t][b]) / cs.SQ2
                    bwp_p[:,t] += 1.j * np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b] * (np.conj(bt_exc[t][a]) - bt_exc[t][b]) / cs.SQ2
                    bwp_2x[:,t] +=   0.5 * np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b] * ( (np.conj(bt_exc[t][a]) + bt_exc[t][b])**2 + 1 )
                    bwp_2p[:,t] +=  -0.5 * np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b] * ( (np.conj(bt_exc[t][a]) - bt_exc[t][b])**2 - 1 )

            t_slot = int(t*cP.dt/cP.t_sample_rate)
            for a in range(E.N_bt_mtp):
                for b in range(E.N_bt_mtp):
                    KEav[t_slot] += -0.25 * (E.w_bt/cs.CM2FS) * np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b] * (np.conj(bt_exc[t][a]) - bt_exc[t][b])**2 / cP.t_sample_rate

            for n in range(M.N_el):
                n_sys_fft = np.fft.fftn(el_exc[t][n])
                for a in range(E.N_bt_mtp):
                    for b in range(E.N_bt_mtp):
                        # Electron energy
                        sep_energy[0][t] += np.sum( np.abs(el_exc[t][n])**2 ) * (M.en_el[n]/cs.CM2FS) * np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b]
                        # Electron-phonon interaction energy
                        sep_energy[4][t] += np.sum( np.abs(el_exc[t][n])**2 ) * np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b] * np.sum( 0.5*E.w_bt*MEI.bt_disp[n]**2 - E.w_bt*MEI.bt_disp[n]*(np.conj(bt_exc[t][a]) + bt_exc[t][b])/cs.SQ2 )/cs.CM2FS
                        # Bath energy (without zero-point)
                        sep_energy[5][t] += np.sum( np.abs(el_exc[t][n])**2 ) * np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b] * np.sum( E.w_bt * np.conj(bt_exc[t][a]) * bt_exc[t][b] )/cs.CM2FS
                        for q in range(M.N_dof):
                            # Kinetic energy
                            sep_energy[1][t] += -0.5*(M.w_dof[q][n]/cs.CM2FS)*np.sum( np.conj(el_exc[t][n])*np.fft.ifftn(M.k_vec_d2[q]*n_sys_fft) ) * np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b] 
                            # Potential energy
                            sep_energy[2][t] += np.sum( np.abs(el_exc[t][n])**2 *M.pot_surf[q][n] ) * np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b] /cs.CM2FS
                            # Vibrational-phonon interaction energy
                            sep_energy[3][t] += np.sum( np.abs(el_exc[t][n])**2 *M.cord_vec[q] )    * np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b] * np.sum( MEI.vp_coef_LL[n][q] *  (np.conj(bt_exc[t][a]) + bt_exc[t][b]) / cs.SQ2 ) /cs.CM2FS      
                            sep_energy[3][t] += np.sum( np.abs(el_exc[t][n])**2 *M.cord_vec[q] )    * np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b] * np.sum( MEI.vp_coef_LQ[n][q] * ((np.conj(bt_exc[t][a]) + bt_exc[t][b])**2 + 1 ) / 2 ) /cs.CM2FS      
                            sep_energy[3][t] += np.sum( np.abs(el_exc[t][n])**2 *M.cord_vec[q]**2 ) * np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b] * np.sum( MEI.vp_coef_QL[n][q] *  (np.conj(bt_exc[t][a]) + bt_exc[t][b]) / cs.SQ2 ) /cs.CM2FS     
                        # Conical intersection coupling energy      
                        for m in range(M.N_el):
                            if n != m:
                                sep_energy[6][t] += np.sum( np.conj(el_exc[t][n])*el_exc[t][m]*M.con_int[n][m] ) * np.conj(mtp_exc[t][a]) * mtp_exc[t][b] * S[a][b] /cs.CM2FS

        return [populations, mean_coord, bath_energy, prob_distr, sep_energy, bath_energy_distr, KEav, diag_populations, diag_prob_distr, bath_density, bwp_x, bwp_p, bwp_2x, bwp_2p]

    def start_ode_solver(self, initial_cond, M, E, MEI, cP, tp):
        
        print("I am here.")
        if tp == 'sD2':
            Propagator = sD2_Propagator
        else:
            raise("Undefined wavefunction type.")
        
        if E.thermalization == 0:
            
            prop_exc = solve_ivp(lambda t, y: Propagator(t, y, M, E, MEI, cP), [0, cP.tmax], initial_cond, method=cP.ode_solver, t_eval=cP.t_out, rtol=cP.rtol_precision, atol=cP.atol_precision)
            exc_st_prop = prop_exc.y
            
        elif E.thermalization == 1:
            
            if E.scat_interval*E.scat_rate > 1: 
                raise Exception("Thermalization probability per interval exceeds unity.")
            
            exc_st_prop = np.zeros((len(initial_cond), len(cP.t_out)), dtype=np.complex)
            t_cur_arr = np.array([0.0, E.scat_interval])
            t_cur_out = cP.t_out[np.where(np.logical_and(cP.t_out[int(cP.dt*t_cur_arr[0]):]>=t_cur_arr[0], cP.t_out[int(cP.dt*t_cur_arr[0]):]<=t_cur_arr[1]))]
            cur_state = initial_cond
            dt_start, dt_end = 0, len(t_cur_out)
            
            flag = 1
            while flag:   
                
                if dt_end*cP.dt >= cP.tmax: flag = 0
                prop_exc = solve_ivp(lambda t, y: Propagator(t, y, M, E, MEI, cP), t_cur_arr, cur_state, method=cP.ode_solver, t_eval=t_cur_out, rtol=cP.rtol_precision, atol=cP.atol_precision)
                exc_st_prop[:,dt_start:dt_end] = prop_exc.y[:,:]
                t_cur_arr = t_cur_arr + E.scat_interval 
                t_cur_out = cP.t_out[np.where(np.logical_and(cP.t_out>t_cur_arr[0], cP.t_out<=t_cur_arr[1]))]
                cur_state = prop_exc.y[:,-1]

                toss = np.random.binomial(1, E.scat_interval*E.scat_rate, size=E.N_coh)
                if (E.N_bt_mtp == 1):
                    for p in range(E.N_coh):
                        if toss[p] == 1:
                            new_imag = np.imag(fn.generate_bath(E.w_bt[p], E.T+E.T_offset))
                            cur_state[M.N_el*M.N_xpts + E.N_bt_mtp + p] = np.complex( cur_state[M.N_el*M.N_xpts + E.N_bt_mtp + p].real , new_imag )
                elif (E.N_bt_mtp > 1):
                    cur_state = fn.getThermalBathState(cur_state, toss, M, E, cP)

                dt_start = dt_end
                dt_end = dt_end+len(t_cur_out)
            
        else:
            raise("Invalid thermalization parameter value.")

        return self.calculateExpectationValues(M, E, MEI, cP, exc_st_prop)
  
    def __init__(self, wf):

        if wf.multi == False:
            
            # Things to be calculated
            self.prob_distr          = np.zeros((wf.M.N_el, wf.M.N_el, wf.cP.N_t), dtype=np.ndarray)
            self.mean_coord          = np.zeros((wf.M.N_el, wf.M.N_dof, wf.cP.N_t), dtype=np.float)
            self.populations         = np.zeros((wf.M.N_el, wf.cP.N_t), dtype=np.float)
            self.bath_density        = np.zeros((wf.E.N_bt_mtp, wf.E.N_bt_mtp, wf.cP.N_t), dtype=np.complex)
            self.bath_energy_distr   = np.zeros((wf.cP.N_t, wf.E.N_coh), dtype=np.complex)
            self.bath_energy         = np.zeros(wf.cP.N_t, dtype=np.complex)
            self.lin_pol             = np.zeros(wf.cP.N_t, dtype=np.complex)
            self.sep_energy          = np.zeros((7, wf.cP.N_t), dtype=np.complex)
            self.diag_prob_distr     = np.zeros((wf.M.N_el, wf.M.N_el, wf.cP.N_t, *np.shape(wf.M.ndg)), dtype=np.ndarray)
            self.diag_populations    = np.zeros((wf.M.N_el, wf.cP.N_t), dtype=np.float)  
            self.KEav                = np.zeros((len(wf.cP.t_aver_axis), wf.E.N_coh), dtype=np.complex)
            self.T_bath              = np.zeros((len(wf.cP.t_aver_axis), wf.E.N_coh), dtype=np.float)
            self.bwp_x               = np.zeros((wf.E.N_coh,len(wf.cP.t_out)), dtype=np.complex)
            self.bwp_p               = np.zeros((wf.E.N_coh,len(wf.cP.t_out)), dtype=np.complex)
            self.bwp_2x              = np.zeros((wf.E.N_coh,len(wf.cP.t_out)), dtype=np.complex)
            self.bwp_2p              = np.zeros((wf.E.N_coh,len(wf.cP.t_out)), dtype=np.complex)
            self.wf                  = wf

            pool = mp.Pool(processes=wf.cP.N_cores)
            print("Solving equations of a SINGLE system using ", str(wf.cP.N_cores), " threads.")
            start_time_all = time.time()
            N_calc_runs = int(np.ceil(wf.cP.N_rlz/wf.cP.N_cores))
            for r in range(N_calc_runs):
                start_time_run = time.time()
                if (r+1)*wf.cP.N_cores < wf.cP.N_rlz:
                    in_range = range(r*wf.cP.N_cores,(r+1)*wf.cP.N_cores)
                    print((r+1)*wf.cP.N_cores)
                else:
                    in_range = range(r*wf.cP.N_cores,wf.cP.N_rlz)
                    print(wf.cP.N_rlz)
                results = [pool.apply_async(self.start_ode_solver, args=(wf.initial_cond[rlz], wf.M, wf.E, wf.MEI, wf.cP, wf.tp)) for rlz in in_range]
                res = [p.get() for p in results]
                out_data_vector = np.sum(res, axis=0)
                np.add(self.populations,       out_data_vector[0],  out=self.populations)
                np.add(self.mean_coord,        out_data_vector[1],  out=self.mean_coord)
                np.add(self.bath_energy,       out_data_vector[2],  out=self.bath_energy)
                np.add(self.prob_distr,        out_data_vector[3],  out=self.prob_distr)
                np.add(self.sep_energy,        out_data_vector[4],  out=self.sep_energy)
                np.add(self.bath_energy_distr, out_data_vector[5],  out=self.bath_energy_distr)
                np.add(self.KEav,              out_data_vector[6],  out=self.KEav)
                np.add(self.diag_populations,  out_data_vector[7],  out=self.diag_populations)
                np.add(self.diag_prob_distr,   out_data_vector[8],  out=self.diag_prob_distr)
                np.add(self.bath_density,      out_data_vector[9],  out=self.bath_density)
                np.add(self.bwp_x,             out_data_vector[10], out=self.bwp_x)
                np.add(self.bwp_p,             out_data_vector[11], out=self.bwp_p)
                np.add(self.bwp_2x,            out_data_vector[12], out=self.bwp_2x)
                np.add(self.bwp_2p,            out_data_vector[13], out=self.bwp_2p)
                print(np.round((time.time() - start_time_run),2))

            print(np.round((time.time() - start_time_all),2))

            # Renormalizing 
            self.prob_distr /= wf.cP.N_rlz
            self.diag_prob_distr /= wf.cP.N_rlz
            self.bath_density /= wf.cP.N_rlz
            self.bath_energy /= wf.cP.N_rlz
            self.bath_energy_distr /= wf.cP.N_rlz
            self.populations /= wf.cP.N_rlz
            self.diag_populations /= wf.cP.N_rlz
            self.mean_coord /= wf.cP.N_rlz
            self.sep_energy /= wf.cP.N_rlz
            self.lin_pol /= wf.cP.N_rlz
            self.KEav /= wf.cP.N_rlz
            self.bwp_x /= wf.cP.N_rlz
            self.bwp_p /= wf.cP.N_rlz
            self.bwp_2x /= wf.cP.N_rlz
            self.bwp_2p /= wf.cP.N_rlz
            self.tot_energy = np.sum(self.sep_energy, axis=0)
            
            for i in range(len(wf.cP.t_aver_axis)):
                self.T_bath[i] = np.real( (wf.E.w_bt/cs.CM2FS) / ( np.log(1 + 2 * self.KEav[i] / (wf.E.w_bt/cs.CM2FS)) - np.log( 2 * self.KEav[i] / (wf.E.w_bt/cs.CM2FS) ) )  / cs.KB )
        
        elif wf.multi == True:  

            # # Things to be calculated
            self.prob_distr          = np.empty(wf.N_multi, dtype=np.ndarray)  
            self.mean_coord          = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.populations         = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.bath_density        = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.bath_energy_distr   = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.bath_energy         = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.lin_pol             = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.sep_energy          = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.diag_prob_distr     = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.diag_populations    = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.KEav                = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.T_bath              = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.bwp_x               = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.bwp_p               = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.bwp_2x              = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.bwp_2p              = np.empty(wf.N_multi, dtype=np.ndarray) 
            self.wf                  = wf
            
            for n in range(wf.N_multi):
                self.prob_distr        [n]  = np.zeros((wf.M[n].N_el, wf.M[n].N_el, wf.cP.N_t), dtype=np.ndarray)
                self.mean_coord        [n]  = np.zeros((wf.M[n].N_el, wf.M[n].N_dof, wf.cP.N_t), dtype=np.float)
                self.populations       [n]  = np.zeros((wf.M[n].N_el, wf.cP.N_t), dtype=np.float)
                self.bath_density      [n]  = np.zeros((wf.E[n].N_bt_mtp, wf.E[n].N_bt_mtp, wf.cP.N_t), dtype=np.complex)
                self.bath_energy_distr [n]  = np.zeros((wf.cP.N_t, wf.E[n].N_coh), dtype=np.complex)
                self.bath_energy       [n]  = np.zeros(wf.cP.N_t, dtype=np.complex)
                self.lin_pol           [n]  = np.zeros(wf.cP.N_t, dtype=np.complex)
                self.sep_energy        [n]  = np.zeros((7, wf.cP.N_t), dtype=np.complex)
                self.diag_prob_distr   [n]  = np.zeros((wf.M[n].N_el, wf.M[n].N_el, wf.cP.N_t, *np.shape(wf.M[n].ndg)), dtype=np.ndarray)
                self.diag_populations  [n]  = np.zeros((wf.M[n].N_el, wf.cP.N_t), dtype=np.float)  
                self.KEav              [n]  = np.zeros((len(wf.cP.t_aver_axis), wf.E[n].N_coh), dtype=np.complex)
                self.T_bath            [n]  = np.zeros((len(wf.cP.t_aver_axis), wf.E[n].N_coh), dtype=np.float)
                self.bwp_x             [n]  = np.zeros((wf.E[n].N_coh,len(wf.cP.t_out)), dtype=np.complex)
                self.bwp_p             [n]  = np.zeros((wf.E[n].N_coh,len(wf.cP.t_out)), dtype=np.complex)
                self.bwp_2x            [n]  = np.zeros((wf.E[n].N_coh,len(wf.cP.t_out)), dtype=np.complex)
                self.bwp_2p            [n]  = np.zeros((wf.E[n].N_coh,len(wf.cP.t_out)), dtype=np.complex)

            pool = mp.Pool(processes=wf.cP.N_cores)
            if wf.cP.N_cores == wf.N_multi:
                print("Solving equations of a MULTIPLE systems using", str(wf.cP.N_cores), "threads.")
            else:
                raise("Number of threads and systems being calculated must be equal.")
            start_time_all = time.time()

            for rlz in range(wf.cP.N_rlz):
                start_time_run = time.time()
                results = [pool.apply_async(self.start_ode_solver, args=(wf.initial_cond[n, rlz], wf.M[n], wf.E[n], wf.MEI[n], wf.cP, wf.tp)) for n in range(wf.N_multi)]
                res = [p.get() for p in results]
                for n in range(wf.N_multi):
                    np.add(self.populations[n],       res[n][0],  out=self.populations[n])
                    np.add(self.mean_coord[n],        res[n][1],  out=self.mean_coord[n])
                    np.add(self.bath_energy[n],       res[n][2],  out=self.bath_energy[n])
                    np.add(self.prob_distr[n],        res[n][3],  out=self.prob_distr[n])
                    np.add(self.sep_energy[n],        res[n][4],  out=self.sep_energy[n])
                    np.add(self.bath_energy_distr[n], res[n][5],  out=self.bath_energy_distr[n])
                    np.add(self.KEav[n],              res[n][6],  out=self.KEav[n])
                    np.add(self.diag_populations[n],  res[n][7],  out=self.diag_populations[n])
                    np.add(self.diag_prob_distr[n],   res[n][8],  out=self.diag_prob_distr[n])
                    np.add(self.bath_density[n],      res[n][9],  out=self.bath_density[n])
                    np.add(self.bwp_x[n],             res[n][10], out=self.bwp_x[n])
                    np.add(self.bwp_p[n],             res[n][11], out=self.bwp_p[n])
                    np.add(self.bwp_2x[n],            res[n][12], out=self.bwp_2x[n])
                    np.add(self.bwp_2p[n],            res[n][13], out=self.bwp_2p[n])

                print(np.round((time.time() - start_time_run),2))

            print(np.round((time.time() - start_time_all),2))

            # Renormalizing 
            self.tot_energy = np.empty(wf.N_multi, dtype=np.ndarray)
            for n in range(wf.N_multi):
                self.prob_distr         [n] /= wf.cP.N_rlz
                self.diag_prob_distr    [n] /= wf.cP.N_rlz
                self.bath_density       [n] /= wf.cP.N_rlz
                self.bath_energy        [n] /= wf.cP.N_rlz
                self.bath_energy_distr  [n] /= wf.cP.N_rlz
                self.populations        [n] /= wf.cP.N_rlz
                self.diag_populations   [n] /= wf.cP.N_rlz
                self.mean_coord         [n] /= wf.cP.N_rlz
                self.sep_energy         [n] /= wf.cP.N_rlz
                self.lin_pol            [n] /= wf.cP.N_rlz
                self.KEav               [n] /= wf.cP.N_rlz
                self.bwp_x              [n] /= wf.cP.N_rlz
                self.bwp_p              [n] /= wf.cP.N_rlz
                self.bwp_2x             [n] /= wf.cP.N_rlz
                self.bwp_2p             [n] /= wf.cP.N_rlz
                self.tot_energy         [n] = np.sum(self.sep_energy[n], axis=0)
            
            for n in range(wf.N_multi):
                for i in range(len(wf.cP.t_aver_axis)):
                    self.T_bath[n][i] = np.real( (wf.E[n].w_bt/cs.CM2FS) / ( np.log(1 + 2 * self.KEav[n][i] / (wf.E[n].w_bt/cs.CM2FS)) - np.log( 2 * self.KEav[n][i] / (wf.E[n].w_bt/cs.CM2FS) ) )  / cs.KB )




            
