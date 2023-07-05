import numpy as np
import scipy as sc
import constants as cs
import copy

def createCopies(ref, N):
    copies = np.empty(N, dtype=np.object)
    for n in range(N):
        copies[n] = copy.deepcopy(ref)
        # copies[n] = copy.copy(ref)
        copies[n].id = n+ref.id
    ref.id = -1
    return copies

def getSeperateBaths(vec, M, E, cP):
    vecbt = np.zeros((E.N_bt_mtp, E.N_coh), dtype=np.complex)
    st, nxt = (M.N_el*M.N_xpts) + E.N_bt_mtp, 0
    for i in range(E.N_bt_mtp):
        nxt = st + E.N_coh
        vecbt[i] = vec[st:nxt]
        st = nxt
    return vecbt

def S_overlap(bt, E):
    S = np.zeros((E.N_bt_mtp, E.N_bt_mtp), dtype=np.complex)
    for a in range(E.N_bt_mtp):
        for b in range(E.N_bt_mtp):
            S[a][b] = np.exp( np.sum( np.conj( bt[a] ) * bt[b] - 0.5*( np.abs( bt[a] )**2 + np.abs( bt[b] )**2 ) ) )
    return S

def generate_bath(w, T):
    if T == 0:
        return np.zeros(np.shape(w), dtype=np.complex)
    else:
        avgn = 1.0/(np.exp((w/cs.CM2FS)/(cs.KB*T))-1.0)
        sigma = np.sqrt(avgn/2.0)
        amp = np.random.normal(0.0, sigma)
        ph = np.exp(1.j*np.random.uniform(0.0, 2*np.pi, size=np.shape(amp)))
        return np.sqrt(2)*amp*ph

def generate_bath_displacemenet(w, T):
    if T == 0:
        return np.zeros(np.shape(w), dtype=np.complex)
    else:
        avgn = 1.0/(np.exp((w/cs.CM2FS)/(cs.KB*T))-1.0)
        sigma = np.sqrt(avgn/2.0)
        amp = np.random.normal(0.0, sigma)
        return np.sqrt(2)*amp

def minFun(par, theta, lmb, N_toss, toss, sample_state_kin, old_kin, E, cP):

    new_dis = par[0:N_toss]
    new_scal_y = par[N_toss:2*N_toss]

    toss_ind = np.argwhere(toss == 1)
    ind_toss = toss_ind[:,0]

    new_lmb = np.copy(lmb)
    new_D = np.zeros((E.N_bt_mtp, N_toss), dtype=np.complex)
    for j in range(E.N_bt_mtp):
        new_D[j] = lmb[j][ind_toss] + 1.j * new_dis

    new_C_x = np.sum(new_D.real, axis=0) / E.N_bt_mtp
    new_C_y = np.sum(new_D.imag, axis=0) / E.N_bt_mtp

    for j in range(E.N_bt_mtp):
        new_lmb[j][ind_toss] =        1 * ( new_D[j].real - new_C_x ) + new_C_x 
        new_lmb[j][ind_toss] += 1.j * new_scal_y * ( new_D[j].imag - new_C_y ) + 1.j * new_C_y 

    new_S = S_overlap(new_lmb, E)
    new_kin_dif = np.zeros( (E.N_bt_mtp, E.N_bt_mtp, E.N_coh), dtype=np.complex )
    for a in range(E.N_bt_mtp):
        for b in range(E.N_bt_mtp):
            new_kin_dif[a][b] = (np.conj(new_lmb[a]) - new_lmb[b])**2
    new_kin = -0.25 * (E.w_bt/cs.CM2FS) * np.real( np.einsum("i,j,ij,ijp->p", np.conj(theta), theta, new_S, new_kin_dif) )

    new_stateKin = old_kin
    new_stateKin[ind_toss] = sample_state_kin

    return np.sum( ((new_stateKin - new_kin)/new_stateKin)**2 )


def getNewKineticState(theta, lmb, N_toss, toss, old_kin, xmean, E, cP):
        toss_ind = np.argwhere(toss == 1)
        ind_toss = toss_ind[:,0]
        
        sample_state = generate_bath(E.w_bt[ind_toss], (E.T+E.T_offset))
        sample_state_kin = (E.w_bt[ind_toss]/cs.CM2FS) * np.imag(sample_state)**2 

        # Initial displacements and angle values
        dis  = np.zeros(N_toss, dtype=np.float)
        scal_y = np.ones(N_toss, dtype=np.float)

        x0  = np.hstack((dis, scal_y))
        sol = sc.optimize.minimize(minFun, x0, method="Powell", args=(theta, lmb, N_toss, toss, sample_state_kin, old_kin, E, cP))

        new_dis     = sol.x[0:N_toss]
        new_scal_y  = sol.x[N_toss:2*N_toss]

        new_lmb = np.copy(lmb)
        new_D = np.zeros((E.N_bt_mtp, N_toss), dtype=np.complex)
        for j in range(E.N_bt_mtp):
            new_D[j] = lmb[j][ind_toss] + 1.j * new_dis

        new_C_x = np.sum(new_D.real, axis=0) / E.N_bt_mtp
        new_C_y = np.sum(new_D.imag, axis=0) / E.N_bt_mtp

        for j in range(E.N_bt_mtp):
            new_lmb[j][ind_toss] =        1 * ( new_D[j].real - new_C_x ) + new_C_x 
            new_lmb[j][ind_toss] += 1.j * new_scal_y * ( new_D[j].imag - new_C_y ) + 1.j * new_C_y 

        new_S = S_overlap(new_lmb, E)
        new_kin_dif = np.zeros( (E.N_bt_mtp, E.N_bt_mtp, E.N_coh), dtype=np.complex )
        for a in range(E.N_bt_mtp):
            for b in range(E.N_bt_mtp):
                new_kin_dif[a][b] = (np.conj(new_lmb[a]) - new_lmb[b])**2
        new_kin = -0.25 * (E.w_bt/cs.CM2FS) * np.real( np.einsum("i,j,ij,ijp->p", np.conj(theta), theta, new_S, new_kin_dif) )

        return new_kin, new_lmb, sample_state_kin


def getThermalBathState(cur_state, toss, M, E, cP):

    N_toss = np.sum(toss)
    if N_toss != 0:
        theta = cur_state[M.N_el*M.N_xpts:M.N_el*M.N_xpts+E.N_bt_mtp]
        lmb = getSeperateBaths(cur_state, M, E, cP)

        toss_ind = np.argwhere(toss == 1)
        ind_toss = toss_ind[:,0]

        old_S = S_overlap(lmb, E)
        old_kin_dif = np.zeros( (E.N_bt_mtp, E.N_bt_mtp, E.N_coh), dtype=np.complex )
        for a in range(E.N_bt_mtp):
            for b in range(E.N_bt_mtp):
                old_kin_dif[a][b] = (np.conj(lmb[a]) - lmb[b])**2
        old_kin = -0.25 * (E.w_bt/cs.CM2FS) * np.real( np.einsum("i,j,ij,ijp->p", np.conj(theta), theta, old_S, old_kin_dif) )
        
        xmean = np.zeros(E.N_coh, dtype=np.complex)
        for i in range(E.N_bt_mtp):
            for j in range(E.N_bt_mtp):
                xmean += np.conj(theta[i]) * theta[j] * old_S[i][j] * (np.conj(lmb[i]) + lmb[j]) / cs.SQ2

        nbad = 0
        while True:
            new_kin, new_lmb, sample_state_kin = getNewKineticState(theta, lmb, N_toss, toss, old_kin, xmean.real, E, cP)

            new_stateKin = old_kin
            new_stateKin[ind_toss] = sample_state_kin

            kdif = np.abs((new_stateKin - new_kin)/new_stateKin)

            if (kdif < 0.01).all():
                out_state = np.copy(cur_state)
                out_state[M.N_el*M.N_xpts+E.N_bt_mtp:] = new_lmb.ravel()
                return out_state
            else:
                nbad += 1
            if nbad == 100:
                return cur_state
    else:
        return cur_state
