import numpy as np
import scipy as sc
import constants as cs
import functions as fn
from numba import jit

def getCoefficientMatrix(ky_chi, ky_th, ky_th_cj, ky_lmb, ky_lmb_cj,
                         t_chi,  t_th,  t_th_cj,  t_lmb,  t_lmb_cj,
                         mj_chi, mj_th, mj_th_cj, mj_lmb, mj_lmb_cj,
                         sys_vec_sz,mtp_vec_sz,bt_vec_sz ):

    # Creating blocks of constant indeces
    ky_chi_rf =    np.reshape( ky_chi,              (sys_vec_sz, sys_vec_sz))
    ky_th_rf =     np.reshape( ky_th,               (sys_vec_sz, mtp_vec_sz))
    ky_th_cj_rf =  np.reshape( ky_th_cj,            (sys_vec_sz, mtp_vec_sz))
    ky_lmb_rf =    np.reshape( ky_lmb,              (sys_vec_sz, bt_vec_sz))
    ky_lmb_cj_rf = np.reshape( ky_lmb_cj,           (sys_vec_sz, bt_vec_sz))

    t_chi_rf =    np.reshape( t_chi,                (mtp_vec_sz, sys_vec_sz))
    t_th_rf =     np.reshape( t_th,                 (mtp_vec_sz, mtp_vec_sz))
    t_th_cj_rf =  np.reshape( t_th_cj,              (mtp_vec_sz, mtp_vec_sz))
    t_lmb_rf =    np.reshape( t_lmb,                (mtp_vec_sz, bt_vec_sz))
    t_lmb_cj_rf = np.reshape( t_lmb_cj,             (mtp_vec_sz, bt_vec_sz))

    mj_chi_rf =    np.reshape( mj_chi,              (bt_vec_sz, sys_vec_sz))
    mj_th_rf =     np.reshape( mj_th,               (bt_vec_sz, mtp_vec_sz))
    mj_th_cj_rf =  np.reshape( mj_th_cj,            (bt_vec_sz, mtp_vec_sz))
    mj_lmb_rf =    np.reshape( mj_lmb,              (bt_vec_sz, bt_vec_sz))
    mj_lmb_cj_rf = np.reshape( mj_lmb_cj,           (bt_vec_sz, bt_vec_sz))

    r_chi_rf =    np.copy(t_chi_rf)
    r_th_rf =     np.conj(np.copy( t_th_cj_rf ))
    r_th_cj_rf =  np.conj(np.copy( t_th_rf ))
    r_lmb_rf =    np.conj(np.copy( t_lmb_cj_rf ))
    r_lmb_cj_rf = np.conj(np.copy( t_lmb_rf ))
    
    nh_chi_rf =    np.copy(mj_chi)
    nh_th_rf =     np.conj(np.copy(mj_th_cj_rf))
    nh_th_cj_rf =  np.conj(np.copy(mj_th_rf))
    nh_lmb_rf =    np.conj(np.copy(mj_lmb_cj_rf))
    nh_lmb_cj_rf = np.conj(np.copy(mj_lmb_rf))

    ky_coef_mtr = np.hstack( (ky_chi_rf, ky_th_rf, ky_lmb_rf,  ky_th_cj_rf, ky_lmb_cj_rf) )
    t_coef_mtr =  np.hstack( (t_chi_rf,  t_th_rf,  t_lmb_rf,   t_th_cj_rf,  t_lmb_cj_rf) )
    mj_coef_mtr = np.hstack( (mj_chi_rf, mj_th_rf, mj_lmb_rf,  mj_th_cj_rf, mj_lmb_cj_rf) )
    r_coef_mtr =  np.hstack( (r_chi_rf,  r_th_rf,  r_lmb_rf,   r_th_cj_rf,  r_lmb_cj_rf) )
    nh_coef_mtr = np.hstack( (nh_chi_rf, nh_th_rf, nh_lmb_rf,  nh_th_cj_rf, nh_lmb_cj_rf) )
    coef_mtr =    np.vstack( (ky_coef_mtr, t_coef_mtr, mj_coef_mtr, r_coef_mtr, nh_coef_mtr) )

    return coef_mtr

def getCoefAndRhs(t, vec, M, E, MEI, cP):

    chi_unit   = np.ones(M.mtr_sys_flat_shape, dtype=np.complex)
    theta_unit = np.ones(E.N_bt_mtp, dtype=np.complex)
    lmb_unit   = np.ones((E.N_bt_mtp, E.N_coh), dtype=np.complex)
    dlt_n      = np.eye(M.N_el,M.N_el)
    dlt_x      = np.eye(M.N_xpts,M.N_xpts)
    dlt_t      = np.eye(E.N_bt_mtp,E.N_bt_mtp)
    dlt_j      = np.eye(E.N_coh,E.N_coh)

    sys_vec_sz  = M.N_el*M.N_xpts
    mtp_vec_sz  = E.N_bt_mtp
    bt_vec_sz   = E.N_bt_mtp*E.N_coh
    rhs_sys_out = np.zeros(M.mtr_sys_shape, dtype=np.complex)
    rhs_mtp_out = np.zeros(mtp_vec_sz, dtype=np.complex)
    rhs_bt_out  =  np.zeros((E.N_bt_mtp, E.N_coh), dtype=np.complex)

    mtr_sys      = np.reshape(vec[:M.N_el*M.N_xpts], M.mtr_sys_shape, order='C')
    mtr_sys_flat = np.reshape(vec[:M.N_el*M.N_xpts], M.mtr_sys_flat_shape, order='C') 
    mtp_vec      = np.reshape(vec[M.N_el*M.N_xpts:(M.N_el*M.N_xpts)+E.N_bt_mtp], mtp_vec_sz, order='C') 
    vec_bt       = fn.getSeperateBaths(vec, M, E, cP)
    cj_mtp_vec   = mtp_vec.conj()
    S            = fn.S_overlap(vec_bt, E)

    pop = np.zeros(M.N_el, dtype=np.complex)
    for n in range(M.N_el):
        pop[n] = np.sum(np.abs(mtr_sys[n])**2)

    Xl = np.zeros((M.N_el, M.N_dof), dtype=np.complex)
    for n in range(M.N_el):
        for q in range(M.N_dof):
            Xl[n][q] = np.sum(np.abs(mtr_sys[n])**2 * M.cord_vec[q])

    Xq = np.zeros((M.N_el, M.N_dof), dtype=np.complex)
    for n in range(M.N_el):
        for q in range(M.N_dof):
            Xq[n][q] = np.sum(np.abs(mtr_sys[n])**2 * M.cord_vec[q]**2)

    ### Creating auxilary tensors
    T1 =       np.zeros( ( E.N_bt_mtp, E.N_bt_mtp, E.N_coh ), dtype=np.complex)
    T2 =       np.zeros( ( E.N_bt_mtp, E.N_bt_mtp, E.N_coh ), dtype=np.complex)
    T3 =       np.zeros( ( E.N_bt_mtp, E.N_bt_mtp, E.N_coh ), dtype=np.complex)
    Ke =       np.zeros( ( M.N_el, M.N_dof, *np.shape(M.ndg) ), dtype=np.complex) 
    PTe =      np.zeros( ( M.N_el, M.N_dof, *np.shape(M.ndg) ), dtype=np.complex) 
    ADe =      np.zeros( ( M.N_el, M.N_el, *np.shape(M.ndg) ), dtype=np.complex) 
    g1 =       np.zeros( ( E.N_bt_mtp, E.N_bt_mtp, M.N_el, E.N_coh ), dtype=np.complex)
    g2_l =     np.zeros( ( E.N_bt_mtp, E.N_bt_mtp, M.N_el, M.N_dof, E.N_coh ), dtype=np.complex)
    g2_q =     np.zeros( ( E.N_bt_mtp, E.N_bt_mtp, M.N_el, M.N_dof, E.N_coh ), dtype=np.complex)
    for a in range(E.N_bt_mtp):
        for b in range(E.N_bt_mtp):
            T1[a][b] = ( np.conj(vec_bt[a]) + vec_bt[b] ) 
            T2[a][b] = ( T1[a][b]**2 + 1 ) 
            T3[a][b] = ( np.conj(vec_bt[a]) * vec_bt[b] + 0.5 )
            for n in range(M.N_el):
                g1[a][b][n] = T3[a][b] - MEI.bt_disp[n]*T1[a][b]/cs.SQ2
                for q in range(M.N_dof):
                    g2_l[a][b][n][q] = MEI.vp_coef_LL[n][q]*T1[a][b]/cs.SQ2 + MEI.vp_coef_LQ[n][q]*T2[a][b]/2
                    g2_q[a][b][n][q] = MEI.vp_coef_QL[n][q]*T1[a][b]/cs.SQ2

    ### Creating kinetic, potential energy, adiabatic coupling tensors
    for n in range(M.N_el):
        n_sys_fft = np.fft.fftn(mtr_sys[n])
        for q in range(M.N_dof):
            Ke[n][q]  = 0.5 * M.w_dof[q][n] * np.fft.ifftn(M.k_vec_d2[q]*n_sys_fft)
            PTe[n][q] = M.pot_surf[q][n] * mtr_sys[n]
        for m in range(M.N_el):
            if n != m:
                ADe[n][m] = M.con_int[n][m] * mtr_sys[m]
    
    lmbt_cj = np.zeros((E.N_bt_mtp, E.N_bt_mtp, E.N_coh), dtype=np.complex) 
    for a in range(E.N_bt_mtp):
        for b in range(E.N_bt_mtp):
            lmbt_cj[a][b] = np.conj(vec_bt[a] - 0.5 * vec_bt[b])

    ft =       np.einsum( "a,b,ab->ab",    cj_mtp_vec, mtp_vec, S )
    h1 =       np.einsum( "vb,vbnp->np",   ft, g1 )
    h2_l =     np.einsum( "vb,vbnqp->nqp", ft, g2_l )
    h2_q =     np.einsum( "vb,vbnqp->nqp", ft, g2_q )

    H_n           = np.einsum( "jp,ijp->ijp", lmb_unit, lmbt_cj )
    H_c           = np.einsum( "jp,jp,i->ijp", -0.5*lmb_unit, vec_bt, np.ones(E.N_bt_mtp) )
    H_n_cj        = np.conj(H_c)
    H_c_cj        = np.conj(H_n)

    # For constant indeces {k,y}:
    ky_chi        = np.einsum( "ky,kM,yX->kyMX",  chi_unit, dlt_n, dlt_x )
    ky_th         = np.einsum( "ky,a,b,ab->kyb",  mtr_sys_flat, cj_mtp_vec, theta_unit, S )
    ky_th_cj      = np.zeros( (sys_vec_sz, mtp_vec_sz), dtype=np.complex )
    ky_lmb        = np.einsum( "ky,ab,abp->kybp", mtr_sys_flat, ft, H_n )
    ky_lmb_cj     = np.einsum( "ky,ab,abp->kybp", mtr_sys_flat, ft, H_c )

    # For constant index t:
    t_chi         =     np.zeros( (mtp_vec_sz, sys_vec_sz), dtype=np.complex )
    t_th          =     np.einsum( "a,ta->ta",         theta_unit, S )
    t_th          += -1*np.einsum( "a,i,j,ij,ta->tj",  mtp_vec, cj_mtp_vec, theta_unit, S, S )
    t_th_cj       =     np.zeros( (mtp_vec_sz, mtp_vec_sz), dtype=np.complex )
    t_lmb         =     np.einsum( "tap,a,ta->tap",    H_n, mtp_vec, S )
    t_lmb         += -1*np.einsum( "ij,ijp,a,ta->tjp", ft, H_n, mtp_vec, S )
    t_lmb_cj      =     np.einsum( "tap,a,ta->tap",    H_c, mtp_vec, S )
    t_lmb_cj      += -1*np.einsum( "ij,ijp,a,ta->tjp", ft, H_c, mtp_vec, S )

    # For constant indecex {m,j}:
    mj_chi        =     np.zeros( (bt_vec_sz, sys_vec_sz), dtype=np.complex )
    
    mj_th        =   0.5*np.einsum( "m,a,ma,aj->mja", cj_mtp_vec, theta_unit, S, vec_bt )
    mj_th       += -0.25*np.einsum( "m,a,ma,mj->mja", cj_mtp_vec, theta_unit, S, vec_bt )
    mj_th       += +0.25*np.einsum( "a,m,am,mj,mN->mjN", cj_mtp_vec, theta_unit, S, vec_bt, dlt_t )
    
    mj_th_cj     =   0.5*np.einsum( "a,m,ma,aj,mN->mjN", mtp_vec, theta_unit, S, vec_bt, dlt_t )
    mj_th_cj    += -0.25*np.einsum( "a,m,ma,mj,mN->mjN", mtp_vec, theta_unit, S, vec_bt, dlt_t )
    mj_th_cj    += +0.25*np.einsum( "m,a,am,mj->mja", mtp_vec, theta_unit, S, vec_bt )
    
    mj_lmb       =       np.einsum( "ma,aj,jL->mjaL",        ft, lmb_unit, dlt_j )
    mj_lmb      +=  +0.5*np.einsum( "map,ma,aj->mjap",       H_n, ft, vec_bt )
    mj_lmb      +=  +0.5*np.einsum( "amp,ma,aj,mN->mjNp",    H_n_cj, ft, vec_bt, dlt_t )
    mj_lmb_cj    =  +0.5*np.einsum( "map,ma,aj->mjap",       H_c, ft, vec_bt )
    mj_lmb_cj   +=  +0.5*np.einsum( "amp,ma,aj,mN->mjNp",    H_c_cj, ft, vec_bt, dlt_t )    
    mj_lmb      +=  -0.25*np.einsum( "map,ma,mj->mjap", H_n, ft, vec_bt )
    mj_lmb      +=  -0.25*np.einsum( "amp,ma,mj,mN->mjNp", H_n_cj, ft, vec_bt, dlt_t )
    mj_lmb_cj   +=  -0.25*np.einsum( "map,ma,mj->mjap", H_c, ft, vec_bt )
    mj_lmb_cj   +=  -0.25*np.einsum( "amp,ma,mj,mN->mjNp", H_c_cj, ft, vec_bt, dlt_t )
    mj_lmb      +=  +0.25*np.einsum( "amp,am,mj,mN->mjNp", H_n, ft, vec_bt, dlt_t )
    mj_lmb      +=  +0.25*np.einsum( "map,am,mj->mjap", H_n_cj, ft, vec_bt )
    mj_lmb_cj   +=  +0.25*np.einsum( "amp,am,mj,mN->mjNp", H_c, ft, vec_bt, dlt_t )
    mj_lmb_cj   +=  +0.25*np.einsum( "map,am,mj->mjap", H_c_cj, ft, vec_bt )
    
    coef_mtr  = getCoefficientMatrix(ky_chi,  ky_th,   ky_th_cj, ky_lmb, ky_lmb_cj,
                                     t_chi,   t_th,    t_th_cj,  t_lmb,  t_lmb_cj,
                                     mj_chi,  mj_th,   mj_th_cj, mj_lmb, mj_lmb_cj,
                                     sys_vec_sz, mtp_vec_sz, bt_vec_sz )

    # For constant indeces yk:
    for n in range(M.N_el):
        np.add(rhs_sys_out[n], -1.j * mtr_sys[n] * ( M.en_el[n] + 0.5*np.sum(E.w_bt*MEI.bt_disp[n]**2) ) , out=rhs_sys_out[n])
        if M.con_int_mtp != 0:
            for m in range(M.N_el):
                if m != n:
                    np.add(rhs_sys_out[n], -1.j * ADe[n][m], out=rhs_sys_out[n])
        for q in range(M.N_dof):
            np.add(rhs_sys_out[n], -1.j * ( PTe[n][q] - Ke[n][q] ), out=rhs_sys_out[n])
        for a in range(E.N_bt_mtp):
            for b in range(E.N_bt_mtp):
                np.add(rhs_sys_out[n], -1.j * mtr_sys[n] * ft[a][b] * np.sum( E.w_bt * g1[a][b][n] ) , out=rhs_sys_out[n])
                for q in range(M.N_dof):
                    np.add(rhs_sys_out[n], -1.j * mtr_sys[n] * M.cord_vec[q]    * ft[a][b] * np.sum(g2_l[a][b][n][q]) , out=rhs_sys_out[n])
                    np.add(rhs_sys_out[n], -1.j * mtr_sys[n] * M.cord_vec[q]**2 * ft[a][b] * np.sum(g2_q[a][b][n][q]), out=rhs_sys_out[n])
        
    ### For constant index tau:
    for a in range(E.N_bt_mtp):
        for b in range(E.N_bt_mtp):
            pref = mtp_vec[b] * S[a][b]
            for n in range(M.N_el):
                rhs_mtp_out[a] += -1.j * pop[n] * pref * np.sum( E.w_bt * ( g1[a][b][n] - h1[n] ) )
                for q in range(M.N_dof):
                    rhs_mtp_out[a] += -1.j * Xl[n][q] * pref * np.sum( g2_l[a][b][n][q] - h2_l[n][q] )
                    rhs_mtp_out[a] += -1.j * Xq[n][q] * pref * np.sum( g2_q[a][b][n][q] - h2_q[n][q] )

    ### For constant indeces tau,j:
    for n in range(M.N_el):
        for tau in range(E.N_bt_mtp):
            for b in range(E.N_bt_mtp):
                
                ### j terms
                np.add(rhs_bt_out[tau], -1.j * pop[n] * ft[tau][b] * E.w_bt * ( vec_bt[b] - MEI.bt_disp[n]/cs.SQ2 ), out=rhs_bt_out[tau])
                for q in range(M.N_dof):
                    np.add(rhs_bt_out[tau], -1.j * Xl[n][q] * ft[tau][b] * ( MEI.vp_coef_LL[n][q]/cs.SQ2 + MEI.vp_coef_LQ[n][q]*T1[tau][b] ), out=rhs_bt_out[tau])
                    np.add(rhs_bt_out[tau], -1.j * Xq[n][q] * ft[tau][b] * ( MEI.vp_coef_QL[n][q]/cs.SQ2 ), out=rhs_bt_out[tau])
    
    rhs = np.concatenate((rhs_sys_out.ravel(order='C'), 
                          rhs_mtp_out.ravel(order='C'), 
                          rhs_bt_out.ravel(order='C'),
                          np.conj(rhs_mtp_out).ravel(order='C'),
                          np.conj(rhs_bt_out).ravel(order='C'))
                        )
    
    return coef_mtr, rhs

def sD2_Propagator(t, vec, M, E, MEI, cP):

    coef_mtr, rhs = getCoefAndRhs(t, vec, M, E, MEI, cP)

    sys_vec_sz  = M.N_el*M.N_xpts
    mtp_vec_sz  = E.N_bt_mtp
    bt_vec_sz   = E.N_bt_mtp*E.N_coh

    if cP.dif_solver == 'lstsq':
        out = sc.linalg.lstsq(coef_mtr, rhs, lapack_driver = 'gelsy', cond = cP.rcond_precision, check_finite = False)[0][0:(sys_vec_sz+mtp_vec_sz+bt_vec_sz)]
    elif cP.dif_solver == 'inv':
        out = sc.linalg.solve(coef_mtr, rhs, check_finite = False)[0:(sys_vec_sz+mtp_vec_sz+bt_vec_sz)]
    elif cP.dif_solver == 'mix':
        if (t > cP.dif_solver_mixtime):
            out = sc.linalg.solve(coef_mtr, rhs, check_finite = False)[0:(sys_vec_sz+mtp_vec_sz+bt_vec_sz)]
        else:
            out = sc.linalg.lstsq(coef_mtr, rhs, lapack_driver = 'gelsy', cond = cP.rcond_precision, check_finite = False)[0][0:(sys_vec_sz+mtp_vec_sz+bt_vec_sz)]
    
    if cP.print_calc_time == 1:
        print(t)
    
    return out    
    