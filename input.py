import mkl
import numpy as np
import constants  as cs
from functions    import createCopies
from molecule     import Molecule
from environment  import Environment
from mol_env      import MEInteraction
from parameters   import Parameters
from wavefunction import sD2_Wavefunction
from calculator   import Calculator
from figdata      import PlotAndSaveData

# Computation parameters
p = Parameters()
p.show_figures    = 0
p.plot_animations = 1
p.save_animations = 1
p.save_figures    = 1
p.save_data       = 1
p.tmax = 100.0
p.dt   = 1.0
# p.print_calc_time = 1
# p.N_rlz = 1
# p.dif_solver = 'mix'
# p.dif_solver_mixtime = 25

# Molecule
m = Molecule(2, 1, id=1)

           #  E1      E2 
m.en_el    =  [0,   2500]  
m.init_pop =  [0,   1]

m.con_int_str   = 1000.0
m.adia_cpl_type = 'v'
m.adia_cpl_disp = 0

m.w_dof_gr = [[500]]
m.w_dof    = [[500,  500]]  #Q0

m.disp_dof = [[-1.35,  1.35]]  #Q1

m.xmin, m.xmax, m.dx = [-10], [12], [0.40]
m.PEStypes =  [['morse',  'harm']]  #Q1
m.PESparams = [[ [20000],  [] ]]  #Q1
               
# Environment
e = Environment(id=1)

e.T        = 0 # K
e.N_bt_mtp = 1
e.wmin_bt, e.wmax_bt, e.dw_bt = 25, 1000, 25 # cm-1

e.thermalization = 0
e.scat_interval  = 10 # fs
e.scat_rate      = 0.05   # fs-1
e.T_offset       = 0 # K

# Molecule-Environment interaction
mei = MEInteraction(m, e, id=1)

mei.el_phon_cpl  = 1
mei.spdName      = 'ohmic'
mei.spdArgs      = [100, 2]
mei.reorg_energy = [100, 100]

                # Q1  Q2  Q3
mei.vp_mtp     = [1]

                #  E1     E2 
mei.vp_pow_mtp = [[0,     0], # LL
                  [1,     1], # LQ
                  [0,     0]] # QL

# For list calculations
m_list = createCopies(m, 3)
e_list = createCopies(e, 3)

e_list[0].N_bt_mtp = 1
e_list[1].N_bt_mtp = 2
e_list[2].N_bt_mtp = 3
# e_list[3].N_bt_mtp = 4
# e_list[4].N_bt_mtp = 5
# e_list[5].N_bt_mtp = 6

mei_list = MEInteraction(m_list, e_list, mei)     

wf = sD2_Wavefunction(m_list, e_list, mei_list, p)
data = Calculator(wf)
PlotAndSaveData(data)

### Single molecule calculation
# wf = sD2_Wavefunction(m, e, mei, p)
# data = Calculator(wf)
# PlotAndSaveData(data)

