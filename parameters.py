import numpy as np
import matplotlib.pyplot as plt
import mkl
from parser       import ParsedArgs

class Parameters:

    init_flag = 0

    show_figures    = 0
    plot_properties = 1
    plot_abs        = 0
    save_data       = 1
    save_figures    = 1
    plot_animations = 1
    save_animations = 0

    print_calc_time = 0

    sep_thermal = 'same-same'
    N_rlz = 1
    doff = 0

    ens_name = None

    calc_properties = 1
    # calc_abs = 0

    tmin, tmax, dt  = 0.0, 50.0, 1.0 # fs
    t_sample_rate   = 10.0 # fs
    atol_precision  = 1e-6
    rtol_precision  = 1e-6
    rcond_precision = 1e-10
    ode_solver      = 'RK45'
    dif_solver      = 'lstsq'
    dif_solver_mixtime = tmax

    mkl.set_num_threads(1)

    def __init__(self):

        prs = ParsedArgs()

        print("Parsing command line input parameters.")

        if prs.name:  
            print("Data will be saved to prefix %s folder." % prs.name)
            self.data_save_folder = str(prs.name)
        else:
            self.data_save_folder = None

        if prs.cores:
            self.N_cores = int(prs.cores)
        else:
            self.N_cores = 1

        if prs.show:
            if prs.show == 1:
                self.show_figures = 1
            elif prs.show == 0:
                self.show_figures = 0

        if prs.save:
            if prs.save == 1:
                self.save_figures = 1
                self.save_data = 1
            elif prs.save == 0:
                self.save_figures = 0
                self.save_data = 0

        if prs.dispe:
            if prs.dispe == 1:
                pass
            elif prs.dispe == 0:
                plt.switch_backend('agg')

        if prs.roff:
            self.roff = int(prs.roff)
        else:
            self.roff = 0

    def initialize(self):
        self.t_out = np.arange(self.tmin, self.tmax, self.dt)
        self.t_aver_axis = np.arange(self.tmin, self.tmax, self.t_sample_rate)
        self.N_t = len(self.t_out)
        self.init_flag = 1

