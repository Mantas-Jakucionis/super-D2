import os
import time
import shutil
import numpy as np

class Saver():

    def __init__(self, plotter, data):

        if data.wf.cP.save_data == 1:

            if data.wf.multi == False:

                if data.wf.cP.data_save_folder != None:
                    foldername = data.wf.cP.data_save_folder + '_' + str(time.ctime())
                else :
                    foldername = str(time.ctime())
                os.makedirs(foldername)

                for file in plotter.fileNames:
                    preprf_file = file.replace(str(plotter.id) + '_', '')
                    print("Filename:", preprf_file)
                    shutil.move(file, os.path.join(foldername, preprf_file))

                print("Plots have been moved.")

                os.makedirs(foldername + '/data')
                os.chdir(foldername + '/data')
                np.save('prop_times',       data.wf.cP.t_out)
                np.save('prob_distr',       data.prob_distr)
                np.save('diag_prob_distr',  data.diag_prob_distr)
                np.save('tot_energy',       data.tot_energy)
                np.save('bath_energy_distr',data.bath_energy_distr)
                np.save('bath_energy',      data.bath_energy)
                np.save('bath_densty',      data.bath_density)
                np.save('populations',      data.populations)
                np.save('diag_populations', data.diag_populations)
                np.save('psf_energy',       data.wf.M.psf_energy)
                np.save('mean_coord',       data.mean_coord)
                np.save('sep_energy',       data.sep_energy)
                np.save('bwp_x',            data.bwp_x)
                np.save('bwp_p',            data.bwp_p)
                np.save('bwp_2x',           data.bwp_2x)
                np.save('bwp_2p',           data.bwp_2p)
                np.save('KEav',             data.KEav)
                np.save('lin_pol',          data.lin_pol)

                print("Data have been saved and moved.")