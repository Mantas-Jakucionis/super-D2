import os
import time
import shutil
import numpy as np
import constants as cs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import interpolate

class DataStruct():

    def __init__(self, computedData, n):
        self.prob_distr        = computedData.prob_distr        [n]
        self.mean_coord        = computedData.mean_coord        [n]
        self.populations       = computedData.populations       [n]
        self.bath_density      = computedData.bath_density      [n]
        self.bath_energy_distr = computedData.bath_energy_distr [n]
        self.bath_energy       = computedData.bath_energy       [n]
        self.lin_pol           = computedData.lin_pol           [n]
        self.sep_energy        = computedData.sep_energy        [n]
        self.diag_prob_distr   = computedData.diag_prob_distr   [n]
        self.diag_populations  = computedData.diag_populations  [n]
        self.KEav              = computedData.KEav              [n]
        self.T_bath            = computedData.T_bath            [n]
        self.bwp_x             = computedData.bwp_x             [n]
        self.bwp_p             = computedData.bwp_p             [n]
        self.bwp_2x            = computedData.bwp_2x            [n]
        self.bwp_2p            = computedData.bwp_2p            [n]  
        self.tot_energy        = np.sum(computedData.sep_energy [n], axis=0)
      
def PlotAndSaveData(data):

    if data.wf.multi == False:

        pl = Plotter(data, data.wf.M, data.wf.E, data.wf.MEI, data.wf.cP)
        Saver(pl, data, data.wf.M, data.wf.cP)

    elif data.wf.multi == True:

        for n in range(data.wf.N_multi):
            single_data = DataStruct(data, n)
            pl = Plotter(single_data, data.wf.M[n], data.wf.E[n], data.wf.MEI[n], data.wf.cP)
            Saver(pl, single_data, data.wf.M[n], data.wf.cP)

class Plotter():

    fileNames = []

    def ShowAndSave(self, fig, prefix):
        if self.cP.show_figures == 1:
            plt.show()
        if self.cP.save_figures == 1:
            figname = prefix + '_' + str(self.M.id) + '_' + str(self.E.id) + '_' + str(self.MEI.id)
            saveName = figname+str('.pdf')
            fig.savefig(self.foldername + '/' + saveName)
            self.fileNames.append(saveName)
            plt.close()

    def ShowAndSaveAnimation(self, fig, anim, prefix):
        if self.cP.show_figures == 1:
            plt.show()
        if self.cP.save_figures == 1 and self.cP.save_animations == 1:
            if self.M.N_dof == 1:
                figname = prefix + '_' + str(self.M.id) + '_' + str(self.E.id) + '_' + str(self.MEI.id)
                saveName = self.foldername + '/' + figname+str('.gif')
                anim.save(saveName, writer='imagemagick', fps=60)
                self.fileNames.append(saveName)
                plt.close()
            elif self.M.N_dof == 2:
                # figname = prefix + '_' + str(self.M.id) + '_' + str(self.E.id) + '_' + str(self.MEI.id)
                # saveName = figname+str('.gif')
                # anim.save(saveName, writer='imagemagick', fps=60, bitrate=100)
                # self.fileNames.append(saveName)
                # plt.close()
                print("Implementation of 2D probability density plots are not yet done.")

    def __init__(self, data, M, E, MEI, cP):

        N_el  = M.N_el
        N_dof = M.N_dof
        t_out = cP.t_out
        w_bt  = E.w_bt 
        t_aver_axis = cP.t_aver_axis
        cord_vec = M.cord_vec

        self.cP = cP
        self.M = M
        self.E = E
        self.MEI = MEI

        col = ['red', 'blue', 'green', 'black']
        lt = ['-','--','-.',':']

        if cP.save_data == 1 or cP.save_figures == 1:
            if cP.data_save_folder != None:
                self.foldername = cP.data_save_folder  + '_' + str(self.M.id) + '_' + str(self.E.id) + '_' + str(self.MEI.id) + '_' + str(time.ctime())
            else :
                self.foldername = str(self.M.id) + '_' + str(self.E.id) + '_' + str(self.MEI.id) + '_' + str(time.ctime())
            os.makedirs(self.foldername)

        if cP.plot_animations == 1:

            if M.N_dof == 1:

                weight = 15000

                def animate(t):
                    for n in range(N_el):
                        l = np.real(data.prob_distr[n][n][t])
                        lines[n].set_data(cord_vec[0], M.en_el[n]/cs.CM2FS + weight*l)
                    for m in range(N_el):
                        n = m + N_el
                        lines[n].set_data(cord_vec[0], M.psf_energy[m]/cs.CM2FS)
                    return tuple(lines)

                def init():
                    for n in range(N_el):
                        lines[n].set_data([],[])
                    for n in range(N_el):
                        lines[n].set_data([],[])
                    return lines

                f1, ax = plt.subplots()
                lines = []
                for n in range(N_el):
                    ln, = ax.plot(cord_vec[0], np.real(data.prob_distr[n][n][0]), color=col[n])
                    lines.append(ln)
                for n in range(N_el):
                    ln, = ax.plot(cord_vec[0], M.psf_energy[n]/cs.CM2FS, color='black')
                    lines.append(ln)

                anim = animation.FuncAnimation(f1, animate, np.arange(0, cP.N_t-1), init_func=init, interval=25, blit=True)

                plt.ylim(0, 3000 + max(M.en_el)/cs.CM2FS*1.85)
                plt.xlim(M.xmin[0], M.xmax[0])
                ax.set_xlabel("$Reaction \ coordinate$")
                ax.set_ylabel("$Energy \ (cm^{-1})$")
                self.ShowAndSaveAnimation(f1, anim, 'prob')

            elif M.N_dof == 2:

                weight = 5000

                def animate(t, n):
                    cont = plt.contourf(*cord_vec, np.real(data.prob_distr[n][n][t]))
                    pot  = plt.contour(*cord_vec, M.psf_energy[n], colors=['white'], linewidths=0.1)
                    return cont, pot

                for n in range(N_el):

                    f2 = plt.figure()
                    ax = plt.axes(xlim=(M.xmin[0], M.xmax[0]), ylim=(M.xmin[1], M.xmax[1])) 
                    cont = plt.contourf(*cord_vec, np.real(data.prob_distr[n][n][0]))
                    pot  = plt.contour(*cord_vec, M.psf_energy[n], colors=['white'], linewidths=0.1)
                    plt.title("$Prob. \ density \ of \ energy \ $" + str(M.en_el[n]/cs.CM2FS) + "$ \ cm^{-1} \ electronic \ state.$")

                    anim = animation.FuncAnimation(f2, animate, frames=np.arange(0, cP.N_t-1, 4), repeat=True, fargs=(n,))

                    plt.xlabel("$Reaction \ coordinate \ $" + "$X_1$")
                    plt.ylabel("$Reaction \ coordinate \ $" + "$X_2$")
                    plt.xlim(M.xmin[0],M.xmax[0]-M.dx[0])
                    plt.ylim(M.xmin[1],M.xmax[1]-M.dx[1])
                    self.ShowAndSaveAnimation(f2, anim, 'prob_e' + str(n))

        if cP.plot_properties == 1:

            # Plotting physical properties
            f3, axs = plt.subplots(nrows=1, ncols=4, figsize=(25,5))
            for n in range(N_el):
                for q in range(N_dof):
                    axs[0].plot(t_out, data.mean_coord[n][q])
            for n in range(N_el):
                axs[1].plot(t_out, np.real(data.populations[n]))
                axs[2].plot(t_out, np.real(data.diag_populations[n]))
            axs[3].plot(t_out, np.real(data.bath_energy))
            for i in range(len(axs)-1):
                axs[i].set_xlabel("$Time \ (fs)$")
            axs[0].set_ylabel("$Mean \ coordinate$")
            axs[1].set_ylabel("$Adiabatic \ pop.$")
            axs[2].set_ylabel("$Diabatic \ pop.$")                
            axs[3].set_ylabel("$Bath \ energy \ (cm^{-1})$")
            self.ShowAndSave(f3, 'prop')

            # Plotting energies
            f4, axs = plt.subplots(1, 8, sharey=False, figsize=(24,6))
            for i in range(len(axs)-1):
                axs[i].plot(t_out, data.sep_energy[i].real)
                axs[i].set_xlabel("$Time \ (fs)$")
            axs[7].plot(t_out, data.tot_energy.real)
            axs[7].set_xlabel("$Time \ (fs)$")
            axs[0].set_ylabel("$Energy \ (cm^{-1})$")
            self.ShowAndSave(f4, 'energy')

            f5 = plt.figure()
            plt.plot(t_aver_axis, np.mean(data.T_bath, axis=1))
            plt.xlabel("$Time \ (fs)$")
            plt.ylabel("$Trans. \ Bath \ Temp. \ (K)$")
            self.ShowAndSave(f5, 'T_bath')

            if E.N_coh > 5:

                f6 = plt.figure()
                cf = plt.contourf(w_bt/cs.CM2FS, t_out, np.real(data.bath_energy_distr))
                cbar = plt.colorbar(cf)
                plt.title("$Bath \ energy \ distribution$")
                plt.xlabel("$Bath \ frequency, \ (cm{-1})$")
                plt.ylabel("$Time \ (fs)$")
                self.ShowAndSave(f6, 'Bath_energy_distr')

                # f7 = plt.figure()
                # ctemp = plt.contourf(w_bt/cs.CM2FS, t_aver_axis, data.T_bath)
                # cbar = plt.colorbar(ctemp)
                # plt.title("$Bath \ temperature \ distribution$")
                # plt.xlabel("$Bath \ frequency, \ (cm{-1})$")
                # plt.ylabel("$Time \ (fs)$")
                # self.ShowAndSave(f7, 'Bath_temp_distr')

                m1, m2, m3 = 0, 3, 5
                stdx = data.bwp_2x - data.bwp_x**2
                stdp = data.bwp_2p - data.bwp_p**2
                f8 = plt.figure()
                plt.plot(np.real(data.bwp_x[m1]), np.real(data.bwp_p[m1]), color=col[0])
                plt.plot(np.real(data.bwp_x[m2]), np.real(data.bwp_p[m2]), color=col[1])
                plt.plot(np.real(data.bwp_x[m3]), np.real(data.bwp_p[m3]), color=col[2])
                plt.xlabel("$Bath \ mode \ coordinate$")
                plt.ylabel("$Bath \ mode \ momentum$")
                self.ShowAndSave(f8, 'Bath_phase_space')

                f9 = plt.figure()
                plt.plot(t_out, np.real(stdx[m1]), color=col[0], linestyle=lt[0])
                plt.plot(t_out, np.real(stdp[m1]), color=col[1], linestyle=lt[0])
                plt.plot(t_out, np.real(stdx[m2]), color=col[0], linestyle=lt[1])
                plt.plot(t_out, np.real(stdp[m2]), color=col[1], linestyle=lt[1])
                plt.plot(t_out, np.real(stdx[m3]), color=col[0], linestyle=lt[2])
                plt.plot(t_out, np.real(stdp[m3]), color=col[1], linestyle=lt[2])
                plt.xlabel("$Time \ (fs)$")
                plt.ylabel("$Bath \ mode \ variance.$")
                self.ShowAndSave(f9, 'Bath_mode_var')

        if cP.plot_abs == 1:
            pass
            # lin_pol_smooth = [data.lin_pol[t] * np.exp(-t/1000) for t in range(cP.N_t)]
            # # lin_pol_smooth = lin_pol

            # dense_lin_pol = interpolate.interp1d(t_out, lin_pol_smooth)
            # ddt = 0.01
            # lin_pol_smooth = dense_lin_pol( np.arange(cP.tmin, cP.tmax-cP.dt, ddt) )

            # abs_spec = np.fft.fftshift(np.fft.fft(lin_pol_smooth))
            # abs_freq = 2*np.pi*np.fft.fftfreq(abs_spec.size, d=ddt)
            # f10 = plt.figure()
            # plt.plot(np.fft.fftshift(abs_freq)/cs.CM2FS, abs_spec.real)
            # plt.xlabel("$Frequancy \ (cm^{-1})$")
            # plt.ylabel("$Absorption$")
            # plt.ylim(0, plt.ylim()[1])
            # self.ShowAndSave(f10, 'abs')

class Saver():

    def __init__(self, plotter, data, M, cP):

        if cP.save_data == 1:

            # if plotter.multi == False:
   
            fldName = plotter.foldername + '/data'
            os.makedirs(fldName)
            np.save(fldName + '/' + 'prop_times',       cP.t_out)
            np.save(fldName + '/' + 'prob_distr',       data.prob_distr)
            np.save(fldName + '/' + 'diag_prob_distr',  data.diag_prob_distr)
            np.save(fldName + '/' + 'tot_energy',       data.tot_energy)
            np.save(fldName + '/' + 'bath_energy_distr',data.bath_energy_distr)
            np.save(fldName + '/' + 'bath_energy',      data.bath_energy)
            np.save(fldName + '/' + 'bath_densty',      data.bath_density)
            np.save(fldName + '/' + 'populations',      data.populations)
            np.save(fldName + '/' + 'diag_populations', data.diag_populations)
            np.save(fldName + '/' + 'psf_energy',       M.psf_energy)
            np.save(fldName + '/' + 'mean_coord',       data.mean_coord)
            np.save(fldName + '/' + 'sep_energy',       data.sep_energy)
            np.save(fldName + '/' + 'bwp_x',            data.bwp_x)
            np.save(fldName + '/' + 'bwp_p',            data.bwp_p)
            np.save(fldName + '/' + 'KEav',             data.KEav)
            np.save(fldName + '/' + 'lin_pol',          data.lin_pol)

