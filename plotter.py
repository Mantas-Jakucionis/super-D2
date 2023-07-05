import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import interpolate
import constants as cs
import numpy as np

class Plotter():

    id = np.random.random_integers(1, 10000)
    fileNames = []

    def ShowAndSave(self, fig, data, prefix):
        if data.wf.cP.show_figures == 1:
            plt.show()
        if data.wf.cP.save_figures == 1:
            figname = str(self.id) + '_' + prefix + '_' + str(data.wf.M.id) + '_' + str(data.wf.E.id) + '_' + str(data.wf.MEI.id)
            saveName = figname+str('.pdf')
            fig.savefig(saveName)
            self.fileNames.append(saveName)
            plt.close()

    def ShowAndSaveAnimation(self, fig, data, anim, prefix):
        if data.wf.cP.show_figures == 1:
            plt.show()
        if data.wf.cP.save_figures == 1 and data.wf.cP.save_animations == 1:
            if data.wf.M.N_dof == 1:
                figname = str(self.id) + '_' + prefix + '_' + str(data.wf.M.id) + '_' + str(data.wf.E.id) + '_' + str(data.wf.MEI.id)
                saveName = figname+str('.gif')
                anim.save(saveName, writer='imagemagick', fps=60)
                self.fileNames.append(saveName)
                plt.close()
            elif data.wf.M.N_dof == 2:
                # figname = str(self.id) + '_' + prefix + '_' + str(data.wf.M.id) + '_' + str(data.wf.E.id) + '_' + str(data.wf.MEI.id)
                # saveName = figname+str('.gif')
                # anim.save(saveName, writer='imagemagick', fps=60, bitrate=100)
                # self.fileNames.append(saveName)
                # plt.close()
                print("Implementation of 2D probability density plots are not done.")

    def __init__(self, data):

        N_el  = data.wf.M.N_el
        N_dof = data.wf.M.N_dof
        t_out = data.wf.cP.t_out
        w_bt  = data.wf.E.w_bt 
        t_aver_axis = data.wf.cP.t_aver_axis
        cord_vec = data.wf.M.cord_vec

        col = ['red', 'blue', 'green', 'black']
        lt = ['-','--','-.',':']

        if data.wf.cP.plot_animations == 1:

            if data.wf.M.N_dof == 1:

                weight = 5000

                def animate(t):
                    for n in range(N_el):
                        l = np.real(data.prob_distr[n][n][t])
                        lines[n].set_data(cord_vec[0], data.wf.M.en_el[n]/cs.CM2FS + weight*l)
                    for m in range(N_el):
                        n = m + N_el
                        lines[n].set_data(cord_vec[0], data.wf.M.psf_energy[m]/cs.CM2FS)
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
                    ln, = ax.plot(cord_vec[0], data.wf.M.psf_energy[n]/cs.CM2FS, color='black')
                    lines.append(ln)

                anim = animation.FuncAnimation(f1, animate, np.arange(0, data.wf.cP.N_t-1), init_func=init, interval=25, blit=True)

                plt.ylim(0, 3000 + max(data.wf.M.en_el)/cs.CM2FS*1.85)
                plt.xlim(data.wf.M.xmin[0], data.wf.M.xmax[0])
                ax.set_xlabel("$Reaction \ coordinate$")
                ax.set_ylabel("$Energy \ (cm^{-1})$")
                self.ShowAndSaveAnimation(f1, data, anim, 'prob')

            elif data.wf.M.N_dof == 2:

                weight = 5000

                def animate(t, n):
                    cont = plt.contourf(*cord_vec, np.real(data.prob_distr[n][n][t]))
                    return cont

                for n in range(N_el):

                    f2 = plt.figure()
                    ax = plt.axes(xlim=(data.wf.M.xmin[0], data.wf.M.xmax[0]), ylim=(data.wf.M.xmin[1], data.wf.M.xmax[1])) 
                    cont = plt.contourf(*cord_vec, np.real(data.prob_distr[n][n][0]))
                    plt.title("$Prob. \ density \ of \ energy \ $" + str(data.wf.M.en_el[n]/cs.CM2FS) + "$ \ cm^{-1} \ electronic \ state.$")

                    anim = animation.FuncAnimation(f2, animate, frames=np.arange(0, data.wf.cP.N_t-1, 4), repeat=True, fargs=(n,))

                    plt.xlabel("$Reaction \ coordinate \ $" + "$X_1$")
                    plt.ylabel("$Reaction \ coordinate \ $" + "$X_2$")
                    plt.xlim(data.wf.M.xmin[0],data.wf.M.xmax[0]-data.wf.M.dx[0])
                    plt.ylim(data.wf.M.xmin[1],data.wf.M.xmax[1]-data.wf.M.dx[1])
                    self.ShowAndSaveAnimation(f2, data, anim, 'prob' + '_e' + str(n))

        if data.wf.cP.plot_properties == 1:

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
            self.ShowAndSave(f3, data, 'prop')

            # Plotting energies
            f4, axs = plt.subplots(1, 8, sharey=False, figsize=(24,6))
            for i in range(len(axs)-1):
                axs[i].plot(t_out, data.sep_energy[i].real)
                axs[i].set_xlabel("$Time \ (fs)$")
            axs[7].plot(t_out, data.tot_energy.real)
            axs[7].set_xlabel("$Time \ (fs)$")
            axs[0].set_ylabel("$Energy \ (cm^{-1})$")
            self.ShowAndSave(f4, data, 'energy')

            f5 = plt.figure()
            plt.plot(t_aver_axis, np.mean(data.T_bath, axis=1))
            plt.xlabel("$Time \ (fs)$")
            plt.ylabel("$Trans. \ Bath \ Temp. \ (K)$")
            self.ShowAndSave(f5, data, 'T_bath')

            if data.wf.E.N_coh > 5:

                f6 = plt.figure(2)
                cf = plt.contourf(w_bt/cs.CM2FS, t_out, np.real(data.bath_energy_distr))
                cbar = plt.colorbar(cf)
                plt.title("$Bath \ energy \ distribution$")
                plt.xlabel("$Bath \ frequency, \ (cm{-1})$")
                plt.ylabel("$Time \ (fs)$")
                self.ShowAndSave(f6, data, 'Bath_energy_distr')

                f7 = plt.figure(3)
                ctemp = plt.contourf(w_bt/cs.CM2FS, t_aver_axis, data.T_bath)
                cbar = plt.colorbar(ctemp)
                plt.title("$Bath \ temperature \ distribution$")
                plt.xlabel("$Bath \ frequency, \ (cm{-1})$")
                plt.ylabel("$Time \ (fs)$")
                self.ShowAndSave(f7, data, 'Bath_temp_distr')

                m1, m2, m3 = 0, 3, 5
                stdx = data.bwp_2x - data.bwp_x**2
                stdp = data.bwp_2p - data.bwp_p**2
                f8 = plt.figure(4)
                plt.plot(np.real(data.bwp_x[m1]), np.real(data.bwp_p[m1]), color=col[0])
                plt.plot(np.real(data.bwp_x[m2]), np.real(data.bwp_p[m2]), color=col[1])
                plt.plot(np.real(data.bwp_x[m3]), np.real(data.bwp_p[m3]), color=col[2])
                plt.xlabel("$Bath \ mode \ coordinate$")
                plt.ylabel("$Bath \ mode \ momentum$")
                self.ShowAndSave(f8, data, 'Bath_phase_space')

                f9 = plt.figure(5)
                plt.plot(t_out, np.real(stdx[m1]), color=col[0], linestyle=lt[0])
                plt.plot(t_out, np.real(stdp[m1]), color=col[1], linestyle=lt[0])
                plt.plot(t_out, np.real(stdx[m2]), color=col[0], linestyle=lt[1])
                plt.plot(t_out, np.real(stdp[m2]), color=col[1], linestyle=lt[1])
                plt.plot(t_out, np.real(stdx[m3]), color=col[0], linestyle=lt[2])
                plt.plot(t_out, np.real(stdp[m3]), color=col[1], linestyle=lt[2])
                plt.xlabel("$Time \ (fs)$")
                plt.ylabel("$Bath \ mode \ std.$")
                self.ShowAndSave(f9, data, 'Bath_mode_std')

        if data.wf.cP.plot_abs == 1:
            pass
            # lin_pol_smooth = [data.lin_pol[t] * np.exp(-t/1000) for t in range(data.wf.cP.N_t)]
            # # lin_pol_smooth = lin_pol

            # dense_lin_pol = interpolate.interp1d(t_out, lin_pol_smooth)
            # ddt = 0.01
            # lin_pol_smooth = dense_lin_pol( np.arange(data.wf.cP.tmin, data.wf.cP.tmax-data.wf.cP.dt, ddt) )

            # abs_spec = np.fft.fftshift(np.fft.fft(lin_pol_smooth))
            # abs_freq = 2*np.pi*np.fft.fftfreq(abs_spec.size, d=ddt)
            # f10 = plt.figure()
            # plt.plot(np.fft.fftshift(abs_freq)/cs.CM2FS, abs_spec.real)
            # plt.xlabel("$Frequancy \ (cm^{-1})$")
            # plt.ylabel("$Absorption$")
            # plt.ylim(0, plt.ylim()[1])
            # self.ShowAndSave(f10, data, 'abs')

        print("Plots created:")







