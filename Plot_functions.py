import matplotlib.pyplot as plt
import numpy as np

def myPlot(time, name, variables):
    #if np.size(variables) != len(name):
    #    print("Number of variables doesn't match its number of names")

    index = len(name)
    for i in range(0, index, 1):
        plt.figure()
        plt.plot(time, variables[i])
        plt.xlabel("time (s)")
        plt.ylabel(name[i])
        plt.grid()
        plt.show()

def mySubplot(time, name, variables, ref, rows, pref):
    #if np.size(variables) != len(name):
    #    print("Number of variables doesn't match its number of names")

    n = len(name)
    complete_pic = n // rows
    resto = n % rows

    for k in range(0, complete_pic, 1):
        plt.figure()
        for i in range(0, rows, 1):
            plt.subplot(rows, 1, i+1)
            plt.plot(time, variables[k*i+i])
            plt.xlabel("time (s)")
            plt.ylabel(name[k*i+i])
            if pref == True:
                plt.plot(time, ref[k*i+i], 'g--', linewidth=2)
            plt.grid()
        plt.show(block=True)
        
    for j in range(0, resto, 1):
        plt.figure()
        plt.subplot(resto,1,j+1)
        plt.plot(time, variables[-resto+j])
        if pref == True:
            plt.plot(time, ref[-resto+j], 'g--', linewidth=2)
        plt.xlabel("time (s)")
        plt.ylabel(name[-resto +j])
        plt.grid()
        plt.show(block=True)

def mySubplotWithReferences(xx_opt, uu_opt, xx_ref, uu_ref, tf, TT):

    #variables
    XP = 0
    YP = 1
    ALPHA = 2
    THETA = 3
    VX = 4
    VY = 5
    WALPHA = 6
    WTHETA = 7
    FS = 0
    FD = 1
    ns = 8
    ni = 2

    tt_hor = np.linspace(0,tf,TT)
    fig, axs = plt.subplots(ns+ni, 1, sharex='all')

    axs[0].plot(tt_hor, xx_opt[XP,:], linewidth=2)
    axs[0].plot(tt_hor, xx_ref[XP,:], 'g--', linewidth=2)
    axs[0].grid()
    axs[0].set_ylabel('$x_p$')

    axs[1].plot(tt_hor, xx_opt[YP,:], linewidth=2)
    axs[1].plot(tt_hor, xx_ref[YP,:], 'g--', linewidth=2)
    axs[1].grid()
    axs[1].set_ylabel('$y_p$')

    axs[2].plot(tt_hor, xx_opt[ALPHA,:], linewidth=2)
    axs[2].plot(tt_hor, xx_ref[ALPHA,:], 'g--', linewidth=2)
    axs[2].grid()
    axs[2].set_ylabel('$alpha$')

    axs[3].plot(tt_hor, xx_opt[THETA,:], linewidth=2)
    axs[3].plot(tt_hor, xx_ref[THETA,:], 'g--', linewidth=2)
    axs[3].grid()
    axs[3].set_ylabel('$theta$')

    axs[4].plot(tt_hor, xx_opt[VX,:], linewidth=2)
    axs[4].plot(tt_hor, xx_ref[VX,:], 'g--', linewidth=2)
    axs[4].grid()
    axs[4].set_ylabel('$v_x$')

    axs[5].plot(tt_hor, xx_opt[VY,:], linewidth=2)
    axs[5].plot(tt_hor, xx_ref[VY,:], 'g--', linewidth=2)
    axs[5].grid()
    axs[5].set_ylabel('$v_y$')

    axs[6].plot(tt_hor, xx_opt[WALPHA,:], linewidth=2)
    axs[6].plot(tt_hor, xx_ref[WALPHA,:], 'g--', linewidth=2)
    axs[6].grid()
    axs[6].set_ylabel('$w_alpha$')

    axs[7].plot(tt_hor, xx_opt[WTHETA,:], linewidth=2)
    axs[7].plot(tt_hor, xx_ref[WTHETA,:], 'g--', linewidth=2)
    axs[7].grid()
    axs[7].set_ylabel('$w_theta$')

    axs[8].plot(tt_hor, uu_opt[FS,:],'r', linewidth=2)
    axs[8].plot(tt_hor, uu_ref[FS,:], 'r--', linewidth=2)
    axs[8].grid()
    axs[8].set_ylabel('$F_s$')
    axs[8].set_xlabel('time')

    axs[9].plot(tt_hor, uu_opt[FD,:],'r', linewidth=2)
    axs[9].plot(tt_hor, uu_ref[FD,:], 'r--', linewidth=2)
    axs[9].grid()
    axs[9].set_ylabel('$F_d$')
    axs[9].set_xlabel('time')

    plt.show()