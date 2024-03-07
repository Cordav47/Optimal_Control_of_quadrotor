#visualization
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle
import matplotlib.animation as man
from math import sin, cos, pi, radians
import Dynamics as dyn


step = dyn.dt


def visual(x, z, theta, alpha, ref, fig):
    #fig = plt.figure()
    
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5, 5), ylim=(-0.5, 5))
    ax.set_aspect('equal')
    ax.grid()
    th0 = theta
    th1 = alpha
    goal_x = ref[0]
    goal_y = ref[1]

    patch = ax.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g')) # , rotation_point= 'center'
    circle = ax.add_patch(Circle((0,0), 0, linewidth=1, edgecolor='k', facecolor='g'))
    #setpoint circle
    goal = ax.add_patch(Circle((0,0), 0, linewidth=1, edgecolor='black', facecolor='gold'))

    line1, = ax.plot([], [], marker = "o", lw=3)
    line2, = ax.plot([], [], marker = "o",  lw=3)
    giunto, = ax.plot([], [], '-', lw = 2, color = 'red')
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    #size of the object in visualization
    qcx = 0.4
    qcz = 0.25
    l = 0.15
    radius = 0.08
    radius_goal = 0.05

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        giunto.set_data([], [])
        time_text.set_text('')
        patch.set_xy((-qcx/2, -qcz/2))
        patch.set_width(qcx)
        patch.set_height(qcz)
        circle.set_center((0, -l-radius-qcz/2))
        
        circle.set_radius(radius)
        goal.set_center((goal_x, goal_y))
        goal.set_radius(radius_goal)
        return line1, line2, giunto, time_text, patch


    def animate(i):

        i = i*3 #speed up animation

        xl1 = [x[i]-qcx/2+qcz/2*sin(th0[i]), x[i]-qcx/2-l*cos(th0[i])]
        yl1 = [z[i], z[i]-l*sin(th0[i])]
        line1.set_data(xl1, yl1)
        xl2 = [x[i]+qcx/2, x[i]+l*cos(th0[i])+qcx/2]
        yl2 = [z[i], z[i]+l*sin(th0[i])]
        line2.set_data(xl2, yl2)
        gx = [x[i], x[i]+l*sin(th1[i])]
        gz = [z[i]-qcz/2, z[i]-l*cos(th1[i])-qcz/2]
        giunto.set_data(gx, gz)
        time_text.set_text(time_template % (i*step))
        patch.set_x(x[i]-qcx/2)
        patch.set_y(z[i]-qcz/2)
        patch.set_angle(th0[i]*180/pi)
        cx = (x[i]+l*sin(th1[i])+radius*sin(th1[i]))
        cz = (z[i]-l*cos(th1[i])-qcz/2-radius*cos(th1[i]))
        circle.set_center((cx, cz))
        goal.set_center((goal_x, goal_y))

        return line1, line2, giunto, time_text, patch, circle, goal

    ani = man.FuncAnimation(fig, animate, np.arange(1, len(x)), interval=25, blit=True, init_func=init)
    #ani.save('drone.mp4',fps=60) #to save mp4 animation
    plt.show(block=True)