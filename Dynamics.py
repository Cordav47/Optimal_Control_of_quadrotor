#libraries
import numpy as np
import scipy as sp
from scipy.optimize import fsolve
import math

#Matteo Dynamics

dt = 0.01 # discretization stepsize - Forward Euler 
ns = 8  # state dimension
ni = 2  # input dimension
FFeq = (0.028+0.4)*9.81 # equilibrium thrust

mm = 0.04  # pendulum mass
MM = 0.028    # drone mass
LL = 0.2    # pendulum length
ll = 0.05   # distance from drone center of mass to the propeller axis
gg = 9.81   # gravity
JJ = 0.001  # moment of inertia of the drone

# for convenience
kk = 1/(MM + mm)
hh = (mm*LL)*kk
oo = (mm/MM)*kk
M_times_L = MM*LL
l_over_J = ll/JJ

def dynamics(xx_in,uu_in):
  
  """
    Nonlinear dynamics of quadrotor drone with pendulum

    Args
      - xx \in \R^8 state at time t
      - uu \in \R^2 input at time t
  """

  xx = xx_in.squeeze()         # squeeze removes single-dimensional entries from the shape of an array (for ex. (2,1) --> (2,))
  uu = uu_in.squeeze()
  xxp = np.zeros((ns, ));   # This is a 8x1 vector initialized to zero
  
  # Recurrent operations:
  sin_alpha = np.sin(xx[2])
  sin_theta = np.sin(xx[3])
  sin_alpha_theta = np.sin(xx[2]-xx[3])
  
  cos_alpha = np.cos(xx[2])
  cos_theta = np.cos(xx[3])
  cos_alpha_theta = np.cos(xx[2]-xx[3])
  
  omega_alpha_square = xx[6]**2
  
  Fs = uu[0]
  Fd = uu[1]

  # System's equations (nonlinear dynamics):

  xxp[:4] = xx[:4] + dt * xx[4:8]
  xxp[4] = xx[4] + dt * (hh* omega_alpha_square * sin_alpha - kk*Fs*sin_theta + oo*Fs*sin_alpha_theta*cos_alpha)
  xxp[5] = xx[5] + dt * (-hh* omega_alpha_square * cos_alpha + kk*Fs*cos_theta + oo*Fs*sin_alpha_theta*sin_alpha - gg)
  xxp[6] = xx[6] + dt * ((-Fs*sin_alpha_theta)/M_times_L)
  xxp[7] = xx[7] + dt * l_over_J * Fd

  # Jacobian of the dynamics (transpose of the gradient)
  A = np.array ([[1, 0, 0, 0, dt, 0, 0, 0],
                 [0, 1, 0, 0, 0, dt, 0, 0],
                 [0, 0, 1, 0, 0, 0, dt, 0],
                 [0, 0, 0, 1, 0, 0, 0, dt],
                 [0, 0, dt*hh*omega_alpha_square*cos_alpha + dt*oo*Fs*(cos_alpha_theta*cos_alpha - sin_alpha*sin_alpha_theta), -dt*kk*Fs*cos_theta - dt*oo*Fs*cos_alpha_theta*cos_alpha, 1, 0, 2*dt*hh*xx[6]*sin_alpha, 0],
                 [0, 0, dt*hh*omega_alpha_square*sin_alpha + dt*oo*Fs*(cos_alpha_theta*sin_alpha + cos_alpha*sin_alpha_theta), -dt*kk*Fs*sin_theta - dt*oo*Fs*cos_alpha_theta*sin_alpha, 0, 1, -2*dt*hh*xx[6]*cos_alpha, 0],
                 [0, 0, -dt/M_times_L*Fs*cos_alpha_theta, dt/M_times_L*Fs*cos_alpha_theta, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]])
  
  B = np.array ([[0, 0],
                 [0, 0],
                 [0, 0],
                 [0, 0],
                 [-dt*kk*sin_theta + dt*oo*sin_alpha_theta*cos_alpha, 0],
                 [dt*kk*cos_theta + dt*oo*sin_alpha_theta*sin_alpha, 0],
                 [-dt/M_times_L*sin_alpha_theta, 0],
                 [0, dt* l_over_J]])
  
  """
  print("A: \n", A)
  print("\nB: \n", B)
  print("\nCond A: ", np.linalg.cond(A))
  print("\nCond B: ", np.linalg.cond(B)) """ 
  
  return xxp, A, B

def hovering_equilibria(xdes, ydes):

    #Returns equilibria of the system for which the drone is in hovering position.
    #Position of the c.o.m xp and yp are degrees of freedom.

    xp_eq = xdes
    yp_eq = ydes
    alpha_eq = 0
    theta_eq = 0
    vx_eq = 0
    vy_eq = 0
    walpha_eq = 0
    wtheta_eq = 0
    #input forces
    Fs_eq = (MM+mm)*gg
    Fd_eq = 0

    xx_eq = [xp_eq, yp_eq, alpha_eq, theta_eq, vx_eq, vy_eq, walpha_eq, wtheta_eq]
    uu_eq = [Fs_eq, Fd_eq]

    return xx_eq, uu_eq
