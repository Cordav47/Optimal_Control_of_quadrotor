# Optimal Control [OPTCON] 2023
# Main Quadroptor Project
# Andrea Perna
# Davide Corroppoli
# Riccardo Marras

#######################################
# Libraries
#######################################

import numpy as np
import math as mt
from copy import copy
import scipy as sp
import matplotlib.pyplot as plt
import Visualization_drone as vis #drone visualization
import RefCurve as refc
from scipy.optimize import fsolve
from scipy.linalg import solve_discrete_are
from scipy.integrate import solve_ivp
from LQR_LTI_Solver import lti_LQR
import signal
import Plot_functions as plot

import Dynamics as dyn #quadropter dynamics
import cost as cst #cost functions

import LQR_LTI_Solver as lqr #LQR solver
import MPC_Solver as mpc

#Allow Ctrl-C to work despite plotting
signal.signal(signal.SIGINT, signal.SIG_DFL) 
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 16})

#######################################
# Parameters
#######################################

#constants
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

#-general parameters-
max_iters = int(10) #before int(3e2)
stepsize_0 = 1
term_cond = 1e-8
closed_loop = True
do_LQR = True
do_MPC = True

#-Dynamics-
ns = dyn.ns #number of states
ni = dyn.ni #number of inputs

#-Plots & Visualization
animations = True
plots = True
show_references = False
plot_names = ['XP','YP','ALPHA','THETA','VX','VY','WALPHA','WTHETA','FS','FD']

#-Reference Curve-
ref = "Smooth" #Choose either Step, Smooth or DoubleS
tf1 = 10  #duration of step curve in seconds
tf2 = 30 #duration of smooth curve in seconds
if ref == "Step": tf = tf1
if ref == "Smooth": tf = tf2
if ref == "DoubleS": tf = tf2
dt = dyn.dt #discretization

#time definition
time = np.arange(0, tf, dt)

#-Armijo's parameters-
cc = 0.5
beta = 0.7
armijo_maxiters = 20 # number of Armijo iterations
visu_armijo = True
print_armijo = True
do_Armijo = True
visu_armijo_sampling = 10
stepsize = stepsize_0

#######################################
# Equilibrium Points
#######################################

#first hovering equilibrium point
[xp_eq_hov0, yp_eq_hov0] = [0,0] #desidered c.o.m. position in plane
[xx_eq0, uu_eq0] = dyn.hovering_equilibria(xp_eq_hov0,yp_eq_hov0)
print("\n")
print("First state equilibrium: ", xx_eq0)
print("First input equilibrium: ", uu_eq0)

#second hovering equilibrium point
[xp_eq_hov1, yp_eq_hov1] = [2,1] #desidered c.o.m. position in plane
[xx_eq1, uu_eq1] = dyn.hovering_equilibria(xp_eq_hov1,yp_eq_hov1)
print("\n")
print("Second state equilibrium: ", xx_eq1)
print("Second input equilibrium: ", uu_eq1)

#######################################
# Reference Curves
#######################################

#get state reference curves between the two chosen equilibria
if ref == "Step":
  xx_ref_step, uu_ref_step = refc.build_reference_curves(xx_eq0, xx_eq1, uu_eq0, uu_eq1, tf) #step reference curve
  xx_ref = xx_ref_step
  uu_ref = uu_ref_step
  xx_init, uu_init = refc.build_reference_curves(xx_eq0, xx_eq0, uu_eq0, uu_eq0, tf)
  TT = int(tf/dt) # discrete-time samples

elif ref == "Smooth":
  xx_ref_smooth, uu_ref_smooth = refc.build_smooth_reference_curves(xx_eq0, xx_eq1, uu_eq0, uu_eq1, tf, ref) #smooth reference curve
  xx_ref = xx_ref_smooth
  uu_ref = uu_ref_smooth
  xx_init, uu_init = refc.build_smooth_reference_curves(xx_eq0, xx_eq0, uu_eq0, uu_eq0, tf, ref)
  TT = int(tf/dt) # discrete-time samples

elif ref == "DoubleS":
  xx_ref_smooth, uu_ref_smooth = refc.build_smooth_reference_curves(xx_eq0, xx_eq1, uu_eq0, uu_eq1, tf, ref) #smooth reference curve
  xx_ref = xx_ref_smooth
  uu_ref = uu_ref_smooth
  xx_init, uu_init = refc.build_smooth_reference_curves(xx_eq0, xx_eq0, uu_eq0, uu_eq0, tf, ref)
  TT = int(tf/dt) # discrete-time samples

#convert references from list type to numpy array
xx_ref = np.asarray(xx_ref, dtype=np.float32)
uu_ref = np.asarray(uu_ref, dtype=np.float32)

#########################################
# Initial Plots
#########################################

if plots: 
  plot.myPlot(time, plot_names, np.concatenate((xx_ref, uu_ref))) #plot each reference curve

#######################################
# Costs
#######################################

if ref == "Step":

  QQt = np.eye(ns) #diagunal cost matrix of states
  QQt[XP,XP] = 1e2
  QQt[YP,YP] = 1e2
  QQt[ALPHA,ALPHA] = 1e2
  QQt[THETA,THETA] = 1e2
  QQt[VX,VX] = 1e1
  QQt[VY,VY] = 1e1
  QQt[WALPHA,WALPHA] = 10
  QQt[WTHETA,WTHETA] = 10

  RRt = np.eye(ni) #diagunal cost matrix of inputs
  RRt[FS,FS] = 1e-2
  RRt[FD,FD] = 1e-2

  QQT = np.eye(ns) #cost matrix for the final state
  QQT[XP,XP] = 1e1
  QQT[YP,YP] = 1e2
  QQT[ALPHA,ALPHA] = 10e2
  QQT[THETA,THETA] = 10e2
  QQT[VX,VX] = 10
  QQT[VY,VY] = 10
  QQT[WALPHA,WALPHA] = 10
  QQT[WTHETA,WTHETA] = 10

if ref == "Smooth":

  QQt = np.eye(ns) #diagunal cost matrix of states
  QQt[XP,XP] = 20
  QQt[YP,YP] = 45
  QQt[ALPHA,ALPHA] = 5
  QQt[THETA,THETA] = 5
  QQt[VX,VX] = 4
  QQt[VY,VY] = 5
  QQt[WALPHA,WALPHA] = 20
  QQt[WTHETA,WTHETA] = 4

  RRt = np.eye(ni) #diagunal cost matrix of inputs
  RRt[FS,FS] = 1
  RRt[FD,FD] = 2

  QQT = np.eye(ns) #cost matrix for the final state
  QQT[XP,XP] = 15
  QQT[YP,YP] = 30
  QQT[ALPHA,ALPHA] = 2
  QQT[THETA,THETA] = 2
  QQT[VX,VX] = 4
  QQT[VY,VY] = 4
  QQT[WALPHA,WALPHA] = 8
  QQT[WTHETA,WTHETA] = 2

if ref == "DoubleS":

  QQt = np.eye(ns) #diagunal cost matrix of states
  QQt[XP,XP] = 20
  QQt[YP,YP] = 45
  QQt[ALPHA,ALPHA] = 5
  QQt[THETA,THETA] = 5
  QQt[VX,VX] = 4
  QQt[VY,VY] = 5
  QQt[WALPHA,WALPHA] = 20
  QQt[WTHETA,WTHETA] = 4

  RRt = np.eye(ni) #diagunal cost matrix of inputs
  RRt[FS,FS] = 1
  RRt[FD,FD] = 2

  QQT = np.eye(ns) #cost matrix for the final state
  QQT[XP,XP] = 15
  QQT[YP,YP] = 30
  QQT[ALPHA,ALPHA] = 2
  QQT[THETA,THETA] = 2
  QQT[VX,VX] = 4
  QQT[VY,VY] = 4
  QQT[WALPHA,WALPHA] = 8
  QQT[WTHETA,WTHETA] = 2

######################################
# Initial guess
######################################

#convert initial guess from list type to numpy array
xx_init = np.asarray(xx_init, dtype=np.float32)
uu_init = np.asarray(uu_init, dtype=np.float32)

######################################
# Data Arrays' Initialization
######################################

x0 = xx_ref[:,0] #select the first column containing all initial states
xx = np.zeros((ns, TT, max_iters))   # state seq., multi-dimensional
uu = np.zeros((ni, TT, max_iters))   # input seq., multi-dimensional
lmbd = np.zeros((ns, TT, max_iters)) # lambdas - costate seq.
dJ = np.zeros((ni,TT, max_iters))     # DJ - gradient of J wrt u
JJ = np.zeros(max_iters)      # collect cost
descent = np.zeros(max_iters) # collect descent direction
descent_arm = np.zeros(max_iters) # collect descent direction
deltau = np.zeros((ni,TT, max_iters)) # Du - descent direction

###################################################
# Main
###################################################

print('-*-*-*-*-*-')
print("\n")
print("Maximum number of iterations:", max_iters)
print("\n\ndo Netwon's Method\n")

kk = 0 #first iteration

#initialization of state and input vectors
xx[:,:,0] = xx_init
uu[:,:,0] = uu_init

#Newton's Method
for kk in range(max_iters-1):

  #OPTIMIZATION METHOD'S PROCEDURE
  #1) Cost function computation
  #2) Descent direction computation
  #3) Stepsize selection - Armijo
  #4) Armijo's plot
  #5) Current solution's update
  #6) Termination Condition

  #matrices 'initialization
  AA=np.zeros((ns,ns,TT)) #(8,8)
  BB=np.zeros((ns,ni,TT)) #(8,2)
  QQ=np.zeros((ns,ns,TT)) #(8,8)
  RR=np.zeros((ni,ni,TT)) #(2,2)
  SS=np.zeros((ni,ns,TT)) #(2,8)
  qq=np.zeros((ns,1,TT))  #(8,1)
  rr=np.zeros((ni,1,TT))  #(8,1)
  deltax=np.zeros((ns,1,TT)) #(8,1) 

  KK = np.zeros((ni,ns,TT)) #(2,8)
  PP = np.zeros((ns,ns,TT)) #(8,8)
  sigma = np.zeros((ni,1,TT)) #(2,1) #vector for deltau

  ## ##################################################################
  ## 1) Cost Function Computation
  ## ##################################################################

  JJ[kk] = 0
  for tt in range(TT-1):
    temp_cost = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[0]
    JJ[kk] += temp_cost
  
  temp_cost = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[0]  
  JJ[kk] += temp_cost #total stage cost


  ## ##################################################################
  ## 2) Descent Direction Computation
  ## ##################################################################

  for tt in range(TT):
    AA[:,:,tt], BB[:,:,tt] = dyn.dynamics(xx[:,tt,kk], uu[:,tt,kk])[1:] #gradients of the dynamics
    qq[:,:,tt], rr[:,:,tt] = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[1:] #gradients of the cost

    #cost matricies
    QQ[:,:,tt] = QQt
    RR[:,:,tt] = RRt

  #Starting Condition of Co-State Equation
  lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[1]  #gradient of the terminal cost
  lmbd[:,TT-1,kk] = lmbd_temp.squeeze()
  qqT = lmbd_temp #gradient of final cost

  #Gradient Computation - First Order Equations for Optimality
  for tt in reversed(range(TT-1)):  # integration backward in time
    lmbd_temp = AA[:,:,tt].T@lmbd[:,tt+1,kk][:,None] + qq[:,:,tt]
    dJ_temp = - BB[:,:,tt].T@lmbd[:,tt+1,kk][:,None] - rr[:,:,tt]

    lmbd[:,tt,kk] = lmbd_temp.squeeze()
    dJ[:,tt,kk] = dJ_temp.squeeze()

  #Riccati Equation
  KK, PP, sigma = lqr.lti_LQR(AA,BB,QQ,RR,SS,QQT,TT,qq,rr,qqT, ns, ni)

  #Affine LQR
  for tt in range(TT-1):
    deltau[:,tt,kk] = (KK[:,:,tt] @ deltax[:,:,tt] + sigma[:,:,tt]).squeeze()
    deltax[:,:,tt+1] = AA[:,:,tt] @ deltax[:,:,tt] + BB[:,:,tt] @ deltau[:,tt,kk,None]

  #Descent Direction
  for tt in range(TT-1):
    descent[kk] += deltau[:,tt,kk].T @ deltau[:,tt,kk]
    descent_arm[kk] += dJ[:,tt,kk].T @ deltau[:,tt,kk]

  ## ##################################################################
  ## 3) Stepsize selection - Armijo
  ## ##################################################################
    
  stepsize = stepsize_0

  if do_Armijo:

    stepsizes = []  # list of stepsizes
    costs_armijo = []

    ii_Armijo = 0
    
    for ii_Armijo in range(armijo_maxiters):

      # temp solution update
      xx_temp = np.zeros((ns,TT))
      uu_temp = np.zeros((ni,TT))
      xx_temp[:,0] = x0

      for tt in range(TT-1):
        uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt,kk]
        xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

      # temp cost calculation
      JJ_temp = 0
      for tt in range(TT-1):
        temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[0]
        JJ_temp += temp_cost

      temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1], QQT)[0]
      JJ_temp += temp_cost

      stepsizes.append(stepsize)      # save the stepsize
      costs_armijo.append(JJ_temp)    # save the cost associated to the stepsize

      if JJ_temp > JJ[kk] - cc*stepsize*descent_arm[kk]:
          # update the stepsize
          stepsize = beta*stepsize
      else:
          #print('Armijo stepsize = {:.3e}'.format(stepsize)) 
          print('Armijo stepsize = {}\t ii= {}'.format(stepsize,ii_Armijo))
          break

  ## ##################################################################
  ## 4) Armijo's plot
  ## ##################################################################

  if visu_armijo: 

    steps = np.linspace(0,stepsize_0,int(visu_armijo_sampling))  #Graph Granularity
    costs = np.zeros(len(steps))

    for ii_Armijo in range(len(steps)):

      step = steps[ii_Armijo]

      # temp solution update
      xx_temp = np.zeros((ns,TT))
      uu_temp = np.zeros((ni,TT))

      xx_temp[:,0] = x0

      for tt in range(TT-1):
        uu_temp[:,tt] = uu[:,tt,kk] + step*deltau[:,tt,kk]
        xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

      # temp cost calculation
      JJ_temp = 0

      for tt in range(TT-1):
        temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[0]
        JJ_temp += temp_cost

      temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1], QQT)[0]
      JJ_temp += temp_cost

      costs[ii_Armijo] = JJ_temp 

    plt.figure('Armijo PLOT')
    plt.clf()
    plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
    plt.plot(steps, JJ[kk] - descent_arm[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    plt.plot(steps, JJ[kk] - cc*descent_arm[kk]*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

    plt.scatter(stepsizes, costs_armijo, marker='*', label='Selection of stepsizes') # plot the tested stepsize

    plt.grid()
    plt.xlabel('stepsize')
    plt.ylabel('$J(\\mathbf{u}^k,{stepsize})$')
    plt.legend()
    plt.draw()
    
    plt.title('Stepsize selection, number of iteration: {}' .format(ii_Armijo+1))
    plt.show(block=False)
    plt.pause(1)

  ## ##################################################################
  ## 5) Current solution's update
  ## ##################################################################
    
  xx_temp = np.zeros((ns,TT))
  uu_temp = np.zeros((ni,TT))
  xx_temp[:,0] = x0

  if closed_loop: #closed-loop update

    for tt in range(TT-1):
      uu_temp[:,tt] = uu[:,tt,kk] + (stepsize * sigma[:,:,tt]).squeeze() + KK[:,:,tt] @ (xx_temp[:,tt] - xx[:,tt,kk])
      xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

  else: #normal update

    for tt in range(TT-1):
      uu_temp[:,tt] = uu[:,tt,kk] + stepsize * deltau[:,tt,kk]
      xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

  xx[:,:,kk+1] = xx_temp
  uu[:,:,kk+1] = uu_temp

  #save an intermediate trajectory
  if(kk == 2):
    xx_interm = xx_temp
    uu_interm = uu_temp

  ## ##################################################################
  ## 6) Termination condition
  ## ##################################################################

  print('Iter = {}\t Descent = {}\t Cost = {}'.format(kk,descent_arm[kk],JJ[kk]))

  if descent[kk] <= term_cond:
    max_iters = kk
    break

#optimal trajectory
print("Newton's Method Done.")
xx_star = xx[:,:,max_iters-1]
uu_star = uu[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2] # for plotting purposes

#####################################
# Plots
#####################################

if plots:

  #descent direction
  plt.figure('descent direction')
  plt.plot(np.arange(max_iters), descent_arm[:max_iters])
  plt.xlabel('$k$')
  plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
  plt.yscale('log')
  plt.grid()
  plt.show(block=True)

  #cost function
  plt.figure('cost')
  plt.plot(np.arange(max_iters), JJ[:max_iters])
  plt.xlabel('$k$')
  plt.ylabel('$J(\\mathbf{u}^k)$')
  plt.yscale('log')
  plt.grid()
  plt.show(block=True)

  '''
  #intermediate trajectories
  plot.mySubplot(time, ['XP','YP','ALPHA','THETA'], xx_interm, xx_ref, 4, True)
  plot.mySubplot(time, ['VX','VY','WALPHA','WTHETA'], xx_interm[4:], xx_ref[4:], 4, True)
  plot.mySubplot(time, ['FS','FD'], uu_interm, uu_ref, 2, True)
  '''

  #optimal trajectory
  plot.mySubplot(time, ['XP','YP','ALPHA','THETA'], xx_star, xx_ref, 4, True)
  plot.mySubplot(time, ['VX','VY','WALPHA','WTHETA'], xx_star[4:], xx_ref[4:], 4, True)
  plot.mySubplot(time, ['FS','FD'], uu_star, uu_ref, 2, True)

#####################################
# Linear Quadratic Regulator (LQR)
#####################################

if do_LQR and ref == "Smooth": #Linear Quadratic Regulator
  
  print("\ndo LQR.")
  #matices initialization
  xx_reg = np.zeros((ns,TT))
  uu_reg = np.zeros((ni,TT))
  
  #change the initial state
  xx_0 = copy(xx_ref[:,0])
  
  #initial states' disturbances
  xx_0[ALPHA] = 0.3
  xx_0[THETA] = 0.25
  xx_reg[:,0] = xx_0
  
  #matrices 'initialization
  AA=np.zeros((ns,ns,TT)) #(8,8)
  BB=np.zeros((ns,ni,TT)) #(8,2)
  QQ=np.zeros((ns,ns,TT)) #(8,8)
  RR=np.zeros((ni,ni,TT)) #(2,2)
  SS=np.zeros((ni,ns,TT)) #(2,8)
  qq=np.zeros((ns,1,TT))  #(8,1)
  rr=np.zeros((ni,1,TT))  #(8,1)
  KK = np.zeros((ni,ns,TT)) #(2,8)
  
  for tt in range(TT):
    AA[:,:,tt], BB[:,:,tt] = dyn.dynamics(xx_star[:,tt], uu_star[:,tt])[1:] #gradients of the dynamics
    qq[:,:,tt], rr[:,:,tt] = cst.stagecost(xx_star[:,tt], uu_star[:,tt], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[1:] #gradients of the cost

    #cost matricies
    QQ[:,:,tt] = QQt
    RR[:,:,tt] = RRt

  
  #Starting Condition of Co-State Equation
  lmbd_temp = cst.termcost(xx_star[:,TT-1], xx_ref[:,TT-1], QQT)[1]  #gradient of the terminal cost
  lmbd[:,TT-1,kk] = lmbd_temp.squeeze()
  qqT = lmbd_temp #gradient of final cost

  #Riccati Equation
  KK_LQR = lqr.lti_LQR(AA,BB,QQ,RR,SS,QQT,TT,qq,rr,qqT, ns, ni)[0]

  #LQR computation
  for tt in range(TT-1): 
    uu_reg[:,tt] = uu_star[:,tt] + KK_LQR[:,:,tt] @ (xx_reg[:,tt] - xx_star[:,tt])
    xx_reg[:,tt+1] = dyn.dynamics(xx_reg[:,tt], uu_reg[:,tt])[0]

  uu_reg[:,-1] = uu_reg[:,-2]

  if plots:

    #optimal LQR trajectories
    plot.mySubplot(time, ['XP','YP','ALPHA','THETA'], xx_reg, xx_star, 4, True)
    plot.mySubplot(time, ['VX','VY','WALPHA','WTHETA'], xx_reg[4:], xx_star[4:], 4, True)
    plot.mySubplot(time, ['FS','FD'], uu_reg, uu_star, 2, True)

    #tracking error plots
    plot.mySubplot(time, ['XPerr','YPerr','ALPHAerr','THETAerr'], abs(xx_reg-xx_star), xx_star, 4, False)
    plot.mySubplot(time, ['VXerr','VYerr','WALPHAerr','WTHETAerr'], abs(xx_reg[4:]-xx_star[4:]), xx_star[4:], 4, False)
    plot.mySubplot(time, ['FSerr','FDerr'], abs(uu_reg-uu_star), uu_star, 2, False)

  print("\nLQR done.")

#####################################
# Animation (Task3)
#####################################

if (animations and do_LQR and ref=="Smooth"):
  fig = plt.figure('Quadrotor Animation with LQR Optimal Trajectory')
  vis.visual(xx_reg[XP, :],xx_reg[YP, :],xx_reg[THETA, :],xx_reg[ALPHA, :], (xp_eq_hov1, yp_eq_hov1), fig)

#####################################
# Model Predictive Control (MPC)
#####################################

if do_MPC and ref == "Smooth": #Model Predictive Control

  # state cost
  QQ = np.eye(ns) 
  QQT = 10*QQ

  # input cost
  r = 5e-1
  RR = r*np.eye(ni) 

  xx_0 = xx_ref[:,0] #select the first column containing all initial states
  #xx_0[THETA] = 0.1
  Tsim = TT  #simulation horizon
  T_pred = 20 #MPC Prediction horizon

  #here we will set the inequality contraints (as boundaies) for the state vars
  Fs_max=10
  Fd_max=0.5
  Xmax=2.1
  Xmin=-0.1
  Ymax=1.1
  Ymin=-0.1
  Alpha_max=0.5
  Theta_max=0.5
  vx_max=2
  vx_min=0.1
  vy_max=2
  vy_min=0.1
  wa_max=0.05
  wa_min=-0.05
  wt_max=0.05
  wt_min=-0.05

  #mpc trajectories' initialization
  xx_real_mpc = np.zeros((ns,Tsim))
  uu_real_mpc = np.zeros((ni,Tsim))
  xx_opt = np.zeros((ns,Tsim+T_pred))
  uu_opt = np.zeros((ni,Tsim+T_pred))

  xx_mpc = np.zeros((ns, T_pred, Tsim))
  xx_real_mpc[:,0] = xx_0.squeeze() #initial condition has 0 index
  
  #linearized dynamics WRT founded optimal trajectory
  AA_mpc=np.zeros((ns,ns,TT+T_pred))
  BB_mpc=np.zeros((ns,ni,TT+T_pred))

  #mpc's main loop
  for tt in range(Tsim+T_pred):

    if(tt<=Tsim-1):
      AA_mpc[:,:,tt], BB_mpc[:,:,tt] = dyn.dynamics(xx_star[:,tt], uu_star[:,tt])[1:] #gradients of the dynamics
      xx_opt[:,tt]=xx_star[:,tt]
      uu_opt[:,tt]=uu_star[:,tt]
    else:
      AA_mpc[:,:,tt]=AA_mpc[:,:,Tsim-1]
      BB_mpc[:,:,tt]=BB_mpc[:,:,Tsim-1]
      xx_opt[:,tt]=xx_star[:,Tsim-1]
      uu_opt[:,tt]=uu_star[:,Tsim-1]

  print(AA_mpc.shape)
  print(BB_mpc.shape)

  for tt in range(Tsim-1): #Model Predictive Control

    # System evolution - real with MPC
    xx_t_mpc = xx_real_mpc[:,tt] # get initial condition

    #solve MPC problem
    if tt%5 == 0: # print every 5 time instants
      print('MPC:\t t = {}'.format(tt))

    # Solve MPC problem - compute optimal trajectory - apply first input - compute the dynamics ## Fs_max, Fd_max, ,
    uu_real_mpc[:,tt]= mpc.linear_mpc(xx_opt[:,tt:(tt+T_pred+1)],uu_opt[:,tt:(tt+T_pred+1)],AA_mpc[:,:,tt:(tt+T_pred+1)],BB_mpc[:,:,tt:(tt+T_pred+1)],QQ,RR,QQT,xx_t_mpc,T_pred,Fs_max,Fd_max,
                                      Xmax,Xmin,Ymax,Ymin,Alpha_max,Theta_max,vx_max,vx_min,vy_max,vy_min,wa_max,wa_min,wt_max,wt_min)[0]#,Alpha_max,Theta_max)[0]
    xx_real_mpc[:,tt+1] = dyn.dynamics(xx_real_mpc[:,tt], uu_real_mpc[:,tt])[0]
    
  if plots:

    #optimal trajectories
    plot.mySubplot(time, ['XP','YP','ALPHA','THETA'], xx_real_mpc, xx_star, 4, True)
    plot.mySubplot(time, ['VX','VY','WALPHA','WTHETA'], xx_real_mpc[4:], xx_star[4:], 4, True)
    plot.mySubplot(time, ['FS','FD'], uu_real_mpc, uu_star, 2, True)

    #tracking error plots
    plot.mySubplot(time, ['XPerr','YPerr','ALPHAerr','THETAerr'], abs(xx_real_mpc-xx_star), xx_star, 4, False)
    plot.mySubplot(time, ['VXerr','VYerr','WALPHAerr','WTHETAerr'], abs(xx_real_mpc[4:]-xx_star[4:]), xx_star[4:], 4, False)
    plot.mySubplot(time, ['FSerr','FDerr'], abs(uu_real_mpc-uu_star), uu_star, 2, False)

    print("\nMPC done.")

#####################################
# Animation (Task4)
#####################################

if (animations and do_MPC and ref=="Smooth"):
  fig = plt.figure('Quadrotor Animation with MPC Optimal Trajectory')
  vis.visual(xx_real_mpc[XP, :],xx_real_mpc[YP, :],xx_real_mpc[THETA, :],xx_real_mpc[ALPHA, :], (xp_eq_hov1, yp_eq_hov1), fig)