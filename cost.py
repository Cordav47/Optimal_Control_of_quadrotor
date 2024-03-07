# Cost function

#libraries
import numpy as np
import Dynamics as dyn

ns = dyn.ns #number of states
ni = dyn.ni #number of inputs

def stagecost(xx,uu, xx_ref, uu_ref, QQt, RRt): #stage-cost

  """
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

    Args
      - xx \in \R^8 state at time t
      - xx_ref \in \R^8 state reference at time t

      - uu \in \R^2 input at time t
      - uu_ref \in \R^2 input reference at time t


    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """

  xx = xx[:,None]
  uu = uu[:,None]

  xx_ref = xx_ref[:,None]
  uu_ref = uu_ref[:,None]

  #quadratic cost function
  ll = 0.5*(xx - xx_ref).T @ QQt @ (xx - xx_ref) + 0.5*(uu - uu_ref).T @ RRt @ (uu - uu_ref)

  #gradients
  lx = QQt@(xx - xx_ref) #gradient of l wrt x
  lu = RRt@(uu - uu_ref) #gradient of l wrt u

  return ll, lx, lu #return stage cost and gradients wrt x and u

def termcost(xx,xx_ref, QQT): #terminal cost
  """
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """

  xx = xx[:,None]
  xx_ref = xx_ref[:,None]

  #terminal quadratic cost function
  llT = 0.5*(xx - xx_ref).T@ QQT @ (xx - xx_ref)

  #gradient of the terminal cost wrt xT
  lTx = QQT @ (xx - xx_ref)

  #HERE I'VE ADDED SQUEEZE IN BOTH lx AND lu
  return llT, lTx #returns terminal cost and gradient wrt x