import numpy as np
import cvxpy as cp

def unconstrained_lqr(AA, BB, QQ, RR, QQf, xx0, T_hor):
    
    """
        LQR - given init condition and time horizon, optimal state-input trajectory

        Args
          - AA, BB: linear dynamics
          - QQ,RR,QQf: cost matrices
          - xx0: initial condition
          - T: time horizon
    """

    xx0 = xx0.squeeze()

    ns, ni = BB.shape

    xx_lqr = cp.Variable((ns, T_hor))
    uu_lqr = cp.Variable((ni, T_hor))

    cost = 0
    constr = []

    for tt in range(T_hor-1):
        cost += cp.quad_form(xx_lqr[:,tt], QQ) + cp.quad_form(uu_lqr[:,tt], RR)
        constr += [xx_lqr[:,tt+1] == AA@xx_lqr[:,tt] + BB@uu_lqr[:,tt]]
    # sums problem objectives and concatenates constraints.
    cost += cp.quad_form(xx_lqr[:,T_hor-1], QQf)
    constr += [xx_lqr[:,0] == xx0]
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()

    if problem.status == "infeasible":
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! ")

    return xx_lqr.value, uu_lqr.value

#  x1max, x1min, x2max, x2min ,
def linear_mpc( xx_opt, uu_opt,AA, BB, QQ, RR, QQf, xxt, T_pred,Fs_max, Fd_max,Xmax,Xmin,Ymax,Ymin,Alpha_max,Theta_max,
               vx_max,vx_min,vy_max,vy_min,wa_max,wa_min,wt_max,wt_min):
   
    """
        Linear MPC solver - Constrained LQR

        Given a measured state xxt measured at t
        gives back the optimal input to be applied at t

        Args
          - AA, BB: linear dynamics
          - QQ,RR,QQf: cost matrices
          - xxt: initial condition (at time t)
          - T: time (prediction) horizon
          ################
          - xx_opt founded optimal traj for the state to be tracked
          - uu_opt funded optiaml traj for the input to be tracked

        Returns
          - u_t: input to be applied at t
          - xx, uu predicted trajectory

    """

    #initialization
    xxt = xxt.squeeze()
    #print(BB.shape)
    ns, ni= BB.shape[:2]
    #print('ns \t=',ns,'ni\t=',ni)
    #print(BB)

    xx_mpc = cp.Variable((ns, T_pred+1))
    uu_mpc = cp.Variable((ni, T_pred+1))

    cost = 0
    constr = []

    for tt in range(T_pred-1):

        cost += cp.quad_form(xx_mpc[:,tt]-xx_opt[:,tt], QQ) + cp.quad_form(uu_mpc[:,tt]-uu_opt[:,tt], RR)
        #
        constr += [xx_mpc[:,tt+1]-xx_opt[:,tt+1] == (AA[:,:,tt]@(xx_mpc[:,tt]-xx_opt[:,tt]) + BB[:,:,tt]@(uu_mpc[:,tt]-uu_opt[:,tt])),# dynamics constraint
            uu_mpc[0,tt] <= Fs_max, # other constraints
            uu_mpc[0,tt] >= -Fs_max,
            uu_mpc[1,tt] <= Fd_max, # other constraints
            uu_mpc[1,tt] >= -Fd_max,                
            xx_mpc[0,tt] <= Xmax,
            xx_mpc[0,tt] >= Xmin,
            xx_mpc[1,tt] <= Ymax,
            xx_mpc[1,tt] >= Ymin,
            xx_mpc[2,tt] <= Alpha_max,
            xx_mpc[2,tt] >= -Alpha_max,
            xx_mpc[3,tt] <= Theta_max,
            xx_mpc[3,tt] >= -Theta_max]

    # sums problem objectives and concatenates constraints. #here we are adding the terminal cntraints
    cost += cp.quad_form(xx_mpc[:,T_pred]-xx_opt[:,T_pred], QQf)
    constr += [xx_mpc[:,0]==xxt]
    #constr +=[xx_mpc[:,T_pred]==0]
    
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()

    if problem.status == "infeasible":
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

    return uu_mpc[:,0].value, xx_mpc.value, uu_mpc.value