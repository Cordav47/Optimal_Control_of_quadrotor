#LQR solver

import numpy as np

def lti_LQR(AA, BB, QQ, RR, SS, QQT, TT, qq, rr, qqT, ns, ni):

  #matrix and vector initialization
  KK = np.zeros((ni,ns,TT))
  sigma = np.zeros((ni,1,TT))
  PP = np.zeros((ns,ns,TT))
  pp = np.zeros((ns,1,TT))
  
  PP[:,:,-1] = QQT
  pp[:,:,-1] = qqT  #pp[:,:,-1] = qq[:,:,-1], alternative
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):

    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]
    PPtp = PP[:,:,tt+1]
    pptp = pp[:,:,tt+1]
    qqt = qq[:,:,tt]
    rrt = rr[:,:,tt]

    KK[:,:,tt] = -np.linalg.inv((RRt+ BBt.T@PPtp@BBt))@(SSt+ BBt.T@PPtp@AAt)

    sigma[:,:,tt] = -np.linalg.inv((RRt+ BBt.T@PPtp@BBt))@(rrt+BBt.T@pptp)

    pp[:,:,tt] = qqt + AAt.T@pptp - KK[:,:,tt].T@(RRt+ BBt.T@PPtp@BBt)@sigma[:,:,tt]
    
    PP[:,:,tt] = QQt + AAt.T@PPtp@AAt - KK[:,:,tt].T@(RRt+ BBt.T@PPtp@BBt)@KK[:,:,tt]

  return KK, PP, sigma