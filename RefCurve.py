import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
import Dynamics as dyn #quadropter dynamics

ns = 8 #number of states
ni = 2 #number of inputs
step = 0.01

def ref_curve(y0, v0, a0, yf, vf, af, t0, tf): #poly5 curve generation

    p0 = 0
    p1 = v0
    p2 = a0/2
    t1 = tf-t0
    y1 = yf-y0
    b5 = y1-(p2*t1**2+p1*t1+p0)
    b4 = vf-(2*p2*t1+p1)
    b3 = af-2*p2
    p3 = (b3*t1**2-8*b4*t1+20*b5)/(2*t1**3)
    p4 = (-b3*t1**2+7*b4*t1-15*b5)/(t1**4)
    p5 = (b3*t1**2-6*b4*t1+12*b5)/(2*t1**5)
    i = 0
    t = 0
    counter = int((tf-t0)/step)
    y = np.zeros(counter)
    for i in range (0, counter, 1):
        ris =  p5*t**5+p4*t**4+p3*t**3+p2*t**2+p1*t+p0+ y0
        y[i] = ris
        t+= step
        
    return y
    
def build_reference_curves(xx_eq0, xx_eq1, uu_eq0, uu_eq1, TT): #two-segments reference curve

    state_references = [0,0,0,0,0,0,0,0]
    input_references = [0,0]

    for i in range(0,ns):
        init_const = ref_curve(xx_eq0[i], 0, 0, xx_eq0[i], 0, 0, 0, TT/2)
        final_const = ref_curve(xx_eq1[i], 0, 0, xx_eq1[i], 0, 0, TT/2, TT)
        state_references[i] = np.concatenate((init_const, final_const))
    
    for i in range(0,ni):
        init_const = ref_curve(uu_eq0[i], 0, 0, uu_eq0[i], 0, 0, 0, TT/2)
        final_const = ref_curve(uu_eq1[i], 0, 0, uu_eq1[i], 0, 0, TT/2, TT)
        input_references[i] = np.concatenate((init_const, final_const))

    return state_references, input_references

def build_smooth_reference_curves(xx_eq0, xx_eq1, uu_eq0, uu_eq1, TT, ref): #smooth reference curve
    
    state_references = [0,0,0,0,0,0,0,0]
    input_references = [0,0]

    for i in range(0,ns):
        init_const = ref_curve(xx_eq0[i], 0, 0, xx_eq0[i], 0, 0, 0, TT/3)
        middle_poly = ref_curve(xx_eq0[i], 0, 0, xx_eq1[i], 0, 0, TT/3, 2*TT/3)
        if ref == "Smooth":
            final_const = ref_curve(xx_eq1[i], 0, 0, xx_eq1[i], 0, 0, 2*TT/3, TT)
        elif ref == "DoubleS":
            final_const = ref_curve(xx_eq1[i], 0, 0, xx_eq0[i], 0, 0, 2*TT/3, TT)
        state_references[i] = np.concatenate((init_const, middle_poly, final_const))
    
    for i in range(0,ni):
        init_const = ref_curve(uu_eq0[i], 0, 0, uu_eq0[i], 0, 0, 0, TT/3)
        middle_poly = ref_curve(uu_eq0[i], 0, 0, uu_eq1[i], 0, 0, TT/3, 2*TT/3)
        final_const = ref_curve(uu_eq1[i], 0, 0, uu_eq1[i], 0, 0, 2*TT/3, TT)
        input_references[i] = np.concatenate((init_const, middle_poly, final_const))
    return state_references, input_references