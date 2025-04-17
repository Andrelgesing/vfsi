#+from numba import jit
import numpy as np
import scipy.special as spc


def sing_a(theta, y1, r, lam_):
    f_int = (-(y1 - r*np.sqrt(1 - np.cos(theta)**2))/(4*np.pi*y1*lam_**2*r)
             -(-1+np.exp(lam_*r)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_)
             -(spc.expi(lam_ * r * (-1))
             -spc.expi(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             + (y1 - r*np.sqrt(1-np.cos(theta)**2)*np.exp(lam_*r)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2))
         + lam_*r*y1*np.exp(lam_*r)*spc.expi(lam_ * r * (-1))
         - lam_*r*y1*np.exp(lam_*r)*
         spc.expi(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_**2*r*y1))
    return f_int

def sing_b(theta, x1, r, lam_):
    f_int = (-(x1 - r*np.sqrt(1 - np.sin(theta)**2))/(4*np.pi*x1*lam_**2*r)
             -(-1+np.exp(lam_*r)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_)
             -(spc.expi(lam_ * r * (-1))
             -spc.expi(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2)))/(4*np.pi*lam_)
             +(x1 - r*np.sqrt(1-np.sin(theta)**2)*np.exp(lam_*r)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2))
         + lam_*r*x1*np.exp(lam_*r)*spc.expi(lam_ * r * (-1))
         - lam_*r*x1*np.exp(lam_*r)*
         spc.expi(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_**2*r*x1))
    return f_int


# Dummy call to force compilation
# Dummy calls to force compilation
dummy_theta = np.array([0.1])
dummy_x1 = np.array([1.0])
dummy_x2 = np.array([1.5])
dummy_y1 = np.array([1.0])
dummy_y2 = np.array([1.5])
lam_ = 10.1 + 1j*10
_ = sing_a(dummy_theta, dummy_x1, dummy_y2, lam_)
_ = sing_b(dummy_theta, dummy_x1, dummy_y2, lam_)

print("Functions compiled and cached.")
