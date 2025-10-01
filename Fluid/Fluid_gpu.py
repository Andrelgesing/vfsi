#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import cupy as cp
import numpy as np

class Fluid(object):
    """
    Define fluid properties.

    Parameters
    ----------
    dynamic_viscosity : float
              absolute (or dynamic) viscosity [Ns/(m^2)]
    density:  float
              fluid density [kg/m³]
    """
    def __init__(self, dynamic_viscosity=8.9e-4, density=997):
        self.mu_f = dynamic_viscosity  # Fluid dynamic viscosity in Pa.s
        self.rho_f = density  # Fluid density in kg/m^3
        self.nu_f = self.mu_f / self.rho_f  # Fluid kinematic viscosity in m^2/s


def s_zz(r, lam, dx, dy, dz):
    """
    Calculate the Z-component of the unsteady stokeslet.

    This function computes the Z-component of the unsteady stokeslet based on the method described in:
    Pozrikidis, C. "A singularity method for unsteady linearized flow." 
    Physics of Fluids A: Fluid Dynamics 1.9 (1989): 1508-1520.

    Parameters:
    r (float): Radial distance.
    lam (float): Lambda parameter.
    dx (float): X-component of the distance.
    dy (float): Y-component of the distance.
    dz (float): Z-component of the distance.

    Returns:
    float: The Z-component of the unsteady stokeslet.
    """
    R = lam*r
    a_r = 2*np.exp(-R)*(1 + 1/R + 1/R**2) - 2/R**2
    b_r = -2*np.exp(-R)*(1 + 3/R + 3/R**2) + 6/R**2
    s_33 = 1/np.pi/8*(a_r/r + b_r*dz*dz/r**3)
    return s_33

def Szz(xx, x0, y0, lam, dz):
    """
    Calculate the numerical integral of the Z-component of the unsteady stokeslet.

    This function computes the Z-component of the unsteady stokeslet at a given point (xx) 
    by integrating over the distance components.

    Parameters:
    xx (tuple): Coordinates of the point where the Z-component is calculated (x, y).
    x0 (float): X-coordinate of the source point.
    y0 (float): Y-coordinate of the source point.
    lam (float): Lambda parameter.
    dz (float): Z-component of the distance.

    Returns:
    float: The Z-component of the unsteady stokeslet at the given point.
    """
    x = xx[0]
    y = xx[1]
    dx = x - x0
    dy = y - y0
    r__ = np.sqrt(dx**2 + dy**2 + dz**2, dtype=np.float64)
    #r__ = np.asarray(r__, dtype=np.float64)
    R = lam*r__
    a_r = 2*np.exp(-R)*(1 + 1/R + 1/R**2) - 2/R**2
    b_r = -2*np.exp(-R)*(1 + 3/R + 3/R**2) + 6/R**2
    s_33 = 1/np.pi/8*(a_r/r__ + b_r*dz*dz/r__**3)
    #stk_zz = s_zz(r__, lam, dx, dy, dz)
    return s_33
