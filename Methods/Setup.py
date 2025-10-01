#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt


class Geometry(object):
    """
      Define geometry of the plate.

      Parameters
      ----------
    length: float
          length of the cantilever in meters
    width:  float
          width of the cantilever in meters
    thickness: float
          thickness of the cantilever in meters
    """
    def __init__(self, l_c=400e-6, w_c=100e-6, t_c=10e-6):
        self.l_c = l_c  # Length of the plate [m]
        self.w_c = w_c  # Width of the plate [m]
        self.t_c = t_c  # Thickness of the plate [m]
        self.a_r = self.l_c / self.w_c  # Aspect ratio of the plate [-]
        self.i_c = self.t_c ** 3 / 12  # Inertial moment
        self.k_c = self.w_c*self.t_c ** 3 / 3  # Inertial moment [Only applicable for Green's method]


class Material(object):
    """
    Define geometry of the cantilever.

    Parameters
    ----------
    youngs_modulus: float
                   Young's modulus of the material of the cantilever [Pa]
    density:  float
              material density [kg/m³]
    poisson: float
            Poisson coefficient
    """

    def __init__(self, youngs_modulus=169E9, density=2.33E3, poisson=0.3):
        self.e_c = youngs_modulus
        self.rho_c = density
        self.nu_c = poisson
        self.g_c = self.e_c/2/(1 + self.nu_c)  # Shear modulus


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

    # def frequency_range(self, lower_limit, upper_limit, number_of_steps=100, log_scale=True):
    #     """ Define frequency discretization
    #     Parameters
    #     ----------
    #     lower_limit : float
    #                   Inferior limit of the frequency range [Hz]
    #     upper_limit : float
    #                   Upper limit of the frequency range [Hz]
    #     number_of_steps : int
    #                   Number of frequency steps in the frequency range
    #     log_scale : bool (Optional)
    #                 If True, use logarithmic scale. If False, use linear.
    #                 Standard is logarithmic scale.
    #     """
    #     self.f_low = lower_limit
    #     self.f_up = upper_limit
    #     self.n_f = number_of_steps
    #     if log_scale:
    #         self.f = np.logspace(np.log10(self.f_low), np.log10(self.f_up), self.n_f)
    #     else:
    #         self.f = np.linspace(self.f_low, self.f_up, self.n_f)
