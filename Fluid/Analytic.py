#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal
import scipy.optimize
import scipy.special
import math
import Elastic.plate_fem as pt
from Fluid.Fluid import Fluid

def gamma_function(reynolds):
    """
    Calculate the Gamma Hydrodynamic function for circular and rectangular cantilevers.

    This function computes the hydrodynamic function for cantilevers based on the Reynolds number.
    It first calculates the Gamma function for circular beams and then extrapolates it to rectangular
    cantilevers using empirical equations.

    Parameters:
    -----------
    reynolds : array-like
        The Reynolds number, which can be provided as a vector or a numpy array.

    Returns:
    --------
    gamma_cant : numpy.ndarray
        The hydrodynamic function for rectangular cantilevers, represented as a complex array.

    Notes:
    ------
    - The calculation for circular beams is based on Equation 18, which uses Bessel functions.
    - The extrapolation to rectangular cantilevers is based on empirical equations (Equations 21a, 21b, and 22).
    - The real and imaginary parts of the hydrodynamic function are computed separately and combined.

    References:
    -----------
    The equations used in this function are derived from hydrodynamic modeling of cantilevers
    and are specific to the given Reynolds number range.
    """
    # Equation 18 - Tau for circular beams
    gamma_cylinder = 1 + 4 * 1j * \
        scipy.special.kv(1, -1j * np.sqrt(1j * reynolds)) / (np.sqrt(1j * reynolds) *
                                                             scipy.special.kv(0, -1j * np.sqrt(1j * reynolds)))
    # Equation 22
    tau = np.log10(reynolds)

    # Equation 21a - Real part of Omega
    omega_real = (0.91324 - 0.48274 * tau + 0.46842 * tau ** 2 - 0.12886 * tau ** 3
                  + 0.044055 * tau ** 4 - 0.0035117 * tau ** 5
                  + 0.00069085 * tau ** 6) / \
                 (1 - 0.56964 * tau + 0.48690 * tau ** 2 - 0.13444 * tau ** 3
                  + 0.045155 * tau ** 4 - 0.0035862 * tau ** 5 + 0.00069085 * tau ** 6)
    # Equation 21b - Imaginary part of Omega
    omega_imaginary = (-0.024134 - 0.029256 * tau + 0.016294 * tau ** 2 - 0.00010961 * tau ** 3
                       + 0.000064577 * tau ** 4 - 0.000045510 * tau ** 5) / \
                      (1 - 0.59702 * tau + 0.55182 * tau ** 2 - 0.18357 * tau ** 3
                       + 0.079156 * tau ** 4 - 0.014369 * tau ** 5 + 0.0028361 * tau ** 6)

    omega = omega_real + 1j * omega_imaginary
    gamma_cant = omega * gamma_cylinder

    return gamma_cant


class SaderMethod(object):
    """
    SaderMethod is a class designed to calculate the dynamic response of cantilevers in viscous fluids. 
    It is based on the paper 'Frequency response of cantilever beams immersed in viscous fluids with 
    applications to the atomic force microscope' by John E. Sader (1998).

    Attributes:
        geometry (Geometry): An instance of the Geometry class representing the cantilever's geometry.
        fluid (Fluid): An instance of the Fluid class representing the fluid properties.
        mat (Material): An instance of the Material class representing the cantilever's material properties.
        frequency (numpy.ndarray): A numpy array representing the frequency range for calculations.
        kb (float): Boltzmann constant in Joules per Kelvin.
        T (float): Temperature in Kelvin.
        T_Scaling (float): Scaling factor based on fluid and cantilever properties.
        N_modes (int): Number of modes to consider in the calculations.
        n_x (int): Number of discretization points along the cantilever's length.
        x (numpy.ndarray): Discretized x-axis along the cantilever's length.
        i_c (float): Inertial moment of the cantilever cross-section.
        k (float): Spring constant of the cantilever.
        f_n (numpy.ndarray): Array of natural frequencies of the cantilever.
        w_th (numpy.ndarray): Thermal displacement of the cantilever.
        mu_c (float): Mass per unit length of the cantilever.
        alpha (numpy.ndarray): Mode contribution factor.

    Methods:
        __init__():
            Initializes the SaderMethod class with default parameters.

        get_displacement_per_force():
            Calculates the displacement of the cantilever in the fluid per unit force.

        thermal_displacement():
            Calculates the thermal displacement of the cantilever in the fluid.

        get_roots():
            Finds the first Nn roots of the transcendental equation 1 + cos(c)cosh(c) = 0.

        fn_set():
            Calculates and sets the natural frequencies of the cantilever.

        q_factor():
            Calculates the quality factor (Q-factor) of the cantilever in the fluid.
    """
    def __init__(self):
        # Initialize parameters
        self.geometry = pt.Geometry()
        self.fluid = Fluid()
        self.mat = pt.Material()
        self.frequency = np.linspace(1e3, 1e4, 20)  # Frequency range
        self.kb = 1.38064852 * 1e-23
        self.T = 300
        self.T_Scaling = self.fluid.rho_f * self.geometry.w_c / self.mat.rho_c / self.geometry.t_c
        self.N_modes = 20
        self.n_x = 100
        self.x = np.linspace(0, self.geometry.l_c, self.n_x + 1)  # x-axis discretization
        self.i_c = self.geometry.w_c * self.geometry.t_c ** 3 / 12  # Inertial moment
        self.k = 3 * self.mat.e_c * self.i_c / self.geometry.l_c ** 3
        self.f_n = np.array([10, 10])
        self.w_th = np.array([10, 10])
        self.mu_c = self.mat.rho_c * self.geometry.w_c * self.geometry.t_c  # Mass per length unit
        self.alpha = self.frequency  # Mode contribution

    def get_displacement_per_force(self):
        """
        Calculate the displacement of the cantilever in the fluid per unit force.

        This method computes the displacement of a cantilever beam immersed in a fluid
        under the influence of an applied force. It considers the geometry of the cantilever,
        material properties, fluid properties, and dynamic effects such as Reynolds number
        and frequency response.

        Returns:
            np.ndarray: A 2D array representing the displacement of the cantilever 
                at different positions along its length for the given force.

        Notes:
            - The method uses the inertial moment, material stiffness, and mass per unit length
              of the cantilever to compute its dynamic response.
            - The Reynolds number and a gamma function are used to account for fluid-structure
              interaction effects.
            - The displacement is calculated using a series of mathematical expressions involving
              the roots of characteristic equations and scaling factors.
        """
        self.i_c = self.geometry.w_c * self.geometry.t_c ** 3 / 12  # Inertial moment
        self.T_Scaling = self.fluid.rho_f * self.geometry.w_c / self.mat.rho_c / self.geometry.t_c
        self.k = 3 * self.mat.e_c * self.i_c / self.geometry.l_c ** 3
        self.mu_c = self.mat.rho_c * self.geometry.w_c * self.geometry.t_c  # Mass per length unit
        self.x = np.linspace(0, self.geometry.l_c, self.n_x + 1)  # x-axis discretization
        c_roots = self.get_roots()
        self.fn_set()
        reynolds_num = (2 * np.pi * self.frequency) * self.geometry.w_c ** 2 / (4 * self.fluid.nu_f)
        gamma_cant = gamma_function(reynolds_num)
        b = c_roots[0] * np.sqrt(self.frequency / self.f_n[0]) * (
                1 + np.pi * self.fluid.rho_f * self.geometry.w_c ** 2 / 4 / self.mu_c * gamma_cant) ** (1 / 4)
        alpha = 2 * np.sin(c_roots) * np.tan(c_roots) / (c_roots * (c_roots ** 4 - b[:, np.newaxis] ** 4) *
                                                         (np.sin(c_roots) + np.sinh(c_roots)))
        self.alpha = alpha

        w01 = 1 / (2 * b ** 4 * (1 + np.cos(b) * np.cosh(b)))
        w02 = -2 - 2 * np.cos(b) * np.cosh(b) + np.cos(b * self.x[:, np.newaxis] / self.geometry.l_c) + \
              np.cosh(b * self.x[:, np.newaxis] / self.geometry.l_c) + \
              np.cos(b * ((1 - self.x / self.geometry.l_c)[:, np.newaxis])) * np.cosh(b) + \
              np.cos(b) * np.cosh(b * ((1 - self.x / self.geometry.l_c)[:, np.newaxis]))

        w03 = -np.sin(b * ((1 - self.x / self.geometry.l_c)[:, np.newaxis])) * np.sinh(b) + \
              np.sin(b) * np.sinh(b * ((1 - self.x / self.geometry.l_c)[:, np.newaxis]))

        w0 = w01 * (w02 + w03) * self.geometry.l_c ** 4 / self.mat.e_c / self.i_c
        return w0

    def thermal_displacement(self):
        """
        Get the displacement of the cantilever in the fluid.
        """
        self.i_c = self.geometry.w_c * self.geometry.t_c ** 3 / 12  # Inertial moment
        self.mu_c = self.mat.rho_c * self.geometry.w_c * self.geometry.t_c  # Mass per length unit
        self.k = 3 * self.mat.e_c * self.i_c / self.geometry.l_c ** 3
        self.x = np.linspace(0, self.geometry.l_c, self.n_x)  # x-axis discretization
        self.fn_set()
        reynolds_num = (2 * np.pi * self.frequency) * self.geometry.w_c ** 2 / (4 * self.fluid.nu_f)
        gamma_cant = gamma_function(reynolds_num)
        c_roots = self.get_roots()
        b = c_roots[0] * np.sqrt(self.frequency / self.f_n[0]) * (
                1 + np.pi * self.fluid.rho_f * self.geometry.w_c ** 2 / 4 / self.mu_c * gamma_cant) ** (1 / 4)
        alpha = 2 * np.sin(c_roots) * np.tan(c_roots) / (c_roots * (c_roots ** 4 - b[:, np.newaxis] ** 4) *
                                                         (np.sin(c_roots) + np.sinh(c_roots)))
        self.alpha = alpha
        int_alpha = np.trapz(np.abs(alpha) ** 2, self.frequency, axis=0)
        phi = (np.cos(c_roots * self.x[:, np.newaxis] / self.geometry.l_c) - np.cosh(c_roots * self.x[:, np.newaxis] /
                                                                                     self.geometry.l_c)
               + (np.cos(c_roots) + np.cosh(c_roots)) / (np.sin(c_roots) + np.sinh(c_roots))
               * (np.sinh(c_roots * self.x[:, np.newaxis] / self.geometry.l_c) - np.sin(
                    c_roots * self.x[:, np.newaxis] /
                    self.geometry.l_c)))
        w = (np.abs(alpha) ** 2 / c_roots ** 4 / int_alpha)[:, np.newaxis, :] * (phi[np.newaxis, :, :] ** 2)
        self.w_th = np.sqrt(3 * np.pi * self.kb * self.T / self.k * np.sum(w, axis=2))
        return self.w_th

    def get_roots(self):
        """
        Function to find Nn roots of the transcendental Equation 1+Cos(c)Cosh(c) =  0

        Returns
        ----------
        c_vec : numpy.array
                      First Nn roots of the transcendental equation.

        """

        def f_alpha(x_out):
            return math.cos(x_out) * math.cosh(x_out) + 1

        roots = np.zeros(self.N_modes)
        for nn in range(self.N_modes):
            roots[nn] = scipy.optimize.root(f_alpha, np.array((2 * nn + 1) * math.pi / 2))['x'][0]
        return roots

    def fn_set(self):
        self.i_c = self.geometry.w_c * self.geometry.t_c ** 3 / 12  # Inertial moment
        c_roots = self.get_roots()
        self.f_n = c_roots ** 2 / self.geometry.l_c ** 2 * np.sqrt(self.mat.e_c *
                                                                   self.i_c / (self.mat.rho_c *
                                                                                        self.geometry.t_c *
                                                                                        self.geometry.w_c)) / 2 / np.pi

    def q_factor(self):
        fn_indices = scipy.signal.find_peaks(np.abs(self.w_th[:, -1]))[0]
        #fn_indices = scipy.signal.find_peaks(np.abs(self.w0[-1, :]))[0]
        nq = len(fn_indices)
        q = np.zeros(nq)
        self.fn_set()
        for ff in range(nq):
            f_r = (1 + (np.pi * self.fluid.rho_f * self.geometry.w_c ** 2) /
                   (4 * self.mat.rho_c * self.geometry.w_c * self.geometry.t_c)) ** (-0.5) * self.f_n[ff]
            reynolds_num = (2 * np.pi * f_r) * self.geometry.w_c ** 2 / (4 * self.fluid.nu_f)
            gamma_cant = gamma_function(reynolds_num)

            q[ff] = (4 * self.mat.rho_c * self.geometry.w_c * self.geometry.t_c / (np.pi * self.fluid.rho_f *
                                                                                   self.geometry.w_c ** 2) + np.real(
                gamma_cant)) / np.imag(gamma_cant)
            for ii in range(4):
                reynolds_num = (2 * np.pi * f_r) * self.geometry.w_c ** 2 / (4 * self.fluid.nu_f)
                gamma_cant = gamma_function(reynolds_num)
                q[ff] = (4 * self.mat.rho_c * self.geometry.w_c * self.geometry.t_c / (np.pi * self.fluid.rho_f *
                                                                                       self.geometry.w_c ** 2)
                         + np.real(gamma_cant)) / np.imag(gamma_cant)
                f_r = (1 + (np.pi * self.fluid.rho_f * self.geometry.w_c ** 2) /
                       (4 * self.mat.rho_c * self.geometry.w_c * self.geometry.t_c) *
                       np.real(gamma_cant)) ** (-0.5) * self.f_n[ff]
        return q
