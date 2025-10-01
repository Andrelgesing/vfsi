#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal
import scipy.optimize
import scipy.special
import General.Setup as Stp


def gamma_function(reynolds):
    """
    Function to determinate the Gamma Hydrodynamic function of circular beams, and extrapolate to rectangular
    cantilevers.
    The function returns only the cantilever hydrodynamic function.
    Enter with the Reynolds number (as vector or np.array)
    """
    tau = np.log10(reynolds)

    # Equation 21a - Real part of Omega
    gamma_real = (5 * reynolds - 15 * np.log(reynolds) + 8) / (80 * (reynolds + 1)) * \
                 (4.17950 - 0.25269 * tau + 2.88308 * tau ** 2 - 0.08680 * tau ** 3 + 0.33837 * tau ** 4
                  + 0.03318 * tau ** 5 + 0.001884 * tau ** 6)/(1 - 2.27659 * tau + 2.10179 * tau**2 - 0.11365 * tau**3
                                                               + 0.34989 * tau**4 + 0.03779*tau**5 + 0.01884 * tau*+6)

    # Equation 21b - Imaginary part of Omega
    gamma_imaginary = (0.41/np.sqrt(reynolds) + 1/reynolds) * \
                      (0.82494 - 0.66701*tau + 0.41150 * tau**2 - 0.16748 * tau**3 + 0.04897*tau**4 - 0.01107 * tau**5 +
                       0.00148 * tau**6) / (1 - 0.72962 * tau + 0.40663 * tau**2 - 0.16517 * tau**3 + 0.04907*tau**4
                                            - 0.01110*tau**5 + 0.00148 * tau**6)
    gamma = gamma_real + 1j * gamma_imaginary

    return gamma


class GreenMethod(object):
    """
    Class to calculate the dynamic response of torsional modes of cantilevers in viscous fluids.
    Based on the paper 'Torsional frequency response of cantilever beams immersed in viscous
    fluids with applications to the atomic force microscope', by Christopher E. Green and John E. Sader [1998].
    """

    def __init__(self):
        # Initialize parameters
        self.geometry = Stp.Geometry()
        self.fluid = Stp.Fluid()
        self.mat = Stp.Material()
        self.mu_c = self.mat.rho_c * self.geometry.w_c * self.geometry.t_c  # Mass per length unit
        self.k_phi = self.mat.g_c * self.geometry.k_c / self.geometry.l_c
        self.frequency = np.linspace(1e3, 1e4, 20)  # Frequency range
        self.kb = 1.38064852 * 1e-23
        self.T = 300
        self.N_modes = 4
        self.n_x = 100
        self.x = np.linspace(0, self.geometry.l_c, self.n_x + 1)  # x-axis discretization
        self.f_n = np.array([10, 10])
        self.kappa_n = np.pi/2
        self.gamma = np.zeros(1)
        self.alpha = np.zeros(1)
        self.a_f = np.zeros(1)
        self.phi_n = np.sin(self.kappa_n*self.x/self.geometry.l_c)
        self.w_th = np.zeros(1)

    def thermal_displacement(self):
        """
        Get the angle of the cantilever over frequency due to the thermal noise in the fluid.
        """
        self.geometry.i_c = self.geometry.w_c**3*self.geometry.t_c / 12  # Inertial moment
        self.geometry.k_c = self.geometry.w_c*self.geometry.t_c ** 3 / 12  # Inertial moment
        self.mu_c = self.mat.rho_c * self.geometry.w_c * self.geometry.t_c  # Mass per length unit
        self.k_phi = self.mat.g_c * self.geometry.k_c / self.geometry.l_c
        self.x = np.linspace(0, self.geometry.l_c, self.n_x)  # x-axis discretization
        self.fn_set()
        self.mode_shape()
        reynolds_num = (2 * np.pi * self.frequency) * self.geometry.w_c ** 2 / (4 * self.fluid.nu_f)
        self.gamma = gamma_function(reynolds_num)
        self.a_f = np.pi/2 * self.frequency/self.f_n[0]*np.sqrt(1 + np.pi*self.fluid.rho_f*self.geometry.w_c**4 /
                                                                (8*self.mat.rho_c*self.geometry.i_c)*self.gamma)
        self.alpha = 2/(self.kappa_n[:, np.newaxis]*(self.a_f[np.newaxis, :]**2 - self.kappa_n[:, np.newaxis]**2))
        int_alpha = np.trapz(np.abs(self.alpha) ** 2, self.frequency, axis=1)

        self.w_th = np.sqrt(2*np.pi*self.kb*self.T/self.k_phi*np.sum(np.abs(self.alpha[:, :, np.newaxis])**2 /
                                                                     (self.kappa_n[:, np.newaxis, np.newaxis]**2 *
                                                                      int_alpha[:, np.newaxis, np.newaxis])
                                                                     * self.phi_n[:, np.newaxis, :]**2, axis=0))
        return self.w_th

    def mode_shape(self):
        self.phi_n = np.sin(self.kappa_n[:, np.newaxis]*self.x[np.newaxis, :]/self.geometry.l_c)

    def fn_set(self):
        self.geometry.i_c = self.geometry.w_c**3*self.geometry.t_c / 12  # Inertial moment
        self.geometry.k_c = self.geometry.w_c*self.geometry.t_c ** 3 / 12  # Inertial moment
        nn = np.arange(1, self.N_modes+1, 1)
        self.kappa_n = (2*nn - 1)*np.pi/2
        self.f_n = self.kappa_n/self.geometry.l_c * np.sqrt(self.mat.g_c*self.geometry.k_c/(self.mat.rho_c
                                                                                            * self.geometry.i_c))

    def q_factor(self):
        fn_indices = scipy.signal.find_peaks(np.abs(self.w_th[:, -1]))[0]
        nq = len(fn_indices)
        q = np.zeros(nq)
        self.fn_set()
        for ff in range(nq):
            f_r = (1 + (np.pi * self.fluid.rho_f * self.geometry.w_c ** 4) /
                       (8 * self.mat.rho_c * self.geometry.i_c) *
                   np.real(self.gamma[fn_indices[ff]]))**(-0.5) * self.f_n[ff]
            for ii in range(4):
                reynolds_num = (2 * np.pi * f_r) * self.geometry.w_c ** 2 / (4 * self.fluid.nu_f)
                gamma_cant = gamma_function(reynolds_num)
                f_r = (1 + (np.pi * self.fluid.rho_f * self.geometry.w_c ** 4) /
                           (8 * self.mat.rho_c * self.geometry.i_c) * np.real(gamma_cant))**(-0.5) * self.f_n[ff]
                q[ff] = (8 * self.mat.rho_c * self.geometry.i_c / (np.pi * self.fluid.rho_f * self.geometry.w_c ** 4)
                         + np.real(gamma_cant)) / np.imag(gamma_cant)
        return q
