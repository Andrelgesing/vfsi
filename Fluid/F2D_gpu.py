#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fenics as fe
import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.special as special
import sys
sys.path.insert(0, '..')
import Elastic.plate_gpu as pt
from Fluid.Fluid_gpu import Fluid
       
class F2D(object):
    """
    A 2D fluid flow formulation class.

    This class initializes and manages the parameters and properties required 
    for simulating 2D fluid flow. It integrates fluid properties, finite element 
    methods (FEM), and geometry definitions to set up the computational domain 
    and associated variables.

    Attributes:
    ----------
    fluid : Fluid
        An instance of the Fluid class representing the fluid properties.
    fem : pt.Kirchhoff
        An instance of the Kirchhoff class for FEM computations.
    geometry : pt.Geometry
        The geometry of the computational domain.
    n_x_fluid : int
        Number of cross-sections in the x-direction for the fluid grid.
    n_y_fluid : int
        Number of points in the y-direction for the fluid grid.
    l_c : float
        Characteristic length of the geometry.
    w_c : float
        Characteristic width of the geometry.
    t_c : float
        Characteristic thickness of the geometry.
    x : numpy.ndarray
        Array of x-coordinates for the fluid grid.
    dx : float
        Grid spacing in the x-direction.
    wx : float
        Weighting factor for the x-direction grid.
    y : numpy.ndarray
        Array of y-coordinates for the fluid grid.
    wy : float
        Weighting factor for the y-direction grid.
    a : numpy.ndarray
        Placeholder for the complex-valued A matrix.
    w : numpy.ndarray
        Placeholder for the weight matrix.
    p : numpy.ndarray
        Placeholder for the pressure matrix.
    """
    def __init__(self):
        # Initialize parameters
        self.fluid = Fluid()
        self.fem = pt.Kirchhoff()
        self.geometry = self.fem.geometry
        self.n_x_fluid = 16
        self.n_y_fluid = 64
        self.l_c = self.geometry.l_c
        self.w_c = self.geometry.w_c
        self.t_c = self.geometry.t_c
        self.x = cp.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.dx = self.x[1] - self.x[0]
        self.wx = 1 / 3 * self.dx
        self.y = cp.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.wy = 1 / 3 * self.dx
        self.a = cp.ones((10, 10), dtype=complex)
        self.w = cp.ones((10, 10), dtype=float)
        self.p = cp.ones((10, 10), dtype=complex)

    def setup_quadrature(self, n_x_fluid, n_y_fluid):
        """
        Define the fluid grid for the pressure discretization.
        In x, 1/3 Simpson's rule is implemented. In Y, Chebyshev-Gauss quadrature.

        Parameters
        ----------
        n_x_fluid : integer
                  Number of cross-sections in x-direction
                  Must be odd, if an even number is given, it is automatically set to n_x_fluid + 1
        n_y_fluid : integer
                  Number of points in y-direction
        """
        if n_x_fluid % 2 == 0:
            self.n_x_fluid = n_x_fluid + 1
        else:
            self.n_x_fluid = n_x_fluid
        self.l_c = self.geometry.l_c
        self.w_c = self.geometry.w_c
        self.t_c = self.geometry.t_c
        self.n_y_fluid = n_y_fluid
        self.x = cp.linspace(0, self.l_c, self.n_x_fluid)
        self.dx = self.x[1] - self.x[0]
        i = cp.arange(1, self.n_y_fluid + 1)
        # self.y = -cp.cos( i / (self.n_y_fluid + 1) * cp.pi)*self.w_c/2 # First version, wrong
        self.y = -cp.cos(((2 * i - 1) / (2 * self.n_y_fluid)) * cp.pi)*self.w_c/2

        w_x = cp.ones(len(self.x))
        w_x[1:-1:2] = 4
        w_x[2:-2:2] = 2
        self.wx = 1/3*self.dx*w_x
        y_sqrt = cp.sqrt((self.w_c/2)**2 - self.y**2)
        self.wy = cp.pi / self.n_y_fluid * y_sqrt
        self.export_basis_functions()


    def get_h_force(self, omega):
        """
        Calculate the hydrodynamic force for a given angular frequency.

        This function computes the hydrodynamic force by first obtaining the 
        matrix 'a' for the given angular frequency 'omega'. It then calculates 
        the inverse of 'a' and constructs a block diagonal matrix from the 
        inverse. The function proceeds to compute the pressure matrix 'p' and 
        multiplies it with the weights to obtain the final hydrodynamic force.

        Parameters:
        omega (float): The angular frequency at which to compute the hydrodynamic force.

        Returns:
        numpy.ndarray: The computed hydrodynamic force.
        """
        self.get_a_matrix(omega)
        a_inv = cp.linalg.inv(self.a)
        a_dia = [a_inv for ii in range(self.n_x_fluid)]
        del a_inv
        a_inv_dia = sp.block_diag(a_dia, format='csr', dtype='complex64')
        del a_dia
        p = a_inv_dia @ self.w.T
        del a_inv_dia

        p_ = p.T.multiply(cp.repeat(self.wx, self.n_y_fluid).flatten() * cp.tile(self.wy, self.n_x_fluid).flatten()).T
        self.p = self.fluid.mu_f * self.w @ p_
        return self.p

    def f_func(self, z):
        """
        Computes a complex function involving special functions.

        Parameters:
        z (complex): A complex number input.

        Returns:
        complex: The result of the function (1 / z) + special.kerp(z) + 1j * special.keip(z).
        """
        return (1 / z) + special.kerp(z) + 1j * special.keip(z)

    def get_a_matrix(self, omega):
        """Returns the A matrix from Tuck (1969)

            Parameters
            ----------
            omega: float
                 frequency in radians per second
            Returns
            -------
            a : cp.array(type=cp.complex)
                the complex-valued A matrix
        """
        wp = self.w_c
        nu_f = self.fluid.nu_f
        y_lim = cp.concatenate((cp.array([-wp / 2]), 0.5 * (self.y[1:] + self.y[:-1]), cp.array([wp / 2])))
        z1 = cp.sqrt(omega / nu_f) * (y_lim[cp.newaxis, 1:] - self.y[:, cp.newaxis])
        z2 = cp.sqrt(omega / nu_f) * (y_lim[cp.newaxis, :-1] - self.y[:, cp.newaxis])
        f1 = self.f_func(z1)
        f1[z1 < 0] = -self.f_func(-z1[z1 < 0])
        f2 = self.f_func(z2)
        f2[z2 < 0] = -self.f_func(-z2[z2 < 0])
        self.a = 1 / (2 * cp.pi * 1j) * cp.sqrt(nu_f) / cp.sqrt(omega) * (f1 - f2)



    def export_basis_functions(self):
        """
        Export the basis functions for the fluid mesh.
        This function creates a rectangular mesh for the fluid domain, constructs a 
        function space on this mesh, and generates a transfer matrix to map the 
        basis functions from the original function space to the new function space. 
        The resulting transfer matrix is then used to reorder and store the basis 
        functions in a sparse matrix format.
        Returns:
            scipy.sparse.csr_matrix: A sparse matrix containing the reordered basis 
            functions.
        """
        mesh_fluid = fe.RectangleMesh(fe.Point(0, -self.geometry.w_c / 2),
                                      fe.Point(self.geometry.l_c, self.geometry.w_c / 2),
                                     self.n_x_fluid-1, self.n_y_fluid-1, 'left/right')
        (mesh_fluid.coordinates()[:, 1]).sort(axis=0)
        xf, yf = mesh_fluid.coordinates()[:, 0], mesh_fluid.coordinates()[:, 1]
        yf[:] = cp.repeat(self.y, len(self.x))s
        v2 = fe.FunctionSpace(mesh_fluid, 'CG', 1)
        self.fem.function_spaces()
        #transfer_matrix = fe.PETScDMCollection.create_transfer_matrix(self.fem.VCG, v2)
        transfer_matrix = fe.PETScDMCollection.create_transfer_matrix(self.fem.VCG, v2)
        arg_x = cp.argsort(v2.tabulate_dof_coordinates()[:, 0])
        coord_x = v2.tabulate_dof_coordinates()[arg_x, :]
        arg_y = cp.argsort(cp.reshape(coord_x[:, 1], (len(self.x), len(self.y))))
        arg_y2 = (arg_y.T + cp.arange(len(arg_y)) * (len(self.y))).T.flatten()
        row, col, val = fe.as_backend_type(transfer_matrix).mat().getValuesCSR()
        w = sp.csr_matrix((val, col, row), dtype='float64')
        self.w = (w[arg_x])[arg_y2].T
        return self.w