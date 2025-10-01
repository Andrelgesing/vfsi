#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fenics as fe
import numpy as np
import scipy.sparse as sp
import scipy.special as special
import sys
sys.path.insert(0, '..')
import Elastic.plate_fem as pt
import quadpy
import tqdm
from Fluid.h_matrix_app import get_a_matrix, get_a_matrix_uniform_mid
from Fluid.Fluid import Fluid, Szz
import time
from scipy.linalg import lu_factor, lu_solve
from petsc4py import PETSc

class F3D(object):
    """Class with the implementation of the 3D fluid-structure interaction problem.
    The unsteady stokeslet is used to calculate the fluid forces acting on the structure.
    Attributes:
        fluid (Fluid): Instance of the Fluid class.
        fem (InteriorPenalty): Instance of the InteriorPenalty class.
        geometry (Geometry): Geometry of the fluid-structure interaction problem.
        n_x_fluid (int): Number of cross-sections in the x-direction for the fluid grid.
        n_y_fluid (int): Number of points in the y-direction for the fluid grid.
        l_c (float): Characteristic length of the geometry.
        w_c (float): Characteristic width of the geometry.
        t_c (float): Characteristic thickness of the geometry.
        x (ndarray): Discretized x-coordinates for the fluid grid.
        xp (ndarray): Discretized x-coordinates for the pressure discretization.
        wx (ndarray): Weights for the x-coordinates.
        y (ndarray): Discretized y-coordinates for the fluid grid.
        yp (ndarray): Discretized y-coordinates for the pressure discretization.
        wy (ndarray): Weights for the y-coordinates.
        a (ndarray): Complex-valued matrix for calculations.
        w (ndarray): Float-valued matrix for calculations.
        p (ndarray): Complex-valued matrix for pressure calculations.
        tol_stk (float): Tolerance for the stokeslet calculations.
        x_left (ndarray): Left x-coordinates for the rectangular elements.
        x_right (ndarray): Right x-coordinates for the rectangular elements.
        y_low (ndarray): Lower y-coordinates for the rectangular elements.
        y_up (ndarray): Upper y-coordinates for the rectangular elements.
        recs (ndarray): Array of rectangular elements for integration.
        nb (int): Number of basis functions.
        scheme_coarse (quadpy.c2): Quadrature scheme for coarse integration.
        scheme_fine (quadpy.c2): Quadrature scheme for fine integration.
        scheme_finer (quadpy.c2): Quadrature scheme for finer integration.
        scheme_finer2 (quadpy.c2): Quadrature scheme for even finer integration.
        scheme_t (quadpy.t2): Quadrature scheme for triangular integration.
        a_bem (ndarray): Complex-valued matrix for boundary element method calculations.
        lam (complex): Complex-valued parameter for calculations.
        y_sqrt (ndarray): Square root of y-coordinates for calculations.
        x_sqrt (ndarray): Square root of x-coordinates for calculations.
        pbar (tqdm): Progress bar for calculations.

    Methods:
        __init__(): Initializes the F3D class with default parameters.
        gaussian_quad(ny): Computes the Chebyshev-Gauss quadrature points and weights.
        setup_quadrature(n_x_fluid, n_y_fluid): Defines the fluid grid for pressure discretization.
        get_h_force(omega): Computes the hydrodynamic force for a given frequency.
        get_a_matrix(omega): Returns the A matrix from Tuck (1969) for a given frequency.
        export_basis_functions(): Exports the basis functions for the fluid grid.
        around_singularity(ind_x, ind_y): Handles the integration around singularities.
        local_refinement(ind, err, val_fine, x0, y0): Performs local refinement for integration.
    """
    def __init__(self):
        # Initialize parameters
        #self.mat = pt.Material()
        self.fluid = Fluid()
        #self.fem = pt.Kirchhoff()
        #self.geometry = self.fem.geometry#pt.Geometry()
        self.n_x_fluid = 16
        self.n_y_fluid = 64
        self.l_c = 100e-6
        self.w_c = 50e-6
        self.t_c = 2e-6
        self.x = np.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.xp = np.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.wx = 1 / 3 * self.x
        self.y = np.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.yp = np.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.wy = 1 / 3 * self.x
        self.a = np.ones((10, 10), dtype=complex)
        self.w = np.ones((10, 10), dtype=float)
        self.p = np.ones((10, 10), dtype=complex)
        self.tol_stk = 2e-3
        self.x_left = np.tile(self.x[0:-1], len(self.yp))  # [0:n2]
        self.x_right = np.tile(self.x[1:], len(self.yp))  # [0:n2]
        self.y_low = np.repeat(self.y[0:-1], len(self.xp))  # [0:n2]
        self.y_up = np.repeat(self.y[1:], len(self.xp))  # [0:n2]
        recs = np.array([[np.array([self.x_left, self.y_low]), np.array([self.x_right, self.y_low])]
                            , [np.array([self.x_left, self.y_up]), np.array([self.x_right, self.y_up])]])
        self.recs = recs.transpose(0, 1, 3, 2)
        self.nb = 100
        self.scheme_coarse = quadpy.c2.get_good_scheme(2)
        self.scheme_fine = quadpy.c2.get_good_scheme(4)
        self.scheme_finer = quadpy.c2.get_good_scheme(6)
        self.scheme_finer2 = quadpy.c2.get_good_scheme(8)
        self.scheme_t = quadpy.t2.get_good_scheme(3)
        self.a_bem = np.ones((10, 10), dtype=complex)
        self.p = np.ones((10, 10), dtype=complex)
        self.lam = np.sqrt(-1j * 100 * self.l_c ** 2 / self.fluid.nu_f)
        self.y_sqrt = np.sqrt((self.w_c / 2) ** 2 - (self.yp * self.l_c) ** 2) / (self.w_c / 2)
        self.x_sqrt = np.sqrt(1 - self.xp ** 2)
        self.pbar = None
        self.scheme = quadpy.c1.gauss_legendre(8)
        self.w = self.scheme.weights
        self.theta_lin = (self.scheme.points)
        self.x_uniform = True
        self.print_time = False
        self.threshold = 1e-16
        self.sparse = False
        self.dof_corner = 10
        self.wT = None

    def gaussian_quad(self, ny):
        """
        Compute the Gaussian quadrature points.
        This function calculates the Chebyshev-Gauss quadrature points and weights for a given number of quadrature points.
        Parameters:
        ny (int): The number of quadrature points.

        Returns:
        tuple: A tuple containing:
            - y (numpy.ndarray): The array of quadrature points including endpoints -1 and 1.
            - yp (numpy.ndarray): The array of interior quadrature points.
        """
        i = np.arange(1, ny + 1)
        yp = (-np.cos(((2 * i - 1) / (2 * ny)) * np.pi))
        y = np.concatenate((np.array([-1]), 0.5 * (yp[1:] + yp[:-1]), np.array([1])))
        return y, yp
    
    def lin_quad(self, ny):
        """
        Compute the linear quadrature points and weights.

        This function calculates the linear quadrature points and weights for a given number of quadrature points.
        The points are evenly spaced between 0 and 1, including the endpoints. The midpoints between consecutive 
        points are also computed to serve as interior quadrature points.

        Parameters:
        ----------
        ny : int
            The number of quadrature points.

        Returns:
        -------
        tuple
            A tuple containing:
            - y (numpy.ndarray): Array of evenly spaced quadrature points, including endpoints 0 and 1.
            - yp (numpy.ndarray): Array of midpoints between consecutive quadrature points.
        """
        # Create an array of points from 0 to 1 with ny+1 points
        y = np.linspace(0, 1, ny + 1)
        # Calculate the interior points as midpoints between consecutive points
        yp = 1/2 * (y[1:] + y[:-1])
        # Return the points and midpoints
        return y, yp

    def setup_quadrature(self, n_x_fluid, n_y_fluid, x_uniform=False, y_uniform=False):
        """
        Define the fluid grid for the pressure discretization.
        Chebyshev-Gauss quadrature in both directions, unless the x_uniform keyword is set to True.

        Parameters
        ----------
        n_x_fluid : integer
                  Number of cross-sections in x-direction
                  Must be odd, if an even number is given, it is automatically set to n_x_fluid + 1
        n_y_fluid : integer
                  Number of points in y-direction
        """
        self.n_x_fluid = n_x_fluid
        self.n_y_fluid = n_y_fluid
        #self.l_c = self.geometry.l_c
        #self.w_c = self.geometry.w_c
        #self.t_c = self.geometry.t_c
        if x_uniform == False:
                self.x, self.xp = np.array(self.gaussian_quad(2 * n_x_fluid), dtype=object)[0][int(n_x_fluid)::], \
                np.array(self.gaussian_quad(2 * n_x_fluid), dtype=object)[1][int(n_x_fluid)::]
        else:    
            self.x, self.xp = np.array(self.lin_quad(n_x_fluid), dtype=object)
        self.x_uniform = x_uniform
        if y_uniform == False:
            self.y, self.yp = np.array(self.gaussian_quad(n_y_fluid), dtype=object) * self.w_c / 2 / self.l_c
        else:
            self.y, self.yp = (np.array(self.lin_quad(n_y_fluid), dtype=object)-0.5) * self.w_c / 2 / self.l_c
        #self.y, self.yp = np.array(self.gaussian_quad(n_y_fluid), dtype=object) * self.w_c / 2 / self.l_c
        self.x_left = np.tile(self.x[0:-1], len(self.yp))  # [0:n2]
        self.x_right = np.tile(self.x[1:], len(self.yp))  # [0:n2]
        self.y_low = np.repeat(self.y[0:-1], len(self.xp))  # [0:n2]
        self.y_up = np.repeat(self.y[1:], len(self.xp))  # [0:n2]
        recs = np.array([[np.array([self.x_left, self.y_low]), np.array([self.x_right, self.y_low])]
                            , [np.array([self.x_left, self.y_up]), np.array([self.x_right, self.y_up])]])
        self.recs = recs.transpose(0, 1, 3, 2)
        if self.x_uniform:
            self.wx = self.l_c / len(self.xp)*np.ones(len(self.xp))
        else:
            self.x_sqrt = np.sqrt(1 - self.xp ** 2)
            self.wx = self.l_c / 2 * np.pi / len(self.xp) * self.x_sqrt
        self.y_sqrt = np.sqrt((self.w_c / 2) ** 2 - (self.yp * self.l_c) ** 2) / (self.w_c / 2)
        self.wy =  self.w_c / 2 *np.pi/len(self.yp) * self.y_sqrt
        self.weight_int = np.repeat(self.wx, len(self.yp)) * np.tile(self.wy, len(self.xp)).flatten()
        self.wgt = np.asarray(self.weight_int, dtype=np.float64)
        #self.export_basis_functions()


    def get_p_force(self, omega):
        """
        Calculate the hydrodynamic force for a given angular frequency.

        This function computes the hydrodynamic force matrix by first obtaining 
        the A matrix for the given angular frequency, then inverting it and 
        performing a series of matrix operations to derive the pressure matrix.
        The result is stored as a sparse matrix in the instance variable `self.p`.

        Parameters:
        omega (float): The angular frequency at which to calculate the hydrodynamic force.

        Returns:
        scipy.sparse.csr_matrix: The computed hydrodynamic force matrix as a sparse matrix.
        """
        t_init = time.time()
        self.lam = np.sqrt(-1j * omega * self.l_c ** 2 / self.fluid.nu_f).astype(np.complex64)
        if self.x_uniform:
            a_bem = get_a_matrix_uniform_mid(self.xp, self.yp, self.x, self.y, self.lam)#[0]
        else:
            a_bem = get_a_matrix(self.xp, self.yp, self.x, self.y, self.lam)#[0]
        if self.print_time:
            tt = time.time() - t_init
            print('Determine H-matrix was %f seconds' % (tt))
        t_init = time.time()
        #if self.n_x_fluid*self.n_y_fluid < 128*129:
        #lu, piv = lu_factor(a_bem, overwrite_a=False, check_finite=False)
        # 2) Solve in column blocks to limit peak RAM
        # w is your sparse matrix
        #self.wT = self.w.T
        #if sp.isspmatrix_csr(self.wT):
        #    wT = wT.tocsc()
        #else:
        #    pass  # already good
        #t_init = time.time()
        #n = self.w.shape[1]
        #m = self.w.shape[0]
        #p_cpu = np.empty((n, m), dtype=np.complex64)
        #if self.n_x_fluid*self.n_y_fluid < 128*129:
        #    blocksize = m
        #else:
        #    blocksize = 2048

        #X = np.empty((a_bem.shape[0], m), order='F', dtype=a_bem.dtype)
        #for j in range(0, m, blocksize):
        #    k = min(j + blocksize, m)
        #    B = self.wT[:, j:k].toarray(order='F')
        #    X[:, j:k] = lu_solve((lu, piv), B, check_finite=False)

        #p = -self.fluid.mu_f / self.l_c * X
        p = -self.fluid.mu_f / self.l_c *np.linalg.solve(a_bem, self.w.T.toarray())
        #p = -self.fluid.mu_f / self.l_c * lu_solve((lu, piv), self.w.T.toarray())
        #del a_bem, lu, piv, X 
        if self.print_time:
            tt = time.time() - t_init
            print('Invert H-matrix was %f seconds' % (tt))


        #a_inv = np.linalg.inv(self.a_bem * self.l_c)
        #p = -self.fluid.mu_f*a_inv @ self.w.T
        #del a_inv
        t_init = time.time()  
        p = (p.T * self.weight_int).T
        
        #np.multiply(p, self.wgt[:, None], out=p) 
        if self.print_time:
            tt = time.time() - t_init
            print('First multiplication took %f seconds' % (tt))
        t_init = time.time()  
        p =  self.w @ p
        #p = self.w.dot(p)   
        if self.print_time:
            tt = time.time() - t_init
            print('Second multiplication took %f seconds' % (tt))
        #p_sl = sp.csr_matrix(p, dtype=complex)
        #self.p = p_sl
        return p

    
    


    
    