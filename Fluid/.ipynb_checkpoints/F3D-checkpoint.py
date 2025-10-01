#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fenics as fe
import numpy as np
import scipy.sparse as sp
import scipy.special as special
import sys
sys.path.insert(0, '..')
import Elastic.plate_fem as pt
from Fluid.Fluid import Fluid, Szz
import quadpy
import tqdm


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
        self.fem = pt.Kirchhoff()
        self.geometry = self.fem.geometry#pt.Geometry()
        self.n_x_fluid = 16
        self.n_y_fluid = 64
        self.l_c = self.geometry.l_c
        self.w_c = self.geometry.w_c
        self.t_c = self.geometry.t_c
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
        self.scheme = quadpy.c1.gauss_legendre(15)
        self.w = self.scheme.weights
        self.theta_lin = (self.scheme.points)
        self.x_uniform = False

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

    def setup_quadrature(self, n_x_fluid, n_y_fluid, x_uniform=False):
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
        self.l_c = self.geometry.l_c
        self.w_c = self.geometry.w_c
        self.t_c = self.geometry.t_c
        if x_uniform == False:
                self.x, self.xp = np.array(self.gaussian_quad(2 * n_x_fluid), dtype=object)[0][int(n_x_fluid)::], \
                np.array(self.gaussian_quad(2 * n_x_fluid), dtype=object)[1][int(n_x_fluid)::]
        else:    
            self.x, self.xp = np.array(self.lin_quad(n_x_fluid), dtype=object)
        self.x_uniform = x_uniform
        self.y, self.yp = np.array(self.gaussian_quad(n_y_fluid), dtype=object) * self.w_c / 2 / self.l_c
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
        self.export_basis_functions()

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
        if self.x_uniform:
            self.get_h_matrix_uniform_x_discretization(omega)
        else:
            self.get_h_matrix(omega)
        a_inv = np.linalg.inv(self.a_bem * self.l_c)
        p = -self.fluid.mu_f*a_inv @ self.w.T
        del a_inv
        p = (p.T * (np.repeat(self.wx, len(self.yp)) * np.tile(self.wy, len(self.xp)).flatten())).T
        p =  self.w @ p
        p_sl = sp.csr_matrix(p, dtype=complex)
        self.p = p_sl
        return p_sl

    def sing_a(self, theta, y1, r, lam_):
        """
        Computes the 1D integral resulting from the Szz integration between a circle 
        of radius `r` and an edge at a distance `y1` from the circle.

        This function evaluates a complex mathematical expression involving 
        exponential integrals and trigonometric functions.

        Parameters:
        -----------
        theta : float
            The angle in radians used in the trigonometric calculations.
        y1 : float
            The distance from the edge to the circle.
        r : float
            The radius of the circle.
        lam_ : float
            A parameter (lambda) used in the exponential and integral terms.

        Returns:
        --------
        float
            The computed value of the 1D integral.
        """
        f_int = (-(y1 - r*np.sqrt(1 - np.cos(theta)**2))/(4*np.pi*y1*lam_**2*r)
                -(-1+np.exp(lam_*r)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_)
                -(special.expi(lam_ * r * (-1))
                -special.expi(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
                + (y1 - r*np.sqrt(1-np.cos(theta)**2)*np.exp(lam_*r)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2))
            + lam_*r*y1*np.exp(lam_*r)*special.expi(lam_ * r * (-1))
            - lam_*r*y1*np.exp(lam_*r)*
            special.expi(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_**2*r*y1))
        return f_int

    def sing_b(self, theta, x1, r, lam_):
        """
        Compute the 1D integral for Szz integration between a circular region 
        of radius `r` and an edge located at a distance `x1` from the circle.

        This function evaluates a complex mathematical expression involving 
        exponential integrals and trigonometric functions to handle singularities 
        in the integration process.

        Parameters:
        ----------
        theta : float
            The angle in radians used in trigonometric calculations.
        x1 : float
            The distance from the edge to the circle.
        r : float
            The radius of the circular region.
        lam_ : float
            A parameter (lambda) used in exponential and integral terms.

        Returns:
        -------
        float
            The computed value of the 1D integral as a float.
        """
        f_int = (-(x1 - r*np.sqrt(1 - np.sin(theta)**2))/(4*np.pi*x1*lam_**2*r)
                -(-1+np.exp(lam_*r)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_)
                -(special.expi(lam_ * r * (-1))
                -special.expi(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2)))/(4*np.pi*lam_)
                +(x1 - r*np.sqrt(1-np.sin(theta)**2)*np.exp(lam_*r)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2))
            + lam_*r*x1*np.exp(lam_*r)*special.expi(lam_ * r * (-1))
            - lam_*r*x1*np.exp(lam_*r)*
            special.expi(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_**2*r*x1))
        return f_int


    def get_h_matrix(self, omega):
        """Returns the hydrodynamc matrix H, whch arises from the numerical integration of the unsteady stokeslet.

            Parameters
            ----------
            omega: float
                 frequency in radians per second
            Returns
            -------
            h : np.array(type=np.complex)
                the complex-valued H matrix
        """
        # Initialize the hydrodynamic matrix with zeros
        h_bem = np.zeros(([len(self.yp) * len(self.xp), len(self.yp) * len(self.xp)]), dtype=np.complex64)
        
        # Calculate the lambda parameter based on the given frequency
        self.lam = np.sqrt(-1j * omega * self.l_c ** 2 / self.fluid.nu_f)
        
        nn = 0
        
        # Loop over the x-coordinates
        for ii in range(len(self.xp)):
            x0 = self.xp[ii] 
            
            # Loop over half of the y-coordinates
            for jj in range(int(len(self.yp)/2)):
                y0 = self.yp[jj]
                
                # Perform coarse and fine integration
                val_coarse = self.scheme_coarse.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), self.recs)
                val_fine = self.scheme_fine.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), self.recs)
                
                # Calculate the error between coarse and fine integration
                err = np.abs(val_coarse - val_fine) / np.abs(val_fine)
                err[np.where(np.abs(val_fine) / np.max(np.abs(val_fine)) <= 1e-10)] = 0
                val_fine[np.where(np.abs(val_fine) / np.max(np.abs(val_fine)) <= 1e-10)] = 0
                
                # Handle the integration around singularities
                x1 = -(self.x[ii] - x0)
                x2 = self.x[ii+1] - x0
                y1 = -(self.y[jj] - y0)
                y2 = self.y[jj+1] - y0
                xx = np.array([x1, x1, x2, x2])
                yy  = np.array([y1, y2, y1, y2])
                theta_c = np.arctan(yy/xx)
                x_ = np.tile(np.array([xx]).T, len(self.theta_lin))
                y_ = np.tile(np.array([yy]).T, len(self.theta_lin))
                r_min = np.min(np.abs([x1, x2, y1, y2]))
                val_circle = ((-4 * np.pi * self.lam * r_min - 4 * np.pi) 
                        * np.exp(-self.lam * r_min) / self.lam ** 2 / r_min + 4 * np.pi / self.lam ** 2 / r_min) / 8 / np.pi
                val_sing = 0
                for iii in range(4):
                    val_sing += (self.scheme.integrate(lambda theta: self.sing_a(theta, y_[iii], r_min, self.lam), 
                                                      [theta_c[iii], np.pi/2]) +
                                                      self.scheme.integrate(lambda theta: self.sing_b(theta, x_[iii], r_min, self.lam),
                                                                             [theta_c[iii], 0]))
                
                I = val_sing + val_circle
                val_fine[jj*len(self.xp) + ii] = I
                err[jj*len(self.xp) + ii] = self.tol_stk/10
                 
                # Perform finer integration if the error is above the tolerance
                ind = err > self.tol_stk
                recs2 = self.recs[:, :, ind, :]
                val_finer = self.scheme_finer.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), recs2)
                err[ind] = np.abs(val_fine[ind] - val_finer) / np.abs(val_finer)
                val_fine[ind] = val_finer
                
                # Perform even finer integration if the error is still above the tolerance
                ind = err > self.tol_stk
                recs2 = self.recs[:, :, ind, :]
                val_finer = self.scheme_finer2.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), recs2)
                err[ind] = np.abs(val_fine[ind] - val_finer) / np.abs(val_finer)
                val_fine[ind] = val_finer
                
                # Perform local refinement if the error is still above the tolerance
                ind = np.where(err.flatten() > self.tol_stk)[0]
                if len(ind) > 0:
                    val_out, l2_out = self.local_refinement(ind, err[ind], val_fine[ind], x0, y0)
                    val_fine[ind] = val_out
                    err[ind] = l2_out
                
                # Reshape the fine integration results and update the hydrodynamic matrix
                STK_0 = val_fine.reshape(len(self.yp), len(self.xp))
                h_bem[nn, :] = STK_0.T.flatten()
                h_bem[nn + int(len(self.yp)) - 2 * jj - 1, :] = np.flip(STK_0.T, 1).flatten()
                
                nn += 1
            
            
            # Skip the remaining iterations if the loop is halfway through the y-coordinates
            if jj == int(len(self.yp) / 2) - 1:
                nn += int(len(self.yp) / 2)
        
        # Store the hydrodynamic matrix in the instance variable
        self.a_bem = h_bem
        return h_bem

    def export_basis_functions(self):
        """
        Export basis functions for the fluid mesh.

        This function creates a rectangular mesh for the fluid domain, sorts the mesh coordinates,
        and constructs a transfer matrix to map basis functions from one function space to another.
        The resulting transfer matrix is then used to create a sparse matrix representation of the
        basis functions.

        Steps:
        1. Create a rectangular mesh for the fluid domain.
        2. Sort the mesh coordinates.
        3. Define the function space on the fluid mesh.
        4. Create a transfer matrix between the function spaces.
        5. Sort the degrees of freedom coordinates.
        6. Construct a sparse matrix representation of the basis functions.

        Attributes:
            mesh_fluid (fe.Mesh): The rectangular mesh for the fluid domain.
            xf (np.ndarray): The x-coordinates of the mesh points.
            yf (np.ndarray): The y-coordinates of the mesh points.
            v2 (fe.FunctionSpace): The function space on the fluid mesh.
            transfer_matrix (fe.PETScDMCollection): The transfer matrix between function spaces.
            arg_x (np.ndarray): Indices to sort the x-coordinates of the degrees of freedom.
            coord_x (np.ndarray): Sorted coordinates of the degrees of freedom.
            arg_y (np.ndarray): Indices to sort the y-coordinates of the degrees of freedom.
            arg_y2 (np.ndarray): Flattened indices for sorting the y-coordinates.
            row (np.ndarray): Row indices of the CSR matrix.
            col (np.ndarray): Column indices of the CSR matrix.
            val (np.ndarray): Values of the CSR matrix.
            w0 (fe.Function): A function in the original function space.
            self.nb (int): The size of the vector in the original function space.
            w (sp.csr_matrix): The sparse matrix representation of the basis functions.
            self.w (sp.csr_matrix): The transposed and sorted sparse matrix of basis functions.
        """
        mesh_fluid = fe.RectangleMesh(fe.Point(0, -self.geometry.w_c / 2),
                                      fe.Point(self.geometry.l_c, self.geometry.w_c / 2),
                                     self.n_x_fluid-1, self.n_y_fluid-1, 'left/right')
        (mesh_fluid.coordinates()[:, 1]).sort(axis=0)
        xf, yf = mesh_fluid.coordinates()[:, 0], mesh_fluid.coordinates()[:, 1]
        yf[:] = np.repeat(self.yp * self.l_c, len(self.xp))
        xf[np.argsort(xf)] = np.repeat(self.xp * self.l_c, len(self.yp))
        xf = np.repeat(self.xp * self.l_c, len(self.yp))
        # fe.plot(mesh_fluid)
        v2 = fe.FunctionSpace(mesh_fluid, 'CG', 1)
        self.fem.function_spaces()
        # transfer_matrix = fe.PETScDMCollection.create_transfer_matrix(self.fem.VCG, v2)
        transfer_matrix = fe.PETScDMCollection.create_transfer_matrix(self.fem.VCG, v2)
        # print(transfer_matrix.array().shape)

        arg_x = np.argsort(v2.tabulate_dof_coordinates()[:, 0])
        coord_x = v2.tabulate_dof_coordinates()[arg_x, :]
        arg_y = np.argsort(np.reshape(coord_x[:, 1], (len(self.xp), len(self.yp))))
        arg_y2 = (arg_y.T + np.arange(len(arg_y)) * (len(self.yp))).T.flatten()

        # print(arg_y2.shape)
        row, col, val = fe.as_backend_type(transfer_matrix).mat().getValuesCSR()
        w0 = fe.Function(self.fem.VCG)
        self.nb = w0.vector().size()
        w = sp.csr_matrix((val, col, row), shape=(len(self.xp) * len(self.yp), self.nb), dtype='float64')
        self.w = (w[arg_x])[arg_y2].T


    def local_refinement(self, ind, err, val_fine, x0, y0):
        """
        Perform local refinement of the mesh based on error estimates.

        Parameters:
        ind (array-like): Indices where the error exceeds the tolerance.
        err (array-like): Array of error values.
        val_fine (array-like): Array of fine values to be refined.
        x0 (float): X-coordinate of the reference point.
        y0 (float): Y-coordinate of the reference point.

        Returns:
        tuple: A tuple containing:
            - val_prev (array-like): Refined values after local refinement.
            - err_ (array-like): Updated error values after local refinement.
        """
        # ind = np.where(l2_error.flatten() > tol)[0]
        x_left = self.recs[0, 0, ind, 0]
        x_right = self.recs[0, 1, ind, 0]
        y_low = self.recs[0, 0, ind, 1]
        y_up = self.recs[1, 0, ind, 1]
        x_left_init = x_left
        x_right_init = x_right
        y_low_init = y_low
        y_up_init = y_up
        # val_fine = STK.flatten()
        # err = l2_error.flatten()
        err_ = err  # [ind]
        ind_ = np.where(err_ > self.tol_stk)[0]
        # val = val_fine[ind]
        val_prev = val_fine  # [ind]
        nn = 2
        # val_prev[np.where(np.abs(val_prev) == 0)[0]] = 1

        while np.max(err_) >= self.tol_stk:
            for ii in range(len(ind_)):
                if abs(x_right[ii] - x_left[ii]) > abs(y_up[ii] - y_low[ii]):
                    nny = nn
                    nnx = int(nn * (x_right[ii] - x_left[ii]) / (y_up[ii] - y_low[ii]) / 4) + 1
                else:
                    nnx = nn
                    nny = int(nn * (y_up[ii] - y_low[ii]) / (x_right[ii] - x_left[ii]) / 4) + 1
                x_left2 = np.tile(np.linspace(x_left[ii], x_right[ii], nnx)[:nnx - 1], nny - 1)
                x_right2 = np.tile(np.linspace(x_left[ii], x_right[ii], nnx)[1:nnx], nny - 1)
                y_low2 = np.repeat(np.linspace(y_low[ii], y_up[ii], nny)[:nny - 1], nnx - 1)
                y_up2 = np.repeat(np.linspace(y_low[ii], y_up[ii], nny)[1:], nnx - 1)

                recs2 = np.array([[np.array([x_left2, y_low2]), np.array([x_right2, y_low2])]
                                     , [np.array([x_left2, y_up2]), np.array([x_right2, y_up2])]]).transpose(0, 1, 3, 2)
                if ii == 0:
                    recs_f = recs2
                    len_ = [0, len(x_left2) - 1]
                else:
                    recs_f = np.concatenate([recs_f, recs2], axis=2)
                    len_.append(len_[-1] + 1)
                    len_.append(len_[-1] + len(x_left2) - 1)

            val_ = self.scheme_fine.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), recs_f)
            val = np.ones(len(ind_), dtype=complex)
            for ii in range(len(ind_)):
                val[ii] = np.sum(val_[len_[2 * ii]:len_[2 * ii + 1] + 1])
            val_prev[np.where(np.abs(val_prev[ind_]) == 0)] = 1e-14 #  added this to avoid error by dividing by zero
            # it does not change any result
            err_[ind_] = np.abs(val - val_prev[ind_]) / np.abs(val_prev[ind_])
            val_prev[ind_] = val
            ind_ = np.where(err_ > self.tol_stk)[0]
            x_left = x_left_init[ind_]
            x_right = x_right_init[ind_]
            y_low = y_low_init[ind_]
            y_up = y_up_init[ind_]
            nn += 1
        return val_prev, err_
    

    def get_h_matrix_uniform_x_discretization(self, omega):
        """Returns the hydrodynamc matrix H, whch arises from the numerical integration of the unsteady stokeslet.
            Here, an uniform x-discretization scheme is considered, which allows for a more optimized determination of the H-matrix. 
            Parameters
            ----------
            omega: float
                 frequency in radians per second
            Returns
            -------
            h : np.array(type=np.complex)
                the complex-valued H matrix
        """
        
        
        # Calculate the lambda parameter based on the given frequency
        self.lam = np.sqrt(-1j * omega * self.l_c ** 2 / self.fluid.nu_f)
        
        nn = 0
        h_initial = np.zeros((len(self.yp), len(self.yp), 2*len(self.xp)-1), dtype=np.complex64)
        ii = 0
        x0 = self.xp[ii] 
            
            # Loop over half of the y-coordinates
        for jj in range(int(len(self.yp)/2)):
            y0 = self.yp[jj]
            
            # Perform coarse and fine integration
            val_coarse = self.scheme_coarse.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), self.recs)
            val_fine = self.scheme_fine.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), self.recs)
            
            # Calculate the error between coarse and fine integration
            err = np.abs(val_coarse - val_fine) / np.abs(val_fine)
            err[np.where(np.abs(val_fine) / np.max(np.abs(val_fine)) <= 1e-10)] = 0
            val_fine[np.where(np.abs(val_fine) / np.max(np.abs(val_fine)) <= 1e-10)] = 0
            
            # Handle the integration around singularities
            x1 = -(self.x[ii] - x0)
            x2 = self.x[ii+1] - x0
            y1 = -(self.y[jj] - y0)
            y2 = self.y[jj+1] - y0
            xx = np.array([x1, x1, x2, x2])
            yy  = np.array([y1, y2, y1, y2])
            theta_c = np.arctan(yy/xx)
            x_ = np.tile(np.array([xx]).T, len(self.theta_lin))
            y_ = np.tile(np.array([yy]).T, len(self.theta_lin))
            r_min = np.min(np.abs([x1, x2, y1, y2]))
            val_circle = ((-4 * np.pi * self.lam * r_min - 4 * np.pi) 
                    * np.exp(-self.lam * r_min) / self.lam ** 2 / r_min + 4 * np.pi / self.lam ** 2 / r_min) / 8 / np.pi
            val_sing = 0
            for iii in range(4):
                val_sing += (self.scheme.integrate(lambda theta: self.sing_a(theta, y_[iii], r_min, self.lam), 
                                                    [theta_c[iii], np.pi/2]) +
                                                    self.scheme.integrate(lambda theta: self.sing_b(theta, x_[iii], r_min, self.lam),
                                                                            [theta_c[iii], 0]))
            
            I = val_sing + val_circle
            val_fine[jj*len(self.xp) + ii] = I
            err[jj*len(self.xp) + ii] = self.tol_stk/10
                
            # Perform finer integration if the error is above the tolerance
            ind = err > self.tol_stk
            recs2 = self.recs[:, :, ind, :]
            val_finer = self.scheme_finer.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), recs2)
            err[ind] = np.abs(val_fine[ind] - val_finer) / np.abs(val_finer)
            val_fine[ind] = val_finer
            
            # Perform even finer integration if the error is still above the tolerance
            ind = err > self.tol_stk
            recs2 = self.recs[:, :, ind, :]
            val_finer = self.scheme_finer2.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), recs2)
            err[ind] = np.abs(val_fine[ind] - val_finer) / np.abs(val_finer)
            val_fine[ind] = val_finer
            
            # Perform local refinement if the error is still above the tolerance
            ind = np.where(err.flatten() > self.tol_stk)[0]
            if len(ind) > 0:
                val_out, l2_out = self.local_refinement(ind, err[ind], val_fine[ind], x0, y0)
                val_fine[ind] = val_out
                err[ind] = l2_out
            
            # Reshape the fine integration results and update the hydrodynamic matrix
            STK_0 = val_fine.reshape(len(self.yp), len(self.xp))
            h_initial[jj, :, :] = np.concatenate([np.fliplr(STK_0), STK_0[:, 1:]], axis=1)
            h_initial[-jj-1, :, :] = np.flipud(np.concatenate([np.fliplr(STK_0), STK_0[:, 1:]], axis=1))
        
        
        # Skip the remaining iterations if the loop is halfway through the y-coordinates
        if jj == int(len(self.yp) / 2) - 1:
            nn += int(len(self.yp) / 2)
        
        # Initialize the hydrodynamic matrix with zeros
        h_bem = np.zeros(([len(self.yp) * len(self.xp), len(self.yp) * len(self.xp)]), dtype=np.complex64)
        nn = 0#len(yp)
        for ix in range(len(self.xp)):
            for jy in range(int(len(self.yp))):
                STK_out = h_initial[jy, :, len(self.xp)-1-ix:len(self.xp)-1-ix+len(self.xp)]
                h_bem[nn, :] = STK_out.T.flatten()
                nn+=1

        self.a_bem = h_bem
        return h_bem

    
    