#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fenics as fe
import cupy as cp
import numpy as np
import cupyx.scipy.sparse as sp
import cupyx.scipy.special as special
import scipy.special as spc
import sys
sys.path.insert(0, '..')
import Elastic.plate_fem_gpu as pt
from Fluid.Fluid_gpu import Fluid, Szz
import quadpy
import Fluid.h_matrix_gpu as h_matrix


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
        self.x = cp.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.xp = cp.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.wx = 1 / 3 * self.x
        self.y = cp.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.yp = cp.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.wy = 1 / 3 * self.x
        self.a = cp.ones((10, 10), dtype=complex)
        self.w = cp.ones((10, 10), dtype=float)
        self.p = cp.ones((10, 10), dtype=complex)
        self.tol_stk = 2e-3
        self.x_left = cp.tile(self.x[0:-1], len(self.yp))  # [0:n2]
        self.x_right = cp.tile(self.x[1:], len(self.yp))  # [0:n2]
        self.y_low = cp.repeat(self.y[0:-1], len(self.xp))  # [0:n2]
        self.y_up = cp.repeat(self.y[1:], len(self.xp))  # [0:n2]
        recs = cp.array([[cp.array([self.x_left, self.y_low]), cp.array([self.x_right, self.y_low])]
                            , [cp.array([self.x_left, self.y_up]), cp.array([self.x_right, self.y_up])]])
        self.recs = recs.transpose(0, 1, 3, 2)
        self.nb = 100
        self.scheme_coarse = quadpy.c2.get_good_scheme(2)
        self.scheme_fine = quadpy.c2.get_good_scheme(4)
        self.scheme_finer = quadpy.c2.get_good_scheme(6)
        self.scheme_finer2 = quadpy.c2.get_good_scheme(8)
        self.scheme_t = quadpy.t2.get_good_scheme(3)
        self.a_bem = cp.ones((10, 10), dtype=complex)
        self.p = cp.ones((10, 10), dtype=complex)
        self.lam = np.sqrt(-1j * 100 * self.l_c ** 2 / self.fluid.nu_f)
        self.y_sqrt = cp.sqrt((self.w_c / 2) ** 2 - (self.yp * self.l_c) ** 2) / (self.w_c / 2)
        self.x_sqrt = cp.sqrt(1 - self.xp ** 2)
        self.pbar = None
        self.scheme = quadpy.c1.gauss_legendre(12)
        #self.weights = cp.asarray(self.scheme.weights)
        self.theta_lin = cp.asarray(self.scheme.points)
        self.x_uniform = False
        self.dof_corner = 10

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
        i = cp.arange(1, ny + 1)
        yp = (-cp.cos(((2 * i - 1) / (2 * ny)) * cp.pi))
        y = cp.concatenate((cp.array([-1]), 0.5 * (yp[1:] + yp[:-1]), cp.array([1])))
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
        y = cp.linspace(0, 1, ny + 1)
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
            x = self.gaussian_quad(2 * n_x_fluid)[0][int(n_x_fluid)::]
            xp = self.gaussian_quad(2 * n_x_fluid)[1][int(n_x_fluid)::]
            self.x = cp.asarray(x)
            self.xp = cp.asarray(xp)
            #self.x, self.xp = cp.array(self.gaussian_quad(2 * n_x_fluid), dtype=object)[0][int(n_x_fluid)::], \
            #    cp.array(self.gaussian_quad(2 * n_x_fluid), dtype=object)[1][int(n_x_fluid)::]
        else:    
            #self.x, self.xp = cp.array(self.lin_quad(n_x_fluid), dtype=object)
            x, xp = self.lin_quad(n_x_fluid)
            self.x = cp.asarray(x)
            self.xp = cp.asarray(xp)
        self.x_uniform = x_uniform
        y, yp = self.gaussian_quad(n_y_fluid)
        self.y = cp.asarray(y) * self.w_c / 2 / self.l_c
        self.yp = cp.asarray(yp) * self.w_c / 2 / self.l_c
        if self.x_uniform:
            self.wx = self.l_c / len(self.xp)*cp.ones(len(self.xp))
        else:
            self.x_sqrt = cp.sqrt(1 - self.xp ** 2)
            self.wx = self.l_c / 2 * cp.pi / len(self.xp) * self.x_sqrt
        self.y_sqrt = cp.sqrt((self.w_c / 2) ** 2 - (self.yp * self.l_c) ** 2) / (self.w_c / 2)
        self.wy =  self.w_c / 2 *cp.pi/len(self.yp) * self.y_sqrt
        self.export_basis_functions()
        v_space = self.fem.VCG
        # Assume v_space is already defined
        dof_coords = v_space.tabulate_dof_coordinates()
        dof_coords = dof_coords.reshape((-1, v_space.mesh().geometry().dim()))

        # Find the point with maximum x and y
        max_point_index = np.argmax(np.sum(dof_coords, axis=1))  # crude but works for max(x + y)

        # Or use this for strict max x and y:
        mask = np.isclose(dof_coords[:, 0], np.max(dof_coords[:, 0])) & \
            np.isclose(dof_coords[:, 1], np.max(dof_coords[:, 1]))
        indices = np.where(mask)[0]

        # Just take the first one found (in case of duplicates)
        closest_dof = indices[0]
        self.dof_corner = closest_dof

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
        self.lam = cp.sqrt(-1j * omega * self.l_c ** 2 / self.fluid.nu_f).astype(np.complex64)
        if self.x_uniform:
            self.a_bem = h_matrix.get_a_matrix_uniform_mid(self.xp, self.yp, self.x, self.y, self.lam)[0]
        else:
            self.a_bem = h_matrix.get_a_matrix_half(self.xp, self.yp, self.x, self.y, self.lam)[0]
        a_inv = cp.linalg.inv(self.a_bem * self.l_c)
        p = -self.fluid.mu_f*a_inv @ self.w.T
        del a_inv
        p = (p.T * (cp.repeat(self.wx, len(self.yp)) * cp.tile(self.wy, len(self.xp)).flatten())).T
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
                -(spc.expi(lam_ * r * (-1))
                -spc.expi(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
                + (y1 - r*np.sqrt(1-np.cos(theta)**2)*np.exp(lam_*r)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2))
            + lam_*r*y1*np.exp(lam_*r)*spc.expi(lam_ * r * (-1))
            - lam_*r*y1*np.exp(lam_*r)*
            spc.expi(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_**2*r*y1))
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
                -(spc.expi(lam_ * r * (-1))
                -spc.expi(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2)))/(4*np.pi*lam_)
                +(x1 - r*np.sqrt(1-np.sin(theta)**2)*np.exp(lam_*r)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2))
            + lam_*r*x1*np.exp(lam_*r)*spc.expi(lam_ * r * (-1))
            - lam_*r*x1*np.exp(lam_*r)*
            spc.expi(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_**2*r*x1))
        return f_int


    def get_h_matrix(self, omega):
        """Returns the hydrodynamc matrix H, whch arises from the numerical integration of the unsteady stokeslet.

            Parameters
            ----------
            omega: float
                 frequency in radians per second
            Returns
            -------
            h : cp.array(type=cp.complex)
                the complex-valued H matrix
        """
        
        # Calculate the lambda parameter based on the given frequency
        self.lam = np.sqrt(-1j * omega * self.l_c ** 2 / self.fluid.nu_f).astype(np.complex64)
        
        nn = 0
        h_bem = cp.zeros(([len(self.yp) * len(self.xp), len(self.yp) * len(self.xp)]), dtype=cp.complex64)
        h_initial = cp.zeros((len(self.yp), len(self.yp), 2*len(self.xp)-1), dtype=cp.complex64)
        # Loop over the x-coordinates
        for ii in range(len(self.xp)):
            x0 = float(self.xp[ii])
            for jj in range(int(len(self.yp)/2)):
                y0 = float(self.yp.get()[jj])
                
                # Perform coarse and fine integration
                val_coarse = cp.asarray(self.scheme_coarse.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), self.recs))
                val_fine = cp.asarray(self.scheme_fine.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), self.recs))
                
                # Calculate the error between coarse and fine integration
                err = cp.abs(val_coarse - val_fine) / cp.abs(val_fine)
                err[cp.where(cp.abs(val_fine) / cp.max(cp.abs(val_fine)) <= 1e-10)] = 0
                val_fine[cp.where(cp.abs(val_fine) / cp.max(cp.abs(val_fine)) <= 1e-10)] = 0
                
                # Handle the integration around singularities
                x1 = -(self.x[ii] - x0)
                x2 = self.x[ii+1] - x0
                y1 = -(self.y[jj] - y0)
                y2 = self.y[jj+1] - y0
                xx = cp.array([x1, x1, x2, x2])
                yy  = cp.array([y1, y2, y1, y2])
                theta_c = cp.arctan(yy/xx)
                x_ = cp.tile(cp.array([xx]).T, len(self.theta_lin))
                y_ = cp.tile(cp.array([yy]).T, len(self.theta_lin))
                r_min = cp.min(cp.abs(cp.array([x1, x2, y1, y2])))
                val_circle = ((-4 * cp.pi * self.lam * r_min - 4 * cp.pi) 
                        * cp.exp(-self.lam * r_min) / self.lam ** 2 / r_min + 4 * cp.pi / self.lam ** 2 / r_min) / 8 / cp.pi
                val_sing = 0
                for iii in range(4):
                    val_sing += (self.scheme.integrate(lambda theta: self.sing_a(theta, y_.get()[iii], r_min.get(), self.lam), 
                                                        [theta_c.get()[iii], np.pi/2]) +
                                                        self.scheme.integrate(lambda theta: self.sing_b(theta, x_.get()[iii], r_min.get(), self.lam),
                                                                                [theta_c.get()[iii], 0]))
                
                I = val_sing + val_circle
                val_fine[jj*len(self.xp) + ii] = I
                err[jj*len(self.xp) + ii] = self.tol_stk/10
                    
                # Perform finer integration if the error is above the tolerance
                ind = err > self.tol_stk
                recs2 = self.recs[:, :, ind.get(), :]
                val_finer = cp.asarray(self.scheme_finer.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), recs2))
                err[ind] = cp.abs(val_fine[ind] - val_finer) / cp.abs(val_finer)
                val_fine[ind] = val_finer
                
                # Perform even finer integration if the error is still above the tolerance
                ind = err > self.tol_stk
                recs2 = self.recs[:, :, ind.get(), :]
                val_finer = cp.asarray(self.scheme_finer2.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), recs2))
                err[ind] = cp.abs(val_fine[ind] - val_finer) / cp.abs(val_finer)
                val_fine[ind] = val_finer
                
                # Perform local refinement if the error is still above the tolerance
                ind = cp.where(err.flatten() > self.tol_stk)[0]
                if len(ind) > 0:
                    val_out, l2_out = self.local_refinement(ind.get(), err[ind].get(), val_fine[ind].get(), x0, y0)
                    val_fine[ind] = val_out
                    err[ind] = l2_out

                # Reshape the fine integration results and update the hydrodynamic matrix
                STK_0 = val_fine.reshape(len(self.yp.get()), len(self.xp.get()))
                h_bem[nn, :] = STK_0.T.flatten()
                h_bem[nn + int(len(self.yp.get())) - 2 * jj - 1, :] = cp.flip(STK_0.T, 1).flatten()
                
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
            xf (cp.ndarray): The x-coordinates of the mesh points.
            yf (cp.ndarray): The y-coordinates of the mesh points.
            v2 (fe.FunctionSpace): The function space on the fluid mesh.
            transfer_matrix (fe.PETScDMCollection): The transfer matrix between function spaces.
            arg_x (cp.ndarray): Indices to sort the x-coordinates of the degrees of freedom.
            coord_x (cp.ndarray): Sorted coordinates of the degrees of freedom.
            arg_y (cp.ndarray): Indices to sort the y-coordinates of the degrees of freedom.
            arg_y2 (cp.ndarray): Flattened indices for sorting the y-coordinates.
            row (cp.ndarray): Row indices of the CSR matrix.
            col (cp.ndarray): Column indices of the CSR matrix.
            val (cp.ndarray): Values of the CSR matrix.
            w0 (fe.Function): A function in the original function space.
            self.nb (int): The size of the vector in the original function space.
            w (sp.csr_matrix): The sparse matrix representation of the basis functions.
            self.w (sp.csr_matrix): The transposed and sorted sparse matrix of basis functions.
        """
        mesh_fluid = fe.RectangleMesh(fe.Point(0, -self.geometry.w_c / 2),
                                      fe.Point(self.geometry.l_c, self.geometry.w_c / 2),
                                     self.n_x_fluid-1, self.n_y_fluid-1, 'left/right')
        (mesh_fluid.coordinates()[:, 1]).sort(axis=0)
        xf = np.array(mesh_fluid.coordinates()[:, 0])
        yf = np.array(mesh_fluid.coordinates()[:, 1])
        #yf[:] = np.repeat(self.yp * self.l_c, len(self.xp.get()))
        yf = np.repeat(self.yp * self.l_c, len(self.xp.get()))
        xf[np.argsort(xf)] = np.repeat(self.xp.get() * self.l_c, len(self.yp.get()))
        xf = np.repeat(self.xp.get() * self.l_c, len(self.yp.get()))
        v2 = fe.FunctionSpace(mesh_fluid, 'CG', 1)
        self.fem.function_spaces()
        transfer_matrix = fe.PETScDMCollection.create_transfer_matrix(self.fem.VCG, v2)

        arg_x = np.argsort(v2.tabulate_dof_coordinates()[:, 0])
        coord_x = v2.tabulate_dof_coordinates()[arg_x, :]
        arg_y = np.argsort(np.reshape(coord_x[:, 1], (len(self.xp), len(self.yp))))
        arg_y2 = (arg_y.T + np.arange(len(arg_y)) * (len(self.yp))).T.flatten()

        # print(arg_y2.shape)
        row, col, val = fe.as_backend_type(transfer_matrix).mat().getValuesCSR()
        row = cp.asarray(row)
        col = cp.asarray(col)
        val = cp.asarray(val)
        A_csr = sp.csr_matrix((val, col, row), dtype=complex)
        w0 = fe.Function(self.fem.VCG)
        self.nb = w0.vector().size()
        #w = sp.csr_matrix((val, col, row), shape=(len(self.xp) * len(self.yp), self.nb), dtype='float64')
        w = sp.csr_matrix((val, col, row), shape=(len(self.xp.get()) * len(self.yp.get()), self.nb), dtype='float64')
        w_sub = (w[arg_x])[arg_y2].T
        #w_sub = w[arg_x, :][:, arg_y2].T  # Apply both slices at once, then transpose

        # Step 3: Ensure it is canonical CSR
        w_sub = sp.csr_matrix(w_sub)         # Rebuild in CSR format if needed
        w_sub.sort_indices()
        #w_sub.eliminate_zeros()

        # Now safe to assign:
        self.w = w_sub
        return w_sub
        #self.w = (w[arg_x])[arg_y2].T


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
        # ind = cp.where(l2_error.flatten() > tol)[0]
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
            h : cp.array(type=cp.complex)
                the complex-valued H matrix
        """
        
        
        # Calculate the lambda parameter based on the given frequency
        self.lam = np.sqrt(-1j * omega * self.l_c ** 2 / self.fluid.nu_f).astype(np.complex64)
        
        nn = 0
        h_initial = cp.zeros((len(self.yp), len(self.yp), 2*len(self.xp)-1), dtype=cp.complex64)
        ii = 0
        x0 = float(self.xp.get()[ii])
        #self.recs = np.array(self.recs.get())    
            # Loop over half of the y-coordinates
        for jj in range(int(len(self.yp)/2)):
            y0 = float(self.yp.get()[jj])
            
            # Perform coarse and fine integration
            val_coarse = cp.asarray(self.scheme_coarse.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), self.recs))
            val_fine = cp.asarray(self.scheme_fine.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), self.recs))
            
            # Calculate the error between coarse and fine integration
            err = cp.abs(val_coarse - val_fine) / cp.abs(val_fine)
            err[cp.where(cp.abs(val_fine) / cp.max(cp.abs(val_fine)) <= 1e-10)] = 0
            val_fine[cp.where(cp.abs(val_fine) / cp.max(cp.abs(val_fine)) <= 1e-10)] = 0
            
            # Handle the integration around singularities
            x1 = -(self.x[ii] - x0)
            x2 = self.x[ii+1] - x0
            y1 = -(self.y[jj] - y0)
            y2 = self.y[jj+1] - y0
            xx = cp.array([x1, x1, x2, x2])
            yy  = cp.array([y1, y2, y1, y2])
            theta_c = cp.arctan(yy/xx)
            x_ = cp.tile(cp.array([xx]).T, len(self.theta_lin))
            y_ = cp.tile(cp.array([yy]).T, len(self.theta_lin))
            r_min = cp.min(cp.abs(cp.array([x1, x2, y1, y2])))
            #diff_ = cp.abs(cp.pi/2-theta_c)
            #theta_a = (cp.pi/2 + theta_c[:, cp.newaxis])/2 + diff_[:, cp.newaxis]/2*self.theta_lin
            #weights_a = self.w*diff_[:, cp.newaxis]/2
            #diff_ = cp.abs(0-theta_c)
            #theta_b = (0 + theta_c[:, cp.newaxis])/2 + diff_[:, cp.newaxis]/2*self.theta_lin
            #weights_b = self.w*diff_[:, cp.newaxis]/2
            val_circle = ((-4 * cp.pi * self.lam * r_min - 4 * cp.pi) 
                    * cp.exp(-self.lam * r_min) / self.lam ** 2 / r_min + 4 * cp.pi / self.lam ** 2 / r_min) / 8 / cp.pi
            #val_sing = cp.sum(self.sing_a(theta_a, y_, r_min, self.lam)*weights_a
            #                    + self.sing_b(theta_b, x_, r_min, self.lam)*weights_b)
        
            val_sing = 0
            for iii in range(4):
                val_sing += (self.scheme.integrate(lambda theta: self.sing_a(theta, y_.get()[iii], r_min.get(), self.lam), 
                                                    [theta_c.get()[iii], np.pi/2]) +
                                                    self.scheme.integrate(lambda theta: self.sing_b(theta, x_.get()[iii], r_min.get(), self.lam),
                                                                            [theta_c.get()[iii], 0]))
            
            I = val_sing + val_circle
            val_fine[jj*len(self.xp) + ii] = I
            err[jj*len(self.xp) + ii] = self.tol_stk/10
                
            # Perform finer integration if the error is above the tolerance
            ind = err > self.tol_stk
            recs2 = self.recs[:, :, ind.get(), :]
            val_finer = cp.asarray(self.scheme_finer.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), recs2))
            err[ind] = cp.abs(val_fine[ind] - val_finer) / cp.abs(val_finer)
            val_fine[ind] = val_finer
            
            # Perform even finer integration if the error is still above the tolerance
            ind = err > self.tol_stk
            recs2 = self.recs[:, :, ind.get(), :]
            val_finer = cp.asarray(self.scheme_finer2.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), recs2))
            err[ind] = cp.abs(val_fine[ind] - val_finer) / cp.abs(val_finer)
            val_fine[ind] = val_finer
            
            # Perform local refinement if the error is still above the tolerance
            ind = cp.where(err.flatten() > self.tol_stk)[0]
            if len(ind) > 0:
                val_out, l2_out = self.local_refinement(ind.get(), err[ind].get(), val_fine[ind].get(), x0, y0)
                val_fine[ind] = val_out
                err[ind] = l2_out
            
            # Reshape the fine integration results and update the hydrodynamic matrix
            STK_0 = val_fine.reshape(len(self.yp), len(self.xp))
            h_initial[jj, :, :] = cp.concatenate([cp.fliplr(STK_0), STK_0[:, 1:]], axis=1)
            h_initial[-jj-1, :, :] = cp.flipud(cp.concatenate([cp.fliplr(STK_0), STK_0[:, 1:]], axis=1))
        
        
        # Skip the remaining iterations if the loop is halfway through the y-coordinates
        if jj == int(len(self.yp) / 2) - 1:
            nn += int(len(self.yp) / 2)
        
        # Initialize the hydrodynamic matrix with zeros
        h_bem = cp.zeros(([len(self.yp) * len(self.xp), len(self.yp) * len(self.xp)]), dtype=cp.complex64)
        nn = 0#len(yp)
        for ix in range(len(self.xp)):
            for jy in range(int(len(self.yp))):
                STK_out = h_initial[jy, :, len(self.xp)-1-ix:len(self.xp)-1-ix+len(self.xp)]
                h_bem[nn, :] = STK_out.T.flatten()
                nn+=1

        self.a_bem = h_bem
        return h_bem

    
    