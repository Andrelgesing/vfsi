#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cupy as cp
import numpy as np
import cupyx.scipy.sparse as sp
import scipy.sparse as sp_cpu
#import cupyx.scipy.special as special
#import scipy.special as spc
import sys
sys.path.insert(0, '..')
#import Elastic.plate_fem_gpu as pt
from Fluid.Fluid_gpu import Fluid, Szz
import Fluid.h_matrix_gpu as h_matrix
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
        self.x = cp.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.xp = cp.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.wx = 1 / 3 * self.x
        self.y = cp.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.yp = cp.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.wy = 1 / 3 * self.x
        self.a = cp.ones((10, 10), dtype=complex)
        self.w = cp.ones((10, 10), dtype=float)
        self.p = cp.ones((10, 10), dtype=complex)
        self.a_bem = cp.ones((10, 10), dtype=complex)
        self.p = cp.ones((10, 10), dtype=complex)
        self.lam = np.sqrt(-1j * 100 * self.l_c ** 2 / self.fluid.nu_f)
        self.y_sqrt = cp.sqrt((self.w_c / 2) ** 2 - (self.yp * self.l_c) ** 2) / (self.w_c / 2)
        self.x_sqrt = cp.sqrt(1 - self.xp ** 2)
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
        self.lam = cp.sqrt(-1j * omega * self.l_c ** 2 / self.fluid.nu_f).astype(np.complex64)
        if self.x_uniform:
            self.a_bem = h_matrix.get_a_matrix_uniform_mid(self.xp, self.yp, self.x, self.y, self.lam)[0]
        else:
            self.a_bem = h_matrix.get_a_matrix_half(self.xp, self.yp, self.x, self.y, self.lam)[0]
        #a_inv = cp.linalg.inv(self.a_bem)
        #p = -self.fluid.mu_f/self.l_c*a_inv @ self.w.T
        #del a_inv

        a_bem_np = self.a_bem.get()  # CuPy CSR to NumPy CSR
        w_np = self.w.get()          # CuPy CSR to NumPy CSR

        # Transpose w, so we solve a_bem @ p = col of w^T
        w_T_np = w_np.tocsr()  # Can be csc for fast column slicing
        n = a_bem_np.shape[0]
        m = w_T_np.shape[0]#w_T_np.shape[1]

        #m = w_T_np.shape[1]
        
        # Sparse Sparse solve using PETSc
        sparse = False
        sparse_v2 = False
        if sparse:
            a_bem_csr = sp_cpu.csr_matrix(a_bem_np)
            A = PETSc.Mat().createAIJ(size=(n, n),
                                    csr=(a_bem_csr.indptr, a_bem_csr.indices, a_bem_csr.data),
                                    comm=PETSc.COMM_WORLD)    
            A.assemble()
            ## Allocate result matrix
            p_result = np.empty((n, m), dtype=np.complex64)
            ksp = PETSc.KSP().create()
            ksp.setType('preonly')         # Skip iterations
            ksp.setOperators(A)
            pc = ksp.getPC()
            pc.setType('lu')
            ksp.setFromOptions()

            ## Reuse x and b
            x = PETSc.Vec().createSeq(n, comm=PETSc.COMM_WORLD)
            b = PETSc.Vec().createSeq(n, comm=PETSc.COMM_WORLD)

            # Solve for each column of w^T
            for j in range(m):
                b.setArray(w_T_np[:, j].toarray().ravel())
                ksp.solve(b, x)
                p_result[:, j] = x.getArray()

            # Final scaled result
            del A, ksp, x, b, w_T_np, a_bem_np, w_np
        elif sparse_v2:
            a_bem_csr = sp_cpu.csr_matrix(a_bem_np)
            A = PETSc.Mat().createAIJ(size=(n, n),
                                    csr=(a_bem_csr.indptr, a_bem_csr.indices, a_bem_csr.data),
                                    comm=PETSc.COMM_WORLD)    
            A.setUp()
            A.assemble()
            B_dense = w_T_np.toarray()
            B = PETSc.Mat().createDense(size=(n, m), comm=PETSc.COMM_WORLD)
            B.setUp()
            B.setValues(range(n), range(m), B_dense)
            B.assemble()
            #B = PETSc.Mat().createAIJ(size=(n, m),
            #                        csr=(w_T_np.indptr, w_T_np.indices, w_T_np.data),
            #                        comm=PETSc.COMM_WORLD) 
            #B = PETSc.Mat().createAIJ(size=(n, m),
            #                        csr=(self.w.indptr, self.w.indices, self.w.data),
            #                        comm=PETSc.COMM_WORLD) 
            #B.setUp()
            #B.assemble()

            X = PETSc.Mat().createDense(size=(n, m), comm=PETSc.COMM_WORLD)
            X.setUp()
            X.assemble()
            print("A shape:", A.getSize())
            print("B shape:", B.getSize())
            print("X shape:", X.getSize())

            # --- Set up solver (required for matSolve to work)
            ksp = PETSc.KSP().create()
            ksp.setOperators(A)
            ksp.setType('preonly')
            pc = ksp.getPC()
            pc.setType('lu')
            pc.setFactorSolverType('mumps')  # REQUIRED for matSolve
            ksp.setFromOptions()
            A.matSolve(B, X)
            p_result = X.getDenseArray().copy()  
            # Final scaled result
            #del A, ksp, x, b, a_bem_np, w_np
        else:
            
            # DENSE PETSC MATRIX MULTIPLICATION 
            A = PETSc.Mat().createDense(size=a_bem_np.shape, comm=PETSc.COMM_WORLD)
            A.setUp()
            A.setValues(range(a_bem_np.shape[0]), range(a_bem_np.shape[1]), a_bem_np)
            A.assemble()
            ## W_dense: numpy array of shape (n, m)
            W_dense = w_T_np.toarray()
            B = PETSc.Mat().createDense(size=W_dense.shape, comm=PETSc.COMM_WORLD)
            B.setUp()
            B.setValues(range(n), range(m), W_dense)
            B.assemble()#

            X = PETSc.Mat().createDense(size=(n, m), comm=PETSc.COMM_WORLD)
            X.setUp()
            A.matSolve(B, X)
            p_result = X.getDenseArray().copy()  # Copy to avoid referencing issues
        p = -self.fluid.mu_f / self.l_c*cp.asarray(p_result)
        weights = cp.repeat(self.wx, len(self.yp)) * cp.tile(self.wy, len(self.xp)).flatten()
        #p = (p.T * (cp.repeat(self.wx, len(self.yp)) * cp.tile(self.wy, len(self.xp)).flatten())).T
        #p = (p.T * weights).T
        p_scaled = p * weights[:, None] 
        p_result = self.w @ p_scaled # Final matrix multiplication
        #p =  self.w @ p
        p_sl = sp.csr_matrix(p_result, dtype=complex)
        #p_sl = sp.csr_matrix(p, dtype=complex)
        self.p = p_sl
        return p_sl


def petsc_solve_csr(A_csr, b_vec):
    A = PETSc.Mat().createAIJ(size=A_csr.shape,
                              csr=(A_csr.indptr, A_csr.indices, A_csr.data.astype(complex)),
                              comm=PETSc.COMM_WORLD)
    A.assemble()
    b = PETSc.Vec().createWithArray(b_vec, comm=PETSc.COMM_WORLD)
    x = PETSc.Vec().createSeq(A_csr.shape[1], comm=PETSc.COMM_WORLD)

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType('preonly')         # Skip iterations
    #ksp.getPC().setType('lu')      # Direct LU solver (stable)
    pc = ksp.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')
    #ksp.setType('gmres')
    #ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=500)
    ksp.setFromOptions()
    ksp.solve(b, x)

    return x.getArray()

    