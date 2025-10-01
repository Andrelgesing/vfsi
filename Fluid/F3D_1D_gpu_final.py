#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cupy as cp
import numpy as np
import cupyx.scipy.sparse as sp
import cupyx.scipy.special as special
import scipy.special as spc
import sys
sys.path.insert(0, '..')
#import Elastic.plate_fem_gpu as pt
from Fluid.Fluid_gpu import Fluid, Szz
import quadpy
import time
#import Fluid.h_matrix_gpu_efficient as h_matrix
import Fluid.h_matrix_gpu_efficient as h_matrix
from Fluid.fused_kernels import *#wp_product_cuda, wp_product_numba, wp_product_v1, wp_product_v2
from cupyx.scipy.linalg import lu_factor, lu_solve
from petsc4py import PETSc
import gc
#from scipy.linalg import lu_factor as lu_factor_cpu
#from scipy.linalg import lu_solve as lu_solve_cpu

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
        self.x_uniform = True
        self.print_time = False
        self.threshold = 1e-16
        self.sparse = False


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
        self.weight_int = cp.repeat(self.wx, len(self.yp)) * cp.tile(self.wy, len(self.xp)).flatten()
        #del self.x_sqrt, self.y_sqrt, self.wx, self.wy, self.xp, self.yp
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
        self.lam = cp.sqrt(-1j * omega * self.l_c ** 2 / self.fluid.nu_f).astype(np.complex64)
        if self.x_uniform:
            a_bem = h_matrix.get_a_matrix_uniform_mid(self.xp, self.yp, self.x, self.y, self.lam)#[0]
        else:
            a_bem = h_matrix.get_a_matrix_half(self.xp, self.yp, self.x, self.y, self.lam)#[0]
        if self.print_time:
            tt = time.time() - t_init
            print('Determine H-matrix was %f seconds' % (tt))
        #a_bem = a_bem.astype(cp.complex64)
        #a_inv = cp.linalg.inv(self.a_bem * self.l_c)
        #p = -self.fluid.mu_f*a_inv @ self.w.T
        t_init = time.time()
        #if type(self.a_bem) == cp.ndarray:
        #if self.n_x_fluid*self.n_y_fluid < 64*65:
        #try:
        if self.n_x_fluid*self.n_y_fluid < 128*129:
            lu, piv = lu_factor(a_bem)
            p = -self.fluid.mu_f / self.l_c * lu_solve((lu, piv), self.w.T.toarray())
            del a_bem, lu, piv 
            free_gpu_now()
            #p = -self.fluid.mu_f/self.l_c*cp.linalg.solve(self.a_bem , self.w.T.toarray())
        #else:
        #except:
        else:
            if self.print_time:
                print("\nUsing PETSc for large matrix inversion\n")
            # DENSE PETSC MATRIX MULTIPLICATION 
            DT = PETSc.ScalarType
            a_bem_np = cp.asnumpy(a_bem).astype(DT, copy=False)
            del a_bem 
            free_gpu_now()
            A = PETSc.Mat().createDense(size=a_bem_np.shape, array=a_bem_np, comm=PETSc.COMM_WORLD)
            A.assemble()

            #F = A.copy()
            #F.factorLU(None, None)
            ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
            ksp.setOperators(A)
            ksp.setType('preonly')
            pc = ksp.getPC()
            pc.setType('lu')
            #pc.setFactorSolverType('mumps')  # REQUIRED for matSolve
            ksp.setFromOptions()
            #ksp.setUp()

            n = self.w.shape[1]
            m = self.w.shape[0]
            p_cpu = np.empty((n, m), dtype=DT)
            #block_size = min(2048, m, n)
            block_size = int(np.floor(m/3)) + 1
            for j in range(0, m, block_size):
                end_ = min(j + block_size, m)
                W_dense = self.w[j:end_, :].get()
                W_dense = W_dense.toarray().T  # shape (n, block_size)
                #W_dense = self.w[j:end_, :].get().toarray().T.astype(DT, copy=False)
                n2 = W_dense.shape[1]
                m2 = W_dense.shape[0]
                B = PETSc.Mat().createDense(size=W_dense.shape, array=W_dense, comm=PETSc.COMM_WORLD)
                B.assemble()
                #B = PETSc.Mat().createDense(size=W_dense.shape, comm=PETSc.COMM_WORLD)
                #B.setUp()
                #B.setValues(range(m2), range(n2), W_dense)
                #B.assemble()#
                X = PETSc.Mat().createDense(size=W_dense.shape, comm=PETSc.COMM_WORLD)  # output buffer
                X.setUp()
                X.assemble()
                #X = PETSc.Mat().createDense(size=(m2, n2), comm=PETSc.COMM_WORLD)
                #X.setUp()
                #A.matSolve(B, X)
                #F.matSolve(B, X)
                ksp.matSolve(B, X)
                p_result = X.getDenseArray()#.copy()
                #print(j)
                p_cpu[:, j:end_] = -self.fluid.mu_f / self.l_c * p_result
            del a_bem_np, A, B, X, p_result
            #del a_bem 
            free_gpu_now()
            #del p_cpu
            p_cpu = p_cpu.astype(np.complex64, copy=False)
            p = cp.asarray(p_cpu, blocking=True)  
               


        if self.print_time:
            tt = time.time() - t_init
            print('Invert H-matrix was %f seconds' % (tt))
        
        
        t_init = time.time()    
        #p = cp.ascontiguousarray(p.astype(cp.complex64))
        #p = wp_product_v1(p, self.weight_int)
        p *= self.weight_int[:, None]
        if self.print_time:
            tt = time.time() - t_init
            print('First multiplication took %f seconds' % (tt))
        t_init = time.time()  

        if self.n_x_fluid*self.n_y_fluid < 128*127:
        #try:
            #p = self.w.dot(p)
            p =  self.w @ p
        #except:
        else:
            n_cols = p.shape[1]
            n_rows = self.w.shape[0]
            block_size = int(np.floor(n_cols/2)) + 1
            #block_size = min(4096, n_cols)
            p_result = cp.empty((n_rows, n_cols), dtype=cp.complex64)

            col_offset = 0
            while p.shape[1] > 0:
                current_block_size = min(block_size, p.shape[1])
                p_block = p[:, :current_block_size]  # take first block
                p_mapped = self.w @ p_block
                p_result[:, col_offset:col_offset + current_block_size] = p_mapped

                # Delete processed columns from p to free memory
                p = cp.delete(p, cp.s_[:current_block_size], axis=1)
                col_offset += current_block_size
            p = p_result.T
        #else:
        #    p = self.w.dot(p)
        if self.print_time:
            tt = time.time() - t_init
            print('Second multiplication took %f seconds' % (tt))
        t_init = time.time()
        #p = cp.where(cp.abs(p)/cp.max(cp.abs(p)) < self.threshold, 0, p)
        free_gpu_now()
        #if self.sparse:
        #    p_sl = sp.csr_matrix(p)
        #    if self.print_time:
        #        tt = time.time() - t_init
        #        print('Conversion to sparse matrix took %f seconds' % (tt))
        #        density = cp.count_nonzero(p).item() / (p.shape[0] * p.shape[1])
        #         print(f"Matrix density: {density:.2%}\n")
        #    #self.p = sp.csr_matrix(p, dtype=cp.complex64)#p_sl
        #    return p_sl
        #else:
        return p
    
def sort_csr_matrix(csr):
    """Sort the column indices in each row of a CSR matrix."""
    csr = csr.copy()
    for i in range(csr.shape[0]):
        start = csr.indptr[i]
        end = csr.indptr[i+1]
        row_indices = csr.indices[start:end]
        row_data = csr.data[start:end]
        if not cp.all(row_indices[:-1] <= row_indices[1:]):
            sorted_idx = cp.argsort(row_indices)
            csr.indices[start:end] = row_indices[sorted_idx]
            csr.data[start:end] = row_data[sorted_idx]
    return csr

def free_gpu_now():
    cp.cuda.runtime.deviceSynchronize()                 # wait for all kernels/copies
    cp.get_default_memory_pool().free_all_blocks()     # return GPU blocks to driver
    #cp.cuda.get_pinned_memory_pool().free_all_blocks() # release pinned host buffers
    #gc.collect() 