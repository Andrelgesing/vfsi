#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from ufl import indices as ufl_indices
import scipy.sparse as sps
#import General.Setup as Stp



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
        self.k_c = self.w_c*self.t_c ** 3 / 12  # Inertial moment [Only applicable for Green's method]


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
        self.g_c = youngs_modulus/2/(1 + poisson)  # Shear modulus


class InteriorPenalty(object):
    """ Interior penalty method for the Kirchhoff-Love plate"""
    def __init__(self):
        # Initialize parameters
        self.geometry = Geometry()
        self.mat = Material()
        self.a_r = self.geometry.l_c / self.geometry.w_c  # Aspect ratio of the plate [-]
        self.i_c_fe = fe.Constant(self.geometry.t_c ** 3 / 12)  # Copy of i_c, as a fenics constant
        self.mu_c_fe = fe.Constant(self.mat.e_c / 2 / (1 + self.mat.nu_c))  # Copy of mu_c, as a fenics constant
        self.lam_c_fe = fe.Constant(self.mat.e_c * self.mat.nu_c / (1 - self.mat.nu_c ** 2))  # Copy of lam_c, as a
        # fenics constant
        self.n_x = 16  # Number of elements of the mesh in x-direction
        self.n_y = int(self.n_x / self.geometry.a_r)  # Number of elements of the mesh in y-direction
        self.c_tensor = None  # Initialize c_tensor
        self.mesh = fe.UnitIntervalMesh(self.n_x)  # Initialize mesh
        self.h_e = fe.CellDiameter(self.mesh)
        self.h_avg = (self.h_e('+') + self.h_e('-')) / 2
        self.mesh_type = 'crossed'  # Type of mesh. Default is 'crossed'
        self.normal = fe.FacetNormal(self.mesh)  # normal vector in mesh
        self.l_ = 2  # Degree of the FEM function space
        self.VCG = fe.FunctionSpace(self.mesh, fe.FiniteElement("CG", self.mesh.ufl_cell(), degree=self.l_))
        self.u = fe.TrialFunction(self.VCG)
        self.v = fe.TestFunction(self.VCG)
        self.phi = np.zeros((16 * 16 * 2, 5))  # Displacement array
        self.tau_ip = 16  # Constant of the FEM operator (should be higher than 3)
        self.alpha, self.beta, self.gamma, self.delta = ufl_indices(4)
        self.k_matrix = None
        self.m_matrix = None
        self.dirichlet_bc = None
        self.ds_h = None
        self.ds_moment = None
        self.ds_shear_force = None
        self.l_vec = 0

    def preliminary_setup(self):
        """ Implements preliminary setup for the Interior penalty"""
        kr = fe.Identity(2)
        self.i_c_fe = fe.Constant(self.geometry.t_c ** 3 / 12)  # Copy of i_c, as a fenics constant
        self.mu_c_fe = fe.Constant(self.mat.e_c / 2 / (1 + self.mat.nu_c))  # Copy of mu_c, as a fenics constant
        self.lam_c_fe = fe.Constant(self.mat.e_c * self.mat.nu_c / (1 - self.mat.nu_c ** 2))  # Copy of lam_c, as a
        # Elastic tensor
        self.c_tensor = fe.as_tensor(self.i_c_fe * (self.mu_c_fe * (kr[(self.alpha, self.gamma)] *
                                                                    kr[(self.beta, self.delta)]
                                                                    + kr[(self.alpha, self.delta)] * kr[(self.beta,
                                                                                                         self.gamma)])
                                                    + self.lam_c_fe * kr[(self.alpha, self.beta)] *
                                                    kr[(self.gamma, self.delta)]),
                                     [self.alpha, self.beta, self.gamma, self.delta])
        self.meshing()
        print('\n Setup Mesh %d x %d.' % (self.n_x, self.n_y))
        self.function_spaces()
        self.set_bcs()

    def meshing(self, nx=None, ny=None, type_=None):
        """
        Define the mesh of the plate in Fenics.

        Parameters
        ----------
        nx: int
                Number of elements in length l_c (x-direction).
        ny: int
                Number of elements in length w_c (y-direction).
        type_: str
                Type of mesh, either 'crossed', 'left', 'right', 'left/right' or 'right/left'.
        """
        if nx is not None:
            self.n_x = nx  # Number of elements of the mesh in x-direction
            self.n_y = ny  # Number of elements of the mesh in y-direction
            self.mesh_type = type_  # Type of mesh
        self.mesh = fe.RectangleMesh(fe.Point(0, -self.geometry.w_c / 2), fe.Point(self.geometry.l_c, self.geometry.w_c / 2),
                                     self.n_x, self.n_y, self.mesh_type)
        # fe.MPI.comm_self
        self.normal = fe.FacetNormal(self.mesh)
        # Cell diameter and normal to IP method
        self.h_e = fe.CellDiameter(self.mesh)
        self.h_avg = (self.h_e('+') + self.h_e('-')) / 2

    def function_spaces(self, l_=None):
        """
        Define the function spaces used in Interior penalty method.

        Parameters
        ----------
        l_: int
                Degree of the FEM operator function space. Default is 2.
        """
        if l_ is not None:
            self.l_ = l_  # Type of mesh
        self.VCG = fe.FunctionSpace(self.mesh, fe.FiniteElement("CG", self.mesh.ufl_cell(), degree=self.l_))
        self.u = fe.TrialFunction(self.VCG)
        self.v = fe.TestFunction(self.VCG)

    def set_bcs(self):
        """ Define Dirichlet and Neumann boundary conditions"""
        tol = 1e-14

        def left(x, on_boundary):
            return on_boundary and fe.near(x[0], 0, tol)

        self.dirichlet_bc = fe.DirichletBC(self.VCG, fe.Constant(0), left)

        self.ds_h, self.ds_moment, self.ds_shear_force = self.setup_neumann_bcs(self.mesh, self.geometry.l_c,
                                                                           self.geometry.w_c)

    def setup_eigenvalues_problem(self):
        # print('\n         Solving first %d eigenvalues.' % self.n_eig)
        self.k_and_m_matrices()

    def solve_n_eigenvalues(self, n_eig, k, m):
        if k is not None:
            self.k_matrix = k
            self.m_matrix = m
        else:
            self.setup_eigenvalues_problem()
        eigensolver = fe.SLEPcEigenSolver(self.k_matrix, self.m_matrix)
        eigensolver.parameters['problem_type'] = 'gen_hermitian'
        eigensolver.parameters['solver'] = 'arnoldi'  # 'arnoldi', 'krylov-schur', 'lapack'
        eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
        eigensolver.parameters['spectral_shift'] = 0.
        # eigensolver.parameters['spectrum'] = 'smallest real'  # This doesn't seem to work =/
        # eigensolver.parameters["verbose"] = False
        eigensolver.parameters['tolerance'] = 1e-10
        # eigensolver.parameters['maximum_iterations'] = 200
        eigensolver.solve(int(n_eig))
        # assert eigensolver.get_number_converged() > 0

        self.phi = np.zeros([self.k_matrix.size(0), n_eig], dtype=float)
        self.fn = np.zeros(n_eig, dtype=float)
        for ii in range(n_eig):
            r, c, mode, s = eigensolver.get_eigenpair(ii)
            self.phi[:, ii] = mode
            # self.phi[:, ii] = mode.get_local()
            self.fn[ii] = np.sqrt(np.abs(r)) / 2 / np.pi
            print('         Mode %d = %1.1f kHz' % (ii + 1, self.fn[ii] / 1e3))
        return self.phi, self.fn

    def plot_eigenmodes(self, n_eig):
        phi_fun = fe.Function(self.VCG)
        plt.figure(figsize=(10, n_eig))
        for ii in range(n_eig):
            plt.subplot(3, int(n_eig/3)+1, ii+1)
            plt.title('fn= '+str(int(self.fn[ii]/1e3))+' kHz')
            phi_fun.vector()[:] = np.array(self.phi[:, ii])
            fe.plot(phi_fun)
        #plt.show(block=False)
        plt.tight_layout()
        plt.show()

    def save(self, folder_name):
        os.makedirs(os.path.dirname(folder_name), exist_ok=True)
        np.save(folder_name + 'Phi.npy', self.phi)
        np.save(folder_name + 'Fn.npy', self.fn)
    def k_and_m_matrices(self):
        # Elasticity part of bilinear form
        k_kl = fe.inner(fe.grad(fe.grad(self.u))[(self.alpha, self.beta)],
                        fe.as_tensor(self.c_tensor[(self.alpha, self.beta, self.gamma, self.delta)] *
                                     fe.grad(fe.grad(self.v))[(self.gamma, self.delta)])) * fe.dx

        k_kl -= fe.jump(fe.grad(self.v), self.normal) * fe.avg(self.c_tensor[self.alpha, self.beta,
                                                                             self.gamma, self.delta] *
                                                               fe.grad(fe.grad(self.u))[(self.alpha, self.beta)] *
                                                               self.normal[self.gamma] * self.normal[self.delta])*fe.dS

        k_kl -= fe.avg(self.c_tensor[(self.alpha, self.beta, self.gamma, self.delta)] *
                       fe.grad(fe.grad(self.v))[(self.alpha, self.beta)] * self.normal[self.gamma]
                       * self.normal[self.delta]) * fe.jump(fe.grad(self.u), self.normal) * fe.dS

        k_kl -= self.c_tensor[(self.alpha, self.beta, self.gamma, self.delta)] * \
            fe.grad(fe.grad(self.u))[(self.alpha, self.beta)] * self.normal[self.gamma] * self.normal[self.delta] \
            * fe.inner(fe.grad(self.v), self.normal) * self.ds_h

        k_kl -= self.c_tensor[(self.alpha, self.beta, self.gamma, self.delta)] * \
            fe.grad(fe.grad(self.v))[(self.alpha, self.beta)] * self.normal[self.gamma] * self.normal[self.delta] \
            * fe.inner(fe.grad(self.u), self.normal) * self.ds_h

        k_kl += self.tau_ip / self.h_avg * self.c_tensor[(self.alpha, self.beta, self.gamma, self.delta)] \
            * self.normal[self.alpha]('+') * self.normal[self.beta]('+') * \
            self.normal[self.gamma]('+') * self.normal[self.delta]('+') * \
            fe.jump(fe.grad(self.v), self.normal) * fe.jump(fe.grad(self.u), self.normal) * fe.dS

        k_kl += self.tau_ip / self.h_e * self.c_tensor[(self.alpha, self.beta, self.gamma, self.delta)] * \
            self.normal[self.alpha] * self.normal[self.beta] * self.normal[self.gamma] * self.normal[self.delta]\
            * fe.inner(fe.grad(self.v), self.normal) * fe.inner(fe.grad(self.u), self.normal) * self.ds_h

        l_kl = fe.Constant(0) * self.v * fe.dx

        m_kl = self.mat.rho_c * self.geometry.t_c * fe.inner(self.u, self.v) * fe.dx

        k_matrix = fe.PETScMatrix()
        b_vector = fe.PETScVector()

        fe.assemble_system(k_kl, l_kl, self.dirichlet_bc, A_tensor=k_matrix, b_tensor=b_vector)
        m_matrix = fe.PETScMatrix()
        fe.assemble(m_kl, tensor=m_matrix)
        self.m_matrix = fe.as_backend_type(m_matrix)
        self.k_matrix = fe.as_backend_type(k_matrix)

    def get_linear_form(self, force=1):

        l_form = fe.Constant(force) * self.v * fe.dx

        self.l_vec = fe.assemble(l_form)
        self.dirichlet_bc.apply(self.l_vec)
        return self.l_vec

    def apply_point_force(self, force, xf, yf):
        self.get_linear_form(force=0)
        ps = fe.PointSource(self.VCG, fe.Point(xf, yf), force)
        ps.apply(self.l_vec)
        return self.l_vec

    def get_point_value(self, phi, x_p=None, y_p=None):
        if x_p is None:
            x_p = self.geometry.l_c
            y_p = self.geometry.w_c/2
        phi_real = fe.Function(self.VCG)
        phi_real.vector()[:] = np.array(np.real(phi))
        phi_real_out = phi_real(x_p, y_p)

        phi_im = fe.Function(self.VCG)
        phi_im.vector()[:] = np.array(np.imag(phi))
        phi_im_out = phi_im(x_p, y_p)
        return phi_real_out + 1j * phi_im_out

    def get_Q_from_energy(self, phi, omega):
        W_entry = np.pi*(np.real(1j*self.get_point_value(phi, self.geometry.l_c, -self.geometry.w_c/2)))
        i, j, k, l = ufl_indices(4)
        phi_r = fe.Function(self.VCG)
        phi_r.vector()[:] = np.array(np.real(phi))
        phi_i = fe.Function(self.VCG)
        phi_i.vector()[:] = np.array(np.imag(phi))
        W1 =  fe.assemble(fe.grad(fe.grad(phi_r))[(i, j)]*
        fe.as_tensor(self.c_tensor[(i, j, k, l)] * fe.grad(fe.grad(phi_r))[(k, l)]) * fe.dx)

        W2 =  fe.assemble(fe.grad(fe.grad(phi_r))[(i, j)]*
                        fe.as_tensor(self.c_tensor[(i, j, k, l)] * fe.grad(fe.grad(phi_i))[(k, l)]) * fe.dx)

        W3 =  fe.assemble(fe.grad(fe.grad(phi_i))[(i, j)]*
                        fe.as_tensor(self.c_tensor[(i, j, k, l)] * fe.grad(fe.grad(phi_r))[(k, l)]) * fe.dx)


        W4 =  fe.assemble(fe.grad(fe.grad(phi_i))[(i, j)]*
                        fe.as_tensor(self.c_tensor[(i, j, k, l)] * fe.grad(fe.grad(phi_i))[(k, l)]) * fe.dx)

        tt = np.linspace(0, np.pi/omega, 400)
        Wb_tt = W1*np.cos(omega*tt)*np.cos(omega*tt)
        Wb_tt -= W2*np.cos(omega*tt)*np.sin(omega*tt)
        Wb_tt -= W3*np.cos(omega*tt)*np.sin(omega*tt)
        Wb_tt += W4*np.sin(omega*tt)*np.sin(omega*tt)
        Wmax = max(1/2*Wb_tt)
        Q = np.abs(2*np.pi*Wmax/W_entry)
        return Q



    def setup_neumann_bcs(self, mesh, l_c, w_c):
        tol = 1e-14

        # Define the Clamped Boundary conditions
        class BoundaryX0(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], 0, tol)

        # Define the Free End Boundary conditions
        class BoundaryX1(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (fe.near(x[0], l_c, tol) or fe.near(x[1], -w_c/2, tol) or fe.near(x[1], w_c/2, tol))
        #
        boundary_markers = fe.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
        boundary_markers.set_all(0)
        bx0 = BoundaryX0()
        bx1 = BoundaryX1()
        bx0.mark(boundary_markers, 1)
        bx1.mark(boundary_markers, 2)
        ds = fe.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
        ds_h = ds(1)
        ds_moment = ds(2)
        ds_shear_force = ds(2)
        return ds_h, ds_moment, ds_shear_force
