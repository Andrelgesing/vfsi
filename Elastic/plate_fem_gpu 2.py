#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import fenics as fe
import numpy as np
import cupy as cp
import copy
import matplotlib.pyplot as plt
import sys
import os
from ufl import indices as ufl_indices
#import scipy.sparse as sps
import cupyx.scipy.sparse as sps
#import General.Setup as Stp



class Geometry(object):
    """
    Initialize a rectangular geometry of a MEMS resonator.

    l_c : float, optional
        Length of the cantilever in meters (default is 400e-6).
    w_c : float, optional
        Width of the cantilever in meters (default is 100e-6).
    t_c : float, optional
        Thickness of the cantilever in meters (default is 10e-6).

    Attributes
    l_c : float
        Length of the cantilever in meters.
    w_c : float
        Width of the cantilever in meters.
    t_c : float
        Thickness of the cantilever in meters.
    a_r : float
        Aspect ratio of the cantilever (length/width).
    i_c : float
        Inertial moment of the cantilever.
    k_c : float
        Inertial moment for Green's method.
    """
    def __init__(self, l_c=400e-6, w_c=100e-6, t_c=10e-6):
        self.l_c = l_c  # Length of the plate [m]
        self.w_c = w_c  # Width of the plate [m]
        self.t_c = t_c  # Thickness of the plate [m]
        self.a_r = self.l_c / self.w_c  # Aspect ratio of the plate [-]
        #self.k_c = self.w_c*self.t_c ** 3 / 12  # Inertial moment [Only applicable for Green's method]

class Material(object):
    """
    Define the material of the cantilever. For isotropic, give Young's modulus, density, and Poisson ratio.
    For anisotropic, give the 5x1 vector of the c-tensor in the form [c_xxxx, c_yyyy, c_xxyy, c_xyxy, cyxxy]

    Parameters
    ----------
    youngs_modulus: float, optional
                    Young's modulus of the material of the cantilever [Pa] (for isotropic)
    density:  float, optional
                Material density [kg/m³] (for isotropic)
    poisson: float, optional
            Poisson coefficient (for isotropic)
    c_values: list, optional
            5x1 vector of anisotropic material parameters (for anisotropic)

    Attributes
    ----------
    e_c : float
        Young's modulus of the material (for isotropic).
    rho_c : float
        Density of the material (for isotropic).
    nu_c : float
        Poisson's ratio of the material (for isotropic).
    g_c : float
        Shear modulus of the material (for isotropic).
    c_tensor : ufl.tensor
        Elasticity tensor of the material.
    """

    def __init__(self, youngs_modulus=169E9, density=2.33E3, poisson=0.3,  c_values=None):
        self.c_vector = np.zeros((2, 2, 2, 2))
        self.indices_ = ufl_indices(4)
        self.rho_c = density
        self.c_values = c_values
        if c_values is None:
            #print('Isotropic material')
            self.material_type = 'isotropic'
            self.e_c = youngs_modulus
            self.nu_c = poisson
            self.g_c = youngs_modulus / 2 / (1 + poisson)  # Shear modulus
            #kr = fe.Identity(2)
            self.alpha, self.beta, self.gamma, self.delta = self.indices_
            self.mu_c = self.e_c / 2 / (1 + self.nu_c)
            self.lam_c = self.e_c * self.nu_c / (1 - self.nu_c ** 2)
            #self.c_tensor = fe.as_tensor( (self.mu_c_fe * (kr[(self.alpha, self.gamma)] 
            #                                                            * kr[(self.beta, self.delta)] + 
            #               kr[(self.alpha, self.delta)] * kr[(self.beta, self.gamma)])
            #               + self.lam_c_fe * kr[(self.alpha, self.beta)] *  kr[(self.gamma, self.delta)]),  [self.alpha, self.beta, self.gamma, self.delta])
        elif c_values is not None and len(c_values) == 5:
            print('Anisotropic material')
            self.material_type = 'anisotropic'
            self.c_values = c_values
            c_xxxx, c_yyyy, c_xxyy, c_xyxy, c_yxxy = self.c_values
            c_yyxx = c_xxyy
            c_yxyx = c_yxxy
            c_xyyx = c_yxyx
            self.component_list = np.zeros((2, 2, 2, 2))
            self.component_list[0][0][0][0] = c_xxxx
            self.component_list[0][0][1][1] = c_xxyy
            self.component_list[1][1][0][0] = c_yyxx
            self.component_list[1][1][1][1] = c_yyyy
            self.component_list[0][1][0][1] = c_xyxy
            self.component_list[0][1][1][0] = c_xyyx
            self.component_list[1][0][1][0] = c_yxxy
            self.component_list[1][0][0][1] = c_yxyx
            self.c_vector = self.component_list
        else:
            raise ValueError(
                "Invalid parameters for material initialization. Either start the material as 'isotropic' and provide "
                "Young's modulus, density, and Poisson ratio, or provide a list of 5 anisotropic values for the c-tensor. "
                "For example, for silicon in 110 orientation, that would be [194.5, 194.5, 35.7, 50.9, 50.9]"
            )
            

class Kirchhoff(object):
    """
    Class for solving the Kirchhoff-Love plate problem using the Interior Penalty Method as proposed in
    "Engel, Gerald, et al. Computer Methods in Applied Mechanics and Engineering 191.34 (2002): 3669-3750".
    This class provides tools for finite element analysis of plate elasticity problems, 
    including isotropic and anisotropic materials. It supports meshing, function space 
    definition, boundary condition setup, eigenvalue analysis, and visualization of 
    eigenmodes.
    Attributes:
        geometry (Geometry): Plate geometry parameters.
        material (Material): Material properties.
        indices_ (tuple): UFL indices for tensor operations.
        c_values (list or None): Stiffness coefficients for anisotropic materials.
        a_r (float): Aspect ratio of the plate.
        i_c_fe, mu_c_fe, lam_c_fe (fenics.Constant): Material constants.
        n_x, n_y (int): Mesh dimensions.
        c_tensor (fenics.Tensor or None): Elastic tensor.
        mesh (fenics.Mesh): Finite element mesh.
        h_e, h_avg (fenics.Expression): Cell diameter and average for IP method.
        mesh_type (str): Type of mesh ('crossed' by default).
        normal (fenics.FacetNormal): Normal vector of the mesh.
        l_ (int): Degree of FEM function space.
        VCG (fenics.FunctionSpace): Function space for FEM.
        u, v (fenics.Function): Trial and test functions.
        phi (numpy.ndarray): Displacement array.
        tau_ip (float): Penalty parameter for FEM operator.
        k_matrix, m_matrix (fenics.Matrix): Stiffness and mass matrices.
        dirichlet_bc (fenics.DirichletBC): Dirichlet boundary conditions.
        ds_h, ds_moment, ds_shear_force (fenics.Measure): Boundary condition measures.
        l_vec (fenics.Vector): Linear form vector.
    Methods:
        preliminary_setup(): Initializes parameters and tensors for FEM analysis.
        meshing(nx=None, ny=None, type_=None): Defines the plate mesh.
        function_spaces(l_=None): Sets up FEM function spaces.
        set_bcs(): Configures boundary conditions for the plate.
        solve_n_eigenvalues(n_eig=6, k=None, m=None): Solves for eigenvalues and eigenfrequencies.
        plot_eigenmodes(n_eig): Visualizes eigenmodes of the system.
        save(folder_name): Saves eigenvectors and frequencies to files.
        k_and_m_matrices(): Computes stiffness and mass matrices.
        added_mass_matrix(kappa=1): Calculates added mass matrix for uniform density.
        get_linear_form(force=1): Assembles the linear form vector for a given force.
        apply_point_force(force, xf, yf): Applies a point force at a specified location.
        get_point_value(phi, x_p=None, y_p=None): Retrieves the value of a mode shape at a point.
        get_Q_from_energy(phi, omega): Computes the quality factor from system energy.
    """ 
    def __init__(self):
        # Initialize parameters
        self.geometry = Geometry()
        self.material = Material()
        self.indices_ = ufl_indices(4)
        self.c_values = None
        #self.geometry = geometry
        #self.mat = material
        self.a_r = self.geometry.l_c / self.geometry.w_c  # Aspect ratio of the plate [-]
        self.i_c_fe = 0#fe.Constant(self.geometry.t_c ** 3 / 12)  # Copy of i_c, as a fenics constant
        self.mu_c_fe = 0#fe.Constant(self.mat.e_c / 2 / (1 + self.mat.nu_c))  # Copy of mu_c, as a fenics constant
        self.lam_c_fe = 0#fe.Constant(self.mat.e_c * self.mat.nu_c / (1 - self.mat.nu_c ** 2))  # Copy of lam_c, as a
        # fenics constant
        self.n_x = 32  # Number of elements of the mesh in x-direction
        self.n_y = int(self.n_x / self.geometry.a_r)  # Number of elements of the mesh in y-direction
        self.c_tensor = None  # Initialize c_tensor
        self.mesh = fe.UnitIntervalMesh(self.n_x)  # Initialize mesh
        self.h_e = fe.CellDiameter(self.mesh)
        self.h_avg = (self.h_e('+') + self.h_e('-')) / 2
        self.mesh_type = 'crossed'  # Type of mesh. Default is 'crossed'
        self.normal = fe.FacetNormal(self.mesh)  # normal vector in mesh
        self.l_ = 2  # Degree of the FEM function space
        self.VCG = None#fe.FunctionSpace(self.mesh, fe.FiniteElement("CG", self.mesh.ufl_cell(), degree=self.l_))
        self.u = None#fe.TrialFunction(self.VCG)
        self.v = None#fe.TestFunction(self.VCG)
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
        """
        Performs the preliminary setup for the Interior Penalty method.

        This method initializes various parameters and tensors required for the 
        finite element analysis of the plate elasticity problem. It handles both 
        isotropic and anisotropic material types and sets up the corresponding 
        elastic tensors.

        Key steps include:
        - Setting up geometric parameters such as thickness, width, and length.
        - Defining the moment of inertia as a FEniCS constant.
        - For isotropic materials:
            - Initializing the elastic tensor using the Lamé parameters (mu and lambda).
        - For anisotropic materials:
            - Constructing the elastic tensor from the provided stiffness coefficients.
        - Calling subsequent methods to handle meshing, function spaces, and boundary conditions.

        Prints:
            A message indicating the mesh dimensions after setup.

        Note:
            This method assumes that the `geometry` and `material` attributes are 
            already defined in the class instance, and that they provide the necessary 
            properties such as material type, stiffness coefficients, and geometric dimensions.
        """
    
        self.indices_ = ufl_indices(4)
        self.alpha, self.beta, self.gamma, self.delta = self.indices_
        self.t_c, self.w_c, self.l_c = self.geometry.t_c, self.geometry.w_c, self.geometry.l_c
        self.i_c_fe = fe.Constant(self.t_c ** 3 / 12)

        if self.material.material_type == 'isotropic':
            kr = fe.Identity(2)
            self.i_c_fe = fe.Constant(self.t_c ** 3 / 12)  # Copy of i_c, as a fenics constant
            self.mu_c_fe = fe.Constant(self.material.mu_c) # Copy of mu_c, as a fenics constant
            self.lam_c_fe = fe.Constant(self.material.lam_c)
            ## Elastic tensor
            self.c_tensor_fe = fe.as_tensor(self.i_c_fe * (self.mu_c_fe * (kr[(self.alpha, self.gamma)] *
                                                                    kr[(self.beta, self.delta)]
                                                                        + kr[(self.alpha, self.delta)] * kr[(self.beta,
                                                                                                            self.gamma)])
                                                        + self.lam_c_fe * kr[(self.alpha, self.beta)] *
                                                        kr[(self.gamma, self.delta)]),
                                        [self.alpha, self.beta, self.gamma, self.delta])
        else:
            self.c_values = self.material.c_values
            c_xxxx, c_yyyy, c_xxyy, c_xyxy, c_yxxy = self.c_values
            c_yyxx = copy.copy(c_xxyy)
            c_yxyx = copy.copy(c_yxxy)
            c_xyyx = copy.copy(c_yxyx)
            c_vector = np.zeros((2, 2, 2, 2))
            c_vector[0][0][0][0] = c_xxxx
            c_vector[0][0][1][1] = c_xxyy
            c_vector[1][1][0][0] = c_yyxx
            c_vector[1][1][1][1] = c_yyyy
            c_vector[0][1][0][1] = c_xyxy
            c_vector[0][1][1][0] = c_xyyx
            c_vector[1][0][1][0] = c_yxxy
            c_vector[1][0][0][1] = c_yxyx
            self.c_tensor_fe = self.i_c_fe*fe.as_tensor(c_vector)
        self.meshing()
        #print('\n Setup Mesh %d x %d.' % (self.n_x, self.n_y))
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
        """
        Parameters
        ----------
        mesh : fenics.Mesh
            The mesh of the plate.
        l_c : float
            Length of the cantilever in meters.
        w_c : float
            Width of the cantilever in meters.

        Returns
        -------
        dim = mesh.topology().dim() - 1
        boundary_markers = fe.MeshFunction('size_t', mesh, dim)
            Measure for the clamped boundary condition.
        ds_moment : fenics.Measure
            Measure for the free end boundary condition (moment).
        ds_shear_force : fenics.Measure
            Measure for the free end boundary condition (shear force).

        This method sets up the boundary markers for the clamped and free end boundary conditions
        and returns the corresponding measures for use in defining boundary conditions in the FEM problem.
        """
        tol = 1e-14

        # Define the Clamped Boundary conditions
        class BoundaryX0(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], 0, tol)
        geometry = self.geometry
        # Define the Free End Boundary conditions
        class BoundaryX1(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (fe.near(x[0], geometry.l_c, tol) or 
                                        fe.near(x[1], -geometry.w_c/2, tol) or fe.near(x[1], geometry.w_c/2, tol))
        #
        boundary_markers = fe.MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1)
        boundary_markers.set_all(0)
        bx0 = BoundaryX0()
        bx1 = BoundaryX1()
        bx0.mark(boundary_markers, 1)
        bx1.mark(boundary_markers, 2)
        ds = fe.Measure('ds', domain=self.mesh, subdomain_data=boundary_markers)
        self.ds_h = ds(1)
        self.ds_moment = ds(2)
        self.ds_shear_force = ds(2)

        def left(x, on_boundary):
            return on_boundary and fe.near(x[0], 0, tol)

        self.dirichlet_bc = fe.DirichletBC(self.VCG, fe.Constant(0), left)


    def solve_n_eigenvalues(self, n_eig=6, k=None, m=None):
        """
        Solve the first n_eig eigenvalues of the generalized eigenvalue problem.

        Parameters:
        n_eig (int): Number of eigenvalues to solve for.
        k (numpy.ndarray or None): Stiffness matrix. If None, the stiffness matrix will be set up internally.
        m (numpy.ndarray or None): Mass matrix. If None, the mass matrix will be set up internally.

        Returns:
        tuple: A tuple containing:
            - phi (numpy.ndarray): Matrix of eigenvectors.
            - fn (numpy.ndarray): Array of eigenfrequencies in Hz.

        Notes:
        - Uses the SLEPcEigenSolver from the FEniCS library.
        - The eigenvalue problem is solved using the 'arnoldi' solver with 'shift-and-invert' spectral transform.
        - Eigenfrequencies are computed from the eigenvalues and converted to Hz.
        """
        if k is not None and m is not None:
            self.k_matrix = k
            self.m_matrix = m
        else:
            self.setup_eigenvalues_problem()
        eigensolver = fe.SLEPcEigenSolver(self.k_matrix, self.m_matrix)
        eigensolver.parameters['problem_type'] = 'gen_hermitian'
        eigensolver.parameters['solver'] = 'arnoldi'  # 'arnoldi', 'krylov-schur', 'lapack'
        eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
        eigensolver.parameters['spectral_shift'] = 0.
        eigensolver.parameters['tolerance'] = 1e-10
        eigensolver.solve(int(n_eig))

        self.phi = np.zeros([self.k_matrix.size(0), n_eig], dtype=float)
        self.fn = np.zeros(n_eig, dtype=float)
        for ii in range(n_eig):
            r, c, mode, s = eigensolver.get_eigenpair(ii)
            self.phi[:, ii] = mode.get_local()
            self.fn[ii] = np.sqrt(np.abs(r)) / 2 / np.pi
            print('         Mode %d = %1.1f kHz' % (ii + 1, self.fn[ii] / 1e3))
        return self.phi, self.fn

    def plot_eigenmodes(self, n_eig):
        """
        Plots the eigenmodes of the system.

        Parameters:
        n_eig (int): Number of eigenmodes to plot.

        This function creates a figure with subplots for each eigenmode. Each subplot
        displays the eigenmode shape and its corresponding frequency in kHz.
        """
        phi_fun = fe.Function(self.VCG)
        plt.figure(figsize=(10, n_eig))
        for ii in range(n_eig):
            plt.subplot(3, int(np.ceil(n_eig / 3)), ii + 1)
            plt.title('fn= '+str(int(self.fn[ii]/1e3))+' kHz')
            phi_fun.vector()[:] = np.array(self.phi[:, ii].copy())
            fe.plot(phi_fun)
        #plt.show(block=False)
        plt.tight_layout()
        plt.show()

    def save(self, folder_name):
        """
        Save the phi and fn attributes to .npy files in the specified folder.

        Parameters:
        folder_name (str): The path to the folder where the files will be saved. 
                           The folder will be created if it does not exist.

        Files Saved:
        - Phi.npy: Contains the self.phi attribute.
        - Fn.npy: Contains the self.fn attribute.
        """
        os.makedirs(os.path.dirname(folder_name), exist_ok=True)
        np.save(folder_name + 'Phi.npy', self.phi)
        np.save(folder_name + 'Fn.npy', self.fn)

    def k_and_m_matrices(self):
        """
        Calculates the stiffness and mass matrices for the plate based on the interior penalty method.

        This method computes the stiffness matrix (k_matrix) and the mass matrix (m_matrix) for a plate
        using the finite element method (FEM) with an interior penalty approach. The method involves
        assembling the system of equations and applying Dirichlet boundary conditions.

        Returns:
            tuple: A tuple containing the stiffness matrix (k_matrix) and the mass matrix (m_matrix).
        """
        self.preliminary_setup()
        k_kl = fe.inner(fe.grad(fe.grad(self.u))[(self.alpha, self.beta)],
                fe.as_tensor(self.c_tensor_fe[(self.alpha, self.beta, self.gamma, self.delta)] *
                         fe.grad(fe.grad(self.v))[(self.gamma, self.delta)])) * fe.dx

        k_kl -= fe.jump(fe.grad(self.v), self.normal) * fe.avg(self.c_tensor_fe[self.alpha, self.beta,
                                             self.gamma, self.delta] *
                                       fe.grad(fe.grad(self.u))[(self.alpha, self.beta)] *
                                       self.normal[self.gamma] * self.normal[self.delta])*fe.dS

        k_kl -= fe.avg(self.c_tensor_fe[(self.alpha, self.beta, self.gamma, self.delta)] *
                   fe.grad(fe.grad(self.v))[(self.alpha, self.beta)] * self.normal[self.gamma]
                   * self.normal[self.delta]) * fe.jump(fe.grad(self.u), self.normal) * fe.dS

        k_kl -= self.c_tensor_fe[(self.alpha, self.beta, self.gamma, self.delta)] * \
            fe.grad(fe.grad(self.u))[(self.alpha, self.beta)] * self.normal[self.gamma] * self.normal[self.delta] \
            * fe.inner(fe.grad(self.v), self.normal) * self.ds_h

        k_kl -= self.c_tensor_fe[(self.alpha, self.beta, self.gamma, self.delta)] * \
            fe.grad(fe.grad(self.v))[(self.alpha, self.beta)] * self.normal[self.gamma] * self.normal[self.delta] \
            * fe.inner(fe.grad(self.u), self.normal) * self.ds_h

        k_kl += self.tau_ip / self.h_avg * self.c_tensor_fe[(self.alpha, self.beta, self.gamma, self.delta)] \
            * self.normal[self.alpha]('+') * self.normal[self.beta]('+') * \
            self.normal[self.gamma]('+') * self.normal[self.delta]('+') * \
            fe.jump(fe.grad(self.v), self.normal) * fe.jump(fe.grad(self.u), self.normal) * fe.dS

        k_kl += self.tau_ip / self.h_e * self.c_tensor_fe[(self.alpha, self.beta, self.gamma, self.delta)] * \
            self.normal[self.alpha] * self.normal[self.beta] * self.normal[self.gamma] * self.normal[self.delta]\
            * fe.inner(fe.grad(self.v), self.normal) * fe.inner(fe.grad(self.u), self.normal) * self.ds_h

        l_kl = fe.Constant(1) * self.v * fe.dx

        m_kl = self.material.rho_c * self.t_c * fe.inner(self.u, self.v) * fe.dx

        k_matrix = fe.PETScMatrix()
        b_vector = fe.PETScVector()

        fe.assemble_system(k_kl, l_kl, self.dirichlet_bc, A_tensor=k_matrix, b_tensor=b_vector)
        m_matrix = fe.PETScMatrix()
        fe.assemble(m_kl, tensor=m_matrix)
        self.m_matrix = fe.as_backend_type(m_matrix)
        self.k_matrix = fe.as_backend_type(k_matrix)
        return self.k_matrix, self.m_matrix

    def added_mass_matrix(self,kappa=1):
        """Calculate an added mass matrix referring to a uniform density kappa """
        m_ad = kappa*fe.inner(self.u, self.v) * fe.dx
        m_matrix = fe.PETScMatrix()
        fe.assemble(m_ad, tensor=m_matrix)
        #M_ad = fe.as_backend_type(m_ad)
        self.added_mass_matrix = m_matrix
        return m_matrix

    def get_linear_form(self, force=1):
        """
        Assemble the linear form vector for the given force.

        Parameters:
        force (float, optional): The force to be applied. Default is 1.

        Returns:
        dolfin.cpp.la.PETScVector: The assembled linear form vector with applied Dirichlet boundary conditions.
        """

        l_form = fe.Constant(force) * self.v * fe.dx

        self.l_vec = fe.assemble(l_form)
        self.dirichlet_bc.apply(self.l_vec)
        return self.l_vec

    def apply_point_force(self, force, xf, yf):
        """
        Apply a point force to the finite element model at a specified location.

        Parameters:
        force (float): The magnitude of the force to be applied.
        xf (float): The x-coordinate of the point where the force is applied.
        yf (float): The y-coordinate of the point where the force is applied.

        Returns:
        dolfin.cpp.la.PETScVector: The resulting vector after applying the point force.
        """
        self.get_linear_form(force=0)
        ps = fe.PointSource(self.VCG, fe.Point(xf, yf), force)
        ps.apply(self.l_vec)
        return self.l_vec

    def get_point_value(self, phi, x_p=None, y_p=None):
        """
        Get the value of the complex function `phi` at a specific point (x_p, y_p).

        Parameters:
        -----------
        phi : array-like
            The complex function values.
        x_p : float, optional
            The x-coordinate of the point where the value is to be evaluated. 
            If not provided, defaults to the center length of the geometry.
        y_p : float, optional
            The y-coordinate of the point where the value is to be evaluated. 
            If not provided, defaults to half the width of the geometry.

        Returns:
        --------
        complex
            The value of the complex function `phi` at the point (x_p, y_p).
        """
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

    def get_stored_energy(self, phi, omega):
        """
        Calculate the stored energy of the system assuming a complex displacement field phi
        and frequency omega.

        Parameters:
        -----------
        phi : complex
            The complex mode shape function.
        omega : float
            The angular frequency of the system.

        Returns:
        --------
        Q : float
            The calculated quality factor.
        """
        #W_entry = np.pi*(np.real(1j*self.get_point_value(phi, self.geometry.l_c, -self.geometry.w_c/2)))
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
        #Q = np.abs(2*np.pi*Wmax/W_entry)
        return Wmax


    def external_force(self, Mode):
        xp = []
        yp = []
        xn = []
        yn = []
        test = Mode.split(':')
        Mode_x = int(test[0])
        Mode_y = int(test[1])
        l_c = self.geometry.l_c
        w_c = self.geometry.w_c
        if Mode_y == 0:  # Euler Bernoulli mode
            #yp = list(np.linspace(-w_c / 2, w_c / 2, 30))
            yp = []
            yn = [-62*w_c/256, 62*w_c/256]
        ### Torsional
        elif Mode_y == 1:
            yp = list(np.linspace(0, w_c / 2, 5))
            yn = list(-np.linspace(0, w_c / 2, 5))
        elif Mode_y == 2: #Double
            #yp = list(np.linspace(-62*w_c/256, 62*w_c/256, 10))
            #yn =list(np.linspace(-w_c/2,-62*w_c/256, 5)) + list(np.linspace(62*w_c/256,w_c/2, 5))
            yp = [0]
            yn = [-w_c/2, w_c/2]
        elif Mode_y == 3: #Triple
            yp = list(np.linspace(-81*w_c/256,0,  5)) + list(np.linspace(81*w_c/256, w_c/2, 5))
            yn =list(np.linspace(-w_c/2,-81*w_c/256, 5)) + list(np.linspace(0,81/256*w_c, 5))
        elif Mode_y == 4: # Quadruple
            yp = list(np.linspace(-98/256*w_c, -32/256*w_c,  7)) + list(np.linspace(32*w_c/256, 98/256*w_c, 7))
            yn =list(np.linspace(-w_c/2,-98*w_c/256, 5)) + list(np.linspace(-32/256*w_c,32/256*w_c, 5)) + list(np.linspace(98/256*w_c,w_c/2, 5))
        elif Mode_y == 5:# Quintuple
            yp = [-w_c/2, -28/256*w_c, 81/256*w_c]
            yn = [w_c/2, 28/256*w_c, -81/256*w_c]
        elif Mode_y == 6:# Sextuple
            yn = [69/256*w_c, 0, 69/256*w_c]
            yp = [-w_c/2, -23/256*w_c, +23/256*w_c, w_c/2]#
        # elif mode == 50: # Sectuple
        #    yn = [w_c/2, 58/256*w_c, -20/256*w_c, -95/256*w_c]
        #    yp = [-w_c/2, -58/256*w_c, 20/256*w_c, 95/256*w_c]
        # else:
        #    yp = list(np.linspace(-w_c/2, w_c/2, 30))
        #    yn = []

        if Mode_x == 1:  # nx = 1
            xp = list(np.linspace(0, l_c, 20))
            xn = []
        elif Mode_x == 2:  # nx = 2
            xn = [l_c]#list(np.linspace(0, 176 * l_c / 256, 10))
            xp = []#list(np.linspace(176 * l_c / 256, l_c, 5))
        elif Mode_x == 3:  # nx = 3
            xn = [120/256*l_c]
            xp = [40 / 256 * l_c, l_c]
        elif Mode_x == 4:  # nx = 4
            xn = [71 * l_c / 256, 199 * l_c / 256]
            xp = [130 * l_c / 256, l_c]
        elif Mode_x == 5:
            #xp = [l_c]
            #xn = [90/256*l_c]
            xp = list(np.linspace(0, 50 / 256 * l_c, 5)) + list(np.linspace(114 * l_c / 256, 175 / 256 * l_c, 5)) + list(
                np.linspace(230 / 256 * l_c, l_c, 5))
            xn = list(np.linspace(50 / 256 * l_c, 114 * l_c / 256, 5)) + list(
                np.linspace(175 * l_c / 256, 230 / 256 * l_c, 5))
        elif Mode_x == 6:  # nx = 6
            xp = [ 61/256*l_c, 177/256*l_c, l_c]
            xn = [23/256*l_c, 127/256*l_c, 222/256*l_c]
        elif Mode_x == 7:
            xp = [25/256*l_c, 110/256*l_c, 210/256*l_c,  l_c]
            xn = [60/256*l_c, 155/256*l_c, 230/256*l_c]
        elif Mode_x == 8:   #nx = 8
            xp = [l_c]#[60/256*l_c, 129/256*l_c, 196/256*l_c,  l_c]
            xn = []#[25/256*l_c, 94/256*l_c, 163/256*l_c, 229/256*l_c]
        elif Mode_x == 9:  # nx = 9
            xp = [23/256*l_c, 82/256*l_c, 145/256*l_c, 205/256*l_c,  l_c]
            xn = [56/256*l_c, 112/256*l_c, 173/256*l_c, 235/256*l_c]
        elif Mode_x == 10: #nx = 10
            xp = [52/256*l_c, 100/256*l_c, 154/256*l_c, 185/256*l_c,  239/256*l_c]
            xn = [21/256*l_c, 72/256*l_c, 130/256*l_c,  210/256*l_c, l_c]
        elif Mode_x == 11: #nx = 11
            xp = [38/256*l_c, 84/256*l_c, 128/256*l_c, 150/256*l_c,  204/256*l_c, l_c]
            xn = [16/256*l_c, 61/256*l_c, 107/256*l_c,  175/256*l_c, 231/256*l_c]
        else:
            xp = [l_c]
            xn = []
        return xp, xn, yp, yn


