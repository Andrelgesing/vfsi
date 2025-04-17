#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fenics as fe
import numpy as np
import scipy.sparse as sp
import scipy.special as special
import Plate.plate_fem as pt

class Fluid(object):
    """
    Define fluid properties.

    Parameters
    ----------
    dynamic_viscosity : float
              absolute (or dynamic) viscosity [Ns/(m^2)]
    density:  float
              fluid density [kg/m³]
    """
    def __init__(self, dynamic_viscosity=8.9e-4, density=997):
        self.mu_f = dynamic_viscosity  # Fluid dynamic viscosity in Pa.s
        self.rho_f = density  # Fluid density in kg/m^3
        self.nu_f = self.mu_f / self.rho_f  # Fluid kinematic viscosity in m^2/s

class F2D(object):
    """ 2D fluid flow formulation"""
    def __init__(self):
        # Initialize parameters
        #self.mat = pt.Material()
        self.fluid = Fluid()
        self.fem = pt.InteriorPenalty()
        self.geometry = self.fem.geometry#pt.Geometry()
        self.n_x_fluid = 16
        self.n_y_fluid = 64
        self.l_c = self.geometry.l_c
        self.w_c = self.geometry.w_c
        self.t_c = self.geometry.t_c
        self.x = np.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.dx = self.x[1] - self.x[0]
        self.wx = 1 / 3 * self.dx
        self.y = np.linspace(0, self.l_c, self.n_x_fluid + 1)
        self.wy = 1 / 3 * self.dx
        self.a = np.ones((10, 10), dtype=complex)
        self.w = np.ones((10, 10), dtype=float)
        self.p = np.ones((10, 10), dtype=complex)

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
        self.x = np.linspace(0, self.l_c, self.n_x_fluid)
        self.dx = self.x[1] - self.x[0]
        i = np.arange(1, self.n_y_fluid + 1)
        # self.y = -np.cos( i / (self.n_y_fluid + 1) * np.pi)*self.w_c/2 # First version, wrong
        self.y = -np.cos(((2 * i - 1) / (2 * self.n_y_fluid)) * np.pi)*self.w_c/2

        w_x = np.ones(len(self.x))
        w_x[1:-1:2] = 4
        w_x[2:-2:2] = 2
        self.wx = 1/3*self.dx*w_x
        y_sqrt = np.sqrt((self.w_c/2)**2 - self.y**2)
        self.wy = np.pi / self.n_y_fluid * y_sqrt
        #nyf = self.n_y_fluid
        #w_c = self.w_c
        #i = np.arange(1, nyf + 1)
        #y_temp = -w_c / 2 * np.cos(((2 * i - 1) / (2 * nyf)) * np.pi)
        #y_sqrt = np.sqrt((w_c / 2) ** 2 - y_temp ** 2)
        #self.wy = np.pi / self.n_y_fluid * y_sqrt
        self.export_basis_functions()


    def get_h_force(self, omega):
        self.get_a_matrix(omega)
        a_inv = np.linalg.inv(self.a)
        a_dia = [a_inv for ii in range(self.n_x_fluid)]
        del a_inv
        a_inv_dia = sp.block_diag(a_dia, format='csr', dtype='complex64')
        del a_dia
        p = a_inv_dia @ self.w.T
        del a_inv_dia

        p_ = p.T.multiply(np.repeat(self.wx, self.n_y_fluid).flatten() * np.tile(self.wy, self.n_x_fluid).flatten()).T
        self.p = self.fluid.mu_f * self.w @ p_
        return self.p

    def f_func(self, z):
        return (1 / z) + special.kerp(z) + 1j * special.keip(z)

    def get_a_matrix(self, omega):
        """Returns the A matrix from Tuck (1969)

            Parameters
            ----------
            omega: float
                 frequency in radians per second
            Returns
            -------
            a : np.array(type=np.complex)
                the complex-valued A matrix
        """

        #i = np.arange(1, ny + 1)
        #y = -wp / 2 * np.cos(((2 * i - 1) / (2 * ny)) * np.pi)
        wp = self.w_c
        nu_f = self.fluid.nu_f
        y_lim = np.concatenate((np.array([-wp / 2]), 0.5 * (self.y[1:] + self.y[:-1]), np.array([wp / 2])))
        z1 = np.sqrt(omega / nu_f) * (y_lim[np.newaxis, 1:] - self.y[:, np.newaxis])
        z2 = np.sqrt(omega / nu_f) * (y_lim[np.newaxis, :-1] - self.y[:, np.newaxis])
        f1 = self.f_func(z1)
        f1[z1 < 0] = -self.f_func(-z1[z1 < 0])
        f2 = self.f_func(z2)
        f2[z2 < 0] = -self.f_func(-z2[z2 < 0])
        self.a = 1 / (2 * np.pi * 1j) * np.sqrt(nu_f) / np.sqrt(omega) * (f1 - f2)



    def export_basis_functions(self):
        #x, y = setup_fluid_grid_simps(l_c, w_c, n_x_fluid, n_y_fluid)
        #self.fem.geometry = self.geometry
        #self.fem.meshing(self.n_x_fluid-1, self.n_y_fluid-1, 'left/right')
        #mesh_fluid = self.fem.mesh
        mesh_fluid = fe.RectangleMesh(fe.Point(0, -self.geometry.w_c / 2),
                                      fe.Point(self.geometry.l_c, self.geometry.w_c / 2),
                                     self.n_x_fluid-1, self.n_y_fluid-1, 'left/right')
        (mesh_fluid.coordinates()[:, 1]).sort(axis=0)
        xf, yf = mesh_fluid.coordinates()[:, 0], mesh_fluid.coordinates()[:, 1]
        yf[:] = np.repeat(self.y, len(self.x))
        v2 = fe.FunctionSpace(mesh_fluid, 'CG', 1)
        self.fem.function_spaces()
        #transfer_matrix = fe.PETScDMCollection.create_transfer_matrix(self.fem.VCG, v2)
        transfer_matrix = fe.PETScDMCollection.create_transfer_matrix(self.fem.VCG, v2)
        arg_x = np.argsort(v2.tabulate_dof_coordinates()[:, 0])
        coord_x = v2.tabulate_dof_coordinates()[arg_x, :]
        arg_y = np.argsort(np.reshape(coord_x[:, 1], (len(self.x), len(self.y))))
        arg_y2 = (arg_y.T + np.arange(len(arg_y)) * (len(self.y))).T.flatten()
        row, col, val = fe.as_backend_type(transfer_matrix).mat().getValuesCSR()
        w = sp.csr_matrix((val, col, row), dtype='float64')
        self.w = (w[arg_x])[arg_y2].T
        #return w


    #
    # def setup_fluid_grid_simps(l_c, w_c, n_x_fluid, n_y_fluid):
    #     x = np.linspace(0, l_c, n_x_fluid + 1)
    #     i = np.arange(1, n_y_fluid + 1)
    #     #y = -np.cos(((2 * i - 1) / (2 * n_y_fluid)) * np.pi)*w_c/2
    #     y = -np.cos(i / (n_y_fluid + 1) * np.pi) * w_c / 2
    #     return x, y
    #
    # def export_basis_functions_stokeslet(v, l_c, w_c, xp, yp):
    #     mesh_fluid = fe.RectangleMesh(fe.Point(0, -w_c/2), fe.Point(l_c, w_c/2), len(xp)-1, len(yp)-1, 'left/right')
    #     (mesh_fluid.coordinates()[:, 1]).sort(axis=0)
    #     xf, yf = mesh_fluid.coordinates()[:, 0], mesh_fluid.coordinates()[:, 1]
    #     yf[:] = np.repeat(yp*l_c, len(xp))
    #     xf[np.argsort(xf)] = np.repeat(xp*l_c, len(yp))
    #     xf = np.repeat(xp*l_c, len(yp))
    #     #fe.plot(mesh_fluid)
    #     v2 = fe.FunctionSpace(mesh_fluid, 'CG', 1)
    #     transfer_matrix = fe.PETScDMCollection.create_transfer_matrix(v, v2)
    #     #print(transfer_matrix.array().shape)
    #     arg_x = np.argsort(v2.tabulate_dof_coordinates()[:, 0])
    #     #print(arg_x.shape)
    #     coord_x = v2.tabulate_dof_coordinates()[arg_x, :]
    #     arg_y = np.argsort(np.reshape(coord_x[:, 1], (len(xp), len(yp))))
    #     arg_y2 = (arg_y.T + np.arange(len(arg_y)) * (len(yp))).T.flatten()
    #     #print(arg_y2.shape)
    #     row, col, val = fe.as_backend_type(transfer_matrix).mat().getValuesCSR()
    #     w0 = fe.Function(v)
    #     nb = w0.vector().size()
    #     w = sp.csr_matrix((val, col, row), shape=(len(xp)*len(yp), nb),  dtype='float64')
    #     #w = sp.csr_matrix((val, col, row), dtype='float64')
    #     #print(w.shape)
    #     w = (w[arg_x])[arg_y2].T
    #     return w
    #
    #
    #
    # def project_int_simps(w, omega, n_x_fluid, n_y_fluid, l_c, w_c, nu_f, mu_f):
    # #def project_int_simps(w, omega, n_x_fluid, n_y_fluid, l_c, w_c, nu_f, mu_f):
    #     # nb = w.shape[0]
    #     a = tuck.get_a_matrix(n_y_fluid, w_c, omega, nu_f)
    #     a_inv = np.linalg.inv(a)
    #     a_dia = [a_inv for ii in range(n_x_fluid+1)]
    #     del a, a_inv
    #     #a_inv_dia = sp.block_diag(a_dia)
    #     a_inv_dia = sp.block_diag(a_dia, format='csr', dtype='complex64')
    #     del a_dia
    #     p = a_inv_dia @ w.T
    #     del a_inv_dia
    #     # Chebyshev approach
    #     # X integration
    #     corr_int = 1 / 3 * (l_c / n_x_fluid)
    #     w_x = np.ones(n_x_fluid+1)
    #     w_x[1:-1:2] = 4
    #     w_x[2:-2:2] = 2
    #
    #     i = np.arange(1, n_y_fluid + 1)
    #     #yp = (-np.cos(((2 * i - 1) / (2 * n_y_fluid)) * np.pi))
    #     #yp = w_c/2*(-np.cos(((2 * i - 1) / (2 * n_y_fluid)) * np.pi))
    #     #y_temp = -w_c/2*np.cos(((2 * i - 1) / (2 * n_y_fluid+1)) * np.pi)
    #     #y_sqrt = np.sqrt((w_c/2)**2 - yp**2)
    #     weight_y = np.pi/(n_y_fluid+1)*np.sin(i/(n_y_fluid+1)*np.pi)**2*w_c/2
    #     #p = (p.T.multiply(np.repeat(w_x, nyf).flatten())).multiply(np.tile(y_sqrt, nxf).flatten())
    #     p = p.T.multiply(np.repeat(w_x, n_y_fluid).flatten() * np.tile(weight_y, n_x_fluid+1).flatten()).T
    #     #p = (p.T * (np.repeat(w_x, len(yp)) * np.tile(y_sqrt, n_x_fluid+1).flatten())).T
    #     # Chebyshev-Gauss quadrature
    #     #corr_int = corr_int*(w_c/2*np.pi / nyf)
    #     #corr_int = corr_int*(np.pi / n_y_fluid)
    #     p = mu_f * corr_int * w @ p
    #     return p
    #
    # def project_int_stk(w, xp, yp, x, y, l_c, w_c, lam, tol, dz):
    #     x_left = np.tile(x[0:-1], len(yp))  # [0:n2]
    #     x_right = np.tile(x[1:], len(yp))  # [0:n2]
    #     y_low = np.repeat(y[0:-1], len(xp))  # [0:n2]
    #     y_up = np.repeat(y[1:], len(xp))  # [0:n2]
    #     recs = np.array([[np.array([x_left, y_low]), np.array([x_right, y_low])]
    #                         , [np.array([x_left, y_up]), np.array([x_right, y_up])]])
    #     recs = recs.transpose(0, 1, 3, 2)
    #     a = stk.Stokeslet_A_matrix(xp, yp, x, y, dz, lam, recs, tol)
    #     # t2 = np.eye(len(yp)*len(xp)) - 2*t
    #     a_inv = np.linalg.inv(a * l_c)
    #     # a2 = a_inv.dot(t2)
    #     # p = a_inv @ w.T
    #     p = (a_inv[0:len(xp) * len(yp), 0:len(xp) * len(yp)] @ w.T + a_inv[len(xp) * len(yp):, 0:len(xp) * len(yp)] @ w.T
    #          + a_inv[0:len(xp) * len(yp), len(xp) * len(yp):] @ w.T + a_inv[len(xp) * len(yp):, len(xp) * len(yp):] @ w.T)
    #     x_sqrt = np.sqrt(1 - xp ** 2)
    #     y_sqrt = np.sqrt((w_c / 2) ** 2 - (yp * l_c) ** 2) / (w_c / 2)
    #     p = (p.T * (np.repeat(x_sqrt, len(yp)) * np.tile(y_sqrt, len(xp)).flatten())).T
    #     corr_int = (l_c/2 * np.pi / len(xp)) * (w_c / 2 * np.pi / len(yp))
    #     p = corr_int * w @ p
    #     p_sl = sp.csr_matrix(p, dtype=complex)
    #
    #     # a_inv = np.linalg.inv(a*l_c*l_c)
    #     # p = a_inv @ w.T
    #     # x_sqrt = np.sqrt(1 - xp**2)
    #     # y_sqrt = np.sqrt((w_c/2)**2 - (yp*l_c)**2)/(w_c/2)
    #     # p = (p.T*(np.repeat(x_sqrt, len(yp)) * np.tile(y_sqrt, len(xp)).flatten())).T
    #     # corr_int = (l_c*np.pi / 2 / len(xp))*(w_c/2*np.pi / len(yp))
    #     # p = l_c*corr_int* w @ p
    #     # p_sl = sp.csr_matrix(p, dtype=complex)
    #     return p_sl
    #
    # def project_int_stk_sbt(w, xp, yp, x, y, l_c, w_c, lam, tol):
    #     x_left = np.tile(x[0:-1], len(yp))  # [0:n2]
    #     x_right = np.tile(x[1:], len(yp))  # [0:n2]
    #     y_low = np.repeat(y[0:-1], len(xp))  # [0:n2]
    #     y_up = np.repeat(y[1:], len(xp))  # [0:n2]
    #     recs = np.array([[np.array([x_left, y_low]), np.array([x_right, y_low])]
    #                         , [np.array([x_left, y_up]), np.array([x_right, y_up])]])
    #     recs = recs.transpose(0, 1, 3, 2)
    #     a = stk.Stokeslet_A_matrix_sbt(xp, yp, x, y,  lam, recs, tol)
    #     # t2 = np.eye(len(yp)*len(xp)) - 2*t
    #     a_inv = np.linalg.inv(a * l_c * l_c)
    #     # a2 = a_inv.dot(t2)
    #     # p = a_inv @ w.T
    #     p = a_inv[0:len(xp) * len(yp), 0:len(xp) * len(yp)] @ w.T# + a_inv[len(xp) * len(yp):, 0:len(xp) * len(yp)] @ w.T
    #          #+ a_inv[0:len(xp) * len(yp), len(xp) * len(yp):] @ w.T + a_inv[len(xp) * len(yp):, len(xp) * len(yp):] @ w.T)
    #     x_sqrt = np.sqrt(1 - xp ** 2)
    #     y_sqrt = np.sqrt((w_c / 2) ** 2 - (yp * l_c) ** 2) / (w_c / 2)
    #     p = (p.T * (np.repeat(x_sqrt, len(yp)) * np.tile(y_sqrt, len(xp)).flatten())).T
    #     corr_int = (l_c * np.pi / 2 / len(xp)) * (w_c / 2 * np.pi / len(yp))
    #     p = l_c * corr_int * w @ p
    #     p_sl = sp.csr_matrix(p, dtype=complex)
    #
    #     # a_inv = np.linalg.inv(a*l_c*l_c)
    #     # p = a_inv @ w.T
    #     # x_sqrt = np.sqrt(1 - xp**2)
    #     # y_sqrt = np.sqrt((w_c/2)**2 - (yp*l_c)**2)/(w_c/2)
    #     # p = (p.T*(np.repeat(x_sqrt, len(yp)) * np.tile(y_sqrt, len(xp)).flatten())).T
    #     # corr_int = (l_c*np.pi / 2 / len(xp))*(w_c/2*np.pi / len(yp))
    #     # p = l_c*corr_int* w @ p
    #     # p_sl = sp.csr_matrix(p, dtype=complex)
    #     return p_sl
    #
    #
    # def export_basis_functions_new(v, l_c, w_c, n_x_fluid, n_y_fluid, x_rule='1/3simps'):
    #     x, y = setup_fluid_grid_simps(l_c, w_c, n_x_fluid, n_y_fluid)
    #     mesh_fluid = kl.setup_mesh(l_c, w_c, n_x_fluid, n_y_fluid, 'left/right', x_rule)
    #     (mesh_fluid.coordinates()[:, 1]).sort(axis=0)
    #     xf, yf = mesh_fluid.coordinates()[:, 1], mesh_fluid.coordinates()[:, 1]
    #     #print(len(yf))
    #     yf[:] = np.repeat(y, len(x))
    #     v2 = fe.FunctionSpace(mesh_fluid, 'CG', 1)
    #     transfer_matrix = fe.PETScDMCollection.create_transfer_matrix(v, v2)
    #     #print(transfer_matrix.array().shape)
    #     arg_x = np.argsort(v2.tabulate_dof_coordinates()[:, 0])
    #     #print(arg_x.shape)
    #     coord_x = v2.tabulate_dof_coordinates()[arg_x, :]
    #     arg_y = np.argsort(np.reshape(coord_x[:, 1], (len(x), len(y))))
    #     arg_y2 = (arg_y.T + np.arange(len(arg_y)) * (len(y))).T.flatten()
    #     #print(arg_y2.shape)
    #     row, col, val = fe.as_backend_type(transfer_matrix).mat().getValuesCSR()
    #     #del transfer_matrix
    #     #w0 = fe.Function(v)
    #     #nb = w0.vector().size()
    #     #w = sp.csr_matrix((val, col, row), shape=(len(x)*len(y), nb))
    #     w = sp.csr_matrix((val, col, row), dtype='float64')
    #     #print(w.shape)
    #     w = (w[arg_x])[arg_y2].T
    #     return w
    #
    #
    # def export_basis_functions(v, l_c, w_c, n_x_fluid, n_y_fluid, nbox=2, x_rule='1/3simps'):
    #     x, y = setup_fluid_grid(l_c, w_c, n_x_fluid, n_y_fluid, x_rule)
    #     nxf = n_x_fluid + 1
    #     if x_rule == '1/3simps':
    #         nxf = n_x_fluid+1
    #     elif x_rule == 'midpoint':
    #         nxf = n_x_fluid
    #     xc, yc = v.tabulate_dof_coordinates()[:, 0], v.tabulate_dof_coordinates()[:, 1]
    #     xx = np.sort(np.unique(v.mesh().coordinates()[:, 0]))
    #     dx = xx[1] - xx[0]
    #     yy = np.sort(np.unique(v.mesh().coordinates()[:, 1]))
    #     dy = yy[1] - yy[0]
    #     # Initialize empty matrix with coo_matrix! Efficient memory allocation!
    #     # Avoid running out of memory in initialization!
    # #     w = sps.coo_matrix((nb, n_y_fluid * n_x_fluid))
    #
    #     # Convert w to dok_matrix!
    #     # This is an efficient structure for constructing sparse matrices incrementally.
    #     w0 = fe.Function(v)
    #     nb = w0.vector().size()
    #     #print(nb)
    #     w = sp.lil_matrix((nb, n_y_fluid * nxf), dtype=float)
    #     #w = w.tolil()
    #
    #     for ib in range(nb):
    #         w_local = np.zeros((n_y_fluid, nxf))
    #         w0 = fe.Function(v)
    #         w0.vector()[ib] = 1
    #         for iy in range(len(y)):
    #             if abs(yc[ib] - y[iy]) <= nbox*dy:
    #                 for ix in range(len(x)):
    #                     if abs(xc[ib] - x[ix]) <= nbox*dx:
    #                         w_local[iy, ix] = w0(x[ix], y[iy])
    #                     else:
    #                         pass
    #             else:
    #                 pass
    #         w[ib] = np.reshape(w_local.T, n_y_fluid * nxf)
    #     w = w.tocsr()
    #     return w
    #
    #
    # def project_int(w, omega, n_x_fluid, n_y_fluid, l_c, w_c, nu_f, mu_f, x_rule='1/3simps'):
    #     # nb = w.shape[0]
    #     nxf = n_x_fluid
    #     if x_rule == '1/3simps':
    #         if n_x_fluid % 2 == 1:
    #             n_x_fluid += 1
    #         else:
    #             pass
    #         nxf = n_x_fluid+1
    #     elif x_rule == 'midpoint':
    #         nxf = n_x_fluid
    #     nyf = n_y_fluid
    #     a = tuck.get_a_matrix(nyf, w_c, omega, nu_f)
    #     a_inv = np.linalg.inv(a)
    #     a_dia = [a_inv for ii in range(nxf)]
    #     del a, a_inv
    #     #a_inv_dia = sp.block_diag(a_dia)
    #     a_inv_dia = sp.block_diag(a_dia, format='csr', dtype='complex64')
    #     del a_dia
    #     p = a_inv_dia @ w.T
    #     del a_inv_dia
    #     # Chebyshev approach
    #     # X integration
    #     w_x = np.ones(nxf)
    #     corr_int = 1 / 3 * l_c / n_x_fluid
    #     if x_rule == '1/3simps':
    #         corr_int = 1/3*l_c/n_x_fluid
    #         w_x = np.ones(nxf)
    #         w_x[1::2] = 4*w_x[1::2]
    #         w_x[2:-1:2] = 2*w_x[2:-1:2]
    #     elif x_rule == 'midpoint':
    #         corr_int = l_c/n_x_fluid
    #
    #     i = np.arange(1, nyf + 1)
    #     #y_temp = -w_c/2*np.cos(((2 * i - 1) / (2 * nyf)) * np.pi)
    #     #y_sqrt = np.sqrt((w_c/2)**2 - y_temp**2)
    #     #yp = w_c/2*(-np.cos(((2 * i - 1) / (2 * n_y_fluid)) * np.pi))
    #     #y_temp = -w_c/2*np.cos(((2 * i - 1) / (2 * n_y_fluid+1)) * np.pi)
    #     #y_sqrt = np.sqrt((w_c/2)**2 - yp**2)
    #     weight_y = np.pi/(n_y_fluid+1)*np.sin(i/(n_y_fluid+1)*np.pi)**2*w_c/2
    #     #p = (p.T.multiply(np.repeat(w_x, nyf).flatten())).multiply(np.tile(y_sqrt, nxf).flatten())
    #     p = p.T.multiply(np.repeat(w_x, nyf).flatten() * np.tile(weight_y, nxf).flatten()).T
    #     # Chebyshev-Gauss quadrature
    #     #corr_int = corr_int*(w_c/2*np.pi / nyf)
    #     #corr_int = corr_int*(np.pi / nyf)
    #     p = mu_f * corr_int * w @ p
    #
    #     return p
    #
    # def external_force(Mode, l_c, w_c):
    #     xp = []
    #     yp = []
    #     xn = []
    #     yn = []
    #     test = Mode.split(':')
    #     Mode_x = int(test[0])
    #     Mode_y = int(test[1])
    #     if Mode_y == 0:  # Euler Bernoulli mode
    #         yp = list(np.linspace(-w_c / 2, w_c / 2, 30))
    #         yn = []
    #     ### Torsional
    #     elif Mode_y == 1:
    #         yp = list(np.linspace(0, w_c / 2, 20))
    #         yn = list(-np.linspace(0, w_c / 2, 20))
    #     elif Mode_y == 2: #Double
    #         yp = list(np.linspace(-62*w_c/256, 62*w_c/256, 10))
    #         yn =list(np.linspace(-w_c/2,-62*w_c/256, 5)) + list(np.linspace(62*w_c/256,w_c/2, 5))
    #     # elif mode == 10 or mode == 14 or mode == 18 or mode == 22 or mode == 29 or mode == 36 or mode == 46: #Triple
    #     #    yp = list(np.linspace(-81*w_c/256,0,  5)) + list(np.linspace(81*w_c/256, w_c/2, 5))
    #     #    yn =list(np.linspace(-w_c/2,-81*w_c/256, 5)) + list(np.linspace(0,81/256*w_c, 5))
    #     # elif mode == 16 or mode == 21 or mode == 25 or mode == 32 or mode == 38 or mode == 47: # Quadruple
    #     #    yp = list(np.linspace(-98/256*w_c, -32/256*w_c,  7)) + list(np.linspace(32*w_c/256, 98/256*w_c, 7))
    #     #    yn =list(np.linspace(-w_c/2,-98*w_c/256, 5)) + list(np.linspace(-32/256*w_c,32/256*w_c, 5)) + list(np.linspace(98/256*w_c,w_c/2, 5))
    #     # elif mode == 26 or mode == 28 or mode == 34 or mode == 42 or mode == 49:# Quintuple
    #     #    yp = [-w_c/2, -28/256*w_c, 81/256*w_c]
    #     #   yn = [w_c/2, 28/256*w_c, -81/256*w_c]
    #     # lif mode == 35 or mode == 41 or mode == 43: # Sextuple
    #     ##   yn = [69/256*w_c, 0, 69/256*w_c]
    #     #    yp = [-w_c/2, -23/256*w_c, +23/256*w_c, w_c/2]#
    #     # elif mode == 50: # Sectuple
    #     #    yn = [w_c/2, 58/256*w_c, -20/256*w_c, -95/256*w_c]
    #     #    yp = [-w_c/2, -58/256*w_c, 20/256*w_c, 95/256*w_c]
    #     # else:
    #     #    yp = list(np.linspace(-w_c/2, w_c/2, 30))
    #     #    yn = []
    #
    #     if Mode_x == 1:  # nx = 1
    #         xp = list(np.linspace(0, l_c, 20))
    #         xn = []
    #     elif Mode_x == 2:  # nx = 2
    #         xn = list(np.linspace(0, 196 * l_c / 256, 10))
    #         xp = list(np.linspace(196 * l_c / 256, l_c, 5))
    #     elif Mode_x == 3:  # nx = 3
    #         xn = [176 / 256 * l_c]
    #         xp = [72 / 256 * l_c, l_c]
    #     elif Mode_x == 4:  # nx = 4
    #         xn = [51 * l_c / 256, 199 * l_c / 256]
    #         xp = [125 * l_c / 256, l_c]
    #     elif Mode_x == 5:
    #         xp = list(np.linspace(0, 67 / 256 * l_c, 5)) + list(np.linspace(124 * l_c / 256, 185 / 256 * l_c, 5)) + list(
    #             np.linspace(239 / 256 * l_c, l_c, 5))
    #         xn = list(np.linspace(67 / 256 * l_c, 124 * l_c / 256, 5)) + list(
    #             np.linspace(185 * l_c / 256, 239 / 256 * l_c, 5))
    #     elif Mode_x == 6:# nx = 6
    #         xp = [81/256*l_c, 177/256*l_c, l_c]
    #         xn = [33/256*l_c, 127/256*l_c, 222/256*l_c]
    #     elif Mode_x == 7: #nx = 7
    #         xp = [28/256*l_c, 108/256*l_c, 190/256*l_c,  l_c]
    #         xn = [65/256*l_c, 150/256*l_c, 228/256*l_c]
    #     # elif mode == 27 or mode == 37 or mode == 48: #nx = 8
    #     #    xp = [60/256*l_c, 129/256*l_c, 196/256*l_c,  l_c]
    #     #    xn = [25/256*l_c, 94/256*l_c, 163/256*l_c, 229/256*l_c]
    #     # elif mode == 33 or mode == 45: # nx = 9
    #     #    xp = [23/256*l_c, 82/256*l_c, 145/256*l_c, 205/256*l_c,  l_c]
    #     #    xn = [56/256*l_c, 112/256*l_c, 173/256*l_c, 235/256*l_c]
    #     # elif mode == 40: #nx = 10
    #     #    xp = [52/256*l_c, 100/256*l_c, 154/256*l_c, 185/256*l_c,  239/256*l_c]
    #     #    xn = [21/256*l_c, 72/256*l_c, 130/256*l_c,  210/256*l_c, l_c]
    #     # elif mode == 44: #nx = 11
    #     #   xp = [38/256*l_c, 84/256*l_c, 128/256*l_c, 150/256*l_c,  204/256*l_c, l_c]
    #     #    xn = [16/256*l_c, 61/256*l_c, 107/256*l_c,  175/256*l_c, 231/256*l_c]
    #     #    yp = [-43/512*w_c, -132/512*w_c, -215/512*w_c, 43/512*w_c, 132/512*w_c, 215/512*w_c]
    #     return xp, xn, yp, yn
    #
    # def sho_func(fq, a0, f0, Q, b):
    #     return abs(a0*f0**2/(f0**2 - fq**2+1j*fq*f0/Q) + b)