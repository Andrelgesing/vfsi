#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fenics as fe
import numpy as np
import scipy.sparse as sp
import scipy.special as special
import Plate.plate_fem as pt
import quadpy
import tqdm
from Fluid.precompile import *

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


def s_zz(r, lam, dx, dy, dz):
    R = lam*r
    a_r = 2*np.exp(-R)*(1 + 1/R + 1/R**2) - 2/R**2
    b_r = -2*np.exp(-R)*(1 + 3/R + 3/R**2) + 6/R**2
    s_33 = 1/np.pi/8*(a_r/r + b_r*dz*dz/r**3)
    return s_33

def Szz(xx, x0, y0, lam, dz):
    x = xx[0]
    y = xx[1]
    dx = x-x0
    dy = y-y0
    r__ = np.sqrt(dx**2 + dy**2 + dz**2)
    stk_zz = s_zz(r__, lam, dx, dy, dz)
    return stk_zz
class F3D(object):
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
        self.weights = self.scheme.weights
        self.theta_lin = (self.scheme.points)
    def gaussian_quad(self, ny):
        i = np.arange(1, ny + 1)
        yp = (-np.cos(((2 * i - 1) / (2 * ny)) * np.pi))
        y = np.concatenate((np.array([-1]), 0.5 * (yp[1:] + yp[:-1]), np.array([1])))
        return y, yp

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
        self.n_x_fluid = n_x_fluid
        self.n_y_fluid = n_y_fluid
        self.l_c = self.geometry.l_c
        self.w_c = self.geometry.w_c
        self.t_c = self.geometry.t_c

        self.x, self.xp = np.array(self.gaussian_quad(2 * n_x_fluid), dtype=object)[0][int(n_x_fluid)::], \
            np.array(self.gaussian_quad(2 * n_x_fluid), dtype=object)[1][int(n_x_fluid)::]
        self.y, self.yp = np.array(self.gaussian_quad(n_y_fluid), dtype=object) * self.w_c / 2 / self.l_c
        self.x_left = np.tile(self.x[0:-1], len(self.yp))  # [0:n2]
        self.x_right = np.tile(self.x[1:], len(self.yp))  # [0:n2]
        self.y_low = np.repeat(self.y[0:-1], len(self.xp))  # [0:n2]
        self.y_up = np.repeat(self.y[1:], len(self.xp))  # [0:n2]
        recs = np.array([[np.array([self.x_left, self.y_low]), np.array([self.x_right, self.y_low])]
                            , [np.array([self.x_left, self.y_up]), np.array([self.x_right, self.y_up])]])
        self.recs = recs.transpose(0, 1, 3, 2)
        self.x_sqrt = np.sqrt(1 - self.xp ** 2)
        self.wx = self.l_c / 2 * np.pi / len(self.xp) * self.x_sqrt
        self.y_sqrt = np.sqrt((self.w_c / 2) ** 2 - (self.yp * self.l_c) ** 2) / (self.w_c / 2)
        self.wy =  self.w_c / 2 *np.pi/len(self.yp) * self.y_sqrt
        self.export_basis_functions()

    def get_h_force(self, omega):
        self.get_a_matrix(omega)
        a_inv = np.linalg.inv(self.a_bem)
        p = -self.fluid.mu_f*a_inv @ self.w.T/self.l_c
        del a_inv
        p = (p.T * (np.repeat(self.wx, len(self.yp)) * np.tile(self.wy, len(self.xp)).flatten())).T
        p =  self.w @ p
        p_sl = sp.csr_matrix(p, dtype=complex)
        self.p = p_sl
        return p_sl

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

        a_bem = np.zeros(([len(self.yp) * len(self.xp), len(self.yp) * len(self.xp)]), dtype=np.complex64)
        self.lam = np.sqrt(-1j * omega * self.l_c ** 2 / self.fluid.nu_f)
        nn = 0
        # dz_ = 0
        #pbar = tqdm.tqdm(total=len(self.xp)*len(self.yp)/2)
        for ii in range(len(self.xp)):
            x0 = self.xp[ii]
            for jj in range(int(len(self.yp) / 2)):
                y0 = self.yp[jj]
                val_coarse = self.scheme_coarse.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), self.recs)
                val_fine = self.scheme_fine.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), self.recs)
                err = np.abs(val_coarse - val_fine) / np.abs(val_fine)
                err[np.where(np.abs(val_fine) / np.max(np.abs(val_fine)) <= 1e-10)] = 0
                val_fine[np.where(np.abs(val_fine) / np.max(np.abs(val_fine)) <= 1e-10)] = 0
                # Singularity
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
                    val_sing += (self.scheme.integrate(lambda xx: sing_a(xx, y_[iii], r_min, self.lam),
                                [theta_c[iii], np.pi/2]) 
                                + self.scheme.integrate(lambda xx: sing_b(xx, x_[iii], r_min, self.lam), [theta_c[iii], 0]))
                I = val_sing + val_circle
                val_fine[jj*len(self.xp) + ii] = I
                err[jj*len(self.xp) + ii] = self.tol_stk/10
 
                ind = err > self.tol_stk
                recs2 = self.recs[:, :, ind, :]
                val_finer = self.scheme_finer.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), recs2)
                err[ind] = np.abs(val_fine[ind] - val_finer) / np.abs(val_finer)
                val_fine[ind] = val_finer

                ind = err > self.tol_stk
                recs2 = self.recs[:, :, ind, :]
                val_finer = self.scheme_finer2.integrate(lambda xx: Szz(xx, x0, y0, self.lam, 0), recs2)
                err[ind] = np.abs(val_fine[ind] - val_finer) / np.abs(val_finer)
                val_fine[ind] = val_finer

                ind = np.where(err.flatten() > self.tol_stk)[0]
                if len(ind) > 0:
                    val_out, l2_out = self.local_refinement(ind, err[ind], val_fine[ind], x0, y0)
                    val_fine[ind] = val_out
                    err[ind] = l2_out
                STK_0 = val_fine.reshape(len(self.yp), len(self.xp))

                a_bem[nn, :] = STK_0.T.flatten()
                a_bem[nn + int(len(self.yp)) - 2 * jj - 1, :] = np.flip(STK_0.T, 1).flatten()
                nn += 1
                if self.pbar is not None:
                    self.pbar.update(1)
                if jj == int(len(self.yp) / 2) - 1:
                    nn += int(len(self.yp) / 2)
        self.a_bem = a_bem
        return a_bem

    def export_basis_functions(self):
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
        # ind = np.where(l2_error.flatten() > tol)[0]
        x_left = self.recs[0, 0, ind, 0]
        x_right = self.recs[0, 1, ind, 0]
        y_low = self.recs[0, 0, ind, 1]
        y_up = self.recs[1, 0, ind, 1]
        x_left_init = x_left
        x_right_init = x_right
        y_low_init = y_low
        y_up_init = y_up
        err_ = err  # [ind]
        ind_ = np.where(err_ > self.tol_stk)[0]
        val_prev = val_fine  # [ind]
        nn = 2

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