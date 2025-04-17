
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
#import scipy.integrate
#from scipy.spatial import Delaunay
import quadpy
#import copy
import fenics as fe

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

def gaussian_quad(ny):
    i = np.arange(1, ny + 1)
    yp = (-np.cos(((2 * i - 1) / (2 * ny)) * np.pi))
    y  = np.concatenate((np.array([-1]), 0.5*(yp[1:] + yp[:-1]), np.array([1])))
    return y, yp

def lin_quad(ny):
    # midpoint quadrature
    y = np.linspace(0, 1, ny+1)
    yp = 1/2*(y[1:] + y[:-1])
    return y, yp


def around_singularity(ind_x, ind_y, x, xp, y, yp, lam, tol, scheme_t, scheme_q):
    x0 = xp[ind_x]
    y0 = yp[ind_y]
    r_ = np.min(np.abs([x[ind_x] - x0, x[ind_x + 1] - x0, y[ind_y] - y0, y[ind_y + 1] - y0])) / 1

    # scheme_t = quadpy.t2.get_good_scheme(2)
    err = 0.1
    cc = 3
    Int_analytic = ((-4 * np.pi * lam * r_ - 4 * np.pi) * np.exp(
        -lam * r_) / lam ** 2 / r_ + 4 * np.pi / lam ** 2 / r_) / 8 / np.pi
    Int_prev = Int_analytic
    while err > tol:
        n_el = 3 * cc
        theta = np.linspace(np.pi, 3 * np.pi / 2 - np.pi / 4, int(n_el) + 1)  #
        ypp = y0 + np.sin(theta) * r_
        xpp = x0 + np.cos(theta) * r_
        xleft = xpp[1:]
        xright = xpp[:-1]
        yup = ypp[1:]
        ylow = ypp[:-1]
        ypp2 = np.linspace(ypp[-1], y0 - r_, int(n_el))
        ylow_t = ypp2[1:]
        yup_t = ypp2[:-1]
        xpp2 = np.linspace(xpp[-1], x0 - r_, int(n_el))
        xright_t = xpp2[:-1]
        xleft_t = xpp2[1:]

        xleft2 = np.concatenate([xleft, xleft_t])
        xright2 = np.concatenate([xright, xright_t])
        ylow2 = np.concatenate([ylow, ylow_t])
        yup2 = np.concatenate([yup, yup_t])

        # triangles = np.concatenate([[np.array([xleft, ylow]), np.array([xright, ylow])
        #             , np.array([xleft, yup])]]).transpose(0, 2, 1)
        triangles = np.concatenate([[np.array([xleft2, ylow2]), np.array([xright2, ylow2])
                                        , np.array([xleft2, yup2])]]).transpose(0, 2, 1)
        val = scheme_t.integrate(lambda xx: Szz(xx, x0, y0, lam, 0), triangles).sum()
        xleft = np.repeat(xpp[:-2], np.arange(n_el - 1, 0, -1))
        xright = np.repeat(xpp[1:-1], np.arange(n_el - 1, 0, -1))
        yupgrid, xrgrid = np.meshgrid(ypp[1:-1], xpp[1:-1])
        ylgrid, xlgrid = np.meshgrid(ypp[2:], xpp[:-2])
        yupgrid, xrgrid = np.meshgrid(ypp[1:-1], xpp[1:-1])

        yup = np.triu(yupgrid)
        yup = yup[yup.nonzero()]

        ylow = np.triu(ylgrid)
        ylow = ylow[ylow.nonzero()]

        xleft_l = np.repeat(xpp2[2:], np.arange(1, int(n_el) - 1))
        xright_l = np.repeat(xpp2[1:-1], np.arange(1, int(n_el) - 1))

        yupgrid, xrgrid = np.meshgrid(ypp2[1:-1], xpp2[1:-1])
        ylgrid, xlgrid = np.meshgrid(ypp2[2:], xpp2[:-2])
        yup_l = yupgrid[(np.tril(yupgrid)).nonzero()]
        ylow_l = ylgrid[(np.tril(ylgrid)).nonzero()]
        xleft3 = np.concatenate([xleft, xleft_l])
        xright3 = np.concatenate([xright, xright_l])
        ylow3 = np.concatenate([ylow, ylow_l])
        yup3 = np.concatenate([yup, yup_l])
        recs = np.array([[np.array([xleft3, ylow3]), np.array([xright3, ylow3])]
                            , [np.array([xleft3, yup3]), np.array([xright3, yup3])]]).transpose(0, 1, 3, 2)
        # recs = recs.transpose(0, 2, 1)
        val += scheme_q.integrate(lambda xx: Szz(xx, x0, y0, lam, 0), recs).sum()
        # val += scheme_q.integrate(lambda xx: Szz(xx, x0, y0, lam, 0), recs).sum()

        Int = Int_analytic + 8 * val
        cc += 1
        err = np.abs(Int - Int_prev) / np.abs(Int_prev)
        Int_prev = Int
    nn = 3
    err_ = 0.1
    val_prev = Int
    while err_ > tol:
        # ind_x = ii
        # ind_y = jj
        if (x[ind_x + 1] - x[ind_x]) > (y[ind_y + 1] - y[ind_y]):
            nny = nn
            nnx = int(np.ceil(nn * (x[ind_x + 1] - x[ind_x]) / (y[ind_y + 1] - y[ind_y]) / 2))
        else:
            nnx = nn
            nny = int(np.ceil(nn * (y[ind_y + 1] - y[ind_y]) / (x[ind_x + 1] - x[ind_x]) / 2))

            ##Left region
        if np.abs(x[ind_x] - (x0 - r_)) > 1e-20:
            x_left = np.tile(np.linspace(x[ind_x], x0 - r_, nnx)[:nnx - 1], nny - 1)
            x_right = np.tile(np.linspace(x[ind_x], x0 - r_, nnx)[1:], nny - 1)
            y_low = np.repeat(np.linspace(y[ind_y], y[ind_y + 1], nny)[:nny - 1], nnx - 1)
            y_up = np.repeat(np.linspace(y[ind_y], y[ind_y + 1], nny)[1:], nnx - 1)

            recs_left = np.array([[np.array([x_left, y_low]), np.array([x_right, y_low])]
                                     , [np.array([x_left, y_up]), np.array([x_right, y_up])]])
            recs_left = recs_left.transpose(0, 1, 3, 2)

        ##Right region
        if np.abs(x[ind_x + 1] - (x0 + r_)) > 1e-20:
            x_left = np.tile(np.linspace(x0 + r_, x[ind_x + 1], nnx)[:nnx - 1], nny - 1)
            x_right = np.tile(np.linspace(x0 + r_, x[ind_x + 1], nnx)[1:nnx], nny - 1)
            y_low = np.repeat(np.linspace(y[ind_y], y[ind_y + 1], nny)[:nny - 1], nnx - 1)
            y_up = np.repeat(np.linspace(y[ind_y], y[ind_y + 1], nny)[1:], nnx - 1)
            recs_right = np.array([[np.array([x_left, y_low]), np.array([x_right, y_low])]
                                      , [np.array([x_left, y_up]), np.array([x_right, y_up])]])
            recs_right = recs_right.transpose(0, 1, 3, 2)

        ##Top region
        if np.abs(y[ind_y + 1] - (y0 + r_)) > 1e-20:
            x_left = np.tile(np.linspace(x0 - r_, x0 + r_, nnx)[:nnx - 1], nny - 1)
            x_right = np.tile(np.linspace(x0 - r_, x0 + r_, nnx)[1:nnx], nny - 1)
            y_low = np.repeat(np.linspace(y0 + r_, y[ind_y + 1], nny)[:nny - 1], nnx - 1)
            y_up = np.repeat(np.linspace(y0 + r_, y[ind_y + 1], nny)[1:], nnx - 1)
            recs_top = np.array([[np.array([x_left, y_low]), np.array([x_right, y_low])]
                                    , [np.array([x_left, y_up]), np.array([x_right, y_up])]])
            recs_top = recs_top.transpose(0, 1, 3, 2)

        ##Bottom region
        if np.abs(y[ind_y] - (y0 - r_)) > 1e-20:
            x_left = np.tile(np.linspace(x0 - r_, x0 + r_, nnx)[:nnx - 1], nny - 1)
            x_right = np.tile(np.linspace(x0 - r_, x0 + r_, nnx)[1:nnx], nny - 1)
            y_low = np.repeat(np.linspace(y[ind_y], y0 - r_, nny)[:nny - 1], nnx - 1)
            y_up = np.repeat(np.linspace(y[ind_y], y0 - r_, nny)[1:], nnx - 1)
            recs_bot = np.array([[np.array([x_left, y_low]), np.array([x_right, y_low])]
                                    , [np.array([x_left, y_up]), np.array([x_right, y_up])]])
            recs_bot = recs_bot.transpose(0, 1, 3, 2)
        if np.abs(x[ind_x] - (x0 - r_)) < 1e-20:
            recs_quad = np.concatenate([recs_right, recs_top, recs_bot], axis=2)
        elif np.abs(x[ind_x + 1] - (x0 + r_)) < 1e-20:
            recs_quad = np.concatenate([recs_left, recs_top, recs_bot], axis=2)
        elif np.abs(y[ind_y + 1] - (y0 + r_)) < 1e-20:
            recs_quad = np.concatenate([recs_left, recs_right, recs_bot], axis=2)
        elif np.abs(y[ind_y] - (y0 - r_)) < 1e-20:
            recs_quad = np.concatenate([recs_left, recs_right, recs_top], axis=2)
        else:
            recs_quad = np.concatenate([recs_left, recs_right, recs_top, recs_bot], axis=2)
        val = np.sum(scheme_q.integrate(lambda xx: Szz(xx, x0, y0, lam, 0), recs_quad))
        # val3 = np.sum(scheme_fine.integrate(lambda xx: Szz(xx, x0, y0), recs_quad))
        # val2 = np.sum(scheme_coarse.integrate(Szz, recs_quad))
        # val3 = np.sum(scheme_fine.integrate(Szz, recs_quad))
        err_ = np.abs(val - val_prev) / np.abs(val)
        val_prev = val
        nn += 1
    Int += val
    return Int, np.max([err_, err])


def local_refinement(ind, err, val_fine, tol, x0, y0, recs, scheme_q, lam, dz):
    # ind = np.where(l2_error.flatten() > tol)[0]
    x_left = recs[0, 0, ind, 0]
    x_right = recs[0, 1, ind, 0]
    y_low = recs[0, 0, ind, 1]
    y_up = recs[1, 0, ind, 1]
    x_left_init = x_left
    x_right_init = x_right
    y_low_init = y_low
    y_up_init = y_up
    # val_fine = STK.flatten()
    # err = l2_error.flatten()
    err_ = err  # [ind]
    ind_ = np.where(err_ > tol)[0]
    # val = val_fine[ind]
    val_prev = val_fine  # [ind]
    nn = 2
    # val_prev[np.where(np.abs(val_prev) == 0)[0]] = 1

    while np.max(err_) >= tol:
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

        val_ = scheme_q.integrate(lambda xx: Szz(xx, x0, y0, lam, dz), recs_f)
        val = np.ones(len(ind_), dtype=complex)
        for ii in range(len(ind_)):
            val[ii] = np.sum(val_[len_[2 * ii]:len_[2 * ii + 1] + 1])
        # val = np.add.reduceat(val_, len_)[::2]
        err_[ind_] = np.abs(val - val_prev[ind_]) / np.abs(val_prev[ind_])
        val_prev[ind_] = val
        ind_ = np.where(err_ > tol)[0]
        x_left = x_left_init[ind_]
        x_right = x_right_init[ind_]
        y_low = y_low_init[ind_]
        y_up = y_up_init[ind_]
        nn += 1

        # val_fine[ind] = val_prev
    # err[ind] = err_
    return val_prev, err_

def Stokeslet_A_matrix_sbt(xp, yp, x, y, lam, recs, tol=1e-4):
	dz = 0
	scheme_coarse = quadpy.c2.get_good_scheme(2)
	# val_coarse = scheme.integrate(Szz, recs)
	scheme_fine = quadpy.c2.get_good_scheme(4)

	scheme_finer = quadpy.c2.get_good_scheme(6)

	scheme_finer2 = quadpy.c2.get_good_scheme(8)

	scheme_t = quadpy.t2.get_good_scheme(2)
	STK_v1 = np.zeros((len(yp), len(yp), 2*len(xp)-1), dtype=np.complex64)
	x0 = xp[0]
	ii = 0
	for jj in range(int(len(yp)/2)):
	    y0 = yp[jj]
	    val_coarse = scheme_coarse.integrate(lambda xx: Szz(xx, x0, y0, lam, dz), recs)
	    val_fine = scheme_fine.integrate(lambda xx: Szz(xx, x0, y0, lam, dz), recs)
	    err = np.abs(val_coarse - val_fine) / np.abs(val_fine)
	    I, l2_ = around_singularity(ii, jj, x, xp, y, yp, lam, tol, scheme_t, scheme_fine)
	    val_fine[jj * len(xp) + ii] = I
	    err[jj * len(xp) + ii] = l2_
	    ind = err > tol
	    recs_ = recs[:, :, ind, :]
	    
	    val_finer = scheme_finer.integrate(lambda xx: Szz(xx, x0, y0, lam, 0), recs_)
	    err[ind] = np.abs(val_fine[ind] - val_finer) / np.abs(val_finer)
	    val_fine[ind] = val_finer
	    
	    ind = np.where(err.flatten() > tol)[0]
	    if len(ind) > 0:
	        val_out, l2_out = local_refinement(ind, err[ind], val_fine[ind],
	                                           tol, x0, y0, recs, scheme_fine, lam, 0)
	        val_fine[ind] = val_out
	        err[ind] = l2_out

	    STK = val_fine.reshape(len(yp), len(xp))
	    STK_v1[jj, :, :] = np.concatenate([np.fliplr(STK), STK[:, 1:]], axis=1)
	    STK_v1[-jj-1, :, :] = np.flipud(np.concatenate([np.fliplr(STK), STK[:, 1:]], axis=1))


	A_sbt = np.zeros(([len(yp) * len(xp), len(yp) * len(xp)]), dtype=np.complex64)
	nn = 0#len(yp)
	plot_ = False
	for ix in range(len(xp)):
	    for jy in range(int(len(yp))):
	        STK_out = STK_v1[jy, :, len(xp)-1-ix:len(xp)-1-ix+len(xp)]
	        A_sbt[nn, :] = STK_out.T.flatten()
	        nn+=1
	return A_sbt
	