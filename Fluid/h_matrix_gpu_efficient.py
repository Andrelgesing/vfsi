import cupy as cp
import sys
sys.path.insert(0, '..')
from Fluid.precompile_gpu import *
import quadpy
import scipy.sparse as sp
import cupyx.scipy.sparse as cpsp
import numpy as np
#from scipy.sparse import csr_matrix

def get_a_matrix_half(xp, yp, x, y, lam):
    n_x_fluid = len(xp)
    n_y_fluid = len(yp)
    yp2 = yp[:int(n_y_fluid/2)]
    threshold = 1e-18
    scheme = quadpy.c1.gauss_legendre(8)
    w = cp.asarray(scheme.weights)
    theta_lin = cp.asarray(scheme.points)

    x1_ = (x[:-1, cp.newaxis] - xp).T.flatten()
    x2_ = (x[1:, cp.newaxis] - xp).T.flatten()
    y1_ = (y[:-1, cp.newaxis] - yp2).T.flatten()
    y2_ = (y[1:, cp.newaxis] - yp2).T.flatten()


    y2_grid, x2_grid = cp.meshgrid(y2_, x2_)
    y1_grid, x1_grid = cp.meshgrid(y1_, x1_)
    n_rows, n_cols = cp.shape(x1_grid)
    del x1_, x2_, y1_, y2_

    ind_r = cp.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid > 0) & (y1_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    stacked = cp.stack([cp.abs(x1), cp.abs(x2), cp.abs(y1), cp.abs(y2)], axis=0)
    r_min = cp.min(stacked, axis=0)
    #r_min = cp.min(cp.abs([x1, x2, y1, y2]), axis=0)
    xx = cp.array([x1, x1, x2, x2])
    yy = cp.array([y1, y2, y1, y2])
    x_ = cp.tile(cp.array([xx]).T, len(theta_lin))
    y_ = cp.tile(cp.array([yy]).T, len(theta_lin))
    theta_c = cp.arctan(yy/xx)
    diff_ = cp.abs(cp.pi/2-theta_c)
    theta_a = (cp.pi/2 + theta_c[:, :, cp.newaxis])/2 + diff_[:, :, cp.newaxis]/2*theta_lin
    weights_a = w*diff_[:, :, cp.newaxis]/2
    weights_a = weights_a.transpose(1, 0, 2)
    theta_a = theta_a.transpose(1, 0, 2)

    diff_ = cp.abs(0-theta_c)
    theta_b = (0 + theta_c[:, :, cp.newaxis])/2 + diff_[:, :, cp.newaxis]/2*theta_lin
    weights_b = w*diff_[:, :, cp.newaxis]/2
    weights_b = weights_b.transpose(1, 0, 2)
    theta_b = theta_b.transpose(1, 0, 2)

    val_circle = ((-4 * cp.pi * lam * r_min - 4 * cp.pi) 
            * cp.exp(-lam * r_min) / lam ** 2 / r_min + 4 * cp.pi / lam ** 2 / r_min) / 8 / cp.pi
    val_sing = (cp.sum(cp.sum(sing_a(theta_a, y_, r_min[:, cp.newaxis, cp.newaxis], lam)*weights_a
                            + sing_b(theta_b, x_, r_min[:, cp.newaxis, cp.newaxis], lam)*weights_b, axis=1), axis=1))
    
    row_sing, col_sing = ind_r
    data_sing = val_sing + val_circle

    #del y0_grid, x0_grid

    #a_vec = cp.zeros(cp.shape(x1_grid), dtype=complex)
    #a_vec = sp.lil_matrix(x1_grid.shape, dtype=complex)
    
    #a_vec[ind_r] = val_sing + val_circle

    del theta_a, theta_b, weights_a, weights_b, val_sing, val_circle

    
    ind_r = cp.where((y1_grid > 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = y1_grid[ind_r]
    y2 = y2_grid[ind_r]
    theta1 = cp.arctan(y2/x1)
    theta2 = cp.arctan(y2/x2)
    theta3 = cp.arctan(y1/x1)
    theta4 = cp.arctan(y1/x2)

    ind_thin = cp.where(theta2<theta3)
    ind_tall = cp.where(theta3<theta2)

    diff_ = cp.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = cp.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = cp.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = cp.expand_dims(x1[ind_thin], axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2[ind_thin], axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1[ind_thin], axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2[ind_thin], axis=1) * cp.ones(len(theta_lin))
    #print('Integration now')
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_thin = cp.sum(val, axis=1)

    x1_ = cp.expand_dims(x1[ind_tall], axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2[ind_tall], axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1[ind_tall], axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2[ind_tall], axis=1) * cp.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = cp.sum(val, axis=1)

    val_vec =  cp.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    mask = cp.abs(val_vec)/cp.max(cp.abs(val_vec)) > threshold
    row_1 = ind_r[0][ mask]
    col_1 = ind_r[1][ mask]
    data_1 = val_vec[mask]

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    #print('Solving top-left panel')
    # Top left region with y>0 and x<0
    ind_r = cp.where((y1_grid > 0) & (x2_grid < 0))
    x1 = -x2_grid[ind_r]
    x2 = -x1_grid[ind_r]
    y1 = y1_grid[ind_r]
    y2 = y2_grid[ind_r]
    #y0_ = y0_grid[ind_r]
    #x0_ = x0_grid[ind_r]
    theta1 = cp.arctan(y2/x1)
    theta2 = cp.arctan(y2/x2)
    theta3 = cp.arctan(y1/x1)
    theta4 = cp.arctan(y1/x2)

    ind_thin = cp.where(theta2<theta3)
    ind_tall = cp.where(theta3<theta2)

    diff_ = cp.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = cp.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = cp.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = cp.expand_dims(x1[ind_thin], axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2[ind_thin], axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1[ind_thin], axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2[ind_thin], axis=1) * cp.ones(len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)
    val_thin = cp.sum(val, axis=1)

    x1_ = cp.expand_dims(x1[ind_tall], axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2[ind_tall], axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1[ind_tall], axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2[ind_tall], axis=1) * cp.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = cp.sum(val, axis=1)

    val_vec =  cp.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    mask = cp.abs(val_vec)/cp.max(cp.abs(val_vec)) > threshold
    row_2 = ind_r[0][ mask]
    col_2 = ind_r[1][ mask]
    data_2 = val_vec[mask]
    #a_vec[ind_r] = val_vec
    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    #print('Solving bottom-left panel')
    # Bottom left region with y<0 and x<0
    ind_r = cp.where((y2_grid < 0) & (x2_grid < 0))
    x1 = -x2_grid[ind_r]
    x2 = -x1_grid[ind_r]
    y1 = -y2_grid[ind_r]
    y2 = -y1_grid[ind_r]
    #y0_ = y0_grid[ind_r]
    #x0_ = x0_grid[ind_r]
    theta1 = cp.arctan(y2/x1)
    theta2 = cp.arctan(y2/x2)
    theta3 = cp.arctan(y1/x1)
    theta4 = cp.arctan(y1/x2)

    ind_thin = cp.where(theta2<theta3)
    ind_tall = cp.where(theta3<theta2)

    diff_ = cp.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = cp.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = cp.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T
    x1_ = cp.expand_dims(x1[ind_thin], axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2[ind_thin], axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1[ind_thin], axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2[ind_thin], axis=1) * cp.ones(len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)
    val_thin = cp.sum(val, axis=1)

    x1_ = cp.expand_dims(x1[ind_tall], axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2[ind_tall], axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1[ind_tall], axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2[ind_tall], axis=1) * cp.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = cp.sum(val, axis=1)

    val_vec =  cp.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    mask = cp.abs(val_vec)/cp.max(cp.abs(val_vec)) > threshold
    row_3 = ind_r[0][ mask]
    col_3 = ind_r[1][ mask]
    data_3 = val_vec[mask]
    #a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    #print('Solving bottom-right panel')
    # Bottom right region with y<0 and x>0
    ind_r = cp.where((y2_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = -y2_grid[ind_r]
    y2 = -y1_grid[ind_r]

    theta1 = cp.arctan(y2/x1)
    theta2 = cp.arctan(y2/x2)
    theta3 = cp.arctan(y1/x1)
    theta4 = cp.arctan(y1/x2)

    ind_thin = cp.where(theta2<theta3)
    ind_tall = cp.where(theta3<theta2)

    diff_ = cp.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = cp.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = cp.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T
    x1_ = cp.expand_dims(x1[ind_thin], axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2[ind_thin], axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1[ind_thin], axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2[ind_thin], axis=1) * cp.ones(len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)
    val_thin = cp.sum(val, axis=1)

    x1_ = cp.expand_dims(x1[ind_tall], axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2[ind_tall], axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1[ind_tall], axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2[ind_tall], axis=1) * cp.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = cp.sum(val, axis=1)

    val_vec =  cp.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    mask = cp.abs(val_vec)/cp.max(cp.abs(val_vec)) > threshold
    row_4 = ind_r[0][ mask]
    col_4 = ind_r[1][ mask]
    data_4 = val_vec[mask]
    #a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    # Top panels
    ind_r = cp.where((x2_grid > 0) & (x1_grid < 0) & (y1_grid > 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]

    x1_ = cp.expand_dims(x1, axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2, axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1, axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2, axis=1) * cp.ones(len(theta_lin))

    theta1 = cp.arctan(y2/x1)
    theta2 = cp.arctan(y2/x2)
    theta3 = cp.arctan(y1/x1)
    theta4 = cp.arctan(y1/x2)

    diff_ = cp.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_31 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta1-cp.pi/2)
    theta_10 = (theta1 + cp.pi/2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_10 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(cp.pi/2-theta2)
    theta_02 = (cp.pi/2 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_02 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_24 = w[:, cp.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = cp.sum(val, axis=1)
    mask = cp.abs(val_vec)/cp.max(cp.abs(val_vec)) > threshold
    row_5 = ind_r[0][ mask]
    col_5 = ind_r[1][ mask]
    data_5 = val_vec[mask]
    #a_vec[ind_r] = val_vec

    # Bottom panels
    ind_r = cp.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y2_grid[ind_r]#[0]
    y2 = -y1_grid[ind_r]#[0]

    x1_ = cp.expand_dims(x1, axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2, axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1, axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2, axis=1) * cp.ones(len(theta_lin))

    theta1 = cp.arctan(y2/x1)
    theta2 = cp.arctan(y2/x2)
    theta3 = cp.arctan(y1/x1)
    theta4 = cp.arctan(y1/x2)

    diff_ = cp.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_31 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta1-cp.pi/2)
    theta_10 = (theta1 + cp.pi/2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_10 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(cp.pi/2-theta2)
    theta_02 = (cp.pi/2 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_02 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_24 = w[:, cp.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = cp.sum(val, axis=1)
    mask = cp.abs(val_vec)/cp.max(cp.abs(val_vec)) > threshold
    row_6 = ind_r[0][ mask]
    col_6 = ind_r[1][ mask]
    data_6 = val_vec[mask]
    #a_vec[ind_r] = val_vec

    del theta_31, theta_10, theta_02, theta_24
    del weights_31, weights_10, weights_02, weights_24
    del val_vec, val
    
    # Right-side panels

    ind_r = cp.where((y2_grid > 0) & (y1_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    
    theta1 = cp.arctan(y2/x1)
    theta2 = cp.arctan(y2/x2)
    theta3 = cp.arctan(y1/x1)
    theta4 = cp.arctan(y1/x2)

    x1_ = cp.tile(cp.array([x1]).T, len(theta_lin))
    x2_ = cp.tile(cp.array([x2]).T, len(theta_lin))
    y1_ = cp.tile(cp.array([y1]).T, len(theta_lin))
    y2_ = cp.tile(cp.array([y2]).T, len(theta_lin))


    diff_ = cp.abs(theta2-theta1)
    theta_12 = (theta1 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_12 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(0-theta2)
    theta_20 = (0 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_20 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta4-0)
    theta_04 = (theta4 + 0)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_04 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta3-theta4)
    theta_43 = (theta3 + theta4)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_43 = w[:, cp.newaxis]*diff_/2

    val = (side_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + side_b(theta_20.T, x1_, x2_, lam)*weights_20.T
        + side_b(theta_04.T, x1_, x2_, lam)*weights_04.T
        + side_a(theta_43.T, x1_, y1_, lam)*weights_43.T)# + 
    val_vec = cp.sum(val, axis=1)
    mask = cp.abs(val_vec)/cp.max(cp.abs(val_vec)) > threshold
    row_7 = ind_r[0][ mask]
    col_7 = ind_r[1][ mask]
    data_7 = val_vec[mask]
    #a_vec[ind_r] = val_vec

    # Left-side panels

    ind_r = cp.where((y2_grid > 0) & (y1_grid < 0) & (x2_grid < 0) & (x1_grid < 0))
    x1 = -x2_grid[ind_r]#[0]
    x2 = -x1_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]

    theta1 = cp.arctan(y2/x1)
    theta2 = cp.arctan(y2/x2)
    theta3 = cp.arctan(y1/x1)
    theta4 = cp.arctan(y1/x2)

    x1_ = cp.tile(cp.array([x1]).T, len(theta_lin))
    x2_ = cp.tile(cp.array([x2]).T, len(theta_lin))
    y1_ = cp.tile(cp.array([y1]).T, len(theta_lin))
    y2_ = cp.tile(cp.array([y2]).T, len(theta_lin))


    diff_ = cp.abs(theta2-theta1)
    theta_12 = (theta1 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_12 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(0-theta2)
    theta_20 = (0 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_20 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta4-0)
    theta_04 = (theta4 + 0)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_04 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta3-theta4)
    theta_43 = (theta3 + theta4)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_43 = w[:, cp.newaxis]*diff_/2

    val = (side_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + side_b(theta_20.T, x1_, x2_, lam)*weights_20.T
        + side_b(theta_04.T, x1_, x2_, lam)*weights_04.T
        + side_a(theta_43.T, x1_, y1_, lam)*weights_43.T)# + 
    val_vec = cp.sum(val, axis=1)
    mask = cp.abs(val_vec)/cp.max(cp.abs(val_vec)) > threshold
    row_8 = ind_r[0][ mask]
    col_8 = ind_r[1][ mask]
    data_8 = val_vec[mask]
    #a_vec[ind_r] = val_vec

    del val_vec, val, theta_12, theta_20, theta_04, theta_43
    del weights_12, weights_20, weights_04, weights_43
    del y2_grid, x2_grid, y1_grid, x1_grid#, y0_grid, x0_grid
    
    #a_out = a_vec.reshape((n_x_fluid, n_x_fluid, int(n_y_fluid/2), n_y_fluid))
    #a2_flipped = cp.flip(a_out, axis=(2, 3))#[:, :, :, ::-1]
    #a2_final = cp.concatenate([a_out, a2_flipped], axis=2)
    #del a2_flipped, a_out
    #a_out_1d = (a2_final.swapaxes(1,2)).reshape(n_x_fluid*n_y_fluid, n_x_fluid*n_y_fluid)
    row_all = cp.concatenate([row_sing, row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8])
    col_all = cp.concatenate([col_sing, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8])
    data_all = cp.concatenate([data_sing, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8])  
    #n_rows = n_x_fluid*n_x_fluid
    #n_cols = int(n_y_fluid*n_y_fluid/2)
    #a_bem_coo = cpsp.coo_matrix((data_all, (row_all, col_all)), shape=(n_rows, n_cols))
    # Attempt to reproduce dense matrix logic
    # data, row, col = computed earlier
    new_test = False
    if new_test:
        n_xf = n_x_fluid
        n_yf = n_y_fluid
        ny2 = int(n_yf // 2)

        # Decompose row and col into 2D indices
        j = row_all % n_xf
        l = row_all // n_xf
        i = col_all% ny2
        k = col_all // ny2

        # Flip part: flip i and k as in cp.flip on axes (2, 3)
        i_flip = ny2 - 1 - i
        k_flip = n_yf - 1 - k

        # Reconstruct full column indices
        col_full = i + k * n_yf
        col_flip = i_flip + k_flip * n_yf

        # Reconstruct row indices (stay the same)
        row_full = j + l * n_xf

        # Concatenate both original and flipped parts
        row_final = cp.concatenate([row_full, row_full])
        col_final = cp.concatenate([col_full, col_flip])
        data_final = cp.concatenate([data_1, data_1])

        # Create full matrix in coo format (GPU)
        #a2_full = cpsp.coo_matrix((data_final, (row_final, col_final)),
        #                    shape=(n_xf * n_xf, n_yf * n_yf))

        # ===== OPTIONAL AXIS SWAP STEP =====
        # swapaxes(1,2) equivalent → map (j, l, i, k) to (j, i, l, k)

        # Redecode new row/col indices
        j = row_final % n_xf
        l = row_final // n_xf
        i = col_final % n_yf
        k = col_final // n_yf

        # Swap axes 1 <-> 2 → new row = j + i*n_xf, col = l + k*n_yf
        row_swapped = j + i * n_xf
        col_swapped = l + k * n_yf

        a_out_1d = cpsp.coo_matrix((data_final, (row_swapped, col_swapped)),
                                    shape=(n_xf * n_yf, n_xf * n_yf))
    else:
    # Alternative with dense matrix
        a_bem_coo = cpsp.coo_matrix((data_all, (row_all, col_all)), shape=(n_rows, n_cols))
        a_vec = a_bem_coo.todense()
        a_out = a_vec.reshape((n_x_fluid, n_x_fluid, int(n_y_fluid/2), n_y_fluid))
        a2_flipped = cp.flip(a_out, axis=(2, 3))#[:, :, :, ::-1]
        a2_final = cp.concatenate([a_out, a2_flipped], axis=2)
        del a2_flipped, a_out
        a_out_1d = (a2_final.swapaxes(1,2)).reshape(n_x_fluid*n_y_fluid, n_x_fluid*n_y_fluid)
    return a_out_1d#, a_vec


def get_a_matrix_uniform_mid(xp, yp, x, y, lam):
    n_x_fluid = len(xp)
    n_y_fluid = len(yp)
    #yp2 = yp[:int(n_y_fluid/2)]
    #yp2 = yp#[:int(n_y_fluid/2)]
    yp2 = yp[:int(n_y_fluid/2)]
    scheme = quadpy.c1.gauss_legendre(8)
    w = cp.asarray(scheme.weights)
    theta_lin = cp.asarray(scheme.points)
    threshold = 1e-18
    
    x0 = xp[0]
    x1_ = x[:-1] - x0
    x2_ = x[1:] - x0
    y1_ = (y[:-1, cp.newaxis] - yp2).T.flatten()
    y2_ = (y[1:, cp.newaxis] - yp2).T.flatten()

    y2_grid, x2_grid = cp.meshgrid(y2_, x2_)
    y1_grid, x1_grid = cp.meshgrid(y1_, x1_)
    n_rows, n_cols = cp.shape(x1_grid)
    del x1_, x2_, y1_, y2_
    free_gpu_now()
    
    #a_vec = cp.zeros(cp.shape(x1_grid), dtype=complex)
    
    ### 'Solving top-right panel'
    ind_r = cp.where((y1_grid > 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = y1_grid[ind_r]
    y2 = y2_grid[ind_r]
    theta1 = cp.arctan(y2/x1)
    theta2 = cp.arctan(y2/x2)
    theta3 = cp.arctan(y1/x1)
    theta4 = cp.arctan(y1/x2)

    ind_thin = cp.where(theta2<theta3)
    ind_tall = cp.where(theta3<theta2)

    diff_ = cp.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = cp.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = cp.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = cp.expand_dims(x1[ind_thin], axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2[ind_thin], axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1[ind_thin], axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2[ind_thin], axis=1) * cp.ones(len(theta_lin))
    #print('Integration now')
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_thin = cp.sum(val, axis=1)

    x1_ = cp.expand_dims(x1[ind_tall], axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2[ind_tall], axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1[ind_tall], axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2[ind_tall], axis=1) * cp.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = cp.sum(val, axis=1)

    val_vec =  cp.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    mask = cp.abs(val_vec)/cp.max(cp.abs(val_vec)) > threshold
    row_1 = ind_r[0][ mask]
    col_1 = ind_r[1][ mask]
    data_1 = val_vec[mask]
    #a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34    
    free_gpu_now()

    #print('Solving bottom-right panel')
    # Bottom right region with y<0 and x>0
    ind_r = cp.where((y2_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = -y2_grid[ind_r]
    y2 = -y1_grid[ind_r]

    theta1 = cp.arctan(y2/x1)
    theta2 = cp.arctan(y2/x2)
    theta3 = cp.arctan(y1/x1)
    theta4 = cp.arctan(y1/x2)

    ind_thin = cp.where(theta2<theta3)
    ind_tall = cp.where(theta3<theta2)

    diff_ = cp.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, cp.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = cp.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = cp.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = cp.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, cp.newaxis]*diff_/2).T[ ind_thin].T
    x1_ = cp.expand_dims(x1[ind_thin], axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2[ind_thin], axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1[ind_thin], axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2[ind_thin], axis=1) * cp.ones(len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)
    val_thin = cp.sum(val, axis=1)

    x1_ = cp.expand_dims(x1[ind_tall], axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2[ind_tall], axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1[ind_tall], axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2[ind_tall], axis=1) * cp.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = cp.sum(val, axis=1)

    val_vec =  cp.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    #a_vec[ind_r] = val_vec
    mask = cp.abs(val_vec)/cp.max(cp.abs(val_vec)) > threshold
    row_2 = ind_r[0][ mask]
    col_2 = ind_r[1][ mask]
    data_2 = val_vec[mask]

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34    
    free_gpu_now()
    # Top panels
    ind_r = cp.where((x2_grid > 0) & (x1_grid < 0) & (y1_grid > 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]

    x1_ = cp.expand_dims(x1, axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2, axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1, axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2, axis=1) * cp.ones(len(theta_lin))

    theta1 = cp.arctan(y2/x1)
    theta2 = cp.arctan(y2/x2)
    theta3 = cp.arctan(y1/x1)
    theta4 = cp.arctan(y1/x2)

    diff_ = cp.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_31 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta1-cp.pi/2)
    theta_10 = (theta1 + cp.pi/2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_10 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(cp.pi/2-theta2)
    theta_02 = (cp.pi/2 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_02 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_24 = w[:, cp.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = cp.sum(val, axis=1)
    mask = cp.abs(val_vec)/cp.max(cp.abs(val_vec)) > threshold
    row_3 = ind_r[0][ mask]
    col_3 = ind_r[1][ mask]
    data_3 = val_vec[mask]
    #a_vec[ind_r] = val_vec

    # Bottom panels
    ind_r = cp.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y2_grid[ind_r]#[0]
    y2 = -y1_grid[ind_r]#[0]

    x1_ = cp.expand_dims(x1, axis=1) * cp.ones(len(theta_lin))
    x2_ = cp.expand_dims(x2, axis=1) * cp.ones(len(theta_lin))
    y1_ = cp.expand_dims(y1, axis=1) * cp.ones(len(theta_lin))
    y2_ = cp.expand_dims(y2, axis=1) * cp.ones(len(theta_lin))

    theta1 = cp.arctan(y2/x1)
    theta2 = cp.arctan(y2/x2)
    theta3 = cp.arctan(y1/x1)
    theta4 = cp.arctan(y1/x2)

    diff_ = cp.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_31 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta1-cp.pi/2)
    theta_10 = (theta1 + cp.pi/2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_10 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(cp.pi/2-theta2)
    theta_02 = (cp.pi/2 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_02 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_24 = w[:, cp.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = cp.sum(val, axis=1)
    #a_vec[ind_r] = val_vec
    mask = cp.abs(val_vec)/cp.max(cp.abs(val_vec)) > threshold
    row_4 = ind_r[0][ mask]
    col_4 = ind_r[1][ mask]
    data_4 = val_vec[mask]

    del theta_31, theta_10, theta_02, theta_24
    del weights_31, weights_10, weights_02, weights_24
    del val_vec, val
    free_gpu_now()
    # Right-side panels

    ind_r = cp.where((y2_grid > 0) & (y1_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    
    theta1 = cp.arctan(y2/x1)
    theta2 = cp.arctan(y2/x2)
    theta3 = cp.arctan(y1/x1)
    theta4 = cp.arctan(y1/x2)

    x1_ = cp.tile(cp.array([x1]).T, len(theta_lin))
    x2_ = cp.tile(cp.array([x2]).T, len(theta_lin))
    y1_ = cp.tile(cp.array([y1]).T, len(theta_lin))
    y2_ = cp.tile(cp.array([y2]).T, len(theta_lin))


    diff_ = cp.abs(theta2-theta1)
    theta_12 = (theta1 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_12 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(0-theta2)
    theta_20 = (0 + theta2)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_20 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta4-0)
    theta_04 = (theta4 + 0)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_04 = w[:, cp.newaxis]*diff_/2

    diff_ = cp.abs(theta3-theta4)
    theta_43 = (theta3 + theta4)/2 + diff_/2*theta_lin[:, cp.newaxis]
    weights_43 = w[:, cp.newaxis]*diff_/2

    val = (side_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + side_b(theta_20.T, x1_, x2_, lam)*weights_20.T
        + side_b(theta_04.T, x1_, x2_, lam)*weights_04.T
        + side_a(theta_43.T, x1_, y1_, lam)*weights_43.T)# + 
    val_vec = cp.sum(val, axis=1)
    #a_vec[ind_r] = val_vec
    mask = cp.abs(val_vec)/cp.max(cp.abs(val_vec)) > threshold
    row_5 = ind_r[0][ mask]
    col_5 = ind_r[1][ mask]
    data_5 = val_vec[mask]

    del val_vec, val, theta_12, theta_20, theta_04, theta_43
    del weights_12, weights_20, weights_04, weights_43
    # Singularity
    #print('Solving singularity')


    ind_r = cp.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid > 0) & (y1_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    #r_min = cp.min(cp.abs([x1, x2, y1, y2]), axis=0)
    stacked = cp.stack([cp.abs(x1), cp.abs(x2), cp.abs(y1), cp.abs(y2)], axis=0)
    r_min = cp.min(stacked, axis=0)
    xx = cp.array([x1, x1, x2, x2])
    yy = cp.array([y1, y2, y1, y2])
    x_ = cp.tile(cp.array([xx]).T, len(theta_lin))
    y_ = cp.tile(cp.array([yy]).T, len(theta_lin))
    theta_c = cp.arctan(yy/xx)
    diff_ = cp.abs(cp.pi/2-theta_c)
    theta_a = (cp.pi/2 + theta_c[:, :, cp.newaxis])/2 + diff_[:, :, cp.newaxis]/2*theta_lin
    weights_a = w*diff_[:, :, cp.newaxis]/2
    weights_a = weights_a.transpose(1, 0, 2)
    theta_a = theta_a.transpose(1, 0, 2)

    diff_ = cp.abs(0-theta_c)
    theta_b = (0 + theta_c[:, :, cp.newaxis])/2 + diff_[:, :, cp.newaxis]/2*theta_lin
    weights_b = w*diff_[:, :, cp.newaxis]/2
    weights_b = weights_b.transpose(1, 0, 2)
    theta_b = theta_b.transpose(1, 0, 2)

    val_circle = ((-4 * cp.pi * lam * r_min - 4 * cp.pi) 
            * cp.exp(-lam * r_min) / lam ** 2 / r_min + 4 * cp.pi / lam ** 2 / r_min) / 8 / cp.pi
    val_sing = (cp.sum(cp.sum(sing_a(theta_a, y_, r_min[:, cp.newaxis, cp.newaxis], lam)*weights_a
                            + sing_b(theta_b, x_, r_min[:, cp.newaxis, cp.newaxis], lam)*weights_b, axis=1), axis=1))
    #a_vec[ind_r] = val_sing + val_circle
    row_sing, col_sing = ind_r
    data_sing = val_sing + val_circle

    del theta_a, theta_b, weights_a, weights_b, val_sing, val_circle

    del y2_grid, x2_grid, y1_grid, x1_grid#, y0_grid, x0_grid
    row_all = cp.concatenate([row_sing, row_1, row_2, row_3, row_4, row_5])
    col_all = cp.concatenate([col_sing, col_1, col_2, col_3, col_4, col_5])
    data_all = cp.concatenate([data_sing, data_1, data_2, data_3, data_4, data_5])  
    #n_x_fluid#*n_x_fluid
    #n_cols = int(n_y_fluid*n_y_fluid/2)
    #a_bem_coo = cpsp.coo_matrix((data_all, (row_all, col_all)), shape=(n_rows, n_cols))
    # Attempt to reproduce dense matrix logic
    # data, row, col = computed earlier
    new_test = False
    #if new_test:
    #    pass
    #else:
    a_bem_coo = cpsp.coo_matrix((data_all, (row_all, col_all)), shape=(n_rows, n_cols))
    a_vec = a_bem_coo.todense()
    #print('Shape a_vec is ' + str(a_vec.shape))
    a_out = a_vec.reshape((n_x_fluid, int(n_y_fluid/2), n_y_fluid))
    a2_flipped = cp.flip(a_out, axis=(1, 2))
    a_out = cp.concatenate([a_out, a2_flipped], axis=1)
    a2 = cp.concatenate([(a_out[1::, :, :])[::-1, :, :], a_out])
    del a2_flipped, a_out, a_vec, a_bem_coo, row_all, col_all, data_all, row_1, row_2, row_3, row_4, row_5
    del data_1, data_2, data_3, data_4, data_5
    del row_sing, col_sing, data_sing
    free_gpu_now()
    #a_out_1d = (a2.swapaxes(1,2)).reshape
    if n_x_fluid*n_y_fluid < 128*129:
        dense = True
    else:
        dense = True
    if dense:
        nn = 0
        A_sbt = cp.zeros(([len(yp) * len(xp), len(yp) * len(xp)]), dtype=cp.complex64)
        for ix in range(n_x_fluid):
            for jy in range(n_y_fluid):
                STK_out = a2[len(xp)-1-ix:len(xp)-1-ix+len(xp), :, jy]
                A_sbt[:, nn] = STK_out.ravel()#flatten()
                nn+=1
    else:
        ## NOT WORKING YET
        nn = 0
        data = []
        row = []
        col = []
        for ix in range(n_x_fluid):
            for jy in range(n_y_fluid):
                STK_out = a2[len(xp)-1-ix:len(xp)-1-ix+len(xp), :, jy]  # shape (len(xp), len(yp))
                STK_flat = STK_out.ravel(order="K")#flatten()  # shape (len(xp) * len(yp),)

                idx = cp.arange(STK_flat.size)

                data.append(STK_flat)
                row.append(idx)
                col.append(cp.full_like(idx, nn))
                nn += 1

        # Stack into 1D arrays
        data = cp.concatenate(data)
        row = cp.concatenate(row)
        col = cp.concatenate(col)

        # Move data from GPU to CPU
        data_cpu  = cp.asnumpy(data)
        row_cpu   = cp.asnumpy(row)
        col_cpu   = cp.asnumpy(col)

        # Build sparse COO directly on CPU
        A_sbt_cpu = sp.coo_matrix((data_cpu, (row_cpu, col_cpu)),
                                shape=(n_x_fluid*n_y_fluid, n_x_fluid*n_y_fluid))

        # Convert to dense NumPy (warning: huge memory if matrix is big)
        A_sbt = A_sbt_cpu.toarray()
        # Build sparse matrix in COO format
        #A_sbt = cpsp.coo_matrix((data, (row, col)), shape=(n_x_fluid*n_y_fluid, n_x_fluid*n_y_fluid))#.tocsr()
    return A_sbt


def free_gpu_now():
    cp.cuda.runtime.deviceSynchronize()                 # wait for all kernels/copies
    cp.get_default_memory_pool().free_all_blocks()     # return GPU blocks to driver
    #cp.cuda.get_pinned_memory_pool().free_all_blocks() # release pinned host buffers
    #gc.collect() 