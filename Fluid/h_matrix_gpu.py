import cupy as cp
import sys
sys.path.insert(0, '..')
from Fluid.precompile_gpu import *
import quadpy
#from scipy.sparse import csr_matrix

def get_a_matrix_half(xp, yp, x, y, lam):
    n_x_fluid = len(xp)
    n_y_fluid = len(yp)
    yp2 = yp[:int(n_y_fluid/2)]
    scheme = quadpy.c1.gauss_legendre(8)
    w = cp.asarray(scheme.weights)
    theta_lin = cp.asarray(scheme.points)

    x1_ = (x[:-1, cp.newaxis] - xp).T.flatten()
    x2_ = (x[1:, cp.newaxis] - xp).T.flatten()
    y1_ = (y[:-1, cp.newaxis] - yp2).T.flatten()
    y2_ = (y[1:, cp.newaxis] - yp2).T.flatten()


    y2_grid, x2_grid = cp.meshgrid(y2_, x2_)
    y1_grid, x1_grid = cp.meshgrid(y1_, x1_)
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
    
    #del y0_grid, x0_grid

    a_vec = cp.zeros(cp.shape(x1_grid), dtype=cp.complex64)
    
    a_vec[ind_r] = val_sing + val_circle

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

    val_vec =  cp.zeros(len(x1), dtype=cp.complex64)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

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

    val_vec =  cp.zeros(len(x1), dtype=cp.complex64)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec
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

    val_vec =  cp.zeros(len(x1), dtype=cp.complex64)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

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

    val_vec =  cp.zeros(len(x1), dtype=cp.complex64)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

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
    a_vec[ind_r] = val_vec

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
    a_vec[ind_r] = val_vec

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
    a_vec[ind_r] = val_vec

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
    a_vec[ind_r] = val_vec

    del val_vec, val, theta_12, theta_20, theta_04, theta_43
    del weights_12, weights_20, weights_04, weights_43
    del y2_grid, x2_grid, y1_grid, x1_grid#, y0_grid, x0_grid
    
    #a_out = a_vec.reshape((n_x_fluid, n_x_fluid, int(n_y_fluid/2), n_y_fluid))
    #a2_flipped = cp.flip(a_out, axis=(2, 3))#[:, :, :, ::-1]
    #a2_final = cp.concatenate([a_out, a2_flipped], axis=2)
    #del a2_flipped, a_out
    #a_out_1d = (a2_final.swapaxes(1,2)).reshape(n_x_fluid*n_y_fluid, n_x_fluid*n_y_fluid)

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
    
    x0 = xp[0]
    x1_ = x[:-1] - x0
    x2_ = x[1:] - x0
    y1_ = (y[:-1, cp.newaxis] - yp2).T.flatten()
    y2_ = (y[1:, cp.newaxis] - yp2).T.flatten()

    y2_grid, x2_grid = cp.meshgrid(y2_, x2_)
    y1_grid, x1_grid = cp.meshgrid(y1_, x1_)
    del x1_, x2_, y1_, y2_
    
    a_vec = cp.zeros(cp.shape(x1_grid), dtype=cp.complex64)
    
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

    val_vec =  cp.zeros(len(x1), dtype=cp.complex64)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

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

    val_vec =  cp.zeros(len(x1), dtype=cp.complex64)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

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
    a_vec[ind_r] = val_vec

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
    a_vec[ind_r] = val_vec

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
    a_vec[ind_r] = val_vec

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
    a_vec[ind_r] = val_sing + val_circle

    del theta_a, theta_b, weights_a, weights_b, val_sing, val_circle

    del y2_grid, x2_grid, y1_grid, x1_grid#, y0_grid, x0_grid
    a_out = a_vec.reshape((n_x_fluid, int(n_y_fluid/2), n_y_fluid))
    a2_flipped = cp.flip(a_out, axis=(1, 2))
    a_out = cp.concatenate([a_out, a2_flipped], axis=1)
    a2 = cp.concatenate([(a_out[1::, :, :])[::-1, :, :], a_out])

    #a_out = a_vec.reshape((n_x_fluid, n_y_fluid, n_y_fluid))
    #a2 = cp.concatenate([(a_out[1::, :, :])[::-1, :, :], a_out])
    A_sbt = cp.zeros(([len(yp) * len(xp), len(yp) * len(xp)]), dtype=cp.complex64)
    nn = 0#len(yp)
    for ix in range(len(xp)):
        for jy in range(int(len(yp))):
            STK_out = a2[len(xp)-1-ix:len(xp)-1-ix+len(xp), :, jy]
            A_sbt[:, nn] = STK_out.flatten()
            nn+=1
    #A_sbt = A_sbt.T
    #a_plot = cp.real(
    return A_sbt#, a_vec
