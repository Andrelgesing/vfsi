import numpy as np
import sys
sys.path.insert(0, '..')
from Fluid.precompile_app import *
import quadpy
from scipy.sparse import csr_matrix


def get_a_matrix_half(xp, yp, x, y, lam):
    n_x_fluid = len(xp)
    n_y_fluid = len(yp)
    yp2 = yp[:int(n_y_fluid/2)]
    #yp2 = yp#[:int(n_y_fluid/2)]
    scheme = quadpy.c1.gauss_legendre(8)
    w = scheme.weights
    theta_lin = (scheme.points)

    x1_ = (x[:-1, np.newaxis] - xp).T.flatten()
    x2_ = (x[1:, np.newaxis] - xp).T.flatten()
    y1_ = (y[:-1, np.newaxis] - yp2).T.flatten()
    y2_ = (y[1:, np.newaxis] - yp2).T.flatten()


    #y2_grid, x2_grid = np.meshgrid(y2_, x2_, sparse=True)
    #y1_grid, x1_grid = np.meshgrid(y1_, x1_, sparse=True)

    #x1_grid, y1_grid = np.broadcast_arrays(x1_grid, y1_grid)
    #x2_grid, y2_grid = np.broadcast_arrays(x2_grid, y2_grid)
    y2_grid, x2_grid = np.meshgrid(y2_, x2_)
    y1_grid, x1_grid = np.meshgrid(y1_, x1_)
    del x1_, x2_, y1_, y2_
    #y0_grid, x0_grid = np.meshgrid(np.repeat(yp2, len(yp)).flatten(),
    #                            np.repeat(xp, len(xp)).T.flatten())


    # Singularity
    #print('Solving singularity')
    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid > 0) & (y1_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    r_min = np.min(np.abs([x1, x2, y1, y2]), axis=0)
    xx = np.array([x1, x1, x2, x2])
    yy = np.array([y1, y2, y1, y2])
    x_ = np.tile(np.array([xx]).T, len(theta_lin))
    y_ = np.tile(np.array([yy]).T, len(theta_lin))
    theta_c = np.arctan(yy/xx)
    diff_ = np.abs(np.pi/2-theta_c)
    theta_a = (np.pi/2 + theta_c[:, :, np.newaxis])/2 + diff_[:, :, np.newaxis]/2*theta_lin
    weights_a = w*diff_[:, :, np.newaxis]/2
    weights_a = weights_a.transpose(1, 0, 2)
    theta_a = theta_a.transpose(1, 0, 2)

    diff_ = np.abs(0-theta_c)
    theta_b = (0 + theta_c[:, :, np.newaxis])/2 + diff_[:, :, np.newaxis]/2*theta_lin
    weights_b = w*diff_[:, :, np.newaxis]/2
    weights_b = weights_b.transpose(1, 0, 2)
    theta_b = theta_b.transpose(1, 0, 2)

    val_circle = ((-4 * np.pi * lam * r_min - 4 * np.pi) 
            * np.exp(-lam * r_min) / lam ** 2 / r_min + 4 * np.pi / lam ** 2 / r_min) / 8 / np.pi
    val_sing = (np.sum(np.sum(sing_a(theta_a, y_, r_min[:, np.newaxis, np.newaxis], lam)*weights_a
                            + sing_b(theta_b, x_, r_min[:, np.newaxis, np.newaxis], lam)*weights_b, axis=1), axis=1))
    
    #del y0_grid, x0_grid

    a_vec = np.zeros(np.shape(x1_grid), dtype=complex)
    
    a_vec[ind_r] = val_sing + val_circle

    del theta_a, theta_b, weights_a, weights_b, val_sing, val_circle

    
    #a_vec = {}
    #num_elements = np.prod(x0_grid.shape)  # Total number of elements in a_vec
    #a_vec = csr_matrix((num_elements, 1), dtype=complex)
    
    ind_r = np.where((y1_grid > 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = y1_grid[ind_r]
    y2 = y2_grid[ind_r]
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = np.expand_dims(x1[ind_thin], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_thin], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_thin], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_thin], axis=1) * np.ones(len(theta_lin))
    #print('Integration now')
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_thin = np.sum(val, axis=1)

    x1_ = np.expand_dims(x1[ind_tall], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_tall], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_tall], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_tall], axis=1) * np.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)
 
    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    #print('Solving top-left panel')
    # Top left region with y>0 and x<0
    ind_r = np.where((y1_grid > 0) & (x2_grid < 0))
    x1 = -x2_grid[ind_r]
    x2 = -x1_grid[ind_r]
    y1 = y1_grid[ind_r]
    y2 = y2_grid[ind_r]
    #y0_ = y0_grid[ind_r]
    #x0_ = x0_grid[ind_r]
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = np.expand_dims(x1[ind_thin], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_thin], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_thin], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_thin], axis=1) * np.ones(len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)
    val_thin = np.sum(val, axis=1)

    x1_ = np.expand_dims(x1[ind_tall], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_tall], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_tall], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_tall], axis=1) * np.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec
    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    #print('Solving bottom-left panel')
    # Bottom left region with y<0 and x<0
    ind_r = np.where((y2_grid < 0) & (x2_grid < 0))
    x1 = -x2_grid[ind_r]
    x2 = -x1_grid[ind_r]
    y1 = -y2_grid[ind_r]
    y2 = -y1_grid[ind_r]
    #y0_ = y0_grid[ind_r]
    #x0_ = x0_grid[ind_r]
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T
    x1_ = np.expand_dims(x1[ind_thin], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_thin], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_thin], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_thin], axis=1) * np.ones(len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)
    val_thin = np.sum(val, axis=1)

    x1_ = np.expand_dims(x1[ind_tall], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_tall], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_tall], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_tall], axis=1) * np.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    #print('Solving bottom-right panel')
    # Bottom right region with y<0 and x>0
    ind_r = np.where((y2_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = -y2_grid[ind_r]
    y2 = -y1_grid[ind_r]

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T
    x1_ = np.expand_dims(x1[ind_thin], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_thin], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_thin], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_thin], axis=1) * np.ones(len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)
    val_thin = np.sum(val, axis=1)

    x1_ = np.expand_dims(x1[ind_tall], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_tall], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_tall], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_tall], axis=1) * np.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    # Top panels
    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y1_grid > 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]

    x1_ = np.expand_dims(x1, axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2, axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1, axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2, axis=1) * np.ones(len(theta_lin))

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    diff_ = np.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_31 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta1-np.pi/2)
    theta_10 = (theta1 + np.pi/2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_10 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(np.pi/2-theta2)
    theta_02 = (np.pi/2 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_02 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_24 = w[:, np.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    # Bottom panels
    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y2_grid[ind_r]#[0]
    y2 = -y1_grid[ind_r]#[0]

    x1_ = np.expand_dims(x1, axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2, axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1, axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2, axis=1) * np.ones(len(theta_lin))

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    diff_ = np.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_31 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta1-np.pi/2)
    theta_10 = (theta1 + np.pi/2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_10 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(np.pi/2-theta2)
    theta_02 = (np.pi/2 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_02 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_24 = w[:, np.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    del theta_31, theta_10, theta_02, theta_24
    del weights_31, weights_10, weights_02, weights_24
    del val_vec, val
    
    # Right-side panels

    ind_r = np.where((y2_grid > 0) & (y1_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    x1_ = np.tile(np.array([x1]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2]).T, len(theta_lin))


    diff_ = np.abs(theta2-theta1)
    theta_12 = (theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_12 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(0-theta2)
    theta_20 = (0 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_20 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta4-0)
    theta_04 = (theta4 + 0)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_04 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta3-theta4)
    theta_43 = (theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_43 = w[:, np.newaxis]*diff_/2

    val = (side_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + side_b(theta_20.T, x1_, x2_, lam)*weights_20.T
        + side_b(theta_04.T, x1_, x2_, lam)*weights_04.T
        + side_a(theta_43.T, x1_, y1_, lam)*weights_43.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    # Left-side panels

    ind_r = np.where((y2_grid > 0) & (y1_grid < 0) & (x2_grid < 0) & (x1_grid < 0))
    x1 = -x2_grid[ind_r]#[0]
    x2 = -x1_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    x1_ = np.tile(np.array([x1]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2]).T, len(theta_lin))


    diff_ = np.abs(theta2-theta1)
    theta_12 = (theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_12 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(0-theta2)
    theta_20 = (0 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_20 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta4-0)
    theta_04 = (theta4 + 0)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_04 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta3-theta4)
    theta_43 = (theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_43 = w[:, np.newaxis]*diff_/2

    val = (side_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + side_b(theta_20.T, x1_, x2_, lam)*weights_20.T
        + side_b(theta_04.T, x1_, x2_, lam)*weights_04.T
        + side_a(theta_43.T, x1_, y1_, lam)*weights_43.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    del val_vec, val, theta_12, theta_20, theta_04, theta_43
    del weights_12, weights_20, weights_04, weights_43
    del y2_grid, x2_grid, y1_grid, x1_grid#, y0_grid, x0_grid
    
    a_out = a_vec.reshape((n_x_fluid, n_x_fluid, int(n_y_fluid/2), n_y_fluid))
    #del a_vec
    #a_out = a_out.swapaxes(2, 3)
    a2_flipped = np.flip(a_out, axis=(2, 3))#[:, :, :, ::-1]
    a2_final = np.concatenate([a_out, a2_flipped], axis=2)
    del a2_flipped, a_out
    a_out_1d = (a2_final.swapaxes(1,2)).reshape(n_x_fluid*n_y_fluid, n_x_fluid*n_y_fluid)
    #return a2_final
    #a_out = a_vec.reshape((n_x_fluid, n_y_fluid, n_x_fluid, n_y_fluid))
    #a_out = a_out.transpose(0, 2, 1, 3).reshape(n_x_fluid * n_y_fluid, n_x_fluid * n_y_fluid)


    #a_out_1d = a_vec.reshape((n_x_fluid, n_x_fluid, n_y_fluid, n_y_fluid))
    #a_out_1d = a_out_1d.swapaxes(1, 3)
    #a_out_1d = (a_out_1d.swapaxes(2,3)).reshape(n_x_fluid*n_y_fluid, n_x_fluid*n_y_fluid)
    return a_out_1d#, a_vec

def get_a_matrix(xp, yp, x, y, lam):
    n_x_fluid = len(xp)
    n_y_fluid = len(yp)
    #yp2 = yp[:int(n_y_fluid/2)]
    yp2 = yp#[:int(n_y_fluid/2)]
    scheme = quadpy.c1.gauss_legendre(8)
    w = scheme.weights
    theta_lin = (scheme.points)

    x1_ = (x[:-1, np.newaxis] - xp).T.flatten()
    x2_ = (x[1:, np.newaxis] - xp).T.flatten()
    y1_ = (y[:-1, np.newaxis] - yp2).T.flatten()
    y2_ = (y[1:, np.newaxis] - yp2).T.flatten()


    #y2_grid, x2_grid = np.meshgrid(y2_, x2_, sparse=True)
    #y1_grid, x1_grid = np.meshgrid(y1_, x1_, sparse=True)

    #x1_grid, y1_grid = np.broadcast_arrays(x1_grid, y1_grid)
    #x2_grid, y2_grid = np.broadcast_arrays(x2_grid, y2_grid)
    y2_grid, x2_grid = np.meshgrid(y2_, x2_)
    y1_grid, x1_grid = np.meshgrid(y1_, x1_)
    del x1_, x2_, y1_, y2_


    a_vec = np.zeros(np.shape(x1_grid), dtype=complex)
    #y0_grid, x0_grid = np.meshgrid(np.repeat(yp2, len(yp)).flatten(),
    #                            np.repeat(xp, len(xp)).T.flatten())

    
    #a_vec = {}
    #num_elements = np.prod(x0_grid.shape)  # Total number of elements in a_vec
    #a_vec = csr_matrix((num_elements, 1), dtype=complex)
    
    ind_r = np.where((y1_grid > 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = y1_grid[ind_r]
    y2 = y2_grid[ind_r]
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = np.expand_dims(x1[ind_thin], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_thin], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_thin], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_thin], axis=1) * np.ones(len(theta_lin))
    #print('Integration now')
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_thin = np.sum(val, axis=1)

    x1_ = np.expand_dims(x1[ind_tall], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_tall], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_tall], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_tall], axis=1) * np.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    #print('Solving top-left panel')
    # Top left region with y>0 and x<0
    ind_r = np.where((y1_grid > 0) & (x2_grid < 0))
    x1 = -x2_grid[ind_r]
    x2 = -x1_grid[ind_r]
    y1 = y1_grid[ind_r]
    y2 = y2_grid[ind_r]
    #y0_ = y0_grid[ind_r]
    #x0_ = x0_grid[ind_r]
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = np.expand_dims(x1[ind_thin], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_thin], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_thin], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_thin], axis=1) * np.ones(len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)
    val_thin = np.sum(val, axis=1)

    x1_ = np.expand_dims(x1[ind_tall], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_tall], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_tall], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_tall], axis=1) * np.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec
    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    #print('Solving bottom-left panel')
    # Bottom left region with y<0 and x<0
    ind_r = np.where((y2_grid < 0) & (x2_grid < 0))
    x1 = -x2_grid[ind_r]
    x2 = -x1_grid[ind_r]
    y1 = -y2_grid[ind_r]
    y2 = -y1_grid[ind_r]
    #y0_ = y0_grid[ind_r]
    #x0_ = x0_grid[ind_r]
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T
    x1_ = np.expand_dims(x1[ind_thin], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_thin], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_thin], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_thin], axis=1) * np.ones(len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)
    val_thin = np.sum(val, axis=1)

    x1_ = np.expand_dims(x1[ind_tall], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_tall], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_tall], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_tall], axis=1) * np.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    #print('Solving bottom-right panel')
    # Bottom right region with y<0 and x>0
    ind_r = np.where((y2_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = -y2_grid[ind_r]
    y2 = -y1_grid[ind_r]

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T
    x1_ = np.expand_dims(x1[ind_thin], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_thin], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_thin], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_thin], axis=1) * np.ones(len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)
    val_thin = np.sum(val, axis=1)

    x1_ = np.expand_dims(x1[ind_tall], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_tall], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_tall], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_tall], axis=1) * np.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    # Top panels
    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y1_grid > 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]

    x1_ = np.expand_dims(x1, axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2, axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1, axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2, axis=1) * np.ones(len(theta_lin))

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    diff_ = np.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_31 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta1-np.pi/2)
    theta_10 = (theta1 + np.pi/2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_10 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(np.pi/2-theta2)
    theta_02 = (np.pi/2 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_02 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_24 = w[:, np.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    # Bottom panels
    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y2_grid[ind_r]#[0]
    y2 = -y1_grid[ind_r]#[0]

    x1_ = np.expand_dims(x1, axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2, axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1, axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2, axis=1) * np.ones(len(theta_lin))

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    diff_ = np.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_31 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta1-np.pi/2)
    theta_10 = (theta1 + np.pi/2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_10 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(np.pi/2-theta2)
    theta_02 = (np.pi/2 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_02 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_24 = w[:, np.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    del theta_31, theta_10, theta_02, theta_24
    del weights_31, weights_10, weights_02, weights_24
    del val_vec, val

    # Right-side panels

    ind_r = np.where((y2_grid > 0) & (y1_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    x1_ = np.tile(np.array([x1]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2]).T, len(theta_lin))


    diff_ = np.abs(theta2-theta1)
    theta_12 = (theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_12 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(0-theta2)
    theta_20 = (0 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_20 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta4-0)
    theta_04 = (theta4 + 0)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_04 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta3-theta4)
    theta_43 = (theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_43 = w[:, np.newaxis]*diff_/2

    val = (side_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + side_b(theta_20.T, x1_, x2_, lam)*weights_20.T
        + side_b(theta_04.T, x1_, x2_, lam)*weights_04.T
        + side_a(theta_43.T, x1_, y1_, lam)*weights_43.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    # Left-side panels

    ind_r = np.where((y2_grid > 0) & (y1_grid < 0) & (x2_grid < 0) & (x1_grid < 0))
    x1 = -x2_grid[ind_r]#[0]
    x2 = -x1_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    x1_ = np.tile(np.array([x1]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2]).T, len(theta_lin))


    diff_ = np.abs(theta2-theta1)
    theta_12 = (theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_12 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(0-theta2)
    theta_20 = (0 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_20 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta4-0)
    theta_04 = (theta4 + 0)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_04 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta3-theta4)
    theta_43 = (theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_43 = w[:, np.newaxis]*diff_/2

    val = (side_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + side_b(theta_20.T, x1_, x2_, lam)*weights_20.T
        + side_b(theta_04.T, x1_, x2_, lam)*weights_04.T
        + side_a(theta_43.T, x1_, y1_, lam)*weights_43.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    del val_vec, val, theta_12, theta_20, theta_04, theta_43
    del weights_12, weights_20, weights_04, weights_43


    # Singularity
    #print('Solving singularity')


    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid > 0) & (y1_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    r_min = np.min(np.abs([x1, x2, y1, y2]), axis=0)
    xx = np.array([x1, x1, x2, x2])
    yy = np.array([y1, y2, y1, y2])
    x_ = np.tile(np.array([xx]).T, len(theta_lin))
    y_ = np.tile(np.array([yy]).T, len(theta_lin))
    theta_c = np.arctan(yy/xx)
    diff_ = np.abs(np.pi/2-theta_c)
    theta_a = (np.pi/2 + theta_c[:, :, np.newaxis])/2 + diff_[:, :, np.newaxis]/2*theta_lin
    weights_a = w*diff_[:, :, np.newaxis]/2
    weights_a = weights_a.transpose(1, 0, 2)
    theta_a = theta_a.transpose(1, 0, 2)

    diff_ = np.abs(0-theta_c)
    theta_b = (0 + theta_c[:, :, np.newaxis])/2 + diff_[:, :, np.newaxis]/2*theta_lin
    weights_b = w*diff_[:, :, np.newaxis]/2
    weights_b = weights_b.transpose(1, 0, 2)
    theta_b = theta_b.transpose(1, 0, 2)

    val_circle = ((-4 * np.pi * lam * r_min - 4 * np.pi) 
            * np.exp(-lam * r_min) / lam ** 2 / r_min + 4 * np.pi / lam ** 2 / r_min) / 8 / np.pi
    val_sing = (np.sum(np.sum(sing_a(theta_a, y_, r_min[:, np.newaxis, np.newaxis], lam)*weights_a
                            + sing_b(theta_b, x_, r_min[:, np.newaxis, np.newaxis], lam)*weights_b, axis=1), axis=1))
    a_vec[ind_r] = val_sing + val_circle
    #val_local = val_sing + val_circle

    del theta_a, theta_b, weights_a, weights_b, val_sing, val_circle

    del y2_grid, x2_grid, y1_grid, x1_grid#, y0_grid, x0_grid
    
    #a_out = a_vec.reshape((n_x_fluid, n_x_fluid, int(n_y_fluid/2), n_y_fluid))
    #del a_vec
    #a_out = a_out.swapaxes(2, 3)
    #a2_flipped = np.flip(a_out, axis=(2, 3))#[:, :, :, ::-1]
    #a2_final = np.concatenate([a_out, a2_flipped], axis=3)
    #del a2_flipped, a_out
    #a_out_1d = (a2_final.swapaxes(1,2)).reshape(n_x_fluid*n_y_fluid, n_x_fluid*n_y_fluid)
    #return a2_final
    #a_out = a_vec.reshape((n_x_fluid, n_y_fluid, n_x_fluid, n_y_fluid))
    #a_out = a_out.transpose(0, 2, 1, 3).reshape(n_x_fluid * n_y_fluid, n_x_fluid * n_y_fluid)


    a_out_1d = a_vec.reshape((n_x_fluid, n_x_fluid, n_y_fluid, n_y_fluid))
    #a_out_1d = a_out_1d.swapaxes(1, 3)
    a_out_1d = (a_out_1d.swapaxes(1,2)).reshape(n_x_fluid*n_y_fluid, n_x_fluid*n_y_fluid)
    return a_out_1d#, a_vec



def get_a_matrix_uniform(xp, yp, x, y, lam):
    n_x_fluid = len(xp)
    n_y_fluid = len(yp)
    #yp2 = yp[:int(n_y_fluid/2)]
    yp2 = yp#[:int(n_y_fluid/2)]
    scheme = quadpy.c1.gauss_legendre(8)
    w = scheme.weights
    theta_lin = (scheme.points)
    
    x0 = xp[0]
    x1_ = x[:-1] - x0
    x2_ = x[1:] - x0
    y1_ = (y[:-1, np.newaxis] - yp2).T.flatten()
    y2_ = (y[1:, np.newaxis] - yp2).T.flatten()

    y2_grid, x2_grid = np.meshgrid(y2_, x2_)
    y1_grid, x1_grid = np.meshgrid(y1_, x1_)
    del x1_, x2_, y1_, y2_


    y0_grid, x0_grid = np.meshgrid(np.repeat(yp, len(yp)).flatten(),
                               np.repeat(x0, 1).T.flatten())
    
    a_vec = np.zeros(np.shape(x1_grid), dtype=complex)
    
    ### 'Solving top-right panel'
    ind_r = np.where((y1_grid > 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = y1_grid[ind_r]
    y2 = y2_grid[ind_r]
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = np.expand_dims(x1[ind_thin], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_thin], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_thin], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_thin], axis=1) * np.ones(len(theta_lin))
    #print('Integration now')
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_thin = np.sum(val, axis=1)

    x1_ = np.expand_dims(x1[ind_tall], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_tall], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_tall], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_tall], axis=1) * np.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34


    #print('Solving bottom-right panel')
    # Bottom right region with y<0 and x>0
    ind_r = np.where((y2_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = -y2_grid[ind_r]
    y2 = -y1_grid[ind_r]

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T
    x1_ = np.expand_dims(x1[ind_thin], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_thin], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_thin], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_thin], axis=1) * np.ones(len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)
    val_thin = np.sum(val, axis=1)

    x1_ = np.expand_dims(x1[ind_tall], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_tall], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_tall], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_tall], axis=1) * np.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    # Top panels
    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y1_grid > 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]

    x1_ = np.expand_dims(x1, axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2, axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1, axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2, axis=1) * np.ones(len(theta_lin))

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    diff_ = np.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_31 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta1-np.pi/2)
    theta_10 = (theta1 + np.pi/2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_10 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(np.pi/2-theta2)
    theta_02 = (np.pi/2 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_02 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_24 = w[:, np.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    # Bottom panels
    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y2_grid[ind_r]#[0]
    y2 = -y1_grid[ind_r]#[0]

    x1_ = np.expand_dims(x1, axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2, axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1, axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2, axis=1) * np.ones(len(theta_lin))

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    diff_ = np.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_31 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta1-np.pi/2)
    theta_10 = (theta1 + np.pi/2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_10 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(np.pi/2-theta2)
    theta_02 = (np.pi/2 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_02 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_24 = w[:, np.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    del theta_31, theta_10, theta_02, theta_24
    del weights_31, weights_10, weights_02, weights_24
    del val_vec, val

    # Right-side panels

    ind_r = np.where((y2_grid > 0) & (y1_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    x1_ = np.tile(np.array([x1]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2]).T, len(theta_lin))


    diff_ = np.abs(theta2-theta1)
    theta_12 = (theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_12 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(0-theta2)
    theta_20 = (0 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_20 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta4-0)
    theta_04 = (theta4 + 0)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_04 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta3-theta4)
    theta_43 = (theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_43 = w[:, np.newaxis]*diff_/2

    val = (side_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + side_b(theta_20.T, x1_, x2_, lam)*weights_20.T
        + side_b(theta_04.T, x1_, x2_, lam)*weights_04.T
        + side_a(theta_43.T, x1_, y1_, lam)*weights_43.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    del val_vec, val, theta_12, theta_20, theta_04, theta_43
    del weights_12, weights_20, weights_04, weights_43
    # Singularity
    #print('Solving singularity')


    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid > 0) & (y1_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    r_min = np.min(np.abs([x1, x2, y1, y2]), axis=0)
    xx = np.array([x1, x1, x2, x2])
    yy = np.array([y1, y2, y1, y2])
    x_ = np.tile(np.array([xx]).T, len(theta_lin))
    y_ = np.tile(np.array([yy]).T, len(theta_lin))
    theta_c = np.arctan(yy/xx)
    diff_ = np.abs(np.pi/2-theta_c)
    theta_a = (np.pi/2 + theta_c[:, :, np.newaxis])/2 + diff_[:, :, np.newaxis]/2*theta_lin
    weights_a = w*diff_[:, :, np.newaxis]/2
    weights_a = weights_a.transpose(1, 0, 2)
    theta_a = theta_a.transpose(1, 0, 2)

    diff_ = np.abs(0-theta_c)
    theta_b = (0 + theta_c[:, :, np.newaxis])/2 + diff_[:, :, np.newaxis]/2*theta_lin
    weights_b = w*diff_[:, :, np.newaxis]/2
    weights_b = weights_b.transpose(1, 0, 2)
    theta_b = theta_b.transpose(1, 0, 2)

    val_circle = ((-4 * np.pi * lam * r_min - 4 * np.pi) 
            * np.exp(-lam * r_min) / lam ** 2 / r_min + 4 * np.pi / lam ** 2 / r_min) / 8 / np.pi
    val_sing = (np.sum(np.sum(sing_a(theta_a, y_, r_min[:, np.newaxis, np.newaxis], lam)*weights_a
                            + sing_b(theta_b, x_, r_min[:, np.newaxis, np.newaxis], lam)*weights_b, axis=1), axis=1))
    a_vec[ind_r] = val_sing + val_circle

    del theta_a, theta_b, weights_a, weights_b, val_sing, val_circle

    del y2_grid, x2_grid, y1_grid, x1_grid#, y0_grid, x0_grid
    
    a_out = a_vec.reshape((n_x_fluid, n_y_fluid, n_y_fluid))
    a2 = np.concatenate([(a_out[1::, :, :])[::-1, :, :], a_out])
    A_sbt = np.zeros(([len(yp) * len(xp), len(yp) * len(xp)]), dtype=np.complex64)
    nn = 0#len(yp)
    for ix in range(len(xp)):
        for jy in range(int(len(yp))):
            STK_out = a2[len(xp)-1-ix:len(xp)-1-ix+len(xp), jy, :]
            A_sbt[nn, :] = STK_out.flatten()
            nn+=1
    #a_plot = np.real(
    return A_sbt#, a_vec
 
def lin_quad(ny):
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
        y = np.linspace(0, 1, ny + 1)
        # Calculate the interior points as midpoints between consecutive points
        yp = 1/2 * (y[1:] + y[:-1])
        # Return the points and midpoints
        return y, yp



def get_a_matrix_hyerarchy(xp, yp, x, y, lam, n_x2=5, n_y2=5):
    n_x1 = len(xp)
    n_y1 = len(yp)
    n_x_fluid = n_x1*n_x2#(n_x1-1) + n_x2
    n_x_fluid_ = (n_x1-1) + n_x2
    n_y_fluid = n_y1*n_y2#
    n_y_fluid_ = (n_y1-2) + 2*n_y2
    l_w = np.max(x)/(y[-1]-y[0])
    x, xp = np.array(lin_quad(n_x_fluid), dtype=object)
    y, yp = (np.array(lin_quad(n_y_fluid), dtype=object)-0.5)/l_w
    scheme = quadpy.c1.gauss_legendre(8)
    w = scheme.weights
    theta_lin = (scheme.points)
    x0 = xp[0]
    x1_ = x[:-1] - x0
    x2_ = x[1:] - x0
    y0 = yp[0]
    y1_ = (y[:-1] - y0).T.flatten()
    y2_ = (y[1:] - y0).T.flatten()


    y2_grid, x2_grid = np.meshgrid(y2_, x2_)
    y1_grid, x1_grid = np.meshgrid(y1_, x1_)
    del x1_, x2_, y1_, y2_
    
    a_vec = np.zeros(np.shape(x1_grid), dtype=complex)
    
    ### 'Solving top-right panel'
    ind_r = np.where((y1_grid > 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = y1_grid[ind_r]
    y2 = y2_grid[ind_r]
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = np.expand_dims(x1[ind_thin], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_thin], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_thin], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_thin], axis=1) * np.ones(len(theta_lin))
    #print('Integration now')
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_thin = np.sum(val, axis=1)

    x1_ = np.expand_dims(x1[ind_tall], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_tall], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_tall], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_tall], axis=1) * np.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    # Top panels
    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y1_grid > 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]

    x1_ = np.expand_dims(x1, axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2, axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1, axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2, axis=1) * np.ones(len(theta_lin))

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    diff_ = np.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_31 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta1-np.pi/2)
    theta_10 = (theta1 + np.pi/2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_10 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(np.pi/2-theta2)
    theta_02 = (np.pi/2 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_02 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_24 = w[:, np.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    # Right-side panels

    ind_r = np.where((y2_grid > 0) & (y1_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    x1_ = np.tile(np.array([x1]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2]).T, len(theta_lin))


    diff_ = np.abs(theta2-theta1)
    theta_12 = (theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_12 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(0-theta2)
    theta_20 = (0 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_20 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta4-0)
    theta_04 = (theta4 + 0)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_04 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta3-theta4)
    theta_43 = (theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_43 = w[:, np.newaxis]*diff_/2

    val = (side_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + side_b(theta_20.T, x1_, x2_, lam)*weights_20.T
        + side_b(theta_04.T, x1_, x2_, lam)*weights_04.T
        + side_a(theta_43.T, x1_, y1_, lam)*weights_43.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    del val_vec, val, theta_12, theta_20, theta_04, theta_43
    del weights_12, weights_20, weights_04, weights_43
    # Singularity
    #print('Solving singularity')


    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid > 0) & (y1_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    r_min = np.min(np.abs([x1, x2, y1, y2]), axis=0)
    xx = np.array([x1, x1, x2, x2])
    yy = np.array([y1, y2, y1, y2])
    x_ = np.tile(np.array([xx]).T, len(theta_lin))
    y_ = np.tile(np.array([yy]).T, len(theta_lin))
    theta_c = np.arctan(yy/xx)
    diff_ = np.abs(np.pi/2-theta_c)
    theta_a = (np.pi/2 + theta_c[:, :, np.newaxis])/2 + diff_[:, :, np.newaxis]/2*theta_lin
    weights_a = w*diff_[:, :, np.newaxis]/2
    weights_a = weights_a.transpose(1, 0, 2)
    theta_a = theta_a.transpose(1, 0, 2)

    diff_ = np.abs(0-theta_c)
    theta_b = (0 + theta_c[:, :, np.newaxis])/2 + diff_[:, :, np.newaxis]/2*theta_lin
    weights_b = w*diff_[:, :, np.newaxis]/2
    weights_b = weights_b.transpose(1, 0, 2)
    theta_b = theta_b.transpose(1, 0, 2)

    val_circle = ((-4 * np.pi * lam * r_min - 4 * np.pi) 
            * np.exp(-lam * r_min) / lam ** 2 / r_min + 4 * np.pi / lam ** 2 / r_min) / 8 / np.pi
    val_sing = (np.sum(np.sum(sing_a(theta_a, y_, r_min[:, np.newaxis, np.newaxis], lam)*weights_a
                            + sing_b(theta_b, x_, r_min[:, np.newaxis, np.newaxis], lam)*weights_b, axis=1), axis=1))
    a_vec[ind_r] = val_sing + val_circle

    del theta_a, theta_b, weights_a, weights_b, val_sing, val_circle

    del y2_grid, x2_grid, y1_grid, x1_grid#, y0_grid, x0_grid

    # Assemble matrix over total domain
    a2 = np.concatenate([(a_vec[1::, :])[::-1, :], a_vec])
    a2 = np.concatenate([(a2[:, 1::])[:, ::-1], a2], axis=1)
    
    ind_ys_local = list(list(np.arange(0, n_y2, 1)) + list(np.arange(n_y2-1+int(np.floor(n_y2/2)+1), n_y_fluid-n_y2, n_y2))
             + list(np.arange(n_y_fluid-n_y2, n_y_fluid, 1)))
    ind_xs_local = list(list(np.arange(int(np.floor(n_x2/2)), n_x_fluid-n_x2, n_x2))
             + list(np.arange(n_x_fluid-n_x2, n_x_fluid, 1)))
    H_coarse_final = np.zeros((n_x_fluid_*n_y_fluid_, n_x_fluid_*n_y_fluid_), dtype=complex)
    nn = 0
    ind_x = ind_xs_local[0]
    ind_y = ind_ys_local[0]
    ind_x1 = int(-ind_x+n_x_fluid-1)
    ind_x2 = int(-ind_x+n_x_fluid+n_x_fluid-1)
    ind_y1 = int(-ind_y+n_y_fluid-1)
    ind_y2 = int(-ind_y+n_y_fluid+n_y_fluid-1)
    H_local = a2[ind_x1:ind_x2, ind_y1:ind_y2]
    for ii, ind_x in enumerate(ind_xs_local):
        for jj, ind_y in enumerate(ind_ys_local):
            H_local = a2[int(-ind_x+n_x_fluid-1):int(-ind_x+n_x_fluid+n_x_fluid-1),
                        int(-ind_y+n_y_fluid-1):int(-ind_y+n_y_fluid+n_y_fluid-1)]
            H_coarse = reduce_matrix(H_local, n_x_fluid, n_y_fluid, n_x2, n_y2)
            H_coarse_final[nn, :] = H_coarse.flatten()
            nn += 1
    return H_coarse_final
    
def reduce_matrix(H_local, n_x_fluid, n_y_fluid, n_x2, n_y2):
    H_x = np.add.reduceat(H_local[:n_x_fluid-n_x2, :], np.arange(0, n_x_fluid-n_x2-1, n_x2))
    H_bot = H_x[:, :n_y2]
    H_top = H_x[:, n_y_fluid-n_y2:]
    H_mid = H_x[:, n_y2:n_y_fluid-n_y2]
    H_mid = np.add.reduceat(H_mid, np.arange(0, n_y_fluid-2*n_y2-1, n_y2), axis=1)
    H_out = np.concatenate([H_bot, H_mid, H_top], axis=1)
    H_y = np.add.reduceat(H_local[n_x_fluid-n_x2:, n_y2:n_y_fluid-n_y2], np.arange(0, n_y_fluid-2*n_y2-1, n_y2), axis=1)
    H_free_edge = np.concatenate([H_local[n_x_fluid-n_x2:, 0:n_y2]
                              , H_y, H_local[n_x_fluid-n_x2:, n_y_fluid-n_y2:]], axis=1)
    H_coarse = np.concatenate([H_out, H_free_edge])
    return H_coarse

def get_a_matrix_uniform_mid(xp, yp, x, y, lam):
    n_x_fluid = len(xp)
    n_y_fluid = len(yp)
    #yp2 = yp[:int(n_y_fluid/2)]
    #yp2 = yp#[:int(n_y_fluid/2)]
    yp2 = yp[:int(n_y_fluid/2)]
    scheme = quadpy.c1.gauss_legendre(8)
    w = scheme.weights
    theta_lin = (scheme.points)
    
    x0 = xp[0]
    x1_ = x[:-1] - x0
    x2_ = x[1:] - x0
    y1_ = (y[:-1, np.newaxis] - yp2).T.flatten()
    y2_ = (y[1:, np.newaxis] - yp2).T.flatten()

    y2_grid, x2_grid = np.meshgrid(y2_, x2_)
    y1_grid, x1_grid = np.meshgrid(y1_, x1_)
    del x1_, x2_, y1_, y2_
    
    a_vec = np.zeros(np.shape(x1_grid), dtype=complex)
    
    ### 'Solving top-right panel'
    ind_r = np.where((y1_grid > 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = y1_grid[ind_r]
    y2 = y2_grid[ind_r]
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = np.expand_dims(x1[ind_thin], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_thin], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_thin], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_thin], axis=1) * np.ones(len(theta_lin))
    #print('Integration now')
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_thin = np.sum(val, axis=1)

    x1_ = np.expand_dims(x1[ind_tall], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_tall], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_tall], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_tall], axis=1) * np.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34


    #print('Solving bottom-right panel')
    # Bottom right region with y<0 and x>0
    ind_r = np.where((y2_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = -y2_grid[ind_r]
    y2 = -y1_grid[ind_r]

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T
    x1_ = np.expand_dims(x1[ind_thin], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_thin], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_thin], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_thin], axis=1) * np.ones(len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)
    val_thin = np.sum(val, axis=1)

    x1_ = np.expand_dims(x1[ind_tall], axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2[ind_tall], axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1[ind_tall], axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2[ind_tall], axis=1) * np.ones(len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    del val_vec, val_thin, val_tall, val
    del  theta_12, theta_13, theta_23, theta_24, theta_32, theta_34
    del  weights_12, weights_13,  weights_23, weights_24, weights_32, weights_34

    # Top panels
    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y1_grid > 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]

    x1_ = np.expand_dims(x1, axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2, axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1, axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2, axis=1) * np.ones(len(theta_lin))

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    diff_ = np.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_31 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta1-np.pi/2)
    theta_10 = (theta1 + np.pi/2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_10 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(np.pi/2-theta2)
    theta_02 = (np.pi/2 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_02 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_24 = w[:, np.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    # Bottom panels
    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y2_grid[ind_r]#[0]
    y2 = -y1_grid[ind_r]#[0]

    x1_ = np.expand_dims(x1, axis=1) * np.ones(len(theta_lin))
    x2_ = np.expand_dims(x2, axis=1) * np.ones(len(theta_lin))
    y1_ = np.expand_dims(y1, axis=1) * np.ones(len(theta_lin))
    y2_ = np.expand_dims(y2, axis=1) * np.ones(len(theta_lin))

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    diff_ = np.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_31 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta1-np.pi/2)
    theta_10 = (theta1 + np.pi/2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_10 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(np.pi/2-theta2)
    theta_02 = (np.pi/2 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_02 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_24 = w[:, np.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    del theta_31, theta_10, theta_02, theta_24
    del weights_31, weights_10, weights_02, weights_24
    del val_vec, val

    # Right-side panels

    ind_r = np.where((y2_grid > 0) & (y1_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    x1_ = np.tile(np.array([x1]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2]).T, len(theta_lin))


    diff_ = np.abs(theta2-theta1)
    theta_12 = (theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_12 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(0-theta2)
    theta_20 = (0 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_20 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta4-0)
    theta_04 = (theta4 + 0)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_04 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta3-theta4)
    theta_43 = (theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_43 = w[:, np.newaxis]*diff_/2

    val = (side_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + side_b(theta_20.T, x1_, x2_, lam)*weights_20.T
        + side_b(theta_04.T, x1_, x2_, lam)*weights_04.T
        + side_a(theta_43.T, x1_, y1_, lam)*weights_43.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    del val_vec, val, theta_12, theta_20, theta_04, theta_43
    del weights_12, weights_20, weights_04, weights_43
    # Singularity
    #print('Solving singularity')


    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid > 0) & (y1_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    r_min = np.min(np.abs([x1, x2, y1, y2]), axis=0)
    xx = np.array([x1, x1, x2, x2])
    yy = np.array([y1, y2, y1, y2])
    x_ = np.tile(np.array([xx]).T, len(theta_lin))
    y_ = np.tile(np.array([yy]).T, len(theta_lin))
    theta_c = np.arctan(yy/xx)
    diff_ = np.abs(np.pi/2-theta_c)
    theta_a = (np.pi/2 + theta_c[:, :, np.newaxis])/2 + diff_[:, :, np.newaxis]/2*theta_lin
    weights_a = w*diff_[:, :, np.newaxis]/2
    weights_a = weights_a.transpose(1, 0, 2)
    theta_a = theta_a.transpose(1, 0, 2)

    diff_ = np.abs(0-theta_c)
    theta_b = (0 + theta_c[:, :, np.newaxis])/2 + diff_[:, :, np.newaxis]/2*theta_lin
    weights_b = w*diff_[:, :, np.newaxis]/2
    weights_b = weights_b.transpose(1, 0, 2)
    theta_b = theta_b.transpose(1, 0, 2)

    val_circle = ((-4 * np.pi * lam * r_min - 4 * np.pi) 
            * np.exp(-lam * r_min) / lam ** 2 / r_min + 4 * np.pi / lam ** 2 / r_min) / 8 / np.pi
    val_sing = (np.sum(np.sum(sing_a(theta_a, y_, r_min[:, np.newaxis, np.newaxis], lam)*weights_a
                            + sing_b(theta_b, x_, r_min[:, np.newaxis, np.newaxis], lam)*weights_b, axis=1), axis=1))
    a_vec[ind_r] = val_sing + val_circle

    del theta_a, theta_b, weights_a, weights_b, val_sing, val_circle

    del y2_grid, x2_grid, y1_grid, x1_grid#, y0_grid, x0_grid
    a_out = a_vec.reshape((n_x_fluid, int(n_y_fluid/2), n_y_fluid))
    a2_flipped = np.flip(a_out, axis=(1, 2))
    a_out = np.concatenate([a_out, a2_flipped], axis=1)
    a2 = np.concatenate([(a_out[1::, :, :])[::-1, :, :], a_out])

    #a_out = a_vec.reshape((n_x_fluid, n_y_fluid, n_y_fluid))
    #a2 = np.concatenate([(a_out[1::, :, :])[::-1, :, :], a_out])
    A_sbt = np.zeros(([len(yp) * len(xp), len(yp) * len(xp)]), dtype=np.complex64)
    nn = 0#len(yp)
    for ix in range(len(xp)):
        for jy in range(int(len(yp))):
            STK_out = a2[len(xp)-1-ix:len(xp)-1-ix+len(xp), :, jy]
            A_sbt[:, nn] = STK_out.flatten()
            nn+=1
    #A_sbt = A_sbt.T
    #a_plot = np.real(
    return A_sbt#, a_vec

def local_matrix(xp, yp, x, y, lam):
    x1_ = (x[:-1, np.newaxis] - xp).T.flatten()
    x2_ = (x[1:, np.newaxis] - xp).T.flatten()
    y1_ = (y[:-1, np.newaxis] - yp).T.flatten()
    y2_ = (y[1:, np.newaxis] - yp).T.flatten()


    y2_grid, x2_grid = np.meshgrid(y2_, x2_)
    y1_grid, x1_grid = np.meshgrid(y1_, x1_)


    #y0_grid, x0_grid = np.meshgrid(np.repeat(yp, len(yp)).flatten(),
    #                            np.repeat(xp, len(xp)).T.flatten())

    a_vec = np.zeros(np.shape(x1_grid), dtype=complex)

    scheme = quadpy.c1.gauss_legendre(8)
    w = scheme.weights
    theta_lin = (scheme.points)
    ind_r = np.where((y1_grid > 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = y1_grid[ind_r]
    y2 = y2_grid[ind_r]
    #y0_ = y0_grid[ind_r]
    #x0_ = x0_grid[ind_r]
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = np.tile(np.array([x1[ind_thin]]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2[ind_thin]]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1[ind_thin]]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2[ind_thin]]).T, len(theta_lin))
    #print('Integration now')
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_thin = np.sum(val, axis=1)

    x1_ = np.tile(np.array([x1[ind_tall]]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2[ind_tall]]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1[ind_tall]]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2[ind_tall]]).T, len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    #print('Solving top-left panel')
    # Top left region with y>0 and x<0
    ind_r = np.where((y1_grid > 0) & (x2_grid < 0))
    x1 = -x2_grid[ind_r]
    x2 = -x1_grid[ind_r]
    y1 = y1_grid[ind_r]
    y2 = y2_grid[ind_r]
    #y0_ = y0_grid[ind_r]
    #x0_ = x0_grid[ind_r]
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = np.tile(np.array([x1[ind_thin]]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2[ind_thin]]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1[ind_thin]]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2[ind_thin]]).T, len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_thin = np.sum(val, axis=1)

    x1_ = np.tile(np.array([x1[ind_tall]]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2[ind_tall]]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1[ind_tall]]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2[ind_tall]]).T, len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    #print('Solving bottom-left panel')
    # Bottom left region with y<0 and x<0
    ind_r = np.where((y2_grid < 0) & (x2_grid < 0))
    x1 = -x2_grid[ind_r]
    x2 = -x1_grid[ind_r]
    y1 = -y2_grid[ind_r]
    y2 = -y1_grid[ind_r]
    #y0_ = y0_grid[ind_r]
    #x0_ = x0_grid[ind_r]
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = np.tile(np.array([x1[ind_thin]]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2[ind_thin]]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1[ind_thin]]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2[ind_thin]]).T, len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_thin = np.sum(val, axis=1)

    x1_ = np.tile(np.array([x1[ind_tall]]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2[ind_tall]]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1[ind_tall]]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2[ind_tall]]).T, len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    #print('Solving bottom-right panel')
    # Bottom right region with y<0 and x>0
    ind_r = np.where((y2_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]
    x2 = x2_grid[ind_r]
    y1 = -y2_grid[ind_r]
    y2 = -y1_grid[ind_r]

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    ind_thin = np.where(theta2<theta3)
    ind_tall = np.where(theta3<=theta2)

    diff_ = np.abs(theta2-theta1)
    theta_12 = ((theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_12 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta2)
    theta_23 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_23 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta4)
    theta_34 = ((theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_tall].T
    weights_34 = (w[:, np.newaxis]*diff_/2).T[ ind_tall].T

    diff_ = np.abs(theta3-theta1)
    theta_13 = ((theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_13 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta3-theta2)
    theta_32 = ((theta3 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_32 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    diff_ = np.abs(theta2-theta4)
    theta_24 = ((theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]).T[ ind_thin].T
    weights_24 = (w[:, np.newaxis]*diff_/2).T[ ind_thin].T

    x1_ = np.tile(np.array([x1[ind_thin]]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2[ind_thin]]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1[ind_thin]]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2[ind_thin]]).T, len(theta_lin))
    val = (thin_a(theta_13.T, x1_, y2_, lam)*weights_13.T 
        + thin_b(theta_32.T, y1_, y2_, lam)*weights_32.T
        + thin_c(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_thin = np.sum(val, axis=1)

    x1_ = np.tile(np.array([x1[ind_tall]]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2[ind_tall]]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1[ind_tall]]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2[ind_tall]]).T, len(theta_lin))
    val = (tall_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + tall_b(theta_23.T, x1_, x2_, lam)*weights_23.T
        + tall_c(theta_34.T, x2_, y1_, lam)*weights_34.T)# + 
    val_tall = np.sum(val, axis=1)

    val_vec =  np.zeros(len(x1), dtype=complex)
    val_vec[ind_thin] = val_thin
    val_vec[ind_tall] = val_tall
    a_vec[ind_r] = val_vec

    # Singularity
    #print('Solving singularity')
    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid > 0) & (y1_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    r_min = np.min(np.abs([x1, x2, y1, y2]), axis=0)
    xx = np.array([x1, x1, x2, x2])
    yy = np.array([y1, y2, y1, y2])
    x_ = np.tile(np.array([xx]).T, len(theta_lin))
    y_ = np.tile(np.array([yy]).T, len(theta_lin))
    theta_c = np.arctan(yy/xx)
    diff_ = np.abs(np.pi/2-theta_c)
    theta_a = (np.pi/2 + theta_c[:, :, np.newaxis])/2 + diff_[:, :, np.newaxis]/2*theta_lin
    weights_a = w*diff_[:, :, np.newaxis]/2
    weights_a = weights_a.transpose(1, 0, 2)
    theta_a = theta_a.transpose(1, 0, 2)

    diff_ = np.abs(0-theta_c)
    theta_b = (0 + theta_c[:, :, np.newaxis])/2 + diff_[:, :, np.newaxis]/2*theta_lin
    weights_b = w*diff_[:, :, np.newaxis]/2
    weights_b = weights_b.transpose(1, 0, 2)
    theta_b = theta_b.transpose(1, 0, 2)

    val_circle = ((-4 * np.pi * lam * r_min - 4 * np.pi) 
            * np.exp(-lam * r_min) / lam ** 2 / r_min + 4 * np.pi / lam ** 2 / r_min) / 8 / np.pi
    val_sing = (np.sum(np.sum(sing_a(theta_a, y_, r_min[:, np.newaxis, np.newaxis], lam)*weights_a
                            + sing_b(theta_b, x_, r_min[:, np.newaxis, np.newaxis], lam)*weights_b, axis=1), axis=1))
    a_vec[ind_r] = val_sing + val_circle

    # Top panels
    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y1_grid > 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]

    x1_ = np.tile(np.array([x1]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2]).T, len(theta_lin))

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    diff_ = np.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_31 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta1-np.pi/2)
    theta_10 = (theta1 + np.pi/2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_10 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(np.pi/2-theta2)
    theta_02 = (np.pi/2 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_02 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_24 = w[:, np.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    # Bottom panels
    ind_r = np.where((x2_grid > 0) & (x1_grid < 0) & (y2_grid < 0))
    x1 = -x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y2_grid[ind_r]#[0]
    y2 = -y1_grid[ind_r]#[0]

    x1_ = np.tile(np.array([x1]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2]).T, len(theta_lin))

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    diff_ = np.abs(theta3-theta1)
    theta_31 = (theta1 + theta3)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_31 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta1-np.pi/2)
    theta_10 = (theta1 + np.pi/2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_10 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(np.pi/2-theta2)
    theta_02 = (np.pi/2 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_02 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta2-theta4)
    theta_24 = (theta4 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_24 = w[:, np.newaxis]*diff_/2

    val = (top_a(theta_31.T, x1_, y1_, lam)*weights_31.T 
        + top_b(theta_10.T, y1_, y2_, lam)*weights_10.T
        + top_b(theta_02.T, y1_, y2_, lam)*weights_02.T
        + top_a(theta_24.T, x2_, y1_, lam)*weights_24.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    # Right-side panels

    ind_r = np.where((y2_grid > 0) & (y1_grid < 0) & (x1_grid > 0))
    x1 = x1_grid[ind_r]#[0]
    x2 = x2_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]
    
    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    x1_ = np.tile(np.array([x1]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2]).T, len(theta_lin))


    diff_ = np.abs(theta2-theta1)
    theta_12 = (theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_12 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(0-theta2)
    theta_20 = (0 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_20 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta4-0)
    theta_04 = (theta4 + 0)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_04 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta3-theta4)
    theta_43 = (theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_43 = w[:, np.newaxis]*diff_/2

    val = (side_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + side_b(theta_20.T, x1_, x2_, lam)*weights_20.T
        + side_b(theta_04.T, x1_, x2_, lam)*weights_04.T
        + side_a(theta_43.T, x1_, y1_, lam)*weights_43.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    # Left-side panels

    ind_r = np.where((y2_grid > 0) & (y1_grid < 0) & (x2_grid < 0) & (x1_grid < 0))
    x1 = -x2_grid[ind_r]#[0]
    x2 = -x1_grid[ind_r]#[0]
    y1 = -y1_grid[ind_r]#[0]
    y2 = y2_grid[ind_r]#[0]

    theta1 = np.arctan(y2/x1)
    theta2 = np.arctan(y2/x2)
    theta3 = np.arctan(y1/x1)
    theta4 = np.arctan(y1/x2)

    x1_ = np.tile(np.array([x1]).T, len(theta_lin))
    x2_ = np.tile(np.array([x2]).T, len(theta_lin))
    y1_ = np.tile(np.array([y1]).T, len(theta_lin))
    y2_ = np.tile(np.array([y2]).T, len(theta_lin))


    diff_ = np.abs(theta2-theta1)
    theta_12 = (theta1 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_12 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(0-theta2)
    theta_20 = (0 + theta2)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_20 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta4-0)
    theta_04 = (theta4 + 0)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_04 = w[:, np.newaxis]*diff_/2

    diff_ = np.abs(theta3-theta4)
    theta_43 = (theta3 + theta4)/2 + diff_/2*theta_lin[:, np.newaxis]
    weights_43 = w[:, np.newaxis]*diff_/2

    val = (side_a(theta_12.T, x1_, y2_, lam)*weights_12.T 
        + side_b(theta_20.T, x1_, x2_, lam)*weights_20.T
        + side_b(theta_04.T, x1_, x2_, lam)*weights_04.T
        + side_a(theta_43.T, x1_, y1_, lam)*weights_43.T)# + 
    val_vec = np.sum(val, axis=1)
    a_vec[ind_r] = val_vec

    #a_out = a_vec.reshape((n_x_fluid, n_x_fluid, n_y_fluid, n_y_fluid))
    #a2 = a_out.swapaxes(1, 3)
    #a2_out = (a2.swapaxes(2,3)).reshape(n_x_fluid*n_y_fluid, n_x_fluid*n_y_fluid)
    a_out = a_vec.reshape((n_x_fluid, n_y_fluid, n_x_fluid, n_y_fluid))
    a_out = a_out.transpose(0, 2, 1, 3).reshape(n_x_fluid * n_y_fluid, n_x_fluid * n_y_fluid)

    return a_out
