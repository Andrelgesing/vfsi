#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '..')
from Elastic import plate_fem as pt
import numpy as np
import time

# Specify geometry
l_c = 600e-6  # length of beam in m
h_c = 5e-6  # thickness of the beam in m
w_c = 100e-6  # width in m


# Specify material of plate -> Silicon
e_c = 169E9  # Young's Modulus in Pa
rho_c = 2.33E3  # Density in kg/m^3
nu_c = 0.3  # Poisson Coefficient, dimensionless

n_eig = 5
# Basic objects, shared by all classes
geometry = pt.Geometry(l_c, w_c, h_c)
mat = pt.Material(youngs_modulus=e_c, density=rho_c, poisson=nu_c)

# Anisotropic silicon
#c_values = np.array([194.5, 194.5, 35.7, 50.9, 50.9])*1e9
#mat = pt.Material(c_values=c_values, density=rho_c)

# Mesh specification -> To solve Kirchhoff-Love plate equation
n_x = 30  # Number of elements in length l_c (x-direction)
n_y = int(n_x*w_c/l_c)  # Number of elements in width w_c (y-direction)

print('Solving IP-Method.')
t = time.time()
plate = pt.Kirchhoff()

plate.geometry= geometry
plate.material = mat
plate.meshing(n_x, n_y, 'crossed')
plate.preliminary_setup()
#plate.setup_eigenvalues_problem()
plate.k_and_m_matrices()
plate.solve_n_eigenvalues(n_eig, plate.k_matrix, plate.m_matrix)
plate.plot_eigenmodes(n_eig)
tt = time.time() - t
print('\n         Solved in %d minutes and %d seconds.' % (np.floor(tt/60), tt % 60))

