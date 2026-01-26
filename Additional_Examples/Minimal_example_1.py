# 1-Import modules from the VFSI library
import sys
sys.path.insert(0, 'home/loch/Code/vfi-mems')  # Adjust the path to your VFSI library
sys.path.insert(0, '..')
import Elastic.plate_fem as pt
import Fluid.F3D_1D as F3D
import numpy as np
from scipy.sparse.linalg import spsolve
import tqdm
# 2-Basic objects, shared by all classes for geometry, material and fluid
length = 800e-6  # length of beam in m
width = 100e-6  # width in m
height = 5e-6  # thickness of the beam in m
n_x_fluid = 24  # Number of elements in length l_c (x-direction)
n_y_fluid = 48  # Number of elements in width w_c (y-direction)
# Material of plate -> Silicon
Youngs_modulus = 169E9  # Young's Modulus in Pa
Density = 2.33E3  # Density in kg/m^3
Poisson_Coefficient = 0.3  # Poisson Coefficient, dimensionless
# Fluid -> Water
Dynamic_viscosity = 890e-6  # Fluid dynamic viscosity in Pa.s
Density = 997  # Fluid density in kg/m^3

geometry = pt.Geometry(length, width, height)
mat = pt.Material(Youngs_modulus, Density, Poisson_Coefficient)
fluid = F3D.Fluid(Dynamic_viscosity, Density)
# 3-Determine the elasticity and mass matrices
plate = pt.Kirchhoff()
plate.geometry = geometry
plate.mat = mat
K, M = plate.k_and_m_matrices()
F = np.array(plate.get_linear_form(1))
# 4-Setup fluid interaction problem
vfi = F3D.F3D()
vfi.fluid = fluid
vfi.fem = plate
vfi.geometry = geometry
vfi.setup_quadrature(n_x_fluid, n_y_fluid, True, True)
# 5-Solve the VFSI problem in the frequency domain
frequency = np.linspace(1e3, 500e3, 1000)
for ff in tqdm.tqdm(range(len(frequency))):
    omega = 2*np.pi*frequency[ff]
    P = vfi.get_p_matrix(omega)
    phi = spsolve(K + 1j*omega*P - omega**2*M, F)