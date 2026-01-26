import sys
sys.path.insert(0, '..')
import quadpy
import scipy.special as sp
import matplotlib.colors as colors
import numpy as np
import time
import matplotlib.pyplot as plt
#import scipy.special as sp
from scipy.sparse.linalg import spsolve
import scipy.sparse as sps

#sys.stdout
import tracemalloc

from Elastic import plate_fem as pt
from Fluid.Fluid import Fluid
from Fluid import F3D
from Fluid import F3D_1D

f = 1e5 # Frequency [Hz]
omega = 2*np.pi*f

# Specify geometry
l_c = 500e-6
h_c = 5e-6
w_c = 200e-6

# Specify material of beam -> Silicon
e_c = 169E9  # Young's Modulus in Pa
rho_c = 2.33E3  # Density in kg/m^3
nu_c = 0.064  # Poisson Coefficient, dimensionless
 
####  Water
mu_f = 890e-6  # Fluid dynamic viscosity in Pa.s
rho_f = 997  # Fluid density in kg/m^3
nu_f = mu_f/rho_f


lam = np.sqrt(-1j*omega*l_c**2/nu_f) 

tol_stk = 1e-4

# Basic objects, shared by all classes
geometry = pt.Geometry(l_c, w_c, h_c)
mat = pt.Material(e_c, rho_c, nu_c)
fluid = Fluid(mu_f, rho_f)

# Define grids
n_x_fluid=32
n_y_fluid=96

# Basic objects, shared by all classes
# Mesh specification -> To solve Kirchhoff-Love plate equation
n_x = 32  # Number of elements in length l_c (x-direction)
n_y = int(n_x*w_c/l_c) # Number of elements in width w_c (y-direction)
geometry = pt.Geometry(l_c, w_c, h_c)
mat = pt.Material(e_c, rho_c, nu_c)
fluid = Fluid(mu_f, rho_f)
plate = pt.Kirchhoff()
plate.geometry = geometry
plate.mat = mat
plate.meshing(n_x, n_y, 'crossed')
plate.preliminary_setup()
plate.k_and_m_matrices()

t_init = time.time()
vfi = F3D.F3D()
vfi.fluid = fluid
vfi.fem = plate
vfi.geometry = geometry
vfi.setup_quadrature(n_x_fluid, n_y_fluid, x_uniform=True)
vfi.tol_stk=tol_stk

## Non-uniform
t_init = time.time()
tracemalloc.start()
a_2d = vfi.get_h_matrix(omega)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_2d_vfi = peak
print(f"Current memory usage: {current / 1e6:.2f} MB; Peak was {peak / 1e6:.2f} MB")
tt = (time.time() - t_init)
print('2D Full time: %d minutes and %d seconds' %(np.floor(tt/60), tt%60))
tt_2d_vfi = tt

## Uniform
vfi.setup_quadrature(n_x_fluid, n_y_fluid, x_uniform=True)

x = vfi.x
xp = vfi.xp
y = vfi.y
yp = vfi.yp

y_plot, x_plot = np.meshgrid(y, x)


from Fluid.h_matrix import *
t_init = time.time()
tracemalloc.start()
a_out_1d, a_1d = get_a_matrix(xp, yp, x, y, lam)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_1d_full = peak
print(f"Current memory usage: {current / 1e6:.2f} MB; Peak was {peak / 1e6:.2f} MB")
tt = (time.time() - t_init)
print('1D Full time: %d minutes and %d seconds' %(np.floor(tt/60), tt%60))
tt_1d_full = tt

t_init = time.time()
tracemalloc.start()
a_out_1d_half, a_1d_half = get_a_matrix_half(xp, yp, x, y, lam)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_1d_half = peak
print(f"Current memory usage: {current / 1e6:.2f} MB; Peak was {peak / 1e6:.2f} MB")
tt = (time.time() - t_init)
print('1D Half: %d minutes and %d seconds' %(np.floor(tt/60), tt%60))
tt_1d_half = tt


t_init = time.time()
tracemalloc.start()
a_2d_uni = vfi.get_h_matrix(omega)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_2d_uni = peak
print(f"Current memory usage: {current / 1e6:.2f} MB; Peak was {peak / 1e6:.2f} MB")
tt = (time.time() - t_init)
print('2D Uniform: %d minutes and %d seconds' %(np.floor(tt/60), tt%60))
tt_2d_uni = tt


vfi.setup_quadrature(n_x_fluid, n_y_fluid, x_uniform=True)
t_init = time.time()
tracemalloc.start()
a_out_1d_uni, _ = get_a_matrix_uniform(vfi.xp, vfi.yp, vfi.x, vfi.y, lam)
tt = (time.time() - t_init)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_1d_uni = peak
print(f"Current memory usage: {current / 1e6:.2f} MB; Peak was {peak / 1e6:.2f} MB")
print('1D Uniform: %d minutes and %d seconds' %(np.floor(tt/60), tt%60))
tt_1d_uni = tt


from Fluid.h_matrix_app import *
t_init = time.time()
tracemalloc.start()
a_out_1d_mid_app, _ = get_a_matrix_uniform_mid(vfi.xp, vfi.yp, vfi.x, vfi.y, lam)
tt = (time.time() - t_init)
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1e6:.2f} MB; Peak was {peak / 1e6:.2f} MB")

tracemalloc.stop()
print('1D With approximation: %d minutes and %d seconds' %(np.floor(tt/60), tt%60))
tt_1d_mid_app_uni = tt
peak_1d_mid_app_uni = peak

from Fluid.h_matrix_app import *
t_init = time.time()
tracemalloc.start()
a_out_1d_uni_app, _ = get_a_matrix_uniform(vfi.xp, vfi.yp, vfi.x, vfi.y, lam)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_1d_uni_app = peak
print(f"Current memory usage: {current / 1e6:.2f} MB; Peak was {peak / 1e6:.2f} MB")
tt = (time.time() - t_init)
print('1D Uniform: %d minutes and %d seconds' %(np.floor(tt/60), tt%60))
tt_1d_uni_app = tt


v1 = np.linspace(0, 1, n_x_fluid)
v2 = np.tile(v1, (n_y_fluid, 1))
v = v2.T.flatten()#v = np.ones(n_x_fluid*n_y_fluid)
t_init = time.time()
tracemalloc.start()
p_2d_vfi = np.linalg.solve(a_2d, v)
tt = (time.time() - t_init)
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1e6:.2f} MB; Peak was {peak / 1e6:.2f} MB")

tracemalloc.stop()
print('Solution time: %d minutes and %d seconds' %(np.floor(tt/60), tt%60))
tt_solve = tt
peak_solve = peak

a_2d_sparse = sps.csr_matrix(a_2d) 

t_init = time.time()
tracemalloc.start()
p_2d_vfi_ = spsolve(a_2d_sparse, v)
tt = (time.time() - t_init)
current, peak = tracemalloc.get_traced_memory()
print(f"Sp Solve memory usage: {current / 1e6:.2f} MB; Peak was {peak / 1e6:.2f} MB")

tracemalloc.stop()
print('Spsolve Solution time: %d minutes and %d seconds' %(np.floor(tt/60), tt%60))
tt_spsolve = tt
peak_spsolve = peak


p_2d_uni = np.linalg.solve(a_2d_uni, v)
p_2d_uni = p_2d_uni.T.reshape(n_x_fluid, n_y_fluid)

p_1d_uni = np.linalg.solve(a_out_1d_uni, v)
p_1d_uni = p_1d_uni.T.reshape(n_x_fluid, n_y_fluid)

p_1d_uni_app = np.linalg.solve(a_out_1d_uni_app, v)
p_1d_uni_app = p_1d_uni_app.T.reshape(n_x_fluid, n_y_fluid)


p_1d_uni_mid_app = np.linalg.solve(a_out_1d_mid_app, v)
p_1d_uni_mid_app = p_1d_uni_mid_app.T.reshape(n_x_fluid, n_y_fluid)

p_1d_implemented = np.linalg.solve(a_out_1d, v)
p_1d_implemented_half = np.linalg.solve(a_out_1d_half, v)

p_2d_vfi = p_2d_vfi.T.reshape(n_x_fluid, n_y_fluid)

p_1d = p_1d_implemented.T.reshape(n_x_fluid, n_y_fluid)
p_1d_half = p_1d_implemented_half.T.reshape(n_x_fluid, n_y_fluid)

p_1d = p_1d_implemented.T.reshape(n_x_fluid, n_y_fluid)
p_1d_half = p_1d_implemented_half.T.reshape(n_x_fluid, n_y_fluid)

plt.figure(figsize=(10, 10))
plt.subplot(2, 3, 1)
plt.title('Velocity')
p = plt.pcolormesh(x_plot, y_plot, v2.T)
plt.colorbar(p)

plt.subplot(2, 3, 2)
plt.title('2D - Quadpy')
p = plt.pcolormesh(x_plot, y_plot, np.abs(p_2d_vfi))
plt.colorbar(p)

plt.subplot(2, 3, 3)
plt.title('1D Uniform')
p = plt.pcolormesh(x_plot, y_plot, np.abs(p_1d_uni))
plt.colorbar(p)

plt.subplot(2, 3, 4)
plt.title('1D Uniform approximation')
p = plt.pcolormesh(x_plot, y_plot, np.abs(p_1d_uni_app))
plt.colorbar(p)

plt.subplot(2, 3, 5)
plt.title('VFI - 1D')
p = plt.pcolormesh(x_plot, y_plot, np.abs(p_1d))
plt.colorbar(p)

plt.subplot(2, 3, 6)
plt.title('1D implemented Half')
p = plt.pcolormesh(x_plot, y_plot, np.abs(p_1d_uni_mid_app))
plt.colorbar(p)
plt.tight_layout()
plt.savefig('Figures/Pressure_maps.png')

ix = n_x_fluid-4#10
iy = n_y_fluid-1#2


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
#plt.plot(yp, np.real(p_2d_vfi[ix, :]), label='2D - Full')
plt.plot(yp, np.real(p_2d_uni[ix, :]), label='2D - Uni')
plt.plot(yp, np.real(p_1d[ix, :]), label='VFI - 1d')
plt.plot(yp, np.real(p_1d_uni[ix, :]), label='VFI - 1d Uniform')
plt.plot(yp, np.real(p_1d_uni_app[ix, :]), label='VFI - 1d Uniform App')
plt.plot(yp, np.real(p_1d_uni_mid_app[ix, :]), label='VFI - 1d Uniform Mid App')
plt.legend(fontsize=(10))

plt.subplot(1, 2, 2)
#plt.plot(yp, np.imag(p_2d_vfi[ix, :]), label='VFI - Quadpy')
plt.plot(yp, np.imag(p_1d[ix, :]), label='VFI - 1d')
plt.plot(yp, np.imag(p_1d_uni[ix, :]), label='VFI - 1d Uniform')
plt.plot(yp, np.imag(p_1d_uni_app[ix, :]), label='VFI - 1d Uniform App')
plt.plot(yp, np.imag(p_1d_uni_mid_app[ix, :]), label='VFI - 1d Uniform Mid App')
plt.legend(fontsize=(10))
plt.savefig('Figures/Pressure_comparison.png')


fig = plt.figure(figsize=(10, 4))
#ax = fig.add_axes([0,0,1,1])
ax = plt.subplot(1, 1, 1)
types = ['2D - Full', '2D - Uniform x', '1D - Full','1D Half', '1D - Uniform', 
         '1D - Uniform App' , '1D - Uniform Mid App', 'Linalg: Solution time', 'SP Solve']
times = [tt_2d_vfi, tt_2d_uni,  tt_1d_full,  tt_1d_half, tt_1d_uni,  tt_1d_uni_app,
         tt_1d_mid_app_uni, tt_solve, tt_spsolve]
ax.bar(types,times)
plt.ylabel('Time [s/iteration]', fontsize=14)
plt.xticks(fontsize=14, rotation=45)
plt.tight_layout()
plt.yscale('log')
plt.yticks([1, 10, 100])
plt.savefig('Figures/Computation_time.png', dpi=400)
plt.show()

fig = plt.figure(figsize=(10, 4))
#ax = fig.add_axes([0,0,1,1])
ax = plt.subplot(1, 1, 1)
types = ['2D - Full',   '2D - Uniform x', '1D - Full','1D Half', '1D - Uniform', 
         '1D - Uniform App' , '1D - Uniform Mid App', 'Linalg: Solution', 'SPSolve: Solution']
mem = [peak_2d_vfi, peak_2d_uni,  peak_1d_full,  peak_1d_half, peak_1d_uni,  peak_1d_uni_app,
         peak_1d_mid_app_uni, peak_solve, peak_spsolve]
mem = np.array(mem)/1e6
ax.bar(types,mem)
plt.ylabel('Memory [MB]', fontsize=14)
plt.xticks(fontsize=14, rotation=45)
plt.tight_layout()
plt.yscale('log')
plt.yticks([1, 10, 100])
plt.savefig('Figures/Memory_usage.png', dpi=400)
plt.show()


data = {
    "types": types,
    "memory_MB": mem,
    "time_sec": times,
    "p_2d_vfi": p_2d_vfi,
    "p_1d_uni":p_1d_uni,
    "p_1d": p_1d,
    "p_1d_uni_app":p_1d_uni_app,
    "p_1d_uni_mid_app": p_1d_uni_mid_app,
    "x_plot": x_plot,
    "y_plot": y_plot
}

np.save("Data/results_mem_time_cpu.npy", data)
