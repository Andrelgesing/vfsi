from numba import jit
import numpy as np
import scipy.special as spc

#@jit(nopython=True, parallel=True, cache=True)
#def expi_approx(x):
#    gamma = 0.57721566490153286060  # Euler-Mascheroni constant
#    return -gamma - np.log(x) + x - (x**2)/4  # Approximate formula

#@jit(nopython=True, parallel=True, cache=True)
#def expi_app(x):
#    val = spc.expi(x)
#    #gamma = 0.57721566490153286060  # Euler-Mascheroni constant
#    #val = -gamma - np.log(x) + x - (x**2)/4 
#    return  val 

#import numpy as np
#@jit(parallel=True, cache=True)
def expi_app(z_array):
    """
    Approximate exponential integral Ei(z) for complex z using Numba-compatible code.
    """
    gamma = 0.5772156649015328606
    N = z_array.shape[0]
    out = np.empty_like(z_array, dtype=np.complex128)

    for i in prange(N):
        z = z_array[i]
        if np.abs(z) < 4.0:
            # Power series
            term = 1.0 + 0j
            result = np.log(z) + gamma
            for k in range(1, 40):
                term *= z / k
                result += term / k
            out[i] = result
        else:
            # Asymptotic expansion
            inv_z = 1.0 / z
            term = inv_z
            result = term
            for k in range(1, 10):
                term *= -k / z
                result += term
            out[i] = np.exp(z) * result

    return out

@jit(nopython=True, parallel=True, cache=True)
def thin_a(theta, x1, y2, lam_):
    f_int = ((x1*np.sqrt(1 - np.cos(theta)**2) - y2*np.sqrt(1 - np.sin(theta)**2))/(4*np.pi*lam_**2*x1*y2)
             -(expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
             -expi_app(lam_ * y2 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             -(-np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2)) +
              np.exp(-lam_*y2/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             + (-x1*np.sqrt(1-np.cos(theta)**2)*np.exp(-lam_*y2/np.sqrt(1 - np.cos(theta)**2))
         +y2*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2))
         + lam_*x1*y2*expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         - lam_*x1*y2*expi_app(lam_ * y2 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_**2*x1*y2))
    return f_int

@jit(nopython=True, parallel=True, cache=True)
def thin_b(theta, y1, y2, lam_):
    f_int = ((y1-y2)*np.sqrt(1 - np.cos(theta)**2)/(4*np.pi*lam_**2*y1*y2)
             -(expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2))
             -expi_app(lam_ * y2 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             -(-np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2)) +
              np.exp(-lam_*y2/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             + (-y1*np.sqrt(1-np.cos(theta)**2)*np.exp(-lam_*y2/np.sqrt(1 - np.cos(theta)**2))
         +y2*np.sqrt(1-np.cos(theta)**2)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2))
         + lam_*y1*y2*expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2))
         - lam_*y1*y2*expi_app(lam_ * y2 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_**2*y1*y2))
    return f_int

@jit(nopython=True, parallel=True, cache=True)
def thin_c(theta, x2, y1, lam_):
    f_int = (-(x2*np.sqrt(1 - np.cos(theta)**2) - y1*np.sqrt(1 - np.sin(theta)**2))/(4*np.pi*lam_**2*x2*y1)
             + (expi_app(lam_ * x2 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         -expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_) 
             +(-np.exp(-lam_*x2/np.sqrt(1 - np.sin(theta)**2)) +
              np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             -(-x2*np.sqrt(1-np.cos(theta)**2)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2))
         +y1*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x2/np.sqrt(1 - np.sin(theta)**2))
         + lam_*x2*y1*expi_app(lam_ * x2 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         - lam_*x2*y1*expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_**2*x2*y1))

    return f_int

@jit(nopython=True, parallel=True, cache=True)
def tall_a(theta, x1, y2, lam_):
    f_int = ((x1*np.sqrt(1 - np.cos(theta)**2) - y2*np.sqrt(1 - np.sin(theta)**2))/(4*np.pi*lam_**2*x1*y2)
              -(-np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2)) +
              np.exp(-lam_*y2/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             -(expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
             -expi_app(lam_ * y2 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             + (-x1*np.sqrt(1-np.cos(theta)**2)*np.exp(-lam_*y2/np.sqrt(1 - np.cos(theta)**2))
         +y2*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2))
         + lam_*x1*y2*expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         - lam_*x1*y2*expi_app(lam_ * y2 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_**2*x1*y2))
    return f_int

@jit(nopython=True, parallel=True, cache=True)
def tall_b(theta, x1, x2, lam_):
    f_int = ((x1-x2)*np.sqrt(1 - np.sin(theta)**2)/(4*np.pi*lam_**2*x1*x2)
             -(-np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2)) +
              np.exp(-lam_*x2/np.sqrt(1 - np.sin(theta)**2)))/(4*np.pi*lam_)
             -(expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
             -expi_app(lam_ * x2 * (-1)/np.sqrt(1 - np.sin(theta)**2)))/(4*np.pi*lam_)
             + (-x1*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x2/np.sqrt(1 - np.sin(theta)**2))
         +x2*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2))
         + lam_*x1*x2*expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         - lam_*x1*x2*expi_app(lam_ * x2 * (-1)/np.sqrt(1 - np.sin(theta)**2)))/(4*np.pi*lam_**2*x1*x2))
    return f_int

@jit(nopython=True, parallel=True, cache=True)
def tall_c(theta, x2, y1, lam_):
    f_int = (-(x2*np.sqrt(1 - np.cos(theta)**2) - y1*np.sqrt(1 - np.sin(theta)**2))/(4*np.pi*lam_**2*x2*y1)
             + (-np.exp(-lam_*x2/np.sqrt(1 - np.sin(theta)**2)) +
              np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             + (expi_app(lam_ * x2 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         -expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             -(-x2*np.sqrt(1-np.cos(theta)**2)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2))
         +y1*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x2/np.sqrt(1 - np.sin(theta)**2))
         + lam_*x2*y1*expi_app(lam_ * x2 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         - lam_*x2*y1*expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_**2*x2*y1))
    return f_int

@jit(nopython=True, parallel=True, cache=True)
def side_a(theta, x1, y2, lam_):
    f_int = ((x1*np.sqrt(1 - np.cos(theta)**2) 
             - y2*np.sqrt(1 - np.sin(theta)**2))/(4*np.pi*lam_**2*x1*y2)
             +(-x1*np.sqrt(1-np.cos(theta)**2)*np.exp(-lam_*y2/np.sqrt(1 - np.cos(theta)**2))
         +y2*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2))
         + lam_*x1*y2*expi_app(lam_ * x1 * np.exp(1j*np.pi)/np.sqrt(1 - np.sin(theta)**2))
         - lam_*x1*y2*expi_app(lam_ * y2 * np.exp(1j*np.pi)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_**2*x1*y2)
         -(expi_app(lam_ * x1 * np.exp(1j*np.pi)/np.sqrt(1 - np.sin(theta)**2))
             -expi_app(lam_ * y2 * np.exp(1j*np.pi)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
            -(-np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2)) +
              np.exp(-lam_*y2/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_))
            
    return f_int

@jit(nopython=True, parallel=True, cache=True)
def side_b(theta, x1, x2, lam_):
    f_int = ((x1-x2)*np.sqrt(1 - np.sin(theta)**2)/
            (4*np.pi*lam_**2*x1*x2)
            + (-x1*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x2/np.sqrt(1 - np.sin(theta)**2))
         +x2*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2))
         + lam_*x1*x2*expi_app(lam_ * x1 * np.exp(1j*np.pi)/np.sqrt(1 - np.sin(theta)**2))
         - lam_*x1*x2*expi_app(lam_ * x2 * np.exp(1j*np.pi)/np.sqrt(1 - np.sin(theta)**2)))/(4*np.pi*lam_**2*x1*x2)
             -(expi_app(lam_ * x1 * np.exp(1j*np.pi)/np.sqrt(1 - np.sin(theta)**2))
             -expi_app(lam_ * x2 * np.exp(1j*np.pi)/np.sqrt(1 - np.sin(theta)**2)))/(4*np.pi*lam_)
            -(-np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2)) +
              np.exp(-lam_*x2/np.sqrt(1 - np.sin(theta)**2)))/(4*np.pi*lam_))
    return f_int


@jit(nopython=True, parallel=True, cache=True)
def side_a_old(theta, x1, y2, lam_):
    f_int = ((x1*np.sqrt(1 - np.cos(theta)**2) 
             - y2*np.sqrt(1 - np.sin(theta)**2))/(4*np.pi*lam_**2*x1*y2)
             +(-x1*np.sqrt(1-np.cos(theta)**2)*np.exp(-lam_*y2/np.sqrt(1 - np.cos(theta)**2))
         +y2*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2))
         + lam_*x1*y2*expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         - lam_*x1*y2*expi_app(lam_ * y2 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_**2*x1*y2)
         -(expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
             -expi_app(lam_ * y2 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
            -(-np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2)) +
              np.exp(-lam_*y2/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_))
            
    return f_int

@jit(nopython=True, parallel=True, cache=True)
def side_b_old(theta, x1, x2, lam_):
    f_int = ((x1-x2)*np.sqrt(1 - np.sin(theta)**2)/
            (4*np.pi*lam_**2*x1*x2)
            + (-x1*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x2/np.sqrt(1 - np.sin(theta)**2))
         +x2*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2))
         + lam_*x1*x2*expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         - lam_*x1*x2*expi_app(lam_ * x2 * (-1)/np.sqrt(1 - np.sin(theta)**2)))/(4*np.pi*lam_**2*x1*x2)
             -(expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
             -expi_app(lam_ * x2 * (-1)/np.sqrt(1 - np.sin(theta)**2)))/(4*np.pi*lam_)
            -(-np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2)) +
              np.exp(-lam_*x2/np.sqrt(1 - np.sin(theta)**2)))/(4*np.pi*lam_))
    return f_int

@jit(nopython=True, parallel=True, cache=True)
def side_c(theta, x1, y1, lam_):
    f_int = ((x1*np.sqrt(1 - np.cos(theta)**2) - y1*np.sqrt(1 - np.sin(theta)**2))/
            (4*np.pi*lam_**2*x1*y1)
            -(-x1*np.sqrt(1-np.cos(theta)**2)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2))
         +y1*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2))
         + lam_*x1*y1*expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         - lam_*x1*y1*expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_**2*x1*y1)
           + (expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         -expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_) 
           + (-np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2)) +
              np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_) )
    return f_int

@jit(nopython=True, parallel=True, cache=True)
def top_a(theta, x1, y1, lam_):
    f_int = ((-x1*np.sqrt(1 - np.cos(theta)**2) 
             + y1*np.sqrt(1 - np.sin(theta)**2))/(4*np.pi*lam_**2*x1*y1)
             + (-np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2)) +
              np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             + (expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
             -expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             -(-x1*np.sqrt(1-np.cos(theta)**2)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2))
         +y1*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2))
         + lam_*x1*y1*expi_app(lam_ * x1 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         - lam_*x1*y1*expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_**2*x1*y1)
    )
    return f_int

@jit(nopython=True, parallel=True, cache=True)
def top_b(theta, y1, y2, lam_):
    f_int = ((y1-y2)*np.sqrt(1 - np.cos(theta)**2)/
            (4*np.pi*lam_**2*y1*y2)
            -(-np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2)) +
              np.exp(-lam_*y2/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             -(expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2))
             -expi_app(lam_ * y2 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             + (-y1*np.sqrt(1-np.cos(theta)**2)*np.exp(-lam_*y2/np.sqrt(1 - np.cos(theta)**2))
         +y2*np.sqrt(1-np.cos(theta)**2)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2))
         + lam_*y1*y2*expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2))
         - lam_*y1*y2*expi_app(lam_ * y2 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_**2*y1*y2)
            )
    return f_int

@jit(nopython=True, parallel=True, cache=True)
def top_c(theta, x2, y1, lam_):
    f_int = ((-x2*np.sqrt(1 - np.cos(theta)**2) + y1*np.sqrt(1 - np.sin(theta)**2))/
            (4*np.pi*lam_**2*x2*y1)
            + (-np.exp(-lam_*x2/np.sqrt(1 - np.sin(theta)**2)) +
              np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             + (expi_app(lam_ * x2 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         -expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_) 
             -(-x2*np.sqrt(1-np.cos(theta)**2)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2))
         +y1*np.sqrt(1-np.sin(theta)**2)*np.exp(-lam_*x2/np.sqrt(1 - np.sin(theta)**2))
         + lam_*x2*y1*expi_app(lam_ * x2 * (-1)/np.sqrt(1 - np.sin(theta)**2))
         - lam_*x2*y1*expi_app(lam_ * y1 * (-1)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_**2*x2*y1)
            )
    return f_int

@jit(nopython=True, parallel=True, cache=True)
def sing_a(theta, y1, r, lam_):
    f_int = (-(y1 - r*np.sqrt(1 - np.cos(theta)**2))/(4*np.pi*y1*lam_**2*r)
             -(-1+np.exp(lam_*r)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_)
             -(expi_app(lam_ * r * np.exp(1j*np.pi))
             -expi_app(lam_ * y1 * np.exp(1j*np.pi)/np.sqrt(1 - np.cos(theta)**2)))/(4*np.pi*lam_)
             + (y1 - r*np.sqrt(1-np.cos(theta)**2)*np.exp(lam_*r)*np.exp(-lam_*y1/np.sqrt(1 - np.cos(theta)**2))
         + lam_*r*y1*np.exp(lam_*r)*expi_app(lam_ * r * np.exp(1j*np.pi))
         - lam_*r*y1*np.exp(lam_*r)*
         expi_app(lam_ * y1 * np.exp(1j*np.pi)/np.sqrt(1 - np.cos(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_**2*r*y1))
    return f_int

@jit(nopython=True, parallel=True, cache=True)
def sing_b(theta, x1, r, lam_):
    f_int = (-(x1 - r*np.sqrt(1 - np.sin(theta)**2))/(4*np.pi*x1*lam_**2*r)
             -(-1+np.exp(lam_*r)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_)
             -(expi_app(lam_ * r * np.exp(1j*np.pi))
             -expi_app(lam_ * x1 * np.exp(1j*np.pi)/np.sqrt(1 - np.sin(theta)**2)))/(4*np.pi*lam_)
             +(x1 - r*np.sqrt(1-np.sin(theta)**2)*np.exp(lam_*r)*np.exp(-lam_*x1/np.sqrt(1 - np.sin(theta)**2))
         + lam_*r*x1*np.exp(lam_*r)*expi_app(lam_ * r * np.exp(1j*np.pi))
         - lam_*r*x1*np.exp(lam_*r)*
         expi_app(lam_ * x1 * np.exp(1j*np.pi)/np.sqrt(1 - np.sin(theta)**2)))*np.exp(-lam_*r)/(4*np.pi*lam_**2*r*x1))
    return f_int

# Dummy call to force compilation
# Dummy calls to force compilation
dummy_theta = np.array([0.1])
dummy_x1 = np.array([1.0])
dummy_x2 = np.array([1.5])
dummy_y1 = np.array([1.0])
dummy_y2 = np.array([1.5])
lam_ = 10.1 + 1j*10

_ = thin_a(dummy_theta, dummy_x1, dummy_y2, lam_)
_ = thin_b(dummy_theta, dummy_y1, dummy_y2, lam_)
_ = thin_c(dummy_theta, dummy_x2, dummy_y1, lam_)
_ = tall_a(dummy_theta, dummy_x1, dummy_y2, lam_)
_ = tall_b(dummy_theta, dummy_x1, dummy_y2, lam_)
_ = tall_c(dummy_theta, dummy_x1, dummy_y2, lam_)
_ = side_a(dummy_theta, dummy_x1, dummy_y2, lam_)
_ = side_b(dummy_theta, dummy_x1, dummy_y2, lam_)
_ = side_c(dummy_theta, dummy_x1, dummy_y2, lam_)
_ = top_a(dummy_theta, dummy_x1, dummy_y2, lam_)
_ = top_b(dummy_theta, dummy_x1, dummy_y2, lam_)
_ = top_c(dummy_theta, dummy_x1, dummy_y2, lam_)
_ = sing_a(dummy_theta, dummy_x1, dummy_y2, lam_)
_ = sing_b(dummy_theta, dummy_x1, dummy_y2, lam_)

print("Functions compiled and cached.")