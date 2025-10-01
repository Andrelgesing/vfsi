#from numba import jit
import cupy as cp
import cupyx.scipy.special as spc

#@cp.fuse()
def expi_app(z):
    """
    Approximate exponential integral Ei(z) for complex z using NumPy.
    Uses:
      - Power series for |z| < 4
      - Asymptotic expansion for |z| >= 4
    Parameters
    ----------
    z : numpy.ndarray or scalar
        Complex input (real or complex scalar or array)
    Returns
    -------
    numpy.ndarray or scalar
        Approximate values of Ei(z)
    """
    z = cp.asarray(z, dtype=cp.complex128)
    out = cp.empty_like(z)

    # Euler–Mascheroni constant
    gamma = 0.5772156649015328606

    # Masks for domain
    mask_small = cp.abs(z) < 4
    mask_large = ~mask_small

    # --- Power series for small z ---
    if cp.any(mask_small):
        zs = z[mask_small]
        term = cp.ones_like(zs)
        result = cp.log(zs) + gamma
        for k in range(1, 40):
            term *= zs / k
            result += term / k
        out[mask_small] = result

    # --- Asymptotic expansion for large z ---
    if cp.any(mask_large):
        zl = z[mask_large]
        inv_z = 1 / zl
        term = inv_z.copy()
        result = term.copy()
        for k in range(1, 10):
            term *= -k / zl
            result += term
        out[mask_large] = cp.exp(zl) * result

    return out

#@jit(nopython=True, parallel=True, cache=True)
#@cp.fuse()
def thin_a(theta, x1, y2, lam_):
    f_int = ((x1*cp.sqrt(1 - cp.cos(theta)**2) - y2*cp.sqrt(1 - cp.sin(theta)**2))/(4*cp.pi*lam_**2*x1*y2)
             -(expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
             -expi_app(lam_ * y2 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             -(-cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2)) +
              cp.exp(-lam_*y2/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             + (-x1*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(-lam_*y2/cp.sqrt(1 - cp.cos(theta)**2))
         +y2*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*x1*y2*expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         - lam_*x1*y2*expi_app(lam_ * y2 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_**2*x1*y2))
    return f_int

#@jit(nopython=True, parallel=True, cache=True)
#@cp.fuse()
def thin_b(theta, y1, y2, lam_):
    f_int = ((y1-y2)*cp.sqrt(1 - cp.cos(theta)**2)/(4*cp.pi*lam_**2*y1*y2)
             -(expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2))
             -expi_app(lam_ * y2 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             -(-cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2)) +
              cp.exp(-lam_*y2/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             + (-y1*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(-lam_*y2/cp.sqrt(1 - cp.cos(theta)**2))
         +y2*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2))
         + lam_*y1*y2*expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2))
         - lam_*y1*y2*expi_app(lam_ * y2 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_**2*y1*y2))
    return f_int

#@jit(nopython=True, parallel=True, cache=True)
#@cp.fuse()
def thin_c(theta, x2, y1, lam_):
    f_int = (-(x2*cp.sqrt(1 - cp.cos(theta)**2) - y1*cp.sqrt(1 - cp.sin(theta)**2))/(4*cp.pi*lam_**2*x2*y1)
             + (expi_app(lam_ * x2 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         -expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_) 
             +(-cp.exp(-lam_*x2/cp.sqrt(1 - cp.sin(theta)**2)) +
              cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             -(-x2*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2))
         +y1*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x2/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*x2*y1*expi_app(lam_ * x2 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         - lam_*x2*y1*expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_**2*x2*y1))

    return f_int

#@jit(nopython=True, parallel=True, cache=True)
#@cp.fuse()
def tall_a(theta, x1, y2, lam_):
    f_int = ((x1*cp.sqrt(1 - cp.cos(theta)**2) - y2*cp.sqrt(1 - cp.sin(theta)**2))/(4*cp.pi*lam_**2*x1*y2)
              -(-cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2)) +
              cp.exp(-lam_*y2/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             -(expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
             -expi_app(lam_ * y2 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             + (-x1*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(-lam_*y2/cp.sqrt(1 - cp.cos(theta)**2))
         +y2*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*x1*y2*expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         - lam_*x1*y2*expi_app(lam_ * y2 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_**2*x1*y2))
    return f_int

#@jit(nopython=True, parallel=True, cache=True)
#@cp.fuse()
def tall_b(theta, x1, x2, lam_):
    f_int = ((x1-x2)*cp.sqrt(1 - cp.sin(theta)**2)/(4*cp.pi*lam_**2*x1*x2)
             -(-cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2)) +
              cp.exp(-lam_*x2/cp.sqrt(1 - cp.sin(theta)**2)))/(4*cp.pi*lam_)
             -(expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
             -expi_app(lam_ * x2 * (-1)/cp.sqrt(1 - cp.sin(theta)**2)))/(4*cp.pi*lam_)
             + (-x1*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x2/cp.sqrt(1 - cp.sin(theta)**2))
         +x2*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*x1*x2*expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         - lam_*x1*x2*expi_app(lam_ * x2 * (-1)/cp.sqrt(1 - cp.sin(theta)**2)))/(4*cp.pi*lam_**2*x1*x2))
    return f_int

#@jit(nopython=True, parallel=True, cache=True)
#@cp.fuse()
def tall_c(theta, x2, y1, lam_):
    f_int = (-(x2*cp.sqrt(1 - cp.cos(theta)**2) - y1*cp.sqrt(1 - cp.sin(theta)**2))/(4*cp.pi*lam_**2*x2*y1)
             + (-cp.exp(-lam_*x2/cp.sqrt(1 - cp.sin(theta)**2)) +
              cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             + (expi_app(lam_ * x2 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         -expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             -(-x2*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2))
         +y1*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x2/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*x2*y1*expi_app(lam_ * x2 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         - lam_*x2*y1*expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_**2*x2*y1))
    return f_int

#@cp.fuse()
def side_a(theta, x1, y2, lam_):
    f_int = ((x1*cp.sqrt(1 - cp.cos(theta)**2) 
             - y2*cp.sqrt(1 - cp.sin(theta)**2))/(4*cp.pi*lam_**2*x1*y2)
             +(-x1*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(-lam_*y2/cp.sqrt(1 - cp.cos(theta)**2))
         +y2*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*x1*y2*expi_app(lam_ * x1 * cp.exp(1j*cp.pi)/cp.sqrt(1 - cp.sin(theta)**2))
         - lam_*x1*y2*expi_app(lam_ * y2 * cp.exp(1j*cp.pi)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_**2*x1*y2)
         -(expi_app(lam_ * x1 * cp.exp(1j*cp.pi)/cp.sqrt(1 - cp.sin(theta)**2))
             -expi_app(lam_ * y2 * cp.exp(1j*cp.pi)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
            -(-cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2)) +
              cp.exp(-lam_*y2/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_))
            
    return f_int

#@cp.fuse()
def side_b(theta, x1, x2, lam_):
    f_int = ((x1-x2)*cp.sqrt(1 - cp.sin(theta)**2)/
            (4*cp.pi*lam_**2*x1*x2)
            + (-x1*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x2/cp.sqrt(1 - cp.sin(theta)**2))
         +x2*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*x1*x2*expi_app(lam_ * x1 * cp.exp(1j*cp.pi)/cp.sqrt(1 - cp.sin(theta)**2))
         - lam_*x1*x2*expi_app(lam_ * x2 * cp.exp(1j*cp.pi)/cp.sqrt(1 - cp.sin(theta)**2)))/(4*cp.pi*lam_**2*x1*x2)
             -(expi_app(lam_ * x1 * cp.exp(1j*cp.pi)/cp.sqrt(1 - cp.sin(theta)**2))
             -expi_app(lam_ * x2 * cp.exp(1j*cp.pi)/cp.sqrt(1 - cp.sin(theta)**2)))/(4*cp.pi*lam_)
            -(-cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2)) +
              cp.exp(-lam_*x2/cp.sqrt(1 - cp.sin(theta)**2)))/(4*cp.pi*lam_))
    return f_int


#@jit(nopython=True, parallel=True, cache=True)
#@cp.fuse()
def side_a_old(theta, x1, y2, lam_):
    f_int = ((x1*cp.sqrt(1 - cp.cos(theta)**2) 
             - y2*cp.sqrt(1 - cp.sin(theta)**2))/(4*cp.pi*lam_**2*x1*y2)
             +(-x1*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(-lam_*y2/cp.sqrt(1 - cp.cos(theta)**2))
         +y2*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*x1*y2*expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         - lam_*x1*y2*expi_app(lam_ * y2 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_**2*x1*y2)
         -(expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
             -expi_app(lam_ * y2 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
            -(-cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2)) +
              cp.exp(-lam_*y2/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_))
            
    return f_int

#@jit(nopython=True, parallel=True, cache=True)
#@cp.fuse()
def side_b_old(theta, x1, x2, lam_):
    f_int = ((x1-x2)*cp.sqrt(1 - cp.sin(theta)**2)/
            (4*cp.pi*lam_**2*x1*x2)
            + (-x1*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x2/cp.sqrt(1 - cp.sin(theta)**2))
         +x2*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*x1*x2*expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         - lam_*x1*x2*expi_app(lam_ * x2 * (-1)/cp.sqrt(1 - cp.sin(theta)**2)))/(4*cp.pi*lam_**2*x1*x2)
             -(expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
             -expi_app(lam_ * x2 * (-1)/cp.sqrt(1 - cp.sin(theta)**2)))/(4*cp.pi*lam_)
            -(-cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2)) +
              cp.exp(-lam_*x2/cp.sqrt(1 - cp.sin(theta)**2)))/(4*cp.pi*lam_))
    return f_int

#@jit(nopython=True, parallel=True, cache=True)
#@cp.fuse()
def side_c(theta, x1, y1, lam_):
    f_int = ((x1*cp.sqrt(1 - cp.cos(theta)**2) - y1*cp.sqrt(1 - cp.sin(theta)**2))/
            (4*cp.pi*lam_**2*x1*y1)
            -(-x1*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2))
         +y1*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*x1*y1*expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         - lam_*x1*y1*expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_**2*x1*y1)
           + (expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         -expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_) 
           + (-cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2)) +
              cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_) )
    return f_int

#@jit(nopython=True, parallel=True, cache=True)
#@cp.fuse()
def top_a(theta, x1, y1, lam_):
    f_int = ((-x1*cp.sqrt(1 - cp.cos(theta)**2) 
             + y1*cp.sqrt(1 - cp.sin(theta)**2))/(4*cp.pi*lam_**2*x1*y1)
             + (-cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2)) +
              cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             + (expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
             -expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             -(-x1*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2))
         +y1*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*x1*y1*expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         - lam_*x1*y1*expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_**2*x1*y1)
    )
    return f_int

#@jit(nopython=True, parallel=True, cache=True)
#@cp.fuse()
def top_b(theta, y1, y2, lam_):
    f_int = ((y1-y2)*cp.sqrt(1 - cp.cos(theta)**2)/
            (4*cp.pi*lam_**2*y1*y2)
            -(-cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2)) +
              cp.exp(-lam_*y2/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             -(expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2))
             -expi_app(lam_ * y2 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             + (-y1*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(-lam_*y2/cp.sqrt(1 - cp.cos(theta)**2))
         +y2*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2))
         + lam_*y1*y2*expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2))
         - lam_*y1*y2*expi_app(lam_ * y2 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_**2*y1*y2)
            )
    return f_int

#@jit(nopython=True, parallel=True, cache=True)
def top_c(theta, x2, y1, lam_):
    f_int = ((-x2*cp.sqrt(1 - cp.cos(theta)**2) + y1*cp.sqrt(1 - cp.sin(theta)**2))/
            (4*cp.pi*lam_**2*x2*y1)
            + (-cp.exp(-lam_*x2/cp.sqrt(1 - cp.sin(theta)**2)) +
              cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             + (expi_app(lam_ * x2 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         -expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_) 
             -(-x2*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2))
         +y1*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(-lam_*x2/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*x2*y1*expi_app(lam_ * x2 * (-1)/cp.sqrt(1 - cp.sin(theta)**2))
         - lam_*x2*y1*expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_**2*x2*y1)
            )
    return f_int


def sing_a(theta, y1, r, lam_):
    f_int = (-(y1 - r*cp.sqrt(1 - cp.cos(theta)**2))/(4*cp.pi*y1*lam_**2*r)
             -(-1+cp.exp(lam_*r)*cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2)))*cp.exp(-lam_*r)/(4*cp.pi*lam_)
             -(expi_app(lam_ * r * cp.exp(1j*cp.pi))
             -expi_app(lam_ * y1 * cp.exp(1j*cp.pi)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             + (y1 - r*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(lam_*r)*cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2))
         + lam_*r*y1*cp.exp(lam_*r)*expi_app(lam_ * r * cp.exp(1j*cp.pi))
         - lam_*r*y1*cp.exp(lam_*r)*
         expi_app(lam_ * y1 * cp.exp(1j*cp.pi)/cp.sqrt(1 - cp.cos(theta)**2)))*cp.exp(-lam_*r)/(4*cp.pi*lam_**2*r*y1))
    return f_int
def sing_b(theta, x1, r, lam_):
    f_int = (-(x1 - r*cp.sqrt(1 - cp.sin(theta)**2))/(4*cp.pi*x1*lam_**2*r)
             -(-1+cp.exp(lam_*r)*cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2)))*cp.exp(-lam_*r)/(4*cp.pi*lam_)
             -(expi_app(lam_ * r * cp.exp(1j*cp.pi))
             -expi_app(lam_ * x1 * cp.exp(1j*cp.pi)/cp.sqrt(1 - cp.sin(theta)**2)))/(4*cp.pi*lam_)
             +(x1 - r*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(lam_*r)*cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*r*x1*cp.exp(lam_*r)*expi_app(lam_ * r * cp.exp(1j*cp.pi))
         - lam_*r*x1*cp.exp(lam_*r)*
         expi_app(lam_ * x1 * cp.exp(1j*cp.pi)/cp.sqrt(1 - cp.sin(theta)**2)))*cp.exp(-lam_*r)/(4*cp.pi*lam_**2*r*x1))
    return f_int

#@jit(parallel=True)
def sing_a_old(theta, y1, r, lam_):
    f_int = (-(y1 - r*cp.sqrt(1 - cp.cos(theta)**2))/(4*cp.pi*y1*lam_**2*r)
             -(-1+cp.exp(lam_*r)*cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2)))*cp.exp(-lam_*r)/(4*cp.pi*lam_)
             -(expi_app(lam_ * r * (-1))
             -expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))/(4*cp.pi*lam_)
             + (y1 - r*cp.sqrt(1-cp.cos(theta)**2)*cp.exp(lam_*r)*cp.exp(-lam_*y1/cp.sqrt(1 - cp.cos(theta)**2))
         + lam_*r*y1*cp.exp(lam_*r)*expi_app(lam_ * r * (-1))
         - lam_*r*y1*cp.exp(lam_*r)*
         expi_app(lam_ * y1 * (-1)/cp.sqrt(1 - cp.cos(theta)**2)))*cp.exp(-lam_*r)/(4*cp.pi*lam_**2*r*y1))
    return f_int

#@jit(parallel=True)
def sing_b_old(theta, x1, r, lam_):
    f_int = (-(x1 - r*cp.sqrt(1 - cp.sin(theta)**2))/(4*cp.pi*x1*lam_**2*r)
             -(-1+cp.exp(lam_*r)*cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2)))*cp.exp(-lam_*r)/(4*cp.pi*lam_)
             -(expi_app(lam_ * r * (-1))
             -expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2)))/(4*cp.pi*lam_)
             +(x1 - r*cp.sqrt(1-cp.sin(theta)**2)*cp.exp(lam_*r)*cp.exp(-lam_*x1/cp.sqrt(1 - cp.sin(theta)**2))
         + lam_*r*x1*cp.exp(lam_*r)*expi_app(lam_ * r * (-1))
         - lam_*r*x1*cp.exp(lam_*r)*
         expi_app(lam_ * x1 * (-1)/cp.sqrt(1 - cp.sin(theta)**2)))*cp.exp(-lam_*r)/(4*cp.pi*lam_**2*r*x1))
    return f_int


# Dummy call to force compilation
# Dummy calls to force compilation
dummy_theta = cp.array([0.1])
dummy_x1 = cp.array([1.0])
dummy_x2 = cp.array([1.5])
dummy_y1 = cp.array([1.0])
dummy_y2 = cp.array([1.5])
lam_ = cp.array([10.1 + 1j*10])

#_ = thin_a(dummy_theta, dummy_x1, dummy_y2, lam_)
#_ = thin_b(dummy_theta, dummy_y1, dummy_y2, lam_)
#_ = thin_c(dummy_theta, dummy_x2, dummy_y1, lam_)
#_ = tall_a(dummy_theta, dummy_x1, dummy_y2, lam_)
#_ = tall_b(dummy_theta, dummy_x1, dummy_y2, lam_)
#_ = tall_c(dummy_theta, dummy_x1, dummy_y2, lam_)
#_ = side_a(dummy_theta, dummy_x1, dummy_y2, lam_)
#_ = side_b(dummy_theta, dummy_x1, dummy_y2, lam_)
#_ = side_c(dummy_theta, dummy_x1, dummy_y2, lam_)
#_ = top_a(dummy_theta, dummy_x1, dummy_y2, lam_)
#_ = top_b(dummy_theta, dummy_x1, dummy_y2, lam_)
#_ = top_c(dummy_theta, dummy_x1, dummy_y2, lam_)
#_ = sing_a(dummy_theta, dummy_x1, dummy_y2, lam_)
#_ = sing_b(dummy_theta, dummy_x1, dummy_y2, lam_)

#print("Functions compiled and cached.")
