cimport numpy as np
cimport cython

from libc.math cimport sin
from libc.math cimport cos

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cyVbin(int n, np.ndarray[np.float64_t, ndim=1, mode='c'] Vr,
           np.ndarray[np.float64_t, ndim=1, mode='c'] Vi,
           np.ndarray[np.float64_t, ndim=1, mode='c'] u,
           np.ndarray[np.float64_t, ndim=1, mode='c'] v,
           np.ndarray[np.float64_t, ndim=1, mode='c'] wavel,
           np.ndarray[np.float64_t, ndim=1, mode='c'] dwavel, int nsmear,
           np.float64_t x, np.float64_t y, np.float64_t fc, np.float64_t fres,
           np.float64_t diam, np.float64_t diamc):
    """
    Complex visibility of a binary
    n: size of Vr, Vi, u, v, wavel, dwavel (int)
    Vr, Vi: real and imaginary visibilities (1D)
    u, v: in m (1D)
    wavel, dwavel: in um (1D)
    x, y, diam, diamc: in mas (scalars)
    fc, fres: in % of the primary (scalars)
    """
    cdef Py_ssize_t i, j
    cdef np.float64_t phi, B2, wl, X2
    cdef np.float64_t Vc = 1.0 # unresolved secondary
    i = 0
    while i<n:
        B2 = u[i]**2 + v[i]**2
        phi = -2*0.01523*(u[i]*x + v[i]*y)
        wl = wavel[i]
        if diam>0:
            # -- Uniform disk V function
            X2 = 0.0002319529*diam**2*B2/wl**2
            Vr[i] = 1.0 - X2/8.0 +\
                    X2**2/192.0 -\
                    X2**3/9216.0 +\
                    X2**4/737280.0 -\
                    X2**5/88473600.0 +\
                    X2**6/14863564800.0 -\
                    X2**7/3329438515200.0 +\
                    X2**8/958878292377600.0
            Vi[i] = 0.0
        else:
            # -- unresolved primary
            Vr[i] = 1.0
            Vi[i] = 0.0
        if diamc>0:
            # -- Uniform disk V function
            X2 = 0.0002319529*diamc**2*B2/wl**2
            Vc = 1.0 - X2/8.0 +\
                    X2**2/192.0 -\
                    X2**3/9216.0 +\
                    X2**4/737280.0 -\
                    X2**5/88473600.0 +\
                    X2**6/14863564800.0 -\
                    X2**7/3329438515200.0 +\
                    X2**8/958878292377600.0
        if dwavel[i] == 0 or nsmear<2:
            # -- monochromatic
            Vr[i] += Vc*fc*cos(phi/wl)/100.
            Vi[i] += Vc*fc*sin(phi/wl)/100.
        else:
            # -- bandwidth smeared
            j = 0
            while j<nsmear:
                wl = wavel[i] + (-0.5 + j*1./(nsmear-1))*dwavel[i]
                Vr[i] += Vc*fc*cos(phi/wl)/100./nsmear
                Vi[i] += Vc*fc*sin(phi/wl)/100./nsmear
                j = j+1
        # -- flux normalization
        Vr[i] /= (1. + fc/100. + fres/100.)
        Vi[i] /= (1. + fc/100. + fres/100.)
        i = i+1
    return
