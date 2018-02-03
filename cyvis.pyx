cimport numpy as np
cimport cython

from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport pow
from libc.math cimport exp

# cdef double pi = 3.141592
# cdef double c = pi*0.00154320987654321

def _Vud(double X):
    return 1.0 -pow(X,2)/8.0 + \
            pow(X,4)/192.0 -\
            pow(X,6)/9216.0 +\
            pow(X,8)/737280.0 -\
            pow(X,10)/88473600.0 +\
            pow(X,12)/14863564800.0 -\
            pow(X,14)/3329438515200.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cyVbin(np.ndarray[double, ndim=1] Vr, np.ndarray[double, ndim=1] Vi,
           np.ndarray[double, ndim=1] u,
           np.ndarray[double, ndim=1] v,
           np.ndarray[double, ndim=1] wavel,
           np.ndarray[double, ndim=1] dwavel, int nsmear,
          double x, double y, double fc, double fres,
         double diam, double diamc):
    """
    Complex visibility of a binary
    Vr, Vi: real and imaginary visibilities
    u, v: in m
    wavel, dwavel: in um
    x, y, diam, diamc: in mas
    fc, fres: in % of the primary
    """
    cdef int i, j, k
    cdef double phi, B, wl, X
    cdef double Vc = 1.0
    cdef int n = len(u)
    i = 0
    while i<n:
        B = (u[i]**2 + v[i]**2)**0.5
        phi = -2*0.01523*(u[i]*x + v[i]*y)
        if diam>0:
            X = 0.01523*diam*B/wavel[i]
            Vr[i] = _Vud(X)
            Vi[i] = 0.0
        else:
            Vr[i] = 1.0
            Vi[i] = 0.0
        if diamc>0:
            X = 0.01523*diamc*B/wavel[i]
            Vc = _Vud(X)
        if dwavel[i] == 0 or nsmear<2:
            Vr[i] += Vc*fc*cos(phi/wavel[i])/100.
            Vi[i] += Vc*fc*sin(phi/wavel[i])/100.
        else:
            j = 0
            while j<nsmear:
                wl = wavel[i] + (-0.5 + j*1./(nsmear-1))*dwavel[i]
                Vr[i] += Vc*fc*cos(phi/wl)/100./nsmear
                Vi[i] += Vc*fc*sin(phi/wl)/100./nsmear
                j = j+1
        Vr[i] /= (1. + fc/100. + fres/100.)
        Vi[i] /= (1. + fc/100. + fres/100.)
        i = i+1
    return
