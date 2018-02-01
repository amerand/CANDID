
import numpy as np
cimport numpy as np
from numpy cimport ndarray
import cython
cimport cython

from cpython cimport array

from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport pow

cdef double pi = 3.141592
cdef double c = pi*0.00154320987654321
DTYPE = np.double
ctypedef np.double_t DTYPE_t

def _Vud(double X):
    return 1.0 -pow(X,2)/8.0 + \
            pow(X,4)/192.0 -\
            pow(X,6)/9216.0 +\
            pow(X,8)/737280.0 -\
            pow(X,10)/88473600.0 +\
            pow(X,12)/14863564800.0 -\
            pow(X,14)/3329438515200.0

cdef extern from "cvis.c":
    double complex VbinMonoSingle(double u, double v, double wl,
                                    double x, double y, double fc, double fres, double diam, double diamc)

cdef extern from "cvis.c":
    void VbinMono(int n, double complex* V, double* u, double* v, double* wl,
                double x, double y, double fc, double fres, double diam, double diamc)

cdef extern from "cvis.c":
    void Vbin(int n, double complex* V, double* u, double* v, double* wl, double* dwl, int nsmear,
                double x, double y, double fc, double fres, double diam, double diamc)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cVbinMonoSingle(double u, double v, double wl, double x, double y,
                    double fc, double fres, double diam, double diamc):
    return VbinMonoSingle(u, v, wl, x, y, fc, fres, diam, diamc)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cVbinMono(ndarray[complex, ndim=1, mode="c"] V,
              ndarray[double, ndim=1, mode="c"] u,
              ndarray[double, ndim=1, mode="c"] v,
              ndarray[double, ndim=1, mode="c"] wl,
             double x, double y, double fc, double fres, double diam, double diamc):
    VbinMono(len(u), &V[0], &u[0], &v[0], &wl[0], x, y, fc, fres, diam, diamc)
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cVbin(ndarray[complex, ndim=1] V,
          ndarray[double, ndim=1] u,
          ndarray[double, ndim=1] v,
          ndarray[double, ndim=1] wl,
          ndarray[double, ndim=1] dwl,
          int nsmear, double x, double y,  double fc, double fres, double diam, double diamc):
    Vbin(len(u), &V[0], &u[0], &v[0], &wl[0], &dwl[0], nsmear,
            x, y, fc, fres, diam, diamc)
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cyVbin(ndarray[DTYPE_t, ndim=1] u,
         ndarray[DTYPE_t, ndim=1] v,
         ndarray[DTYPE_t, ndim=1] wavel,
         double dwavel, double diam, double diamc,
         double x, double y, double f, double fres, int nsmear):
    """
    assumes u, v are 1D np.ndarray
    """
    cdef unsigned int i = 0
    cdef unsigned int j, k
    cdef double phi, B
    cdef double Vc = 1.0
    cdef unsigned int n = u.shape[0]
    cdef np.ndarray Vr = np.ones(n, dtype=DTYPE)
    cdef np.ndarray Vi = np.zeros(n, dtype=DTYPE)

    for i in range(n):
        B = pow(u[i]**2 + v[i]**2, 0.5)
        phi = -2*pi*c*(u[i]*x + v[i]*y)
        if diam>0:
            Vr[i] = _Vud(pi*c*diam*B/wavel[i])
        if diamc>0:
            Vc = _Vud(pi*c*diamc*B/wavel[i])
        if dwavel == 0 or nsmear<2:
            Vr[i] += Vc*f*cos(phi/wavel[i])/100.
            Vi[i] += Vc*f*sin(phi/wavel[i])/100.
        else:
            j = 0
            while j<nsmear:
                #wl = wavel[i] + (-0.5 + j*1./(nsmear-1))*dwavel
                Vr[i] += Vc*f*cos(phi/(wavel[i] + (-0.5 + j*1./(nsmear-1))*dwavel))/100./nsmear
                Vi[i] += Vc*f*sin(phi/(wavel[i] + (-0.5 + j*1./(nsmear-1))*dwavel))/100./nsmear
                j += 1
        Vr[i] /= (1. + f/100. + fres/100.)
        Vi[i] /= (1. + f/100. + fres/100.)
    return Vr+1j*Vi
