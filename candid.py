from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
plt.ion() # interactive mode
_fitsLoaded=False
try:
    from astropy.io import fits
    _fitsLoaded=True
except:
    try:
        import pyfits as fits
        _fitsLoaded=True
    except:
        pass

if not _fitsLoaded:
    print('ERROR: astropy.io.fits or pyfits required!')

import time
import scipy.special
import scipy.interpolate
import scipy.stats

# -- defunct ;(
#from scipy import weave
#from scipy.weave import converters
#from scipy.weave import blitz_tools

from scipy.misc import factorial

import multiprocessing
import os
import sys


#__version__ = '0.1 | 2014/11/25'
#__version__ = '0.2 | 2015/01/07'  # big clean
#__version__ = '0.3 | 2015/01/14'  # add Alex contrib and FLAG taken into account
#__version__ = '0.4 | 2015/01/30'  # modified bandwidth smearing handling
#__version__ = '0.5 | 2015/02/01'  # field of view, auto rmin/rmax, bootstrapping
#__version__ = '0.6 | 2015/02/10'  # bug fix in smearing
#__version__ = '0.7 | 2015/02/17'  # bug fix in T3 computation
#__version__ = '0.8 | 2015/02/19'  # can load directories instead of single files, AMBER added, ploting V2, CP
#__version__ = '0.9 | 2015/02/25'  # adding polynomial reduction (as function of wavelength) to V2 et CP
#__version__ = '0.10 | 2015/08/14' # adding LD coef and coding CP in iCP
#__version__ = '0.11 | 2015/09/03' # changing detection limits to 99% and Mag instead of %
#__version__ = '0.12 | 2015/09/17' # takes list of files; bestFit cannot be out rmin/rmax
#__version__ = '0.13 | 2015/09/30' # fixed some bugs in list on minima for fitMap
#__version__ = '0.14 | 2015/09/30' # fixed some BIG bugs in fixed diameter option
#__version__ = '0.15 | 2015/10/02' # np.nanmean instead of np.mean in _chi2func
#__version__ = '0.16 | 2015/10/22' # weave to accelerate the binary visibility!
#__version__ = '0.17 | 2015/10/24' # weave to accelerate the binary T3!
#__version__ = '0.18 | 2015/10/26' # change in injection with a simpler algorithm;
                                   # change a bit in detectionLimit with injeciton,
                                   # doing a UD fit now; auto smearing setting
#__version__ = '0.19 | 2015/12/26' # adding instrument selection
#__version__ = '0.20 | 2016/06/14' # adding numpy (default, slow)/weave selection, bug fixes
#__version__ = '0.21 | 2016/11/22' # some cleaning
#__version__ = '0.22 | 2017/02/23' # bug corrected in smearing computation
#__version__ = '0.23 | 2017/11/08' # minor bugs corrected in plots
#__version__ = '0.3.1 | 2018/02/04' # Cython acceleration
__version__ = '1.0 | 2018/11/08' # Converted to Python3


print("""
========================== This is CANDID ==============================
[C]ompanion [A]nalysis and [N]on-[D]etection in [I]nterferometric [D]ata
                https://github.com/amerand/CANDID
========================================================================
""")
print(' version:', __version__)

# -- some general parameters:
CONFIG = {'color map':'cubehelix', # color map used
          'chi2 scale' : 'lin', # can be log
          'long exec warning': 300, # in seconds
          'suptitle':True, # display over head title
          'progress bar': True,
          'Ncores': None, # default is to use N-1 Cores
          'Nsmear': 4,
          'algo': 'numpy', # numpy / weave
          }

# -- units of the parameters
def paramUnits(s):
    if 'dwavel' in s:
        return 'um'
    else:
        if s.startswith('f_') or s.startswith('fres_'):
            return '% primary'
        return {'x':'mas', 'y':'mas', 'f':'% primary', 'diam*':'mas',
                'diamc':'mas', 'alpha*': 'none', 'fres':'% primary'}[s]

def variables():
    print(' | global parameters (can be updated):')
    for k in CONFIG.keys():
        print("  CONFIG['%s']"%k, CONFIG[k])
    return

variables()

__warningDwavel = True

# -- some general functions:
def _Vud(base, diam, wavel):
    """
    Complex visibility of a uniform disk for parameters:
    - base in m
    - diam in mas
    - wavel in um
    """
    x = 0.01523087098933543*diam*base/wavel
    x += 1e-6*(x==0)
    return 2*scipy.special.j1(x)/(x)

def _Vld(base, diam, wavel, alpha=0.36):
    nu = alpha /2. + 1.
    diam *= np.pi/(180*3600.*1000)
    x = -1.*(np.pi*diam*base/wavel/1.e-6)**2/4.
    V_ = 0
    for k_ in range(50):
        V_ += scipy.special.gamma(nu + 1.)/\
             scipy.special.gamma(nu + 1.+k_)/\
             scipy.special.gamma(k_ + 1.) *x**k_
    return V_

def _VbinSlow(uv, param):
    """
    Analytical complex visibility of a binary composed of a uniform
    disk diameter and an unresolved source. "param" is a dictionnary
    containing:

    'diam*'   : in mas
    'alpha*'  : optional LD coef for main star
    'wavel'   : in um
    'x, y'    : in mas
    'f'       : flux ratio in % -> takes the absolute value
    'f_wl_dwl': addition flux ratio in the line at wl, width dwl
    'fres'    : resolved flux
    'fres_wl_dwl': addition resolved flux ratio in the line at wl, width dwl

    'xg', yg', 'diamg', 'fg': gaussian position, diameters and flux

    """
    if 'f' in param.keys():
        f = np.abs(param['f'])/100.
    else:
        f = 0.
    for f_ in param.keys():
        if f_.startswith('f_'):
            wl = float(f_.split('_')[1])
            dwl = float(f_.split('_')[2])
            f += param[f_]*np.exp(-4*np.log(2)*(param['wavel']-wl)**2/dwl**2)/100.

    if 'fres' in param.keys():
        fres = param['fres']/100.0
    else:
        fres = 0

    for f_ in param.keys():
        if f_.startswith('fres_'):
            wl = float(f_.split('_')[1])
            dwl = float(f_.split('_')[2])
            fres += param[f_]*np.exp(-4*np.log(2)*(param['wavel']-wl)**2/dwl**2)/100.

    if 'fg' in param.keys():
        fg = param['fg']/100.0
    else:
        fg = 0.
    for f_ in param.keys():
        if f_.startswith('fg_'):
            wl = float(f_.split('_')[1])
            dwl = float(f_.split('_')[2])
            fg += param[f_]*np.exp(-4*np.log(2)*(param['wavel']-wl)**2/dwl**2)/100.

    c = np.pi/180/3600000.*1e6
    B = np.sqrt(uv[0]**2+uv[1]**2)
    if 'alpha*' in param.keys() and param['alpha*']>0.0:
        Vstar = _Vld(B, param['diam*'], param['wavel'], alpha=param['alpha*'])
    else:
        Vstar = _Vud(B, param['diam*'], param['wavel'])

    if 'diamc' in param.keys():
        Vcomp = _Vud(B, param['diamc'], param['wavel'])
    else:
        Vcomp = 1.0 + 0j # -- assumes it is unresolved.

    if 'diamg' in param.keys():
        Vg = np.exp(-(np.pi*c*param['diamg']*B/param['wavel'])**2/(4*np.log(2)))
        if 'xg' in param.keys():
            phig = 2*np.pi*c*(uv[0]*param['xg']+uv[1]*param['yg'])
            phig = 0.0
        else:
            phig = 0.0
    else:
        Vg = 0.0
        fg = 0.0
        phig = 0.0

    dl = np.linspace(-0.5,0.5, CONFIG['Nsmear'])

    if CONFIG['Nsmear']<2:
        dl = np.array([0.])

    if np.isscalar(param['wavel'] ):
        wl = param['wavel']+dl*param['dwavel']
        phi = 2*np.pi*c*(uv[0][:,None]*param['x']+uv[1][:,None]*param['y'])/wl[None,:]
        tmp = f*Vcomp[:,None]*np.exp(-1j*phi)
        if np.isscalar(phig):
            phig = phig/wl[None,:] +0.*uv[0][:,None]
            tmp += fg*Vg*np.exp(-1j*phig)
        else:
            phig = phig[:,None]/wl[None,:]
            tmp += fg*Vg[:,None]*np.exp(-1j*phig)
        res = (Vstar[:,None] + tmp)/(1.0 + f + fres + fg)
        res = res.mean(axis=1)
    else:
        # -- assumes u, v, and wavel are 2D
        wl = param['wavel'][:,:,None] + dl[None,None,:]*param['dwavel']
        phi = 2*np.pi*c*(uv[0][:,:,None]*param['x']+uv[1][:,:,None]*param['y'])/wl

        if not np.isscalar(Vcomp):
            Vcomp = Vcomp[:,:,None]*(1.+0*wl)
        tmp = f*Vcomp*np.exp(-1j*phi)
        if not np.isscalar(phig):
            phig = phig[:,:,None]/wl
        if not np.isscalar(fg):
            fg = fg[:,:,None]/(1+0*wl)
        if not np.isscalar(Vg):
            Vg = Vg[:,:,None]/(1+0*wl)
        tmp += fg*Vg*np.exp(-1j*phig)
        res = (Vstar + tmp.mean(axis=2))/(1.0 + f + fres + fg)
    return res

try:
    # -- Using Cython visibility function
    import cyvis
    def _VbinCy(uv, p):
        N = np.size(uv[0])
        if not 'diam*' in p.keys():
            diam = 0.0
        else:
            diam = p['diam*']*1.0 # copy

        if not 'diamc' in p.keys():
            diamc = 0.0
        else:
            diamc = p['diamc']*1.0 # copy

        if not 'fres' in p.keys():
            fres = 0.0
        else:
            fres = p['fres']*1.0 # copy

        if isinstance(p['wavel'], float):
            wavel = np.ones(N)*p['wavel']
        else:
            wavel = p['wavel'].flatten()

        if not 'dwavel' in p.keys():
            dwavel = 0.0
        else:
            dwavel = p['dwavel']*1.0 # copy
        if isinstance(dwavel, float):
            dwavel = np.ones(N)*p['dwavel']
        else:
            dwavel = p['wavel'].flatten()
        Vr = np.ones(np.size(uv[0]), dtype=np.double)
        Vi = np.zeros(np.size(uv[0]), dtype=np.double)

        cyvis.cyVbin(len(Vr), Vr, Vi, uv[0].flatten(), uv[1].flatten(),
                     wavel, dwavel, CONFIG['Nsmear'],
                     p['x'], p['y'], min(p['f'], 100), fres,
                     diam, diamc)
        return np.reshape(Vr+1j*Vi, uv[0].shape)
    _Vbin = _VbinCy
    print('Using Cython visibilities computation (Faster than Numpy)')
except:
    # -- Using Numpy visibility function
    _Vbin = _VbinSlow
    print('Using Numpy visibilities computation (Slower than Cython)')

def _V2binSlow(uv, param):
    """
    using weave
    uv = (u,v) where u,v a are ndarray

    param MUST contain:
    - diam*, x, y: in mas
    - f: in %
    - wavel: in um

    optional:
    - diamc: in mas
    - dwavel: in um
    - fres: fully resolved flux, in fraction of primary flux
    """
    if 'f' in param.keys():
        param['f'] = min(np.abs(param['f']),100)
    return np.abs(_Vbin(uv, param))**2

def _T3binSlow(uv, param):
    """
    using weave
    uv = (u1,v1, u2, v2) where u1,v1, u2,v2 a are ndarray

    param MUST contain:
    - diam*, x, y: in mas
    - f: in %
    - wavel: in um

    optional:
    - diamc: in mas
    - dwavel: in um
    - fres: unresolved flux, in fraction of primary flux
    """
    if 'f' in param.keys():
       param['f'] = min(np.abs(param['f']),100)
    return _Vbin((uv[0], uv[1]), param)*\
           _Vbin((uv[2], uv[3]), param)*\
           np.conj(_Vbin((uv[0]+uv[2], uv[1]+uv[3]), param))

def _approxVUD(x='x', maxM=8):
    """
    bases on polynomial approx of Bessel's J1
    """
    n = 1
    cm = lambda m: 2**(n+2*m-1)*factorial(m)*factorial(n+m)
    return ['%s%s/%.1f'%(' -' if (-1)**m < 0 else ' +',
                        '*'.join(x*(n+2*m-1)) if (n+2*m-1)>0 else '1',
                         cm(m)) for m in range(maxM+1)]

# -- set the approximation for UD visibility
_VUDX = ''.join(_approxVUD('X', maxM=7)).strip()
_VUDX =_VUDX[3:]
#print(_VUDX)
_VUDXeval = eval('lambda X:'+_VUDX)
if False: # -- check approximation
    print(_VUDX)
    plt.close('all')
    plt.subplot(211)
    x = np.linspace(1e-6, 7, 500)
    plt.plot(x, 2*scipy.special.j1(x)/x, '-r', label='Bessel')
    plt.plot(x, _VUDXeval(x), '-b', label='approximation')
    plt.plot(x, 0*x, linestyle='dashed')
    plt.ylim(-0.3,1)
    plt.ylabel('visibility')
    plt.subplot(212)
    plt.plot(x, 100*(2*scipy.special.j1(x)/x-_VUDXeval(x))/(
                     scipy.special.j1(x)/x+_VUDXeval(x)/2.), '-k')
    plt.ylabel('rel. err. %')
    plt.xlabel(r'$\pi$ B $\theta$ / $\lambda$')
    plt.ylim(-1,1)
    plt.legend()

def _V2binFast(uv, param):
    """
    using weave
    uv = (u,v) where u,v a are ndarray

    param MUST contain:
    - diam*, x, y: in mas
    - f: in %
    - wavel: in um

    optional:
    - diamc: in mas
    - dwavel: in um
    - fres: fully resolved flux, in fraction of primary flux
    """
    u, v = uv
    B = np.sqrt(u**2+v**2)
    s = u.shape
    u, v = u.flatten(), v.flatten()
    NU = len(u)
    vr, vi, v2 = np.zeros(NU), np.zeros(NU), np.zeros(NU)

    diam = np.abs(float(param['diam*']))

    wavel = param['wavel']

    if isinstance(wavel, np.ndarray):
        wavel = wavel.flatten()
    else:
        wavel = np.ones(NU)*wavel

    if 'x' in param.keys():
        x = param['x']*1.0
    else:
        x = 0.0

    if 'y' in param.keys():
        y = param['y']*1.0
    else:
        y = 0.0

    if 'f' in param.keys():
        f = np.abs(param['f'])*1.0
    else:
        f = 0.0

    if 'diamc' in param.keys():
        diamc = np.abs(param['diamc'])*1.0
    else:
        diamc = 0.0

    if 'dwavel' in param.keys():
        dwavel = param['dwavel']*1.0
    else:
        dwavel = 0.0

    if __warningDwavel and dwavel==0:
        print(' >>> WARNING: no spectral bandwidth provided!')

    if 'fres' in param.keys():
        fres = param['fres']*1.0
    else:
        fres = 0.0

    #print(diam, x, y, f, wavel, dwavel)
    #print(u.shape, v.shape, wavel.shape, vr.shape)
    Nsmear = CONFIG['Nsmear']
    print('#'*12, f, diam, '#'*12)
    code = u"""
    int i, j;
    float vis, X, pi, phi, visc, wl, t_vr, t_vi, c;

    pi = 3.1415926535;
    c = 0.004848136;
    f /= 100.0; /* flux ratio given in % */
    if (f>1){
        f = 1.0;
        }
    fres /= 100.0; /* flux ratio given in % */

    phi = 0.0;
    /* -- companion visibility, default is unresoved (V=1) */
    visc = 1.0;
    vis = 1.0;
    for (i=0; i<NU; i++){
        /* -- primary star of V_UD */
        phi = -2*pi*c*(u[i]*x + v[i]*y);

        if (Nsmear<2){
            if (diam>0){
                X = pi*c*B[i]*diam/wavel[i];
                vis = VUDX;
            }
            if (diamc>0) {
                X = pi*c*B[i]*diamc/wavel[i];
                visc = VUDX;
                }
            vr[i] = vis/(1.0+f+fres) + f * visc * cos( phi/wavel[i] ) / (1.0 + f + fres);
            vi[i] = f * visc * sin( phi/wavel[i] ) / (1.0 + f + fres);
            v2[i] = vr[i]*vr[i] + vi[i]*vi[i];
        } else {
        for (j=0;j<Nsmear;j++) {
            wl = wavel[i]+(-0.5 + j/(Nsmear-1.0))*dwavel;
            if (diam>0){
                X = pi*c*B[i]*diam/wl;
                vis = VUDX;
            }
            if (diamc>0) {
                X = pi*c*B[i]*diamc/wl;
                visc = VUDX;
                }
            t_vr = (vis + f * visc * cos(phi/wl) ) / (1.0 + f + fres);
            t_vi = (0.0 + f * visc * sin(phi/wl) ) / (1.0 + f + fres);

            vr[i] += t_vr / Nsmear;
            vi[i] += t_vi / Nsmear;
            v2[i] += (t_vr*t_vr + t_vi*t_vi) / Nsmear;
            }
        }
    }""".replace('VUDX', _VUDX)
    err = weave.inline(code, ['u','v','NU','diam','x','y','f','diamc','B',
                              'wavel','dwavel','vr','vi', 'v2', 'Nsmear','fres'],
                       compiler = 'gcc', verbose=0, #extra_compile_args=['-O3'],
                       type_converters = converters.blitz,
                       #headers=['<algorithm>', '<limits>']
                       )
    v2 = v2.reshape(s)
    return v2

def _T3binFast(uv, param):
    """
    using weave
    uv = (u1,v1, u2, v2) where u1,v1, u2,v2 a are ndarray

    param MUST contain:
    - diam*, x, y: in mas
    - f: in %
    - wavel: in um

    optional:
    - diamc: in mas
    - dwavel: in um
    - fres: unresolved flux, in fraction of primary flux

    """
    u1, v1, u2, v2 = uv
    s = u1.shape
    u1, v1 = u1.flatten(), v1.flatten()
    u2, v2 = u2.flatten(), v2.flatten()
    NU = len(u1)
    t3r, t3i = np.zeros(NU), np.zeros(NU)

    diam = np.abs(float(param['diam*']))

    wavel = param['wavel']

    if isinstance(wavel, np.ndarray):
        wavel = wavel.flatten()
    else:
        wavel = np.ones(NU)*wavel

    if 'x' in param.keys():
        x = float(param['x'])
    else:
        x = 0.0

    if 'y' in param.keys():
        y = float(param['y'])
    else:
        y = 0.0

    if 'f' in param.keys():
        f = float(np.abs(param['f']))
        #f = min(f, 1.0)
    else:
        f = 0.0

    if 'diamc' in param.keys():
        diamc = np.abs(float(param['diamc']))
    else:
        diamc = 0.0

    if 'dwavel' in param.keys():
        dwavel = float(param['dwavel'])
    else:
        dwavel = 0.0

    if __warningDwavel and dwavel==0:
        print(' >>> WARNING: no spectral bandwidth provided!')

    if 'fres' in param.keys():
        fres = float(param['fres'])
    else:
        fres = 0.0

    Nsmear = CONFIG['Nsmear']

    code = u"""int i, j;
    /* -- first baseline */
    double B1, vis1, phi1, visc1, vr1, vi1;

    /* -- second baseline */
    double B2, vis2, phi2, visc2, vr2, vi2;

    /* -- third baseline */
    double u12, v12;
    double B12, vis12, phi12, visc12, vr12, vi12;

    double X, pi, wl, c;
    pi = 3.1415926535;
    c = 0.004848136;
    f /= 100.; /* flux ratio given in % */
    if (f>1) {f = 1.0;}
    fres /= 100.; /* flux ratio given in % */

    phi1  = 0.0;
    phi2  = 0.0;
    phi12 = 0.0;
    vis1  = 1.0;
    vis2  = 1.0;
    vis12 = 1.0;

    /* -- companion visibility, default is unresoved (V=1) */
    visc1  = 1.0;
    visc2  = 1.0;
    visc12 = 1.0;


    for (i=0; i<NU; i++){
        /* -- baselines for each u,v coordinates */
        B1 = sqrt(u1[i]*u1[i] + v1[i]*v1[i]);
        B2 = sqrt(u2[i]*u2[i] + v2[i]*v2[i]);

        u12 = u1[i] + u2[i];
        v12 = v1[i] + v2[i];
        B12 = sqrt(u12*u12 + v12*v12);

        phi1  = -2*pi*0.004848136*(u1[i] * x + v1[i] * y);
        phi2  = -2*pi*0.004848136*(u2[i] * x + v2[i] * y);
        phi12 = -2*pi*0.004848136*(  u12 * x +   v12 * y);

        if (Nsmear<2) { /* -- monochromatic */
            if (diam>0) {
                /* -- approximation of V_UD */
                X = pi*c*B1*diam/wavel[i];
                vis1 = VUDX;
                X = pi*c*B2*diam/wavel[i];
                vis2 = VUDX;
                X = pi*c*B12*diam/wavel[i];
                vis12 = VUDX;
            }
            if (diamc>0) {
                /* -- approximation of V_UD */
                X = pi*c*B1*diamc/wavel[i];
                visc1 = VUDX;
                X = pi*c*B2*diamc/wavel[i];
                visc2 = VUDX;
                X = pi*c*B12*diamc/wavel[i];
                visc12 = VUDX;
                }

            /* -- binary visibilities: */
            vr1 = (vis1 + f*cos(phi1/wavel[i])) / (1.0 + f + fres);
            vi1 = f*sin(phi1/wavel[i]) / (1.0 + f + fres);

            vr2 = (vis2 + f*cos(phi2/wavel[i])) / (1.0 + f + fres);
            vi2 = f*sin(phi2/wavel[i]) / (1.0 + f + fres);

            vr12 = (vis12 + f*cos(phi12/wavel[i])) / (1.0 + f + fres);
            vi12 = f*sin(phi12/wavel[i]) / (1.0 + f + fres);

            /* -- T3 = V1 * V2 * conj(V12) */
            t3r[i] = vr1*(vr2*vr12 + vi2*vi12);
            t3r[i] += vi1*(vr2*vi12 - vi2*vr12);
            t3i[i] = vr1*(vi2*vr12 - vr2*vi12);
            t3i[i] += vi1*(vr2*vr12 + vi2*vi12);

        } else { /* -- smeared */
            vr1 = 0.0;
            vi1 = 0.0;
            vr2 = 0.0;
            vi2 = 0.0;
            vr12 = 0.0;
            vi12 = 0.0;
            for (j=0; j<Nsmear; j++) {
                /* -- wavelength in bin: */
                wl = wavel[i] + (-0.5 + j/(Nsmear-1.0)) * dwavel;
                if (diam>0) {
                    /* -- approximation of V_UD */
                    X = pi*c*B1*diam/wl;
                    vis1 = VUDX;
                    X = pi*c*B2*diam/wl;
                    vis2 = VUDX;
                    X = pi*c*B12*diam/wl;
                    vis12 = VUDX;
                }
                if (diamc>0) {
                    /* -- approximation of V_UD */
                    X = pi*c*B1*diamc/wl;
                    visc1 = VUDX;
                    X = pi*c*B2*diamc/wl;
                    visc2 = VUDX;
                    X = pi*c*B12*diamc/wl;
                    visc12 = VUDX;
                    }

                /* == smear in T3 ======================= */
                /* -- binary visibilities: */
                vr1 = (vis1 + f*cos(phi1/wl)) / (1.0 + f + fres);
                vi1 = f*sin(phi1/wl) / (1.0 + f + fres);

                vr2 = (vis2 + f*cos(phi2/wl)) / (1.0 + f + fres);
                vi2 = f*sin(phi2/wl) / (1.0 + f + fres);

                vr12 = (vis12 + f*cos(phi12/wl)) / (1.0 + f + fres);
                vi12 = f*sin(phi12/wl) / (1.0 + f + fres);

                /* -- T3 = V1 * V2 * conj(V12) */
                t3r[i] += vr1*(vr2*vr12 + vi2*vi12)/Nsmear;
                t3r[i] += vi1*(vr2*vi12 - vi2*vr12)/Nsmear;
                t3i[i] += vr1*(vi2*vr12 - vr2*vi12)/Nsmear;
                t3i[i] += vi1*(vr2*vr12 + vi2*vi12)/Nsmear;
            }
        }

    }""".replace('VUDX', _VUDX)
    err = weave.inline(code, ['u1','v1','u2','v2','NU','diam','x','y', 'fres',
                              'f','diamc','wavel','dwavel','t3r','t3i','Nsmear'],
                       #type_factories = blitz_type_factories,
                       compiler = 'gcc', verbose=0)
    res = t3r + 1j*t3i
    res = res.reshape(s)
    return res

def _NsmearForCPaccuracy(errCP, B, sep, wavel, dwavel, f):
    """
    - errCP in degrees
    - B in meters
    - sep in mas
    - wavel, dwavel in um
    - f in percent
    """
    R = wavel/dwavel
    mod = (B/100.*sep/10./wavel)**2/(R/20.)**2*f/2.
    return min(np.ceil((mod/errCP)**(2/3.)), 2)

def _modelObservables(obs, param, flattened=True):
    """
    model observables contained in "obs".
    param -> see _Vbin

    --> will force the contrast ratio to be positive!

    Observations are entered as:
    obs = [('v2;ins', u, v, wavel, ...),
           ('cp;ins', u1, v1, u2, v2, wavel, ...),
           ('t3;ins', u1, v1, u2, v2, wavel, ...)]
    each tuple can be longer, the '...' part will be ignored

    units: u,v in m; wavel in um

    width of the wavelength channels in param:
    - a global "dwavel" is defined, the width of the pixels
    - "dwavel;ins" average value per intrument

    for CP and T3, the third u,v coordinate is computed as u1+u2, v1+v2
    """
    global CONFIG
    #c = np.pi/180/3600000.*1e6
    res = [0.0 for o in obs]
    # -- copy parameters:
    tmp = {k:param[k] for k in param.keys()}
    tmp['f'] = np.abs(tmp['f'])
    for i, o in enumerate(obs):
        if 'dwavel' in param.keys():
            dwavel = param['dwavel']
        elif 'dwavel;'+o[0].split(';')[1] in param.keys():
            dwavel = param['dwavel;'+o[0].split(';')[1]]
        else:
            dwavel = 0.0

        # -- remove dwavel(s)
        tmp = {k:tmp[k] for k in param.keys() if not k.startswith('dwavel')}

        if o[0].split(';')[0]=='v2':
            tmp['wavel'] = o[3]
            tmp['dwavel'] = dwavel
            if CONFIG['algo']=='weave':
                res[i] = _V2binFast([o[1], o[2]], tmp)
            else:
                res[i] = _V2binSlow([o[1], o[2]], tmp)
        elif o[0].split(';')[0].startswith('v2_'): # polynomial fit
            p = int(o[0].split(';')[0].split('_')[1])
            n = int(o[0].split(';')[0].split('_')[2])
            # -- wl range based on min, mean, max
            _wl = np.linspace(o[-4][0], o[-4][2], 2*n+2)
            _v2 = []
            for _l in _wl:
                tmp['wavel']=_l
                _v2.append(_V2binFast([o[1], o[2]], tmp))
            _v2 = np.array(_v2)
            res[i] = np.array([np.polyfit(_wl-o[-4][1], _v2[:,j], n)[n-p]
                                for j in range(_v2.shape[1])])

        elif o[0].split(';')[0]=='cp' or o[0].split(';')[0]=='t3' or\
            o[0].split(';')[0]=='icp':
            tmp['wavel'] = o[5]
            tmp['dwavel'] = dwavel
            if CONFIG['algo']=='weave':
                t3 = _T3binFast((o[1], o[2], o[3], o[4]), tmp)
            else:
                t3 = _T3binSlow((o[1], o[2], o[3], o[4]), tmp)

            if o[0].split(';')[0]=='cp':
                res[i] = np.angle(t3)
            elif o[0].split(';')[0]=='icp':
                res[i] = t3/np.absolute(t3)
            elif o[0].split(';')[0]=='t3':
                res[i] = np.absolute(t3)
        elif o[0].split(';')[0].startswith('cp_'): # polynomial fit
            p = int(o[0].split(';')[0].split('_')[1])
            n = int(o[0].split(';')[0].split('_')[2])
            # -- wl range based on min, mean, max
            _wl = np.linspace(o[-4][0], o[-4][2], 2*n+2)
            # -- remove pix width
            tmp.pop('dwavel')
            _cp = []
            for _l in _wl:
                tmp['wavel']=_l
                _cp.append(np.angle(_T3binFast((o[1], o[2], o[3], o[4]), tmp)))
            _cp = np.array(_cp)
            res[i] = np.array([np.polyfit(_wl-o[-4][1], _cp[:,j], n)[n-p] for j in range(_cp.shape[1])])
        else:
            print('ERROR: unreckognized observable:', o[0])

    if not flattened:
        return res

    res2 = np.array([])
    for r in res:
        res2 = np.append(res2, r.flatten())

    return res2

def _nSigmas(chi2r_TEST, chi2r_TRUE, NDOF):
    """
    - chi2r_TEST is the hypothesis we test
    - chi2r_TRUE is what we think is what described best the data
    - NDOF: numer of degres of freedom

    chi2r_TRUE <= chi2r_TEST

    returns the nSigma detection
    """
    p = scipy.stats.chi2.cdf(NDOF, NDOF*chi2r_TEST/chi2r_TRUE)
    log10p = np.log10(np.maximum(p, 1e-161)) ### Alex: 50 sigmas max
    #p = np.maximum(p, -100)
    res = np.sqrt(scipy.stats.chi2.ppf(1-p,1))
    # x = np.logspace(-15,-12,100)
    # c = np.polyfit(np.log10(x), np.sqrt(scipy.stats.chi2.ppf(1-x,53)), 1)
    c = np.array([-0.25028407,  9.66640457])
    if isinstance(res, np.ndarray):
        res[log10p<-15] = np.polyval(c, log10p[log10p<-15])
        res = np.nan_to_num(res)
        res += 90*(res==0)
    else:
        if log10p<-15:
            res =  np.polyval(c, log10p)
        if np.isnan(res):
            res = 90.
    return res

def _injectCompanionData(data, delta, param):
    """
    Inject analytically a companion defined as 'param' in the 'data' using
    'delta'. 'delta' contains the corresponding V2 for T3 and CP first order
    calculations.

    data and delta have same length
    """
    global CONFIG

    bi = _modelObservables(data, param, flattened=False)
    ud = param.copy(); ud['f'] = 0.0
    ud = _modelObservables(data, ud, flattened=False)
    for i,d in enumerate(data):
        d[-2] += np.sign(param['f'])*(bi[i]-ud[i])
    return data
    return res

def _generateFitData(chi2Data, observables, instruments):
    """
    filter only the meaningful observables
    returns:
    - measurements, flattened
    - errors, flattened
    - uv = B flattened if V2
      uv = max baseline flattened if CP or T3
    - type, flattened
    - wl, flattened
    """
    _meas, _errs, _wl, _uv, _type = np.array([]), np.array([]), np.array([]), [], []
    for c in chi2Data:
        if c[0].split(';')[0] in observables and \
            c[0].split(';')[1] in instruments:
            _type.extend([c[0]]*len(c[-2].flatten()))
            _meas = np.append(_meas, c[-2].flatten())
            _errs = np.append(_errs, c[-1].flatten())
            _wl = np.append(_wl, c[-4].flatten())

            if c[0].split(';')[0] == 'v2':
                _uv = np.append(_uv, np.sqrt(c[1].flatten()**2+c[2].flatten()**2)/c[3].flatten())
            elif c[0].split(';')[0].startswith('v2_'):
                _uv = np.append(_uv, np.sqrt(c[1].flatten()**2+c[2].flatten()**2)/c[3][1])
            elif c[0].split(';')[0] == 't3' or c[0].split(';')[0] == 'cp' or\
                c[0].split(';')[0] == 'icp':
                tmp = np.maximum(np.sqrt(c[1].flatten()**2+c[2].flatten()**2)/c[5].flatten(),
                                                np.sqrt(c[3].flatten()**2+c[4].flatten()**2)/c[5].flatten())
                tmp = np.maximum(tmp, np.sqrt((c[1]+c[3]).flatten()**2+(c[2]+c[4]).flatten()**2)/c[5].flatten() )
                _uv = np.append(_uv, tmp)

    _errs += _errs==0. # remove bad point in a dirty way
    return _meas, _errs, _uv, np.array(_type), _wl

def _fitFunc(param, chi2Data, observables, instruments, fitAlso=[], doNotFit=[]):
    """
    fit the data in "chi2data" (only "observables") using starting parameters

    returns a dpfit dictionnary
    """
    # -- extract meaningfull data
    _meas, _errs, _uv, _types, _wl = _generateFitData(chi2Data, observables,
                                                    instruments)
    # -- guess what needs to be fitted
    fitOnly=[]

    if param['f']!=0:
        fitOnly.extend(['x', 'y', 'f'])

    if 'v2' in observables or 't3' in observables:
       for k in ['diam*', 'diamc', 'fres']:
           if k in param.keys():
               fitOnly.append(k)

    if not fitAlso is None:
        fitOnly.extend(fitAlso)

    fitOnly = list(set(fitOnly))
    for f in doNotFit:
        if f in fitOnly:
            fitOnly.remove(f)


    # -- does the actual fit
    res = _dpfit_leastsqFit(_modelObservables,
                            list(filter(lambda c: c[0].split(';')[0] in observables and
                                             c[0].split(';')[1] in instruments,
                                             chi2Data)),
                            param, _meas, _errs, fitOnly = fitOnly)

    # -- _k used in some callbacks
    if '_k' in param.keys():
        res['_k'] = param['_k']
    # -- diam* and f can only be positive
    if 'diam*' in res['best'].keys():
        res['best']['diam*'] = np.abs(res['best']['diam*'])
    if 'f' in res['best'].keys():
        res['best']['f'] = np.abs(res['best']['f'])
    return res

def _chi2Func(param, chi2Data, observables, instruments):
    """
    Returns the chi2r comparing model of parameters "param" and data "chi2Data", only
    considering "observables" (such as v2, cp, t3)
    """
    _meas, _errs, _uv, _types, _wl = _generateFitData(chi2Data, observables, instruments)

    res = (_meas-_modelObservables(list(filter(lambda c: c[0].split(';')[0] in observables and
                                                    c[0].split(';')[1] in instruments, chi2Data)), param))
    res = np.nan_to_num(res) # FLAG == TRUE are nans in the data
    res[np.iscomplex(res)] = np.abs(res[np.iscomplex(res)])
    res = np.abs(res)**2/_errs**2
    res = np.nanmean(res)

    if '_i' in param.keys() and '_j' in param.keys():
        return param['_i'], param['_j'], res
    else:
        #print('test:', res)
        return res

def _detectLimit(param, chi2Data, observables, instruments, delta=None, method='injection'):
    """
    Returns the flux ratio (in %) for which the chi2 ratio between binary and UD is 3 sigmas.

    Uses the position and diameter given in "Param" and only varies the flux ratio

    - method=="Absil", uses chi2_BIN/chi2_UD, assuming chi2_UD is the best model
    - otherwise, uses chi2_UD/chi2_BIN, after injecting a companion
    """
    fr, nsigma, chi2= [], [], []
    mult = 1.4
    cond = True
    if method=='Absil':
        tmp = {k:param[k] if k!='f' else 0.0 for k in param.keys()}
        # -- reference chi2 for UD
        if '_i' in param.keys() or '_j' in param.keys():
            chi2_0 = _chi2Func(tmp, chi2Data, observables, instruments)[-1]
        else:
            chi2_0 = _chi2Func(tmp, chi2Data, observables, instruments)

    ndata = np.sum([c[-1].size for c in chi2Data if c[0].split(';')[0] in observables and
                                                    c[0].split(';')[1] in instruments])
    n = 0
    while cond:
        if method=='Absil':
            fr.append(param['f'])
            if '_i' in param.keys() or '_j' in param.keys():
                chi2.append(_chi2Func(param, chi2Data, observables, instruments)[-1])
            else:
                chi2.append(_chi2Func(param, chi2Data, observables, instruments))
            nsigma.append(_nSigmas(chi2[-1], chi2_0, ndata))

        elif method=='injection':
            fr.append(param['f'])
            # -- copy data
            data = [[x if i==0 else x.copy() for i,x in enumerate(d)]
                        for d in chi2Data]
            # -- inject companion
            data = _injectCompanionData(data, delta, param)
            # -- compare chi2 UD and chi2 Binary
            tmp = {k:(param[k] if k!='f' else 0.0) for k in param.keys()} # -- UD
            if 'v2' in observables or 't3' in observables:
                fit = _fitFunc(tmp, data, observables, instruments,
                               doNotFit=list(filter(lambda k: k!='diam*', tmp.keys())))
                a = fit['chi2']
            else:
                a = None

            if '_i' in param.keys() or '_j' in param.keys():
                if a is None:
                    a = _chi2Func(tmp, data, observables, instruments)[-1] # -- chi2 UD
                b = _chi2Func(param, data, observables, instruments)[-1] # -- chi2 Binary
            else:
                if a is None:
                    a = _chi2Func(tmp, data, observables, instruments) # -- chi2 UD
                b = _chi2Func(param, data, observables, instruments) # -- chi2 Binary
            chi2.append((a,b))
            nsigma.append(_nSigmas(a, b, ndata))

        # -- Newton method:
        if len(fr)==1:
            if nsigma[-1]<3:
                param['f'] *= mult
            else:
                param['f'] /= mult
        else:
            if nsigma[-1]<3 and nsigma[-2]<3:
                param['f'] *= mult
            elif nsigma[-1]>=3 and nsigma[-2]>=3:
                param['f'] /= mult
            elif nsigma[-1]<3 and nsigma[-2]>=3:
                mult /= 1.5
                param['f'] *= mult
            elif nsigma[-1]>=3 and nsigma[-2]<3:
                mult /= 1.5
                param['f'] /= mult
        n+=1
        # -- stopping:
        if len(fr)>1 and any([s>3 for s in nsigma]) and any([s<3 for s in nsigma]):
            cond = False
        if n>50:
            cond = False

    fr, nsigma = np.array(fr), np.array(nsigma)
    fr = fr[np.argsort(nsigma)]
    nsigma = nsigma[np.argsort(nsigma)]
    if '_i' in param.keys() and '_j' in param.keys():
        return param['_i'], param['_j'], np.interp(3, nsigma, fr)
    else:
        return np.interp(3, nsigma, fr)

# == The main class
class Open:
    global CONFIG, _ff2_data
    def __init__(self, filename, rmin=None, rmax=None,  reducePoly=None,
                 wlOffset=0.0, alpha=0.0, v2bias = 1., instruments=None):
        """
        - filename: an OIFITS file
        - rmin, rmax: minimum and maximum radius (in mas) for plots and search
        - wlOffset (in um) will be *added* to the wavelength table. Mainly for AMBER in LR-HK
        - alpha: limb darkening coefficient
        - instruments: load only certain instruments from the OIfiles (faster)

        load OIFITS file assuming one target, one OI_VIS2, one OI_T3 and one WAVE table
        """
        self.wlOffset = wlOffset # mainly to correct AMBER poor wavelength calibration...
        self.v2bias = v2bias
        self.loadOnlyInstruments = instruments
        if isinstance(filename, list):
            self._initOiData()
            for i,f in enumerate(filename):
                print(' | file %d/%d: %s'%(i+1, len(filename), f))
                #try:
                self._loadOifitsData(f, reducePoly=reducePoly)
                #except:
                #    print('   -> ERROR! could not read', f)
            self.filename = filename
            self.titleFilename = '\n'.join([os.path.basename(f) for f in filename])
        elif os.path.isdir(filename):
            print(' | loading FITS files in ', filename)
            files = os.listdir(filename)
            files = filter(lambda x: ('.fit' in x.lower()) or ('.oifits' in x.lower()), files)
            self._initOiData()
            for i,f in enumerate(files):
                print(' | file %d/%d: %s'%(i+1, len(files), f))
                try:
                    self._loadOifitsData(os.path.join(filename, f), reducePoly=reducePoly)
                except:
                    print('   -> ERROR! could not read', os.path.join(filename, f))
            self.filename = filename
            self.titleFilename = filename
        elif os.path.exists(filename):
            print(' | loading file', filename)
            self.filename = filename
            self.titleFilename = os.path.basename(filename)
            self._initOiData()
            self._loadOifitsData(filename, reducePoly=reducePoly)

        print(' | compute aux data for companion injection')
        self._compute_delta()
        #self.estimateCorrSpecChannels()

        # -- all MJDs in the files:
        mjds = []
        for d in self._rawData:
            mjds.extend(list(set(d[-3].flatten())))
        self.allMJD = np.array(list(set(mjds)))

        self.ALLobservables = list(set([c[0].split(';')[0] for c in self._rawData]))
        self.ALLinstruments = list(set([c[0].split(';')[1] for c in self._rawData]))

        # -- can be updated by user
        self.observables = list(set([c[0].split(';')[0] for c in self._rawData]))
        self.instruments = list(set([c[0].split(';')[1] for c in self._rawData]))

        print(' | observables available: [',end=' ')
        print(', '.join(["'"+o+"'" for o in self.observables])+']')
        print(' | instruments: [',end=' ')
        print(', '.join(["'"+o+"'" for o in self.instruments])+']')

        self._chi2Data = self._copyRawData()

        self.rmin = rmin
        if self.rmin is None:
            self.rmin = self.minSpatialScale
            print(" | rmin= not given, set to smallest spatial scale: rmin=%5.2f mas"%(self.rmin))

        self.rmax = rmax
        if self.rmax is None:
            self.rmax = 1.2*self.smearFov
            print(" | rmax= not given, set to 1.2*Field of View: rmax=%5.2f mas"%(self.rmax))
        self.diam = None
        self.bestFit = None

        self.alpha=alpha

        # if diam is None and\
        #     ('v2' in self.observables or 't3' in self.observables):
        #     print(' | no diameter given, trying to estimate it...')
        #     self.fitUD()
        # else:
        #     self.diam = diam
    def setLDcoefAlpha(self, alpha):
        """
        set the power law LD coef "alpha" for the main star. not fitted in any fit!
        """
        self.alpha=alpha
        return
    def fitUD(self, forcedDiam=None, fitAlso=None, guess=1.0):
        """
        FIT UD diameter model to the data (if )
        """
        # -- fit diameter if possible
        if not forcedDiam is None:
            if isinstance(forcedDiam, float) or \
                isinstance(forcedDiam, int):
                self.diam = forcedDiam
            else:
                print(' > WARNING: assume unresolved diam=0.0mas')
                print(' |          forceDiam= should be the fixed value')
                self.diam = 0.0

        if forcedDiam is None and \
            ('v2' in self.observables or 't3' in self.observables) and\
            ('v2' in [c[0].split(';')[0] for c in self._chi2Data] or
             't3' in [c[0].split(';')[0] for c in self._chi2Data]):
            tmp = {'x':0.0, 'y':0.0, 'f':0.0, 'diam*':guess, 'alpha*':self.alpha}
            if self.alpha>0:
                print(' | LD diam Fit')
            else:
                print(' | UD diam Fit')

            for _k in self.dwavel.keys():
                tmp['dwavel;'+_k] = self.dwavel[_k]
            fit_0 = _fitFunc(tmp, self._chi2Data, self.observables,
                                self.instruments, fitAlso=fitAlso)
            self.chi2_UD = fit_0['chi2']
            print(' | best fit diameter: %5.3f +- %5.3f mas'%(fit_0['best']['diam*'],
                                                           fit_0['uncer']['diam*']))
            if not fitAlso is None:
                for k in fitAlso:
                    print(' | %s: %5.3f +- %5.3f '%(k, fit_0['best'][k],
                                                    fit_0['uncer'][k]))
            self.diam = fit_0['best']['diam*']
            self.ediam = fit_0['uncer']['diam*']
            print(' | chi2 = %4.3f'%self.chi2_UD)
        elif not self.diam is None:
            print(' | single star CHI2 (no fit!)')
            if self.alpha == 0:
                print(' | Chi2 UD for diam=%4.3fmas'%self.diam)
            else:
                print(' | Chi2 LD for diam=%4.3fmas, alpha=%4.3f'%self.diam)

            tmp = {'x':0.0, 'y':0.0, 'f':0.0, 'diam*':self.diam, 'alpha*':self.alpha}
            for _k in self.dwavel.keys():
                tmp['dwavel;'+_k] = self.dwavel[_k]
            #fit_0 = _fitFunc(tmp, self._chi2Data, self.observables)
            self.chi2_UD = _chi2Func(tmp, self._chi2Data, self.observables,
                                        self.instruments)
            self.ediam = np.nan
            print(' |  chi2 = %4.3f'%self.chi2_UD)
        else:
            print(" > WARNING: a UD cannot be determined, and a valid 'diam' is not defined for Chi2 computation. ASSUMING CHI2UD=1.0")
            self.diam = None
            self.ediam = np.nan
            self.chi2_UD = 1.0
        return

    def _initOiData(self):
        self.wavel = {} # wave tables for each instrumental setup
        self.dwavel = {} # -- average width of the spectral channels
        self.all_dwavel = {} # -- width of all the spectral channels (same shape as wavel)
        self.wavel_3m = {} # -- min, mean, max
        self.telArray = {}
        self._rawData, self._delta = [], []
        self.smearFov = 5e3 # bandwidth smearing FoV, in mas
        self.diffFov = 5e3 # diffraction FoV, in mas
        self.minSpatialScale = 5e3
        self._delta = []
        return
    def _copyRawData(self):
        """
        create a copy of the raw data
        """
        return [[x if i==0 else x.copy() for i,x in enumerate(d)] for d in self._rawData]

    def _loadOifitsData(self, filename, reducePoly=None):
        """
        Note that CP are stored in radians, not degrees like in the OIFITS!

        reducePoly: reduce data by poly fit of order "reducePoly" on as a
        function wavelength
        """
        self._fitsHandler = fits.open(filename)
        self._dataheader={}
        for k in ['X','Y','F']:
            try:
                self._dataheader[k] = self._fitsHandler[0].header['INJCOMP'+k]
            except:
                pass
        if self.loadOnlyInstruments is None:
            testInst = lambda h: True
        else:
            testInst = lambda h: h.header['INSNAME'] in self.loadOnlyInstruments

        # -- load Wavelength and Array: ----------------------------------------------
        for hdu in self._fitsHandler[1:]:
            if hdu.header['EXTNAME']=='OI_WAVELENGTH' and testInst(hdu):
                self.wavel[hdu.header['INSNAME']] = self.wlOffset + hdu.data['EFF_WAVE']*1e6 # in um
                self.wavel_3m[hdu.header['INSNAME']] = (self.wlOffset + self.wavel[hdu.header['INSNAME']].min(),
                                                        self.wlOffset + self.wavel[hdu.header['INSNAME']].mean(),
                                                        self.wlOffset + self.wavel[hdu.header['INSNAME']].max())
                self.all_dwavel[hdu.header['INSNAME']] = hdu.data['EFF_BAND']*1e6

                self.all_dwavel[hdu.header['INSNAME']] *= 2. # assume the limit is not the pixel
                self.dwavel[hdu.header['INSNAME']] = \
                        np.mean(self.all_dwavel[hdu.header['INSNAME']])
            if hdu.header['EXTNAME']=='OI_ARRAY':
                name = hdu.header['ARRNAME']
                diam = hdu.data['DIAMETER'].mean()
                if diam==0:
                    if 'VLTI' in name:
                        if 'AT' in hdu.data['TEL_NAME'][0]:
                            diam = 1.8
                        if 'UT' in hdu.data['TEL_NAME'][0]:
                            diam = 8.2
                self.telArray[name] = diam

        # -- load all data:
        maxRes = 0.0 # -- in Mlambda
        #amberWLmin, amberWLmax = 1.8, 2.4 # -- K
        #amberWLmin, amberWLmax = 1.4, 1.7 # -- H
        #amberWLmin, amberWLmax = 1.0, 1.3 # -- J
        #amberWLmin, amberWLmax = 1.4, 2.5 # -- H+K
        amberWLmin, amberWLmax = 1.0, 2.5 # -- J+H+K

        amberAtmBand = [1.0, 1.35, 1.87]
        for hdu in self._fitsHandler[1:]:
            if hdu.header['EXTNAME'] in ['OI_T3', 'OI_VIS2'] and testInst(hdu):
                ins = hdu.header['INSNAME']
                arr = hdu.header['ARRNAME']

            if hdu.header['EXTNAME']=='OI_T3' and testInst(hdu):
                # -- CP
                data = hdu.data['T3PHI']*np.pi/180
                data[hdu.data['FLAG']] = np.nan # we'll deal with that later...
                data[hdu.data['T3PHIERR']>1e8] = np.nan # we'll deal with that later...
                if len(data.shape)==1:
                    data = np.array([np.array([d]) for d in data])
                err = hdu.data['T3PHIERR']*np.pi/180
                if len(err.shape)==1:
                    err = np.array([np.array([e]) for e in err])
                if 'AMBER' in ins:
                    print(' | !!AMBER: rejecting CP WL<%3.1fum'%amberWLmin)
                    print(' | !!AMBER: rejecting CP WL<%3.1fum'%amberWLmax)
                    wl = hdu.data['MJD'][:,None]*0+self.wavel[ins][None,:]
                    data[wl<amberWLmin] = np.nan
                    data[wl>amberWLmax] = np.nan
                    for b in amberAtmBand:
                        data[np.abs(wl-b)<0.1] = np.nan

                if not reducePoly is None:
                    p = []
                    ep = []
                    for i in range(len(data)):
                        if all(np.isnan(data[i])) or all(np.isnan(err[i])):
                            tmp = {'A'+str(j):np.nan for j in range(reducePoly+1)},\
                                {'A'+str(j):np.nan for j in range(reducePoly+1)}
                        else:
                            tmp = _decomposeObs(self.wavel[ins], data[i], err[i], reducePoly)
                        p.append(tmp[0])
                        ep.append(tmp[1])
                    # -- each order:
                    for j in range(reducePoly+1):
                        self._rawData.append(['cp_%d_%d;'%(j,reducePoly)+ins,
                          hdu.data['U1COORD'],
                          hdu.data['V1COORD'],
                          hdu.data['U2COORD'],
                          hdu.data['V2COORD'],
                          self.wavel_3m[ins],
                          hdu.data['MJD'],
                          np.array([x['A'+str(j)] for x in p]),
                          np.array([x['A'+str(j)] for x in ep])])

                if np.sum(np.isnan(data))<data.size:
                    # self._rawData.append(['cp;'+ins,
                    #       hdu.data['U1COORD'][:,None]+0*self.wavel[ins][None,:],
                    #       hdu.data['V1COORD'][:,None]+0*self.wavel[ins][None,:],
                    #       hdu.data['U2COORD'][:,None]+0*self.wavel[ins][None,:],
                    #       hdu.data['V2COORD'][:,None]+0*self.wavel[ins][None,:],
                    #       self.wavel[ins][None,:]+0*hdu.data['V1COORD'][:,None],
                    #       hdu.data['MJD'][:,None]+0*self.wavel[ins][None,:],
                    #       data, err])
                    # -- complex closure phase
                    self._rawData.append(['icp;'+ins,
                        hdu.data['U1COORD'][:,None]+0*self.wavel[ins][None,:],
                        hdu.data['V1COORD'][:,None]+0*self.wavel[ins][None,:],
                        hdu.data['U2COORD'][:,None]+0*self.wavel[ins][None,:],
                        hdu.data['V2COORD'][:,None]+0*self.wavel[ins][None,:],
                        self.wavel[ins][None,:]+0*hdu.data['V1COORD'][:,None],
                        hdu.data['MJD'][:,None]+0*self.wavel[ins][None,:],
                        np.exp(1j*data), err])
                else:
                    print(' > WARNING: no valid T3PHI values in this HDU')
                # -- T3
                data = hdu.data['T3AMP']
                data /= np.sqrt(self.v2bias)**3
                data[hdu.data['FLAG']] = np.nan # we'll deal with that later...
                data[hdu.data['T3AMPERR']>1e8] = np.nan # we'll deal with that later...
                if len(data.shape)==1:
                    data = np.array([np.array([d]) for d in data])
                err = hdu.data['T3AMPERR']
                if len(err.shape)==1:
                    err = np.array([np.array([e]) for e in err])
                if 'AMBER' in ins:
                    print(' | !!AMBER: rejecting T3 WL<%3.1fum'%amberWLmin)
                    print(' | !!AMBER: rejecting T3 WL>%3.1fum'%amberWLmax)
                    wl = hdu.data['MJD'][:,None]*0+self.wavel[ins][None,:]
                    data[wl<amberWLmin] = np.nan
                    data[wl>amberWLmax] = np.nan

                    for b in amberAtmBand:
                        data[np.abs(wl-b)<0.1] = np.nan

                if np.sum(np.isnan(data))<data.size:
                    self._rawData.append(['t3;'+ins,
                          hdu.data['U1COORD'][:,None]+0*self.wavel[ins][None,:],
                          hdu.data['V1COORD'][:,None]+0*self.wavel[ins][None,:],
                          hdu.data['U2COORD'][:,None]+0*self.wavel[ins][None,:],
                          hdu.data['V2COORD'][:,None]+0*self.wavel[ins][None,:],
                          self.wavel[ins][None,:]+0*hdu.data['V1COORD'][:,None],
                          hdu.data['MJD'][:,None]+0*self.wavel[ins][None,:],
                          data, err])
                else:
                    print(' > WARNING: no valid T3AMP values in this HDU')
                Bmax = (hdu.data['U1COORD']**2+hdu.data['V1COORD']**2).max()
                Bmax = max(Bmax, (hdu.data['U2COORD']**2+hdu.data['V2COORD']**2).max())
                Bmax = max(Bmax, ((hdu.data['U1COORD']+hdu.data['U2COORD'])**2
                                   +(hdu.data['U1COORD']+hdu.data['V2COORD'])**2).max())
                Bmax = np.sqrt(Bmax)
                maxRes = max(maxRes, Bmax/self.wavel[ins].min())
                self.smearFov = min(self.smearFov, 2*self.wavel[ins].min()**2/self.dwavel[ins]/Bmax*180*3.6/np.pi)
                self.diffFov = min(self.diffFov, 1.2*self.wavel[ins].min()/self.telArray[arr]*180*3.6/np.pi)
            if hdu.header['EXTNAME']=='OI_VIS2' and testInst(hdu):
                data = hdu.data['VIS2DATA']
                if len(data.shape)==1:
                    data = np.array([np.array([d]) for d in data])
                err = hdu.data['VIS2ERR']
                if len(err.shape)==1:
                    err = np.array([np.array([e]) for e in err])
                data[hdu.data['FLAG']] = np.nan # we'll deal with that later...

                data /= self.v2bias

                if 'AMBER' in ins:
                    print(' | !!AMBER: rejecting V2 WL<%3.1fum'%amberWLmin)
                    print(' | !!AMBER: rejecting V2 WL>%3.1fum'%amberWLmax)
                    wl = hdu.data['MJD'][:,None]*0+self.wavel[ins][None,:]
                    data[wl<amberWLmin] = np.nan
                    data[wl>amberWLmax] = np.nan
                    print(' | !!AMBER: rejecting bad V2 (<<0 or err too large):', end=' ')
                    data[data<-3*hdu.data['VIS2ERR']] = np.nan
                    print(np.sum(hdu.data['VIS2ERR']>np.abs(data)))
                    data[hdu.data['VIS2ERR']>0.5*np.abs(data)] = np.nan
                    for b in amberAtmBand:
                        data[np.abs(wl-b)<0.1] = np.nan
                if not reducePoly is None:
                    p = []
                    ep = []
                    for i in range(len(data)):
                        if all(np.isnan(data[i])) or all(np.isnan(hdu.data['VIS2ERR'][i])):
                            tmp = {'A'+str(j):np.nan for j in range(reducePoly+1)},\
                                {'A'+str(j):np.nan for j in range(reducePoly+1)}
                        else:
                            tmp = _decomposeObs(self.wavel[ins], data[i], hdu.data['VIS2ERR'][i], reducePoly)
                        p.append(tmp[0])
                        ep.append(tmp[1])
                    # -- each order:
                    for j in range(reducePoly+1):
                        self._rawData.append(['v2_%d_%d;'%(j,reducePoly)+ins,
                          hdu.data['UCOORD'],
                          hdu.data['VCOORD'],
                          self.wavel_3m[ins],
                          hdu.data['MJD'],
                          np.array([x['A'+str(j)] for x in p]),
                          np.array([x['A'+str(j)] for x in ep])])

                self._rawData.append(['v2;'+ins,
                      hdu.data['UCOORD'][:,None]+0*self.wavel[ins][None,:],
                      hdu.data['VCOORD'][:,None]+0*self.wavel[ins][None,:],
                      self.wavel[ins][None,:]+0*hdu.data['VCOORD'][:,None],
                      hdu.data['MJD'][:,None]+0*self.wavel[ins][None,:],
                      data, err])

                Bmax = (hdu.data['UCOORD']**2+hdu.data['VCOORD']**2).max()
                Bmax = np.sqrt(Bmax)
                maxRes = max(maxRes, Bmax/self.wavel[ins].min())
                self.smearFov = min(self.smearFov, self.wavel[ins].min()**2/self.dwavel[ins]/Bmax*180*3.6/np.pi)
                self.diffFov = min(self.diffFov, 1.2*self.wavel[ins].min()/self.telArray[arr]*180*3.6/np.pi)

        self.minSpatialScale = min(1e-6/maxRes*180*3600*1000/np.pi, self.minSpatialScale)
        print(' | Smallest spatial scale:    %7.2f mas'%(self.minSpatialScale))
        print(' | Diffraction Field of view: %7.2f mas'%(self.diffFov))
        print(' | WL Smearing Field of view: %7.2f mas'%(self.smearFov))
        self._fitsHandler.close()
        return
    def _compute_delta(self):
        # -- compute a flatten version of all V2:
        allV2 = {'u':np.array([]), 'v':np.array([]), 'mjd':np.array([]),
                 'wl':np.array([]), 'v2':np.array([])}
        for r in filter(lambda x: x[0].split(';')[0]=='v2', self._rawData):
            allV2['u'] = np.append(allV2['u'], r[1].flatten())
            allV2['v'] = np.append(allV2['v'], r[2].flatten())
            allV2['wl'] = np.append(allV2['wl'], r[3].flatten())
            allV2['mjd'] = np.append(allV2['mjd'], r[4].flatten())
            allV2['v2'] = np.append(allV2['v2'], r[-2].flatten())

        # -- delta for approximation, very long!
        for r in self._rawData:
            if r[0].split(';')[0] in ['cp', 't3', 'icp']:
                # -- this will contain the delta for this r
                vis1, vis2, vis3 = np.zeros(r[-2].shape), np.zeros(r[-2].shape), np.zeros(r[-2].shape)
                for i in range(r[-2].shape[0]):
                    for j in range(r[-2].shape[1]):
                        # -- find Vis
                        k1 = np.argmin((allV2['u']-r[1][i,j])**2+
                                       (allV2['v']-r[2][i,j])**2+
                                       (allV2['wl']-r[5][i,j])**2+
                                       ((allV2['mjd']-r[6][i,j])/10000.)**2)
                        if not np.isnan(r[-2][i,j]):
                            vis1[i,j] = np.sqrt(allV2['v2'][k1])
                        k2 = np.argmin((allV2['u']-r[3][i,j])**2+
                                       (allV2['v']-r[4][i,j])**2+
                                       (allV2['wl']-r[5][i,j])**2+
                                       ((allV2['mjd']-r[6][i,j])/10000.)**2)
                        if not np.isnan(r[-2][i,j]):
                            vis2[i,j] = np.sqrt(allV2['v2'][k2])
                        k3 = np.argmin((allV2['u']-r[1][i,j]-r[3][i,j])**2+
                                       (allV2['v']-r[2][i,j]-r[4][i,j])**2+
                                       (allV2['wl']-r[5][i,j])**2+
                                       ((allV2['mjd']-r[6][i,j])/10000.)**2)
                        if not np.isnan(r[-2][i,j]):
                            vis3[i,j] = np.sqrt(allV2['v2'][k3])
                self._delta.append((vis1, vis2, vis3))
            if r[0].split(';')[0] == 'v2':
                self._delta.append(None)
        return
    def estimateCorrSpecChannels(self, verbose=False):
        """
        estimate correlation between spectral channels
        """
        self.corrSC = []
        if verbose:
            print(' | Estimating correlations in error bars within spectral bands (E**2 == E_stat**2 + E_syst**2):')
        for d in self._rawData:
            tmp = []
            #print(d[0], d[-1].shape)
            for k in range(d[-1].shape[0]):
                n = min(d[-1].shape[1]-2, 2)
                #print(' | ', k, 'Sys / Err =',end=' ')
                model = _dpfit_leastsqFit(_dpfit_polyN, d[-4][k],
                                          {'A'+str(i):0.0 for i in range(n)},
                                          d[-2][k], d[-1][k])['model']
                tmp.append( _estimateCorrelation(d[-2][k], d[-1][k], model))
            self.corrSC.append(tmp)
            if verbose:
                print('   ', d[0], '<E_syst / E_stat> = %4.2f'%(np.mean(tmp)))
        return
    def _estimateNsmear(self):
        _meas, _errs, _uv, _types, _wl = _generateFitData(self._rawData,
                                                          self.observables,
                                                          self.instruments)
        # -- dwavel:
        _dwavel = np.array([])
        for c in self._rawData:
            if c[0].split(';')[0] in self.observables and \
                c[0].split(';')[1] in self.instruments:
                _dwavel = np.append(_dwavel, np.ones(c[-2].shape).flatten()*
                                        self.dwavel[c[0].split(';')[1]])
        res = (_uv*self.rmax/(_wl-0.5*_dwavel)-_uv*self.rmax/(_wl+0.5*_dwavel))*0.004848136
        #print('DEBUG:', res)
        CONFIG['Nsmear'] = max(int(np.ceil(4*res.max())), 3)
        print(' | setting up Nsmear = %d'%CONFIG['Nsmear'])
        return

    def ndata(self):
        tmp = 0
        nan = 0
        for i, c in enumerate(self._chi2Data):
            if c[0].split(';')[0] in self.observables and\
               c[0].split(';')[1] in self.instruments:
                tmp += len(c[-1].flatten())
                nan += np.sum(1-(1-np.isnan(c[-1]))*(1-np.isnan(c[-2])))
        return int(tmp-nan)

    def close(self):
        self._fitsHandler.close()
        return
    def _pool(self):
        if CONFIG['Ncores'] is None:
            self.Ncores = max(multiprocessing.cpu_count(),1)
        else:
            self.Ncores = min(multiprocessing.cpu_count(), CONFIG['Ncores'])
        if self.Ncores==1:
            return None
        else:
            return multiprocessing.Pool(self.Ncores)
    def _estimateRunTime(self, function, params):
        # -- estimate how long it will take, in two passes
        p = self._pool()
        t = time.time()
        if p is None:
            # -- single thread:
            for m in params:
                apply(function, m)
        else:
            # -- multithreaded:
            for m in params:
                p.apply_async(function, m)
            p.close()
            p.join()
        return (time.time()-t)/len(params)
    def _cb_chi2Map(self, r):
        """
        callback function for chi2Map()
        """
        try:
            self.mapChi2[r[1], r[0]] = r[2]
            # -- completed / to be computed
            f = np.sum(self.mapChi2>0)/float(np.sum(self.mapChi2>=0))
            if f>self._prog and CONFIG['progress bar']:
                n = int(50*f)
                print('\033[F',end =' ')
                print('|'+'='*(n+1)+' '*(50-n)+'|', end=' ')
                print('%2d%%'%(int(100*f)), end=' ')
                self._progTime[1] = time.time()
                print('%3d s remaining'%(int((self._progTime[1]-self._progTime[0])/f*(1-f))))
                self._prog = max(self._prog+0.01, f+0.01)
        except:
            print('did not work')
        return

    def chi2Map(self, step=None, fratio=None, addCompanion=None, removeCompanion=None, fig=0, diam=None, rmin=None, rmax=None):
        """
        Performs a chi2 map between rmin and rmax (should be defined) with step
        "step". The diameter is taken as the best fit UD diameter (biased if
        the data indeed contain a companion!).

        If 'addCompanion' or 'removeCompanion' are defined, a companion will
        be analytically added or removed from the data. define the companion
        as {'x':mas, 'y':mas, 'f':fratio in %}. 'f' will be forced to be positive.
        """
        if step is None:
            step = 1/5. * self.minSpatialScale
            print(' | step= not given, using 1/5 X smallest spatial scale = %4.2f mas'%step)

        #--
        if rmin is None:
            self.rmin = self.minSpatialScale
            print(" | rmin= not given, set to smallest spatial scale: rmin=%5.2f mas"%(self.rmin))
        else:
            self.rmin = rmin

        if rmax is None:
            self.rmax = 1.2*self.smearFov
            print(" | rmax= not given, set to 1.2*Field of View: rmax=%5.2f mas"%(self.rmax))
        else:
            self.rmax = rmax
        self._estimateNsmear()

        try:
            N = int(np.ceil(2*self.rmax/step))
        except:
            print('ERROR: you should define rmax first!')
            return
        if fratio is None:
            print(' | fratio= not given -> using 0.01 (==1%)')
            fratio = 1.0
        print(' | observables:', self.observables, 'from', self.ALLobservables)
        print(' | instruments:', self.instruments, 'from', self.ALLinstruments)

        self._chi2Data = self._copyRawData()

        if not addCompanion is None:
            tmp = {k:addCompanion[k] for k in addCompanion.keys()}
            tmp['f'] = np.abs(tmp['f'])
            self._chi2Data = _injectCompanionData(self._chi2Data, self._delta, tmp)
        if not removeCompanion is None:
            tmp = {k:removeCompanion[k] for k in removeCompanion.keys()}
            tmp['f'] = -np.abs(tmp['f'])
            self._chi2Data = _injectCompanionData(self._chi2Data, self._delta, tmp)

        # -- NO! the fitUD should not fix the diameter, the binary fit should:
        #self.fitUD(forcedDiam=diam)
        self.fitUD()

        # -- prepare the grid
        allX = np.linspace(-self.rmax, self.rmax, N)
        allY = np.linspace(-self.rmax, self.rmax, N)
        self.mapChi2 = np.zeros((N,N))
        self.mapChi2[allX[None,:]**2+allY[:,None]**2 > self.rmax**2] = -1
        self.mapChi2[allX[None,:]**2+allY[:,None]**2 < self.rmin**2] = -1
        self._prog = 0.0
        self._progTime = [time.time(), time.time()]

        # -- parallel treatment:
        print(' | Computing Map %dx%d'%(N, N), end=' ')
        if not CONFIG['long exec warning'] is None:
            # -- estimate how long it will take, in two passes
            params, Ntest = [], 20*max(multiprocessing.cpu_count()-1,1)
            for i in range(Ntest):
                o = np.random.rand()*2*np.pi
                tmp = {'x':np.cos(o)*(self.rmax+self.rmin),
                          'y':np.sin(o)*(self.rmax+self.rmin),
                          'f':1.0, 'diam*':self.diam, 'alpha*':self.alpha}
                for _k in self.dwavel.keys():
                    tmp['dwavel;'+_k] = self.dwavel[_k]
                params.append((tmp,self._chi2Data, self.observables, self.instruments))
            est = self._estimateRunTime(_chi2Func, params)
            est *= np.sum(self.mapChi2>=0)
            print('... it should take about %d seconds'%(int(est)))
            if not CONFIG['long exec warning'] is None and\
                 est>CONFIG['long exec warning']:
                print(" > WARNING: this will take too long. ")
                print(" | Increase CONFIG['long exec warning'] if you want to run longer computations.")
                print(" | e.g. "+__name__+".CONFIG['long exec warning'] = %d"%int(1.2*est))
                print(" | set it to None and the warning will disapear... at your own risks!")
                return
        print('')
        # -- done estimating time

        # -- compute actual grid:
        p = self._pool()

        for i,x in enumerate(allX):
            for j,y in enumerate(allY):
                if self.mapChi2[j,i]==0:
                    params = {'x':x, 'y':y, 'f':fratio, 'diam*':self.diam,
                              '_i':i, '_j':j, 'alpha*':self.alpha}
                    for _k in self.dwavel.keys():
                        params['dwavel;'+_k] = self.dwavel[_k]

                    if p is None:
                        # -- single thread:
                        self._cb_chi2Map(_chi2Func(params, self._chi2Data,
                                                    self.observables, self.instruments))
                    else:
                        # -- multi-thread:
                        p.apply_async(_chi2Func, (params, self._chi2Data,
                                    self.observables, self.instruments),
                                    callback=self._cb_chi2Map)
        if not p is None:
            p.close()
            p.join()

        # -- take care of unfitted zone, for esthetics
        self.mapChi2[self.mapChi2<=0] = self.chi2_UD
        X, Y = np.meshgrid(allX, allY)
        x0 = X.flatten()[np.argmin(self.mapChi2.flatten())]
        y0 = Y.flatten()[np.argmin(self.mapChi2.flatten())]
        s0 = _nSigmas(self.chi2_UD,
                       np.minimum(np.min(self.mapChi2), self.chi2_UD),
                       self.ndata())
        print(' | chi2 Min: %5.3f'%(np.min(self.mapChi2)))
        print(' | at X,Y  : %6.2f, %6.2f mas'%(x0, y0))
        #print(' | Nsigma  : %5.2f'%s0)
        print(' | NDOF=%d'%( self.ndata()-1),end=' ')
        print(' | n sigma detection: %5.2f (fully uncorrelated errors)'%s0)

        plt.close(fig)
        if CONFIG['suptitle']:
            plt.figure(fig, figsize=(12/1.2,5.8/1.2))
            plt.subplots_adjust(top=0.78, bottom=0.08,
                                left=0.08, right=0.99, wspace=0.10)
            title = "CANDID: $\chi^2$ Map for f$_\mathrm{ratio}$=%3.1f%% and "%(fratio)
            if self.ediam>0:
                title += r'fitted $\theta_\mathrm{UD}=%4.3f$ mas.'%(self.diam)
            else:
                title += r'fixed $\theta_\mathrm{UD}=%4.3f$ mas.'%(self.diam)
            title += ' Using '+', '.join(self.observables)
            title += '\nfrom '+', '.join(self.instruments)
            title += '\n'+self.titleFilename
            plt.suptitle(title, fontsize=14, fontweight='bold')
        else:
            plt.figure(fig, figsize=(12/1.2,5.4/1.2))
            plt.subplots_adjust(top=0.88, bottom=0.10,
                                left=0.08, right=0.99, wspace=0.10)

        ax1 = plt.subplot(121)
        if CONFIG['chi2 scale']=='log':
            tit = 'log10[$\chi^2_\mathrm{BIN}/\chi^2_\mathrm{UD}$] with $\chi^2_\mathrm{UD}$='
            plt.pcolormesh(X, Y, np.log10(self.mapChi2/self.chi2_UD), cmap=CONFIG['color map'])
        else:
            tit = '$\chi^2_\mathrm{BIN}/\chi^2_\mathrm{UD}$ with $\chi^2_\mathrm{UD}$='
            plt.pcolormesh(X, Y, self.mapChi2/self.chi2_UD, cmap=CONFIG['color map'])
        if self.chi2_UD>0.05:
            tit += '%4.2f'%(self.chi2_UD)
        else:
            tit += '%4.2e'%(self.chi2_UD)

        plt.title(tit)

        plt.colorbar(format='%0.2f')
        plt.xlabel(r'E $\leftarrow\, \Delta \alpha$ (mas)')
        plt.ylabel(r'$\Delta \delta\, \rightarrow$ N (mas)')
        plt.xlim(self.rmax, -self.rmax)
        plt.ylim(-self.rmax, self.rmax)

        plt.subplot(122, sharex=ax1, sharey=ax1)
        plt.title('detection ($\sigma$)')
        plt.pcolormesh(X, Y,
                       _nSigmas(self.chi2_UD,
                                np.minimum(self.mapChi2, self.chi2_UD),
                                self.ndata()),
                       cmap=CONFIG['color map'])
        plt.colorbar(format='%0.2f')
        plt.xlabel(r'E $\leftarrow\, \Delta \alpha$ (mas)')

        plt.xlim(self.rmax, -self.rmax)
        plt.ylim(-self.rmax, self.rmax)
        ax1.set_aspect('equal', 'datalim')
        # -- make a crosshair around the minimum:
        plt.plot([x0, x0], [y0-0.05*self.rmax, y0-0.1*self.rmax], '-r', alpha=0.5, linewidth=2)
        plt.plot([x0, x0], [y0+0.05*self.rmax, y0+0.1*self.rmax], '-r', alpha=0.5, linewidth=2)
        plt.plot([x0-0.05*self.rmax, x0-0.1*self.rmax], [y0, y0], '-r', alpha=0.5, linewidth=2)
        plt.plot([x0+0.05*self.rmax, x0+0.1*self.rmax], [y0, y0], '-r', alpha=0.5, linewidth=2)
        plt.text(0.9*x0, 0.9*y0, r'n$\sigma$=%3.1f'%s0, color='r')
        return
    def _cb_fitFunc(self, r):
        """
        callback function for fitMap
        """
        # try:
        if '_k' in r.keys():
            self.allFits[r['_k']] = r
            f = np.sum([0 if a=={} else 1 for a in self.allFits])/float(self.Nfits)
            if f>self._prog and CONFIG['progress bar']:
                n = int(50*f)
                print('\033[F',end=' ')
                print('|'+'='*(n+1)+' '*(50-n)+'|', end=' ')
                print('%3d%%'%(int(100*f)),end=' ')
                self._progTime[1] = time.time()
                print('%3d s remaining'%(int((self._progTime[1]-self._progTime[0])/f*(1-f))))
                self._prog = max(self._prog+0.01, f+0.01)
        else:
            print('!!! r should have key "_k"')
        # except:
        #     print('!!! I expect a dict!')
        return

    def fitMap(self, step=None,  fig=1, addfits=False, addCompanion=None,
               removeCompanion=None, rmin=None, rmax=None, fratio=2.0,
               doNotFit=[], addParam={}):
        """
        - filename: a standard OIFITS data file
        - N: starts fits on a NxN grid
        - rmax: size of the grid in mas (20 will search between -20mas to +20mas)
        - rmin: do not look into the inner radius, in mas
        - fig=0: the figure number (default 0)
        - addfits: add the fit and fimap in the FITS file, as a binary table
        """
        if step is None:
            step = np.sqrt(2)*self.minSpatialScale
            print(' | step= not given, using sqrt(2) x smallest spatial scale = %4.2f mas'%step)

        if rmin is None:
            self.rmin = self.minSpatialScale
            print(" | rmin= not given, set to smallest spatial scale: rmin=%5.2f mas"%(self.rmin))
        else:
            self.rmin = rmin

        if rmax is None:
            self.rmax = 1.2*self.smearFov
            print(" | rmax= not given, set to 1.2*Field of View: rmax=%5.2f mas"%(self.rmax))
        else:
            self.rmax = rmax
        self._estimateNsmear()
        try:
            N = int(np.ceil(2*self.rmax/step))
        except:
            print('ERROR: you should define rmax first!')
        #self.rmin = max(step, self.rmin)
        print(' | observables:', self.observables, 'from', self.ALLobservables)
        print(' | instruments:', self.instruments, 'from', self.ALLinstruments)

        # -- start with all data
        self._chi2Data = self._copyRawData()

        if not removeCompanion is None:
            tmp = {k:removeCompanion[k] for k in removeCompanion.keys()}
            tmp['f'] = -np.abs(tmp['f'])
            self._chi2Data = _injectCompanionData(self._chi2Data, self._delta, tmp)
            if 'diam*' in removeCompanion.keys():
                self.diam = removeCompanion['diam*']
        if not addCompanion is None:
            tmp = {k:addCompanion[k] for k in addCompanion.keys()}
            tmp['f'] = np.abs(tmp['f'])
            self._chi2Data = _injectCompanionData(self._chi2Data, self._delta, tmp)
            if 'diam*' in addCompanion.keys():
                self.diam = addCompanion['diam*']

        if self._dataheader!={}:
            print('found injected companion at', self._dataheader)

        #print(' | Preliminary analysis')
        self.fitUD()
        if self.chi2_UD ==0:
            print(' >>> chi2_UD==0 ???, setting it to 1.')
            self.chi2_UD =1
        X = np.linspace(-self.rmax, self.rmax, N)
        Y = np.linspace(-self.rmax, self.rmax, N)
        self.allFits, self._prog = [{} for k in range(N*N)], 0.0
        self._progTime = [time.time(), time.time()]
        self.Nfits = np.sum((X[:,None]**2+Y[None,:]**2>=self.rmin**2)*
                            (X[:,None]**2+Y[None,:]**2<=self.rmax**2))

        print(' | Grid Fitting %dx%d:'%(N, N),end=' ')
        # -- estimate how long it will take, in two passes
        if not CONFIG['long exec warning'] is None:
            params, Ntest = [], 2*max(multiprocessing.cpu_count()-1,1)
            for i in range(Ntest):
                o = np.random.rand()*2*np.pi
                tmp = {'x':np.cos(o)*(self.rmax+self.rmin),
                          'y':np.sin(o)*(self.rmax+self.rmin),
                          'f':fratio, 'diam*':self.diam, 'alpha*':self.alpha}
                for _k in self.dwavel.keys():
                    tmp['dwavel;'+_k] = self.dwavel[_k]
                params.append((tmp, self._chi2Data, self.observables,
                                self.instruments))
            est = self._estimateRunTime(_fitFunc, params)
            est *= self.Nfits
            print('... it should take about %d seconds'%(int(est)))
            if not CONFIG['long exec warning'] is None and\
                 est>CONFIG['long exec warning']:
                print(" > WARNING: this will take too long. ")
                print(" | Increase CONFIG['long exec warning'] if you want to run longer computations.")
                print(" | e.g. "+__name__+".CONFIG['long exec warning'] = %d"%int(1.2*est))
                print(" | set it to 'None' and the warning will disapear... at your own risks!")
                return
        print('')
        # -- parallel on N-1 cores
        p = self._pool()
        k = 0
        #t0 = time.time()
        params = []
        for y in Y:
            for x in X:
                if x**2+y**2>=self.rmin**2 and x**2+y**2<=self.rmax**2:
                    tmp={'diam*': 0.0, 'f':fratio, 'x':x, 'y':y, '_k':k,
                         'alpha*':self.alpha}
                    for _k in self.dwavel.keys():
                        tmp['dwavel;'+_k] = self.dwavel[_k]
                    tmp.update(addParam)
                    params.append(tmp)
                    #print(tmp, doNotFit)
                    if p is None:
                        # -- single thread:
                        self._cb_fitFunc(_fitFunc(params[-1], self._chi2Data,
                                                  self.observables, self.instruments,
                                                  None, doNotFit))
                    else:
                        # -- multiple threads:
                        p.apply_async(_fitFunc, (params[-1], self._chi2Data,
                                                 self.observables, self.instruments,
                                                 None, doNotFit),
                                                 callback=self._cb_fitFunc)
                    k += 1
        if not p is None:
            p.close()
            p.join()

        print(' | Computing map of interpolated Chi2 minima')
        # -- last one to be ignored?
        self.allFits = self.allFits[:k-1]

        for i, f in enumerate(self.allFits):
            f['best']['f'] = np.abs(f['best']['f'])
            if 'diam*' in f['best'].keys():
                f['best']['diam*'] = np.abs(f['best']['diam*'])
            # -- distance from start to finish of the fit
            f['dist'] = np.sqrt((params[i]['x']-f['best']['x'])**2+
                                (params[i]['y']-f['best']['y'])**2)

        # -- count number of unique minima, start with first one, add N sigma:
        allMin = [self.allFits[0]]
        allMin[0]['nsigma'] =  _nSigmas(self.chi2_UD,  allMin[0]['chi2'], self.ndata()-1)
        for f in self.allFits:
            chi2 = []
            for a in allMin:
                tmp, n = 0., 0.
                for k in ['x', 'y', 'f', 'diam']:
                    if k in f['uncer'].keys() and k in a['uncer'].keys() and\
                            f['uncer'][k]!=0 and a['uncer'][k]!=0:
                        if k == 'f':
                            tmp += (f['best'][k]-a['best'][k])**2/(0.01)**2
                        elif k == 'diam':
                            tmp += (f['best'][k]-a['best'][k])**2/(0.1*self.minSpatialScale)**2
                        else: # -- for x and y
                            tmp += (f['best'][k]-a['best'][k])**2/(0.5*self.minSpatialScale)**2
                        n += 1.
                tmp /= n
                chi2.append(tmp)
            if not any([c<=1 for c in chi2]) and \
                    f['best']['x']**2+f['best']['y']**2>=self.rmin**2 and \
                    f['best']['x']**2+f['best']['y']**2<=self.rmax**2:
                allMin.append(f)
                allMin[-1]['nsigma'] = _nSigmas(self.chi2_UD,  allMin[-1]['chi2'], self.ndata()-1)
                try:
                    allMin[-1]['best']['f'] = np.abs(allMin[-1]['best']['f'])
                except:
                    pass

        # -- plot histogram of distance from start to end of fit:
        if False:
            plt.close(10+fig)
            plt.figure(10+fig)
            plt.hist([f['dist'] for f in self.allFits], bins=20, normed=1, label='all fits')
            plt.xlabel('distance start -> best fit (mas)')
            plt.vlines(2*self.rmax/float(N), 0, plt.ylim()[1], color='r', linewidth=2, label='grid size')
            plt.vlines(2*self.rmax/float(N)*np.sqrt(2)/2, 0, plt.ylim()[1], color='r', linewidth=2,
                       label=r'grid size $\sqrt(2)/2$', linestyle='dashed')
            plt.legend()
            plt.text(plt.xlim()[1]-0.03*(plt.xlim()[1]-plt.xlim()[0]),
                     plt.ylim()[1]-0.03*(plt.ylim()[1]-plt.ylim()[0]),
                     '# of unique minima=%d / grid size=%d'%(len(allMin),  N*N),
                     ha='right', va='top')

        # -- is the grid to wide?
        # -> len(allMin) == number of minima
        # -> distance of fit == number of minima
        print(' | %d individual minima for %d fits'%(len(allMin), len(self.allFits)))
        print(' | 10, 50, 90 percentiles for fit displacement: %3.1f, %3.1f, %3.1f mas'%(
                                np.percentile([f['dist'] for f in self.allFits], 10),
                                np.percentile([f['dist'] for f in self.allFits], 50),
                                np.percentile([f['dist'] for f in self.allFits], 90),))

        self.Nopt = self.rmax/np.nanmedian([f['dist'] for f in self.allFits])*np.sqrt(2)
        self.Nopt = max(np.sqrt(2*len(allMin)), self.Nopt)
        self.Nopt = int(np.ceil(self.Nopt))
        self.stepOptFitMap = 2*self.rmax/self.Nopt
        if len(allMin)/float(N*N)>0.6 or\
             2*self.rmax/float(N)*np.sqrt(2)/2 > 1.1*np.median([f['dist'] for f in self.allFits]):
            print(' > WARNING, grid is too wide!!!', end=' ')
            print('--> try step=%4.2fmas'%(2.*self.rmax/float(self.Nopt)))
            reliability = 'unreliable'
        elif N>1.2*self.Nopt:
            print(' | current grid step (%4.2fmas) is too fine!!!'%(2*self.rmax/float(N)),end=' ')
            print('--> %4.2fmas should be enough'%(2.*self.rmax/float(self.Nopt)))
            reliability = 'overkill'
        else:
            print(' | Grid has the correct steps of %4.2fmas, '%step,end=' ')
            print('optimimum step size found to be %4.2fmas'%(
                        2.*self.rmax/float(self.Nopt)))
            reliability = 'reliable'

        # == plot chi2 min map:
        # -- limited in 32 but, hacking something dirty:
        Nx = 2*len(X)+1
        Ny = 2*len(Y)+1
        if len(allMin)**2*Nx*Ny >= sys.maxsize:
            Nx = int(np.sqrt(sys.maxsize/(len(allMin)**2)))-1
            Ny = int(np.sqrt(sys.maxsize/(len(allMin)**2)))-1

        # print('### DEBUG:', sys.maxsize, len(allMin)**2*Nx*Ny,end=' ')
        # print(len(allMin)**2*Nx*Ny < sys.maxsize)

        _X, _Y = np.meshgrid(np.linspace(X[0]-np.diff(X)[0]/2.,
                                         X[-1]+np.diff(X)[0]/2., Nx),
                             np.linspace(Y[0]-np.diff(Y)[0]/2.,
                                         Y[-1]+np.diff(Y)[0]/2., Ny))
        print(' | Rbf interpolating: %d points -> %d pixels map'%(len(allMin), Nx*Ny))

        rbf = scipy.interpolate.Rbf([x['best']['x'] for x in allMin],
                                    [x['best']['y'] for x in allMin],
                                    [x['chi2'] for x in allMin],
                                    function='linear')
        _Z = rbf(_X, _Y)

        _Z[_X**2+_Y**2<self.rmin**2] = self.chi2_UD
        _Z[_X**2+_Y**2>self.rmax**2] = self.chi2_UD

        plt.close(fig)
        if CONFIG['suptitle']:
            plt.figure(fig, figsize=(12/1.2,5.5/1.2))
            plt.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.78,
                                wspace=0.2, hspace=0.2)
        else:
            plt.figure(fig, figsize=(12/1.2,5./1.2))
            plt.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.9,
                                wspace=0.2, hspace=0.2)

        title = "CANDID: companion search"
        title += ' using '+', '.join(self.observables)
        title += '\nfrom '+', '.join(self.instruments)
        title += '\n'+self.titleFilename
        if not removeCompanion is None:
            title += '\ncompanion removed at X=%3.2fmas, Y=%3.2fmas, F=%3.2f%%'%(
                    removeCompanion['x'], removeCompanion['y'], removeCompanion['f'])
        if CONFIG['suptitle']:
            plt.suptitle(title, fontsize=14, fontweight='bold')

        ax1 = plt.subplot(1,2,1)
        if CONFIG['chi2 scale']=='log':
            plt.title('log10[$\chi^2$ best fit / $\chi^2_{UD}$]')
            plt.pcolormesh(_X-0.5*self.rmax/float(N), _Y-0.5*self.rmax/float(N),
                           np.log10(_Z/self.chi2_UD), cmap=CONFIG['color map']+'_r')
        else:
            plt.title('$\chi^2$ best fit / $\chi^2_{UD}$')
            plt.pcolormesh(_X-0.5*self.rmax/float(N), _Y-0.5*self.rmax/float(N),
                           _Z/self.chi2_UD, cmap=CONFIG['color map']+'_r')

        plt.colorbar(format='%0.2f')
        if reliability=='unreliable':
            plt.text(0,0,'!! UNRELIABLE !!', color='r', size=30, alpha=0.5,
                     ha='center', va='center', rotation=45)
            plt.text(self.rmax/3,self.rmax/3,'!! UNRELIABLE !!', color='r', size=30, alpha=0.5,
                     ha='center', va='center', rotation=45)
            plt.text(-self.rmax/3,-self.rmax/3,'!! UNRELIABLE !!', color='r', size=30, alpha=0.5,
                     ha='center', va='center', rotation=45)
        for i, f in enumerate(self.allFits):
            plt.plot([f['best']['x'], params[i]['x']],
                     [f['best']['y'], params[i]['y']], '-y',
                     alpha=0.3, linewidth=2)
        plt.xlabel(r'E $\leftarrow\, \Delta \alpha$ (mas)')

        plt.ylabel(r'$\Delta \delta\, \rightarrow$ N (mas)')
        plt.xlim(self.rmax-0.5*self.rmax/float(N), -self.rmax+0.5*self.rmax/float(N))
        plt.ylim(-self.rmax+0.5*self.rmax/float(N), self.rmax-0.5*self.rmax/float(N))
        ax1.set_aspect('equal', 'datalim')

        # -- http://www.aanda.org/articles/aa/pdf/2011/11/aa17719-11.pdf section 3.2
        n_sigma = _nSigmas(self.chi2_UD, _Z, self.ndata()-1)

        ax2 = plt.subplot(1,2,2, sharex=ax1, sharey=ax1)
        if CONFIG['chi2 scale']=='log':
            plt.title('log10[n$\sigma$] of detection')
            plt.pcolormesh(_X-0.5*self.rmax/float(N), _Y-0.5*self.rmax/float(N),
                           np.log10(n_sigma), cmap=CONFIG['color map'], vmin=0)
        else:
            plt.title('n$\sigma$ of detection')
            plt.pcolormesh(_X-0.5*self.rmax/float(N), _Y-0.5*self.rmax/float(N),
                           n_sigma, cmap=CONFIG['color map'], vmin=0)
        plt.colorbar(format='%0.2f')

        # if self.rmin>0:
        #     c = plt.Circle((0,0), self.rmin, color='k', alpha=0.33, hatch='x')
        #     ax2.add_patch(c)
        ax2.set_aspect('equal', 'datalim')
        # -- invert X axis

        if reliability=='unreliable':
            plt.text(0,0,'!! UNRELIABLE !!', color='r', size=30, alpha=0.5,
                     ha='center', va='center', rotation=45)
            plt.text(self.rmax/2,self.rmax/2,'!! UNRELIABLE !!', color='r', size=30, alpha=0.5,
                     ha='center', va='center', rotation=45)
            plt.text(-self.rmax/2,-self.rmax/2,'!! UNRELIABLE !!', color='r', size=30, alpha=0.5,
                     ha='center', va='center', rotation=45)

        # --
        nsmax = np.max([a['nsigma'] for a in allMin])
        # -- keep nSigma higher than half the max
        allMin2 = list(filter(lambda x: x['nsigma']>nsmax/2. and a['best']['x']**2+a['best']['y']**2 >= self.rmin**2, allMin))
        # -- keep 5 highest nSigma
        allMin2 = [allMin[i] for i in np.argsort([c['chi2'] for c in allMin])[:5]]
        # -- keep highest nSigma
        allMin2 = [allMin[i] for i in np.argsort([c['chi2'] for c in allMin])[:1]]

        for ii, i in enumerate(np.argsort([x['chi2'] for x in allMin2])):
            print(' | BEST FIT %d: chi2=%5.2f'%(ii, allMin2[i]['chi2']))
            allMin2[i]['best']['f'] = np.abs(allMin2[i]['best']['f'] )
            keys = allMin2[i]['best'].keys()
            keys.remove('_k')
            for _k in keys:
                if _k.startswith('dwavel'):
                    keys.remove(_k)
            keys = sorted(keys)

            for s in keys:
                if s in allMin2[i]['uncer'].keys() and allMin2[i]['uncer'][s]>0:
                    print(' | %6s='%s, '%8.4f +- %6.4f [%s]'%(allMin2[i]['best'][s],
                                                     allMin2[i]['uncer'][s], paramUnits(s)))
                else:
                    print(' | %6s='%s, '%8.4f [%s]'%(allMin2[i]['best'][s], paramUnits(s)))



            # -- http://www.aanda.org/articles/aa/pdf/2011/11/aa17719-11.pdf section 3.2
            print(' | chi2r_UD=%4.2f, chi2r_BIN=%4.2f, NDOF=%d'%(self.chi2_UD, allMin2[i]['chi2'], self.ndata()-1),end=' ')
            print('-> n sigma: %5.2f (assumes uncorr data)'%allMin2[i]['nsigma'])

            x0 = allMin2[i]['best']['x']
            ex0 = allMin2[i]['uncer']['x']
            y0 = allMin2[i]['best']['y']
            ey0 = allMin2[i]['uncer']['y']
            f0 = np.abs(allMin2[i]['best']['f'])
            ef0 = allMin2[i]['uncer']['f']
            s0 = allMin2[i]['nsigma']

            # -- draw cross hair
            for ax in [ax1, ax2]:
                ax.plot([x0, x0], [y0-0.05*self.rmax, y0-0.1*self.rmax], '-r',
                        alpha=0.5, linewidth=2)
                ax.plot([x0, x0], [y0+0.05*self.rmax, y0+0.1*self.rmax], '-r',
                        alpha=0.5, linewidth=2)
                ax.plot([x0-0.05*self.rmax, x0-0.1*self.rmax], [y0, y0], '-r',
                        alpha=0.5, linewidth=2)
                ax.plot([x0+0.05*self.rmax, x0+0.1*self.rmax], [y0, y0], '-r',
                        alpha=0.5, linewidth=2)
            ax2.text(0.9*x0, 0.9*y0, r'%3.1f$\sigma$'%s0, color='r')

        plt.xlabel(r'E $\leftarrow\, \Delta \alpha$ (mas)')
        plt.ylabel(r'$\Delta \delta\, \rightarrow$ N (mas)')
        plt.xlim(self.rmax-self.rmax/float(N), -self.rmax+self.rmax/float(N))
        plt.ylim(-self.rmax+self.rmax/float(N), self.rmax-self.rmax/float(N))
        ax2.set_aspect('equal', 'datalim')

        # -- best fit parameters:
        j = np.argmin([x['chi2'] for x in self.allFits if x['best']['x']**2+x['best']['y']**2>=self.rmin**2])
        self.bestFit = [x for x in self.allFits if x['best']['x']**2+x['best']['y']**2>=self.rmin**2][j]
        self.bestFit['best'].pop('_k')
        self.bestFit['nsigma'] = _nSigmas(self.chi2_UD,  self.bestFit['chi2'], self.ndata()-1)
        self.plotModel(fig=fig+1)

        # -- compare with injected companion, if any
        if 'X' in self._dataheader.keys() and 'Y' in self._dataheader.keys() and 'F' in self._dataheader.keys():
            print('injected X:', self._dataheader['X'], 'found at %3.1f sigmas'%((x0-self._dataheader['X'])/ex0))
            print('injected Y:', self._dataheader['Y'], 'found at %3.1f sigmas'%((y0-self._dataheader['Y'])/ey0))
            print('injected F:', self._dataheader['F'], 'found at %3.1f sigmas'%((f0-self._dataheader['F'])/ef0))
            ax1.plot(self._dataheader['X'], self._dataheader['Y'], 'py', markersize=12, alpha=0.3)

        # -- do an additional fit by fitting also the bandwidth smearing
        if False:
            param = self.bestFit['best']
            fitAlso = list(filter(lambda x: x.startswith('dwavel'), param))
            print(' | fixed bandwidth parameters for the map (in um):')
            for s in fitAlso:
                print(' | %s = '%s, param[s])
            print('*'*67)
            print('*'*3, 'do an additional fit by fitting also the bandwidth smearing', '*'*3)
            print('*'*67)

            fit = _fitFunc(param, self._chi2Data, self.observables,
                            self.instruments,fitAlso)
            print('  > chi2 = %5.3f'%fit['chi2'])
            tmp = ['x', 'y', 'f', 'diam*']
            tmp.extend(fitAlso)
            for s in tmp:
                print(' | %5s='%s, '%8.4f +- %6.4f %s'%(fit['best'][s], fit['uncer'][s], paramUnits(s)))
        return

    def fitBoot(self, N=None, param=None, fig=2, fitAlso=None, doNotFit=[], useMJD=True,
                monteCarlo=False, corrSpecCha=None, nSigmaClip=4.5, addCompanion=None,
                removeCompanion=None):
        """
        boot strap fitting around a single position. By default,
        the position is the best position found by fitMap. It can also been entered
        as "param=" using a dictionnary

        use "fitAlso" to list additional parameters you want to fit,
        channel spectra width for instance. fitAlso has to be a list!

        - useMJD bootstraps on the dates, in addition to the spectrla channels. This is enabled by default.

        - monteCarlo: uses a monte Carlo approches, rather than a statistical resampling:
            all data are considered but with random errors added. An additional dict can be added to describe
        """
        if param is None:
            try:
                param = self.bestFit['best']
                print(' | using best solution from last search (fitMap or chi2Map)')
            except:
                print(' | ERROR: please set param= do the initial conditions (or run fitMap)')
                print(" | param={'x':, 'y':, 'f':, 'diam*':}")
                return
        try:
            for k in ['x', 'y', 'f', 'diam*']:
                if not k in param.keys():
                    print(' | ERROR: please set param= do the initial conditions (or run fitMap)')
                    print(" | param={'x':, 'y':, 'f':, 'diam*':}")
                    return
        except:
            pass
        if N is None:
            print(" | 'N=' not given, setting it to Ndata/2")
            N = max(self.ndata()/2, 100)

        self._chi2Data = self._copyRawData()

        if not addCompanion is None:
            tmp = {k:addCompanion[k] for k in addCompanion.keys()}
            tmp['f'] = np.abs(tmp['f'])
            self._chi2Data = _injectCompanionData(self._chi2Data, self._delta, tmp)
        if not removeCompanion is None:
            tmp = {k:removeCompanion[k] for k in removeCompanion.keys()}
            tmp['f'] = -np.abs(tmp['f'])
            self._chi2Data = _injectCompanionData(self._chi2Data, self._delta, tmp)

        print(" | running N=%d fit"%N, end=' ')
        self._progTime = [time.time(), time.time()]
        self.allFits, self._prog = [{} for k in range(N)], 0.0
        self.Nfits = N
        # -- estimate how long it will take, in two passes
        if not CONFIG['long exec warning'] is None:
            params, Ntest = [], 2*max(multiprocessing.cpu_count()-1,1)
            for i in range(Ntest):
                tmp = {k:param[k] for k in param.keys()}
                tmp['_k'] = i
                for _k in self.dwavel.keys():
                    tmp['dwavel;'+_k] = self.dwavel[_k]
                params.append((tmp, self._chi2Data, self.observables,
                                self.instruments))
            est = self._estimateRunTime(_fitFunc, params)
            est *= self.Nfits
            print('... it should take about %d seconds'%(int(est)))
            self.allFits, self._prog = [{} for k in range(N)], 0.0
            self._progTime = [time.time(), time.time()]
            if not CONFIG['long exec warning'] is None and\
                 est>CONFIG['long exec warning']:
                print(" > WARNING: this will take too long. ")
                print(" | Increase CONFIG['long exec warning'] if you want to run longer computations.")
                print(" | e.g. "+__name__+".CONFIG['long exec warning'] = %d"%int(1.2*est))
                print(" | set it to 'None' and the warning will disapear... at your own risks!")
                return

        print('')
        tmp = {k:param[k] for k in param.keys()}
        for _k in self.dwavel.keys():
            tmp['dwavel;'+_k] = self.dwavel[_k]
        # -- reference fit (all data)
        refFit = _fitFunc(tmp, self._chi2Data, self.observables,
                            self.instruments, fitAlso, doNotFit)

        res = {'fit':refFit}
        res['MJD'] = self.allMJD.mean()
        p = self._pool()

        mjds = []
        for d in self._chi2Data:
            mjds.extend(list(set(d[-3].flatten())))
        mjds = set(mjds)
        if len(mjds)<3 and useMJD:
            print(' > Warning: not enough dates to bootstrap on them!')
            useMJD=False

        for i in range(N): # -- looping fits
            if useMJD:
                x = {j:np.random.rand() for j in mjds}
                #print(i, x)
            tmp = {k:param[k] for k in param.keys()}
            for _k in self.dwavel.keys():
                tmp['dwavel;'+_k] = self.dwavel[_k]
            # if i==0:
            #     print(' | inital parameters set to param=', tmp)
            #     print('')

            tmp['_k'] = i
            data = []
            for d in self._chi2Data:
                data.append(list(d)) # recreate a list of data
                if monteCarlo:
                    # -- Monte Carlo noise ----------------------------------------------------------
                    if corrSpecCha is None:
                        data[-1][-2] = d[-2] + 1.0*d[-1] * np.random.randn(d[-1].shape[0], d[-1].shape[1])
                    else:
                        print('error! >> not implemented')
                        return
                else:
                    # -- Bootstrapping: set 'fracIgnored'% of the data to Nan (ignored)
                    if useMJD:
                        mjd = set(d[-3].flatten())
                        mask = d[-1]*0 + 1.0
                        for j in mjd:
                            if x[j]<=1/3.: # reject some MJDs
                                mask[d[-3]==j] = np.nan # ignored
                            if x[j]>=2/3.: # count some MJDs twice
                                mask[d[-3]==j] = 2. # ignored

                        # -- weight the error bars
                        data[-1][-1] = data[-1][-1]/mask

                    mask = np.random.rand(d[-1].shape[-1])
                    mask = np.float_(mask>=1/3.) + np.float_(mask>=2/3.)
                    mask[mask==0] += np.nan
                    # -- weight the error bars
                    if len(d[-1].shape)==2:
                        data[-1][-1] = data[-1][-1]/mask[None,:]
                    else:
                        data[-1][-1] = data[-1][-1]/mask
            if p is None:
                # -- single thread:
                self._cb_fitFunc(_fitFunc(tmp, data, self.observables,
                                        self.instruments, fitAlso, doNotFit))
            else:
                # -- multi thread:
                p.apply_async(_fitFunc, (tmp, data, self.observables,
                                self.instruments, fitAlso, doNotFit),
                                        callback=self._cb_fitFunc)
        if not p is None:
            p.close()
            p.join()


        if not fig is None:
            plt.close(fig)
            if CONFIG['suptitle']:
                plt.figure(fig, figsize=(9/1.2,9/1.2))
                plt.subplots_adjust(left=0.1, bottom=0.07,
                                right=0.98, top=0.88,
                                wspace=0.35, hspace=0.35)
                title = "CANDID: bootstrapping uncertainties, %d rounds"%N
                title += ' using '+', '.join(self.observables)
                title += '\nfrom '+', '.join(self.instruments)
                title += '\n'+self.titleFilename
                plt.suptitle(title, fontsize=14, fontweight='bold')
            else:
                plt.figure(fig, figsize=(10/1.2,10/1.2))
                plt.subplots_adjust(left=0.10, bottom=0.10,
                                right=0.98, top=0.95,
                                wspace=0.3, hspace=0.3)
        kz = self.allFits[0]['fitOnly']
        kz.sort()
        ax = {}

        # -- sigma cliping
        print(' | sigma clipping in position and flux ratio for nSigmaClip= %3.1f'%(nSigmaClip))
        x = np.array([a['best']['x'] for a in self.allFits])
        y = np.array([a['best']['y'] for a in self.allFits])
        f = np.array([a['best']['f'] for a in self.allFits])

        d = np.array([a['best']['diam*'] for a in self.allFits])
        test = np.array([a['best']['x']!=param['x'] or
                         a['best']['y']!=param['y'] or
                         a['best']['f']!=param['f'] for a in self.allFits])

        w = np.where((x <= np.median(x) + nSigmaClip*(np.percentile(x, 84)-np.median(x)))*
                     (x >= np.median(x) - nSigmaClip*(np.median(x)-np.percentile(x, 16)))*
                     (y <= np.median(y) + nSigmaClip*(np.percentile(y, 84)-np.median(y)))*
                     (y >= np.median(y) - nSigmaClip*(np.median(y)-np.percentile(y, 16)))*
                     (f <= np.median(f) + nSigmaClip*(np.percentile(f, 84)-np.median(f)))*
                     (f >= np.median(f) - nSigmaClip*(np.median(f)-np.percentile(f, 16)))*
                     (d <= np.median(d) + nSigmaClip*(np.percentile(d, 84)-np.median(d)))*
                     (d >= np.median(d) - nSigmaClip*(np.median(d)-np.percentile(d, 16)))*
                     test
                    )
        flag = np.ones(len(self.allFits), dtype=bool)
        flag[w] = False
        for i in range(len(self.allFits)):
            self.allFits[i]['sigma clipping flag'] = flag[i]
        print(' | %d fits ignored'%(len(x)-len(w[0])))

        ax = {}
        res['boot']={}
        for i1,k1 in enumerate(kz):
            for i2,k2 in enumerate(kz):
                X = np.array([a['best'][k1] for a in self.allFits])[w]
                Y = np.array([a['best'][k2] for a in self.allFits])[w]
                if i1<i2:
                    nSigma=1
                    p = pca(np.transpose(np.array([(X-X.mean()),
                                                   (Y-Y.mean())])))
                    _a = np.arctan2(p.base[0][0], p.base[0][1])
                    err0 = p.coef[:,0].std()
                    err1 = p.coef[:,1].std()
                    th = np.linspace(0,2*np.pi,100)
                    _x, _y = nSigma*err0*np.cos(th), nSigma*err1*np.sin(th)
                    _x, _y = X.mean() + _x*np.cos(_a+np.pi/2) + _y*np.sin(_a+np.pi/2), \
                             Y.mean() - _x*np.sin(_a+np.pi/2) + _y*np.cos(_a+np.pi/2)
                    res['boot'][k1+k2] = (_x, _y)
                else:
                    med = np.median(X)
                    errp = np.median(X)-np.percentile(X, 16)
                    errm = np.percentile(X, 84)-np.median(X)
                    res['boot'][k1]=(med, errp,errm)

                if i1<i2 and not fig is None:
                    if i1>0:
                        ax[(i1,i2)] = plt.subplot(len(kz), len(kz), i1+len(kz)*i2+1,
                                                  sharex=ax[(i1,i1)],
                                                  sharey=ax[(0,i2)])
                    else:
                        ax[(i1,i2)] = plt.subplot(len(kz), len(kz), i1+len(kz)*i2+1,
                                                  sharex=ax[(i1,i1)])
                    if i1==0 and i2>0:
                        plt.ylabel(k2)
                    plt.plot(X, Y, '.', color='k', alpha=max(0.15, 0.5-0.12*np.log10(N)))
                    #plt.hist2d(X, Y, cmap='afmhot_r', bins=max(5, np.sqrt(N)/3.))

                    plt.errorbar(refFit['best'][k1], refFit['best'][k2],
                                 xerr=refFit['uncer'][k1], yerr=refFit['uncer'][k2],
                                 color='r', fmt='s', markersize=8, elinewidth=3,
                                 alpha=0.8, capthick=3, linewidth=3)
                    # -- error ellipse
                    plt.plot(_x, _y, linestyle='-', color='b', linewidth=3)
                    if len(ax[(i1,i2)].get_yticks())>5:
                        ax[(i1,i2)].set_yticks(ax[(i1,i2)].get_yticks()[::2])

                if i1==i2 and not fig is None:
                    ax[(i1,i1)] = plt.subplot(len(kz), len(kz), i1+len(kz)*i2+1)
                    try:
                        n = int(max(2-np.log10(errp), 2-np.log10(errm)))
                    except:
                        print(' | to few data points? using mean instead of median')
                        #print(X)
                        med = np.mean(X)
                        errp = np.std(X)
                        errm = np.std(X)
                        #print(med, errp, errm)
                        n = int(2-np.log10(errm))

                    form = '%s = $%'+str(int(n+2))+'.'+str(int(n))+'f'
                    form += '^{+%'+str(int(n+2))+'.'+str(int(n))+'f}'
                    form += '_{-%'+str(int(n+2))+'.'+str(int(n))+'f}$'
                    plt.title(form%(k1, med, errp,errm))
                    plt.hist(X, color='0.5', bins=max(15, N/50))
                    ax[(i1,i2)].set_yticks([])
                    if len(ax[(i1,i2)].get_xticks())>5:
                        ax[(i1,i2)].set_xticks(ax[(i1,i2)].get_xticks()[::2])
                    print(' | %8s = %8.4f + %6.4f - %6.4f %s'%(k1, med, errp,errm, paramUnits(k1)))
                if i2==len(kz)-1:
                    plt.xlabel(k1)
        self.bootRes = res
        return
        # -_-_-

    def plotModel(self, param=None, fig=3, spectral=False):
        """
        param: what companion model to plot. If None, will attempt to use self.bestFit
        """
        if param is None:
            try:
                param = self.bestFit['best']
                #print(param)
            except:
                print(' | ERROR: please set param= do the initial conditions (or run fitMap)')
                print(" | param={'x':, 'y':, 'f':, 'diam*':}")
                return
        try:
            for k in ['x', 'y', 'f', 'diam*']:
                if not k in param.keys():
                    print(' | ERROR: please set param= do the initial conditions (or run fitMap)')
                    print(" | param={'x':, 'y':, 'f':, 'diam*':}")
                    return
        except:
            pass
        for _k in self.dwavel.keys():
            param['dwavel;'+_k] = self.dwavel[_k]

        _meas, _errs, _uv, _types, _wl = _generateFitData(self._rawData,
                                                          self.observables,
                                                          self.instruments)
        _mod = _modelObservables(list(filter(lambda c: c[0].split(';')[0] in self.observables
                                    and c[0].split(';')[1] in self.instruments,
                                self._chi2Data)), param)
        #print(_meas.shape)
        plt.close(fig)
        plt.figure(fig, figsize=(11/1.2,8/1.2))
        plt.clf()
        N = len(set(_types))
        for i,t in enumerate(set(_types)): # for each observables
            w = np.where((_types==t)*(1-np.isnan(_meas)))
            ax1 = plt.subplot(2,N,i+1)
            plt.title(t.split(';')[1])
            #print('DEBUG:', set(np.round(_uv[w]*_wl[w],3)))
            if spectral:
                X = _wl[w]
                marker = '.'
                linestyle = '-'
                B = np.int_(1e3*_uv[w]*_wl[w])
                allB = sorted(list(set(B)))[::-1]
                offset = np.array([allB.index(b) for b in B])
                oV2 = 0.4
                oCP = 20
            else:
                X = _uv[w]
                marker = '.'
                linestyle = '-'
                offset = np.zeros(len(w))
                oV2 = 0.0
                oCP = 0.0

            if any(np.iscomplex(_meas[w])):
                res = (np.angle(_meas[w]/_mod[w])+np.pi)%(2*np.pi) - np.pi
                res /= _errs[w]
                res = np.float_(res)
                _measw = np.angle(_meas[w]).real
                _modw = np.angle(_mod[w]).real
                plt.plot(X, 180/np.pi*(np.mod(_modw+np.pi/2, np.pi)-np.pi/2)+oCP*offset,
                        '.k', alpha=0.4)
                plt.scatter(X, 180/np.pi*(np.mod(_measw+np.pi/2,np.pi)-np.pi/2)+oCP*offset,
                            c=_wl[w], marker=marker, cmap='hot_r',
                            alpha=0.5, linestyle=linestyle)
                plt.errorbar(X, 180/np.pi*(np.mod(_measw+np.pi/2, np.pi)-np.pi/2) +oCP*offset,
                            fmt=',', yerr=180/np.pi*_errs[w], marker=None, color='k', alpha=0.2)
                plt.ylabel(t.split(';')[0]+r': deg, mod 180')
            else:
                res = (_meas[w]-_mod[w])/_errs[w]
                plt.errorbar(X, _meas[w]+oV2*offset, fmt=',', yerr=_errs[w], marker=None,
                             color='k', alpha=0.2)
                plt.scatter(X, _meas[w]+oV2*offset, c=_wl[w], marker=marker, cmap='hot_r',
                        alpha=0.5, linestyle=linestyle)
                plt.plot(X, _mod[w]+oV2*offset, '.k', alpha=0.4)
                plt.ylabel(t.split(';')[0])
                if t.split(';')[0]=='v2':
                    plt.ylim(-0.1, 1.1+np.max(offset)*oV2)

            # -- residuals
            plt.subplot(2,N,i+1+N, sharex=ax1)
            plt.plot(X, res, '.k', alpha=0.5)
            if spectral:
                plt.xlabel('wavelength')
            else:
                plt.xlabel('Bmax / $\lambda$')

            plt.ylabel('residuals ($\sigma$)')
            plt.ylim(-np.max(np.abs(plt.ylim())), np.max(np.abs(plt.ylim())))
        return

    def _cb_nsigmaFunc(self, r):
        """
        callback function for detectionLimit()
        """
        try:
            self.f3s[r[1], r[0]] = r[2]
            # -- completed / to be computed
            f = np.sum(self.f3s>0)/float(np.sum(self.f3s>=0))
            if f>self._prog and CONFIG['progress bar']:
                n = int(50*f)
                print('\033[F',end=' ')
                print('|'+'='*(n+1)+' '*(50-n)+'|',end=' ')
                print('%2d%%'%(int(100*f)),end=' ')
                self._progTime[1] = time.time()
                print('%3d s remaining'%(int((self._progTime[1]-self._progTime[0])/f*(1-f))))
                self._prog = max(self._prog+0.01, f+0.01)
        except:
            print('did not work')
        return
    def detectionLimit(self, step=None, diam=None, fig=4, addCompanion=None,
                        removeCompanion=None, drawMaps=True, rmin=None, rmax=None,
                        methods = ['Absil', 'injection'], fratio=1.):
        """
        step: number of steps N in the map (map will be NxN)
        drawMaps: display the detection maps in addition to the radial profile (default)

        Apart from the plots, the radial detection limits are stored in the dictionnary
        'self.f3s'.
        """
        if isinstance(methods, str):
            methods = [methods]
        for method in methods:
            # -- check known methods:
            knownMethods = ['Absil', 'injection']
            assert method in knownMethods, \
                'unknowm detection limit method: '+method+'. known methods are '+str(knownMethods)

        if step is None:
            step = 1/2. * self.minSpatialScale
            print(' | step= not given, using 1/2 X smallest spatial scale = %4.2f mas'%step)

        if rmin is None:
            self.rmin = self.minSpatialScale
            print(" | rmin= not given, set to smallest spatial scale: rmin=%5.2f mas"%(self.rmin))
        else:
            self.rmin = rmin

        if rmax is None:
            self.rmax = 1.2*self.smearFov
            print(" | rmax= not given, set to 1.2*Field of View: rmax=%5.2f mas"%(self.rmax))
        else:
            self.rmax = rmax
        self._estimateNsmear()
        try:
            N = int(np.ceil(2*self.rmax/step))
        except:
            print('ERROR: you should define rmax first!')
            return
        print(' | observables:', self.observables, 'from', self.ALLobservables)
        print(' | instruments:', self.instruments, 'from', self.ALLinstruments)


        # -- start with all data
        self._chi2Data = self._copyRawData()
        if not addCompanion is None:
            tmp = {k:addCompanion[k] for k in addCompanion.keys()}
            tmp['f'] = np.abs(tmp['f'])
            self._chi2Data = _injectCompanionData(self._chi2Data, self._delta, tmp)
        if not removeCompanion is None:
            tmp = {k:removeCompanion[k] for k in removeCompanion.keys()}
            tmp['f'] = -np.abs(tmp['f'])
            self._chi2Data = _injectCompanionData(self._chi2Data, self._delta, tmp)

        self.fitUD(diam)

        print(' | Detection Limit Map %dx%d'%(N,N),end=' ')

        # -- estimate how long it will take
        if not CONFIG['long exec warning'] is None:
            Ntest = 2*max(multiprocessing.cpu_count()-1,1)
            params = []
            for k in range (Ntest):
                tmp = {'x':10*np.random.rand(), 'y':10*np.random.rand(),
                       'f':fratio, 'diam*':self.diam, 'alpha*':self.alpha}
                for _k in self.dwavel.keys():
                    tmp['dwavel;'+_k] = self.dwavel[_k]
                params.append((tmp, self._chi2Data, self.observables,
                                self.instruments, self._delta))
            # -- Absil is twice as fast as injection
            est = 1.5*self._estimateRunTime(_detectLimit, params)
            est *= N**2
            print('... it should take about %d seconds'%(int(est)))
            if not CONFIG['long exec warning'] is None and\
                 est>CONFIG['long exec warning']:
                print(" > WARNING: this will take too long. ")
                print(" | Increase CONFIG['long exec warning'] if you want to run longer computations.")
                print(" | e.g. "+__name__+".CONFIG['long exec warning'] = %d"%int(1.2*est))
                print(" | set it to 'None' and the warning will disapear... at your own risks!")
                return
        else:
            print('')

        #print('estimated time:', est)
        #t0 = time.time()

        # -- prepare grid:
        allX = np.linspace(-self.rmax, self.rmax, N)
        allY = np.linspace(-self.rmax, self.rmax, N)
        self.allf3s = {}
        for method in methods:
            print(" | Method:", method)
            print('')
            self.f3s = np.zeros((N,N))
            self.f3s[allX[None,:]**2+allY[:,None]**2 > self.rmax**2] = -1
            self.f3s[allX[None,:]**2+allY[:,None]**2 < self.rmin**2] = -1
            self._prog = 0.0
            self._progTime = [time.time(), time.time()]
            # -- parallel treatment:
            p = self._pool()
            for i,x in enumerate(allX):
                for j,y in enumerate(allY):
                    if self.f3s[j,i]==0:
                        params = {'x':x, 'y':y, 'f':fratio, 'diam*':self.diam,
                                  '_i':i, '_j':j, 'alpha*':self.alpha}
                        for _k in self.dwavel.keys():
                            params['dwavel;'+_k] = self.dwavel[_k]
                        if p is None:
                            # -- single thread:
                            self._cb_nsigmaFunc(_detectLimit(params, self._chi2Data,
                                        self.observables, self.instruments,
                                           self._delta, method))
                        else:
                            # -- parallel:
                            p.apply_async(_detectLimit, (params, self._chi2Data,
                                        self.observables, self.instruments,
                                       self._delta, method), callback=self._cb_nsigmaFunc)
            if not p is None:
                p.close()
                p.join()
            # -- take care of unfitted zone, for esthetics
            self.f3s[self.f3s<=0] = np.median(self.f3s[self.f3s>0])
            self.allf3s[method] = self.f3s.copy()

        #print('it actually took %4.1f seconds'%(time.time()-t0))
        X, Y = np.meshgrid(allX, allY)
        vmin=min([np.min(self.allf3s[m]) for m in methods]),
        vmax=max([np.max(self.allf3s[m]) for m in methods]),

        if drawMaps:
            # -- draw the detection maps
            vmin, vmax = None, None
            plt.close(fig)
            if CONFIG['suptitle']:
                plt.figure(fig, figsize=(11/1.2,10/1.2))
                plt.subplots_adjust(top=0.85, bottom=0.08,
                                    left=0.08, right=0.97)
                title = "CANDID: flux ratio for 3$\sigma$ detection, "
                if self.ediam>0:
                    title += r'fitted $\theta_\mathrm{UD}=%4.3f$ mas.'%(self.diam)
                else:
                    title += r'fixed $\theta_\mathrm{UD}=%4.3f$ mas.'%(self.diam)
                title += ' Using '+', '.join(self.observables)
                title += '\n'+self.titleFilename
                if not removeCompanion is None:
                    title += '\ncompanion removed at X=%3.2fmas, Y=%3.2fmas, F=%3.2f%%'%(
                            removeCompanion['x'], removeCompanion['y'],
                            removeCompanion['f'])

                plt.suptitle(title, fontsize=14, fontweight='bold')
            else:
                plt.figure(fig, figsize=(11/1.2,9/1.2))
                plt.subplots_adjust(top=0.95, bottom=0.08,
                                    left=0.08, right=0.97)
            for i,m in enumerate(methods):
                ax1=plt.subplot(2, len(methods), i+1)
                plt.title('Method: '+m)
                plt.pcolormesh(X, Y, self.allf3s[m], cmap=CONFIG['color map'],
                    vmin=vmin, vmax=vmax)
                plt.colorbar()
                plt.xlabel(r'E $\leftarrow\, \Delta \alpha$ (mas)')
                plt.ylabel(r'$\Delta \delta\, \rightarrow$ N (mas)')
                plt.xlim(self.rmax, -self.rmax)
                plt.ylim(-self.rmax, self.rmax)
                ax1.set_aspect('equal', 'datalim')

            plt.subplot(212)

        else:
            plt.close(fig)
            plt.figure(fig, figsize=(8/1.2,5/1.2))

        # -- radial profile:
        r = np.sqrt(X**2+Y**2).flatten()
        r_f3s = {}
        for k in self.allf3s.keys():
            r_f3s[k] = self.allf3s[k].flatten()[np.argsort(r)]
        r = r[np.argsort(r)]
        for k in r_f3s.keys():
            r_f3s[k] = r_f3s[k][(r<=self.rmax)*(r>=self.rmin)]
        r = r[(r<=self.rmax)*(r>=self.rmin)]

        self._detectLim = {'r': r}
        for m in methods:

            self._detectLim[m+'_99_M'] = -2.5*np.log10(sliding_percentile(r, r_f3s[m],
                                    self.rmax/float(N), 99)/100.)

        if True: # -- plot in magnitudes:
            for m in methods:
                plt.plot(r, -2.5*np.log10(sliding_percentile(r, r_f3s[m],
                        self.rmax/float(N), 99)/100.),
                        linewidth=3, alpha=0.5, label=m+' (99%)')

            plt.ylabel('$\Delta \mathrm{Mag}_{3\sigma}$')
            plt.ylim(plt.ylim()[1], plt.ylim()[0]) # -- rreverse plot
            plt.legend(loc='upper center')

        plt.xlabel('radial distance (mas)')
        plt.grid()
        # -- store radial profile of detection limit:
        self.f3s = {'r(mas)':r}
        for m in methods:
            self.f3s[m] = sliding_percentile(r, r_f3s[m],
                                                self.rmax/float(N), 90)
        return

def sliding_percentile(x, y, dx, percentile=50, smooth=True):
    res = np.zeros(len(y))
    for i in range(len(x)):
        res[i] = np.percentile(y[np.abs(x-x[i])<dx/2.], percentile)
    if smooth:
        resp = np.zeros(len(y))
        for i in range(len(x)):
            resp[i] = res[np.abs(x-x[i])<dx/4.].mean()
        return resp
    else:
        return res

def _dpfit_leastsqFit(func, x, params, y, err=None, fitOnly=None, verbose=False,
                        doNotFit=[], epsfcn=1e-5, ftol=1e-5, fullOutput=True,
                        normalizedUncer=True, follow=None):
    """
    - params is a Dict containing the first guess.

    - fits 'y +- err = func(x,params)'. errors are optionnal. in case err is a
      ndarray of 2 dimensions, it is treated as the covariance of the
      errors.

      np.array([[err1**2, 0, .., 0],
                [0, err2**2, 0, .., 0],
                [0, .., 0, errN**2]]) is the equivalent of 1D errors

    - follow=[...] list of parameters to "follow" in the fit, i.e. to print(in)
      verbose mode

    - fitOnly is a LIST of keywords to fit. By default, it fits all
      parameters in 'params'. Alternatively, one can give a list of
      parameters not to be fitted, as 'doNotFit='

    - doNotFit has a similar purpose: for example if params={'a0':,
      'a1': 'b1':, 'b2':}, doNotFit=['a'] will result in fitting only
      'b1' and 'b2'. WARNING: if you name parameter 'A' and another one 'AA',
      you cannot use doNotFit to exclude only 'A' since 'AA' will be excluded as
      well...

    - normalizedUncer=True: the uncertainties are independent of the Chi2, in
      other words the uncertainties are scaled to the Chi2. If set to False, it
      will trust the values of the error bars: it means that if you grossely
      underestimate the data's error bars, the uncertainties of the parameters
      will also be underestimated (and vice versa).

    returns dictionary with:
    'best': bestparam,
    'uncer': uncertainties,
    'chi2': chi2_reduced,
    'model': func(x, bestparam)
    'cov': covariance matrix (normalized if normalizedUncer)
    'fitOnly': names of the columns of 'cov'
    """
    # -- fit all parameters by default
    if fitOnly is None:
        if len(doNotFit)>0:
            fitOnly = list(filter(lambda x: x not in doNotFit, params.keys()))
        else:
            fitOnly = params.keys()
        fitOnly.sort() # makes some display nicer

    # -- build fitted parameters vector:
    pfit = [params[k] for k in fitOnly]

    # -- built fixed parameters dict:
    pfix = {}
    for k in params.keys():
        if k not in fitOnly:
            pfix[k]=params[k]
    if verbose:
        print('[dpfit] %d FITTED parameters:'%(len(fitOnly)), fitOnly)
    # -- actual fit
    plsq, cov, info, mesg, ier = \
              scipy.optimize.leastsq(_dpfit_fitFunc, pfit,
                    args=(fitOnly,x,y,err,func,pfix,verbose,follow,),
                    full_output=True, epsfcn=epsfcn, ftol=ftol)
    if isinstance(err, np.ndarray) and len(err.shape)==2:
        print(cov)

    # -- best fit -> agregate to pfix
    for i,k in enumerate(fitOnly):
        pfix[k] = plsq[i]

    # -- reduced chi2
    model = func(x,pfix)
    tmp = _dpfit_fitFunc(plsq, fitOnly, x, y, err, func, pfix)
    try:
        chi2 = (np.array(tmp)**2).sum()
    except:
        chi2 = 0.0
        for x in tmp:
            chi2+=np.sum(x**2)
    reducedChi2 = chi2/float(np.sum([1 if np.isscalar(i) else
                                     len(i) for i in tmp])-len(pfit)+1)
    if not np.isscalar(reducedChi2):
        reducedChi2 = np.mean(reducedChi2)

    # -- uncertainties:
    uncer = {}
    for k in pfix.keys():
        if not k in fitOnly:
            uncer[k]=0 # not fitted, uncertatinties to 0
        else:
            i = fitOnly.index(k)
            if cov is None:
                uncer[k]=-1
            else:
                uncer[k]= np.sqrt(np.abs(np.diag(cov)[i]))
                if normalizedUncer:
                    uncer[k] *= np.sqrt(reducedChi2)

    if verbose:
        print('-'*30)
        print('REDUCED CHI2=', reducedChi2)
        print('-'*30)
        if normalizedUncer:
            print('(uncertainty normalized to data dispersion)')
        else:
            print('(uncertainty assuming error bars are correct)')
        tmp = pfix.keys(); tmp.sort()
        maxLength = np.max(np.array([len(k) for k in tmp]))
        format_ = "'%s':"
        # -- write each parameter and its best fit, as well as error
        # -- writes directly a dictionnary
        print('') # leave some space to the eye
        for ik,k in enumerate(tmp):
            padding = ' '*(maxLength-len(k))
            formatS = format_+padding
            if ik==0:
                formatS = '{'+formatS
            if uncer[k]>0:
                ndigit = -int(np.log10(uncer[k]))+3
                print(formatS%k , round(pfix[k], ndigit), ',',end=' ')
                print('# +/-', round(uncer[k], ndigit))
            elif uncer[k]==0:
                if isinstance(pfix[k], str):
                    print(formatS%k , "'"+pfix[k]+"'", ',')
                else:
                    print(formatS%k , pfix[k], ',')
            else:
                print(formatS%k , pfix[k], ',',end=' ')
                print('# +/-', uncer[k])
        print('}') # end of the dictionnary
        try:
            if verbose>1:
                print('-'*3, 'correlations:', '-'*15)
                N = np.max([len(k) for k in fitOnly])
                N = min(N,20)
                N = max(N,5)
                sf = '%'+str(N)+'s'
                print(' '*N,end=' ')
                for k2 in fitOnly:
                    print(sf%k2,end=' ')
                print('')
                sf = '%-'+str(N)+'s'
                for k1 in fitOnly:
                    i1 = fitOnly.index(k1)
                    print(sf%k1 ,end=' ')
                    for k2 in fitOnly:
                        i2 = fitOnly.index(k2)
                        if k1!=k2:
                            print(('%'+str(N)+'.2f')%(cov[i1,i2]/
                                                      np.sqrt(cov[i1,i1]*cov[i2,i2])), end=' ')
                        else:
                            print(' '*(N-4)+'-'*4,end=' ')
                    print('')
                print('-'*30)
        except:
            pass
    # -- result:
    if fullOutput:
        cor = None
        if not cov is None:
            if normalizedUncer:
                cov *= reducedChi2
            if all(np.diag(cov)>=0):
                cor = np.sqrt(np.diag(cov))
                cor = cor[:,None]*cor[None,:]
                cor = cov/cor

        pfix={'best':pfix, 'uncer':uncer,
              'chi2':reducedChi2, 'model':model,
              'cov':cov, 'fitOnly':fitOnly,
              'info':info, 'cor':cor}
    return pfix

def _dpfit_fitFunc(pfit, pfitKeys, x, y, err=None, func=None, pfix=None,
                    verbose=False, follow=None):
    """
    interface to leastsq from scipy:
    - x,y,err are the data to fit: f(x) = y +- err
    - pfit is a list of the paramters
    - pfitsKeys are the keys to build the dict
    pfit and pfix (optional) and combines the two
    in 'A', in order to call F(X,A)

    in case err is a ndarray of 2 dimensions, it is treated as the
    covariance of the errors.
    np.array([[err1**2, 0, .., 0],
             [ 0, err2**2, 0, .., 0],
             [0, .., 0, errN**2]]) is the equivalent of 1D errors

    """
    global verboseTime
    params = {}
    # -- build dic from parameters to fit and their values:
    for i,k in enumerate(pfitKeys):
        params[k]=pfit[i]
    # -- complete with the non fitted parameters:
    for k in pfix:
        params[k]=pfix[k]
    if err is None:
        err = np.ones(np.array(y).shape)

    # -- compute residuals
    if type(y)==np.ndarray and type(err)==np.ndarray:
        # -- assumes y and err are a numpy array
        y = np.array(y)
        res = (np.abs(func(x,params)-y)/err).flatten()
        # -- avoid NaN! -> make them 0's
        res = np.nan_to_num(res)
    else:
        # much slower: this time assumes y (and the result from func) is
        # a list of things, each convertible in np.array
        res = []
        tmp = func(x,params)
        for k in range(len(y)):
            # -- abs to deal with complex number
            df = np.abs(np.array(tmp[k])-np.array(y[k]))/np.array(err[k])
            # -- avoid NaN! -> make them 0's
            df = np.nan_to_num(df)
            try:
                res.extend(list(df))
            except:
                res.append(df)

    if verbose and time.time()>(verboseTime+1):
        verboseTime = time.time()
        print(time.asctime(),end=' ')
        try:
            chi2=(res**2).sum/(len(res)-len(pfit)+1.0)
            print('CHI2: %6.4e'%chi2,end=' ')
        except:
            # list of elements
            chi2 = 0
            N = 0
            res2 = []
            for r in res:
                if np.isscalar(r):
                    chi2 += r**2
                    N+=1
                    res2.append(r)
                else:
                    chi2 += np.sum(np.array(r)**2)
                    N+=len(r)
                    res2.extend(list(r))

            res = res2
            print('CHI2: %6.4e'%(chi2/float(N-len(pfit)+1)),end=' ')
        if follow is None:
            print('')
        else:
            try:
                print(' '.join([k+'='+'%5.2e'%params[k] for k in follow]))
            except:
                print('')
    return res

def _dpfit_ramdomParam(fit, N=1):
    """
    get a set of randomized parameters (list of dictionnaries) around the best
    fited value, using a gaussian probability, taking into account the
    correlations from the covariance matrix.

    fit is the result of leastsqFit (dictionnary)
    """
    m = np.array([fit['best'][k] for k in fit['fitOnly']])
    res = []
    for k in range(N):
        p = dict(zip(fit['fitOnly'],np.random.multivariate_normal(m, fit['cov'])))
        p.update({fit['best'][k] for k in fit['best'].keys() if not k in
                 fit['fitOnly']})
        res.append(p)
    if N==1:
        return res[0]
    else:
        return res

def _dpfit_polyN(x, params):
    """
    Polynomial function. e.g. params={'A0':1.0, 'A2':2.0} returns
    x->1+2*x**2. The coefficients do not have to start at 0 and do not
    need to be continuous. Actually, the way the function is written,
    it will accept any '_x' where '_' is any character and x is a float.
    """
    res = 0
    for k in params.keys():
        res += params[k]*np.array(x)**float(k[1:])
    return res

def _decomposeObs(wl, data, err, order=1, plot=False):
    """
    decompose data(wl)+-err as a polynomial of order "order" in (wl-wl.mean())
    """
    X = wl-wl.mean()
    fit = _dpfit_leastsqFit(_dpfit_polyN, X, {'A'+str(i):0.0 for i in range(order+1)}, data, err)
    # -- estimate statistical errors, as scatter around model:
    stat = np.nanstd(data - fit['model'])
    # -- estimate median systematic error:
    sys = np.nanmedian(np.sqrt(np.maximum(err**2-stat**2, 0.0)))
    tmp = np.sqrt(fit['uncer']['A0']**2 + sys**2)/fit['uncer']['A0']
    fit['uncer']['A0'] = np.sqrt(fit['uncer']['A0']**2 + sys**2)

    if plot:
        plt.clf()
        plt.errorbar(X, data, yerr=err, fmt='o', color='k', alpha=0.2)
        plt.plot(X, fit['model'], '-r', linewidth=2, alpha=0.5)
        plt.plot(X, fit['model']+sys, '-r', linewidth=2, alpha=0.5, linestyle='dashed')
        plt.plot(X, fit['model']-sys, '-r', linewidth=2, alpha=0.5, linestyle='dashed')

        ps = _dpfit_ramdomParam(fit, 100)
        _x = np.linspace(X.min(), X.max(), 10*len(X))
        _ymin, _ymax = _dpfit_polyN(_x, fit['best']), _dpfit_polyN(_x, fit['best'])
        for p in ps:
            tmp = _dpfit_polyN(_x, p)
            _ymin = np.minimum(_ymin, tmp)
            _ymax = np.maximum(_ymax, tmp)
        plt.fill_between(_x, _ymin, _ymax, color='r', alpha=0.5)
    else:
        return fit['best'], fit['uncer']

def _estimateCorrelation(data, errors, model):
    """
    assumes errors**2 = stat**2 + sys**2, will compute sys so the chi2 from stat error bars is 1.

    estimate the average correlation in a sample (sys/<err>)
    """
    sys = np.min(errors)/2.
    fact = 0.5
    chi2 = np.mean((data-model)**2/(errors**2-sys**2))
    tol = 1.e-3
    # if chi2 <= 1:
    #      return 0.0
    n = 0
    #print('%4.2f'%chi2, '->',end=' ')
    while np.abs(chi2-1)>tol and sys/np.min(errors)>tol and n<20:
        #print(sys, chi2)
        if chi2>1:
            sys *= fact
        else:
            sys /= fact
        _chi2 = np.mean((data-model)**2/(errors**2-sys**2))
        if (chi2-1)*(_chi2-1)<0:
            fact = np.sqrt(fact)
        chi2 = _chi2
        n+=1
    #print('%5.2f : ERR=%5.3f SYS=%5.3f STA=%5.3f %2d'%(chi2, np.mean(errors),
    #                                                   sys, np.sqrt(np.mean(errors**2)-sys**2), n))
    return sys/np.mean(errors)

class pca():
    """
    principal component analysis
    """
    def __init__(self,data):
        """
        data[i,:] is ith data
        creates 'var' (variances), 'base' (base[i,:] are the base,
        sorted from largest constributor to the least important. .coef
        """
        self.data     = data
        self.data_std = np.std(self.data, axis=0)
        self.data_avg = np.mean(self.data,axis=0)

        # variance co variance matrix
        self.varcovar  = (self.data-self.data_avg[np.newaxis,:])
        self.varcovar = np.dot(np.transpose(self.varcovar), self.varcovar)

        # diagonalization
        e,l,v= np.linalg.svd(self.varcovar)

        # projection
        coef = np.dot(data, e)
        self.var  = l/np.sum(l)
        self.base = e
        self.coef = coef
        return
    def comp(self, i=0):
        """
        returns component projected along one vector
        """
        return self.coef[:,i][:,np.newaxis]*\
               self.base[:,i][np.newaxis,:]
