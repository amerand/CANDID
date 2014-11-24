import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import time
import scipy.special
import scipy.interpolate
import scipy.stats
import multiprocessing
import dpfit
import os

print """
============================ This is CANDID ==========================================
the [C]ompanion [A]nalysis and [N]on-Detection [D]etection in [I]nterferometric [D]ata
======================================================================================
                    https://github.com/amerand/CANDID
"""

"""
import binSimu2
filename = '/Users/amerand/DATA/PIONIER/MIRC_L2.SU_Cas.2012Sep30_tcoh0075ms_JDM_2013Mar24.XCHAN.JDM.AVG15m.oifits'

binSimu2.test1(filename, obs=['cp'])
"""

c = np.pi/180/3600000.*1e6
n_smear = 5 # number of channels for bandwidth smearing
cmap = 'cubehelix' # color map used

def Vud(base, diam, wavel):
    """
    Complex visibility of a uniform disk for parameters:
    - base in m
    - diam in mas
    - wavel in um
    """
    x = c*diam*base/wavel
    x += 1e-6*(x==0)
    x *= np.pi
    res = np.array(2*scipy.special.jv(1, x)/(x))
    return res

def Vbin(uv, param):
    """
    Analytical complex visibility of a binary composed of a uniform
    disk diameter and an unresolved source. "param" is a dictionnary
    containing:

    'diam*': in mas
    'wavel': in um
    'x, y': in mas
    'f': flux ratio -> takes the absolute value
    """
    if 'f' in param:
        f = np.abs(param['f'])
    else:
        f = param['f']
    B = np.sqrt(uv[0]**2+uv[1]**2)
    Vstar = Vud(B, param['diam*'], param['wavel'])
    phi = 2*np.pi*c*(uv[0]*param['x']+uv[1]*param['y'])/param['wavel']
    Vcomp = np.exp(1j*phi)
    res = (Vstar + np.abs(f)*Vcomp)/(1.0+f)
    return res

def observable(obs, param, approx=False):
    """
    model observables contained in "obs".
    param -> see Vbin

    Observations are entered as:
    obs = [('v2', u, v, wavel, ...),
           ('cp', u1, v1, u2, v2, wavel, ...),
           ('t3', u1, v1, u2, v2, wavel, ...)]
    each tuple can be longer, the '...' part will be ignored

    units: u,v in m; wavel in um

    for CP and T3, the third u,v coordinate is computed as u1+u2, v1+v2
    """
    global n_smear
    #res = np.zeros((len(obs), len(obs[0][-1])))
    res = [0.0 for o in obs]
    for i, o in enumerate(obs):
        if o[0]=='v2':
            param['wavel'] = o[3]
            if not approx:
                if not 'dwavel' in param: # -- monochromatic
                    res[i] = np.abs(Vbin([o[1], o[2]], param))**2
                else: # -- bandwidth smearing
                    wl0 = param['wavel']
                    for x in np.linspace(-0.5, 0.5, n_smear):
                        param['wavel'] = wl0 + x*param['dwavel']
                        res[i] += np.abs(Vbin([o[1], o[2]], param))**2
                    param['wavel'] = wl0
                    res[i] /= float(n_smear)
            else:
                # -- approximation
                B = np.sqrt(o[1]**2+o[2]**2)
                if not 'dwavel' in param: # -- monochromatic
                    phi = 2*np.pi*c*(param['x']*o[1]+param['y']*o[2])/param['wavel']
                    Vstar = Vud(B, param['diam*'], param['wavel'])
                    res[i] = np.abs((Vstar + param['f']*Vstar*np.cos(phi))/\
                                      (1 + param['f']))**2
                else: # -- with bandwidth smearing:
                    for x in np.linspace(-0.5, 0.5, n_smear):
                        phi = 2*np.pi*c*(param['x']*o[1]+param['y']*o[2])/\
                                        (param['wavel']+x*param['dwavel'])
                        Vstar = Vud(B, param['diam*'],
                                    param['wavel']+x*param['dwavel'])
                        res[i] += np.abs((Vstar + param['f']*Vstar*np.cos(phi))/\
                                      (1 + param['f']))**2
                    res[i] /= float(n_smear)

        elif o[0]=='cp' or o[0]=='t3':
            param['wavel'] = o[5]
            if not approx: # -- approximation
                if not 'dwavel' in param: # -- monochromatic
                    t3 = Vbin([o[1], o[2]], param)*\
                         Vbin([o[3], o[4]], param)*\
                         np.conj(Vbin([o[1]+o[3], o[2]+o[4]], param))
                else: # -- bandwidth smearing
                    wl0 = param['wavel']
                    t3 = 0.0
                    for x in np.linspace(-0.5, 0.5, n_smear):
                        param['wavel'] = wl0 + x*param['dwavel']
                        t3 += Vbin([o[1], o[2]], param)*\
                              Vbin([o[3], o[4]], param)*\
                              np.conj(Vbin([o[1]+o[3], o[2]+o[4]], param))
                    param['wavel'] = wl0
                    t3 /= float(n_smear)
                if o[0]=='cp':
                    res[i] = -np.angle(t3)
                else:
                    res[i] = np.abs(t3)
            else: # -- no bandwidth smearing yet ;(
                # -- assumes star is not resolved (first lobe)
                phi12 = 2*np.pi*c*(param['x']*o[1]+param['y']*o[2])/param['wavel']
                phi23 = 2*np.pi*c*(param['x']*o[3]+param['y']*o[4])/param['wavel']
                phi31 = 2*np.pi*c*(param['x']*(o[1]+o[3])+
                                   param['y']*(o[2]+o[4]))/param['wavel']
                B12 = np.sqrt(o[1]**2+o[2]**2)
                B23 = np.sqrt(o[3]**2+o[4]**2)
                B31 = np.sqrt((o[1]+o[3])**2+(o[2]+o[4])**2)
                Vstar12 = np.abs(Vud(B12, param['diam*'], param['wavel']))
                Vstar23 = np.abs(Vud(B23, param['diam*'], param['wavel']))
                Vstar31 = np.abs(Vud(B31, param['diam*'], param['wavel']))
                if o[0]=='cp':
                    cp = param['f']*(np.sin(phi12)/Vstar12 +
                                     np.sin(phi23)/Vstar23 -
                                     np.sin(phi31)/Vstar31)
                    res[i] = -cp
                else:
                    # --
                    res[i] = (Vstar12*Vstar23*Vstar31 + param['f']*(
                              Vstar12*np.cos(phi12) +
                              Vstar23*np.cos(phi23) -
                              Vstar31*np.cos(phi31))
                              )/(1+param['f'])**3
    res2 = np.array([])
    for r in res:
        res2 = np.append(res2, r.flatten())
    return res2

def loadOifits2chi2Data(filename, obs=['cp', 'v2']):
    """
    obs can contain "cp", "v2" and/or "t3"

    load OIFITS file assuming one target, one OI_VIS2, one OI_T3 and one WAVE table
    """
    global _chi2Data, _dataheader
    f = fits.open(filename)
    _dataheader={}
    for k in ['X','Y','F']:
        try:
            _dataheader[k] = f[0].header['INJCOMP'+k]
        except:
            pass

    # -- load wavelength:
    wavel = {}
    for hdu in f[1:]:
        if hdu.header['EXTNAME']=='OI_WAVELENGTH':
            wavel[hdu.header['INSNAME']] = hdu.data['EFF_WAVE']*1e6 # in um
            print 'dwavel=', np.abs(np.diff(hdu.data['EFF_WAVE']*1e6).mean())

    # -- load all data:
    res, delta = [], []
    for hdu in f[1:]:
        if hdu.header['EXTNAME']=='OI_T3':
            ins = hdu.header['INSNAME']
            # -- load data
            res.append(('cp',
                  hdu.data['U1COORD'][:,None]+0*wavel[ins][None,:],
                  hdu.data['V1COORD'][:,None]+0*wavel[ins][None,:],
                  hdu.data['U2COORD'][:,None]+0*wavel[ins][None,:],
                  hdu.data['V2COORD'][:,None]+0*wavel[ins][None,:],
                  wavel[ins][None,:]+0*hdu.data['V1COORD'][:,None],
                  hdu.data['MJD'][:,None]+0*wavel[ins][None,:],
                  ins,
                  hdu.data['T3PHI']*np.pi/180,
                  hdu.data['T3PHIERR']*np.pi/180))
            res.append(('t3',
                  hdu.data['U1COORD'][:,None]+0*wavel[ins][None,:],
                  hdu.data['V1COORD'][:,None]+0*wavel[ins][None,:],
                  hdu.data['U2COORD'][:,None]+0*wavel[ins][None,:],
                  hdu.data['V2COORD'][:,None]+0*wavel[ins][None,:],
                  wavel[ins][None,:]+0*hdu.data['V1COORD'][:,None],
                  hdu.data['MJD'][:,None]+0*wavel[ins][None,:], ins,
                  hdu.data['T3AMP'],
                  hdu.data['T3AMPERR']))
        if hdu.header['EXTNAME']=='OI_VIS2':
            ins = hdu.header['INSNAME']
            res.append(('v2',
                  hdu.data['UCOORD'][:,None]+0*wavel[ins][None,:],
                  hdu.data['VCOORD'][:,None]+0*wavel[ins][None,:],
                  wavel[ins][None,:]+0*hdu.data['VCOORD'][:,None],
                  hdu.data['MJD'][:,None]+0*wavel[ins][None,:], ins,
                  hdu.data['VIS2DATA'],
                  hdu.data['VIS2ERR']))
    f.close()

    # -- compute a flatten version of all V2:
    allV2 = {'u':np.array([]), 'v':np.array([]), 'mjd':np.array([]),
             'wl':np.array([]), 'v2':np.array([])}

    for r in filter(lambda x: x[0]=='v2', res):
        allV2['u'] = np.append(allV2['u'], r[1].flatten())
        allV2['v'] = np.append(allV2['v'], r[2].flatten())
        allV2['wl'] = np.append(allV2['wl'], r[3].flatten())
        allV2['mjd'] = np.append(allV2['mjd'], r[4].flatten())
        allV2['v2'] = np.append(allV2['v2'], r[-2].flatten())

    # -- delta for approximation, very long!
    delta = []
    for r in res:
        if r[0] in ['cp', 't3']:
            # -- this will contain the delta for this r
            vis1, vis2, vis3 = np.zeros(r[-2].shape), np.zeros(r[-2].shape), np.zeros(r[-2].shape)
            for i in range(r[-2].shape[0]):
                for j in range(r[-2].shape[1]):
                    # -- find Vis
                    k1 = np.argmin((allV2['u']-r[1][i,j])**2+
                                   (allV2['v']-r[2][i,j])**2+
                                   (allV2['wl']-r[5][i,j])**2+
                                   ((allV2['mjd']-r[6][i,j])/10000.)**2)
                    vis1[i,j] = np.sqrt(allV2['v2'][k1])
                    k2 = np.argmin((allV2['u']-r[3][i,j])**2+
                                   (allV2['v']-r[4][i,j])**2+
                                   (allV2['wl']-r[5][i,j])**2+
                                   ((allV2['mjd']-r[6][i,j])/10000.)**2)
                    vis2[i,j] = np.sqrt(allV2['v2'][k2])
                    k3 = np.argmin((allV2['u']-r[1][i,j]-r[3][i,j])**2+
                                   (allV2['v']-r[2][i,j]-r[4][i,j])**2+
                                   (allV2['wl']-r[5][i,j])**2+
                                   ((allV2['mjd']-r[6][i,j])/10000.)**2)
                    vis3[i,j] = np.sqrt(allV2['v2'][k3])
            delta.append((vis1, vis2, vis3))
        if r[0] == 'v2':
            delta.append(None)

    delta = [delta[i] for i in range(len(res)) if res[i][0] in obs]
    res = [res[i] for i in range(len(res)) if res[i][0] in obs]

    _chi2Data = res
    return res, delta

def injectCompanionData(data, delta, param):
    """
    data and delta have same length
    """
    global c, n_smear
    res = [[x if isinstance(x, str) else x.copy() for x in d] for d in data] # copy the data

    for i,d in enumerate(res):
        if d[0]=='v2':
            if not 'dwavel' in param.keys():
                phi = c*2*np.pi*(d[1]*param['x']+d[2]*param['y'])/d[3]
                d[-2] = (d[-2] + 2*param['f']*np.sqrt(np.abs(d[-2]))*np.cos(phi))/(1+param['f'])**2
            else:
                # -- assumes bandwidthSmearing is a spectral Resolution:
                tmp = 0.0
                for x in np.linspace(-0.5, 0.5, n_smear)*param['dwavel']:
                    phi = c*2*np.pi*(d[1]*param['x']+d[2]*param['y'])/(d[3]+x)
                    tmp += (d[-2] + 2*param['f']*np.sqrt(np.abs(d[-2]))*np.cos(phi))/(1+param['f'])**2
                d[-2] = tmp/float(n_smear)

        if d[0]=='cp':
            if not 'dwavel' in param.keys():
                phi0 = c*2*np.pi*(d[1]*param['x']+d[2]*param['y'])/d[5]
                phi1 = c*2*np.pi*(d[3]*param['x']+d[4]*param['y'])/d[5]
                phi2 = c*2*np.pi*((d[1]+d[3])*param['x']+(d[2]+d[4])*param['y'])/d[5]
                d[-2] -= param['f']*(np.sin(phi0)/delta[i][0]+
                                     np.sin(phi1)/delta[i][1]-
                                     np.sin(phi2)/delta[i][2])
            else:
                # -- assumes bandwidthSmearing is a sepctral Resolution:
                tmp = 0.0
                for x in np.linspace(-0.5, 0.5, n_smear)*param['dwavel']:
                    phi0 = c*2*np.pi*(d[1]*param['x']+d[2]*param['y'])/(d[5]+x)
                    phi1 = c*2*np.pi*(d[3]*param['x']+d[4]*param['y'])/(d[5]+x)
                    phi2 = c*2*np.pi*((d[1]+d[3])*param['x']+(d[2]+d[4])*param['y'])/(d[5]+x)
                    tmp += param['f']*(np.sin(phi0)/delta[i][0]+
                                       np.sin(phi1)/delta[i][1]-
                                       np.sin(phi2)/delta[i][2])
                d[-2] -= tmp/float(n_smear)

        if d[0]=='t3':
            if not 'dwavel' in param.keys():
                phi0 = c*2*np.pi*(d[1]*param['x']+d[2]*param['y'])/d[5]
                phi1 = c*2*np.pi*(d[3]*param['x']+d[4]*param['y'])/d[5]
                phi2 = c*2*np.pi*((d[1]+d[3])*param['x']+(d[2]+d[4])*param['y'])/d[5]
                d[-2] += 2*param['f']*(np.cos(phi0)/delta[i][0]+
                                       np.cos(phi1)/delta[i][1]-
                                       np.cos(phi2)/delta[i][2])/(1+param['f'])**3
            else:
                # -- assumes bandwidthSmearing is a sepctral Resolution:
                tmp = 0.0
                for x in np.linspace(-0.5, 0.5, n_smear)*param['dwavel']:
                    phi0 = c*2*np.pi*(d[1]*param['x']+d[2]*param['y'])/(d[5]+x)
                    phi1 = c*2*np.pi*(d[3]*param['x']+d[4]*param['y'])/(d[5]+x)
                    phi2 = c*2*np.pi*((d[1]+d[3])*param['x']+(d[2]+d[4])*param['y'])/(d[5]+x)
                    tmp += 2*param['f']*(np.cos(phi0)/delta[i][0]+
                                       np.cos(phi1)/delta[i][1]-
                                       np.cos(phi2)/delta[i][2])/(1+param['f'])**3
                d[-2] += tmp/float(n_smear)
    return res

def chi2Func(param, approx=False):
    """
    - assumes _chi2Data is defined (loaded from OIFITS file using loadOifits2chi2Data)
    - assumes all observations have same wavelength table
    """
    global _chi2Data
    res, err = np.array([]), np.array([])
    for c in _chi2Data:
        res = np.append(res, c[-2].flatten())
        err = np.append(err, c[-1].flatten())
    err += err==0. # remove bad point in a dirty way

    return np.mean((res-observable(_chi2Data, param, approx=approx))**2/err**2)

def nSigmas(chi2r_TEST, chi2r_TRUE, NDOF):
    """
    - chi2r_TEST is the hypothesis we test
    - chi2r_TRUE is what we think is what described best the data
    - NDOF: numer of degres of freedom

    chi2r_TRUE <= chi2r_TEST

    returns the nSigma detection
    """
    p = scipy.stats.chi2.cdf(NDOF, NDOF*chi2r_TEST/chi2r_TRUE)
    log10p = np.log10(p)
    p = np.maximum(p, 1e-100)
    res = np.sqrt(scipy.stats.chi2.ppf(1-p,1))
    # x = np.logspace(-15,-12,100)
    # c = np.polyfit(np.log10(x), np.sqrt(scipy.stats.chi2.ppf(1-x,53)), 1)
    c = np.array([-0.25028407,  9.66640457])
    if isinstance(res, np.ndarray):
        res[log10p<-15] = np.polyval(c, log10p[log10p<-15])
    else:
        if log10p<-15:
            res =  np.polyval(c, log10p)
    return res

def fitFunc(param):
    global _chi2Data, _progress, _nfit, _next
    res, err = np.array([]), np.array([])
    for c in _chi2Data:
        res = np.append(res, c[-2].flatten())
        err = np.append(err, c[-1].flatten())
    err += err==0. # remove bad point in a dirty way

    fitOnly=[]
    if param['f']!=0:
        fitOnly.extend(['x', 'y', 'f'])
    if 'v2' in [c[0] for c in _chi2Data] or 't3' in [c[0] for c in _chi2Data]:
       fitOnly.append('diam*')
    return dpfit.leastsqFit(observable, _chi2Data, param,
                            res, err, fitOnly = fitOnly)

def fitMap(filename, observables=['cp', 'v2', 't3'], N=40, rmax=20, rmin=0.0, diam=1.0,  fig=0, addfits=False, dwavel=None,addCompanion=None):
    """
    - filename: a standard OIFITS data file
    - observables = ['cp', 'v2', 't3'] list of observables to take into account in the file. Default is all
    - N: starts fits on a NxN grid
    - rmax: size of the grid in mas (20 will search between -20mas to +20mas)
    - rmin: do not look into the inner radius, in mas
    - diam: angular diameter of the central star, in mas. If 'v2' are present and in "observables",
        the UD diam will actually be fitted
    - fig=0: the figure number (default 0)
    - addfits: add the fit and fimap in the FITS file, as a binary table
    """
    fr = 0.02
    global cmap
    global _dataheader, _chi2Data, _rawData, _delta
    if isinstance(filename, list):
        _rawData, _delta = [], []
        for f in filename:
            tmp = loadOifits2chi2Data(f, observables)
            _rawData.extend(tmp[0])
            _delta.extend(tmp[0])
    else:
        _rawData, _delta = loadOifits2chi2Data(filename, observables)

    if not addCompanion is None:
        # -- 'f' < 0 to actually remove a companion !!!
        _rawData = injectCompanionData(_rawData, _delta, addCompanion)
        print 'injected companion at', addCompanion
    _chi2Data = _rawData

    if _dataheader!={}:
        print 'found injected companion at', _dataheader
    res = {}

    t0 = time.time()
    params = []

    # -- compute ndata
    ndata = 0
    for c in _chi2Data:
        ndata += len(c[-1].flatten())
    res['NDATA'] = ndata

    if 'v2' in observables or 't3' in observables:
        fit_0 = fitFunc({'x':0, 'y':0, 'f':0, 'diam*':diam})
        chi2_0 = fit_0['chi2']
        print 'best fit diameter: %5.3f +- %5.3f mas'%(fit_0['best']['diam*'],
                                                       fit_0['uncer']['diam*'])
        diam = fit_0['best']['diam*']

    else:
        if not dwavel is None:
            chi2_0 = chi2Func({'x':0, 'y':0, 'f':0, 'diam*':diam, 'dwavel':dwavel})
        else:
            chi2_0 = chi2Func({'x':0, 'y':0, 'f':0, 'diam*':diam})

    print 'Chi2 without companion: %5.3f'%(chi2_0)
    res['FIT GRID DIAM'] = diam
    res['UD FIT CHI2R'] = chi2_0

    X = np.linspace(-rmax, rmax, N)
    Y = np.linspace(-rmax, rmax, N)
    res['FIT GRID N'] = N
    res['FIT GRID rmax'] = rmax
    res['FIT GRID OBS'] = ', '.join(observables)
    for y in Y:
        for x in X:
            if x**2+y**2>rmin**2:
                if not dwavel is None:
                    params.append({'diam*': diam, 'f':fr, 'x':x, 'y':y, 'dwavel':dwavel})
                else:
                    params.append({'diam*': diam, 'f':fr, 'x':x, 'y':y})

    # -- parallel on 6 cores
    p = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    allfits = p.map_async(fitFunc, params)
    p.close()
    p.join()
    allfits = allfits.get()
    # -- single thread
    #allfits = map(fitFunc, params)

    for i, f in enumerate(allfits):
        f['best']['f'] = np.abs(f['best']['f'])
        # -- distance from start to finish
        f['dist'] = np.sqrt((params[i]['x']-f['best']['x'])**2+
                            (params[i]['y']-f['best']['y'])**2)

    # -- count number of unique minima, start with first one, add N sigma:
    allMin = [allfits[0]]
    allMin[-1]['nsigma'] =  nSigmas(chi2_0,  allMin[-1]['chi2'], res['NDATA']-1)
    for f in allfits:
        chi2 = []
        for a in allMin:
            tmp, n = 0., 0.

            for k in ['x', 'y', 'f', 'diam*']:
                if f['uncer'][k]!=0 and a['uncer'][k]!=0:
                    tmp += (f['best'][k]-a['best'][k])**2/(f['uncer'][k]**2+a['uncer'][k]**2)
                    n += 1.
            tmp /= n
            chi2.append(tmp)
        if not any([c<=1 for c in chi2]):
            allMin.append(f)
            allMin[-1]['nsigma'] =  nSigmas(chi2_0,  allMin[-1]['chi2'], res['NDATA']-1)

    # -- plot histogram of distance from start to end of fit:
    if False:
        plt.close(10+fig)
        plt.figure(10+fig)
        plt.hist([f['dist'] for f in allfits], bins=20, normed=1, label='all fits')
        plt.xlabel('distance start -> best fit (mas)')
        plt.vlines(2*rmax/float(N), 0, plt.ylim()[1], color='r', linewidth=2, label='grid size')
        plt.vlines(2*rmax/float(N)*np.sqrt(2)/2, 0, plt.ylim()[1], color='r', linewidth=2,
                   label=r'grid size $\sqrt(2)/2$', linestyle='dashed')
        plt.legend()
        plt.text(plt.xlim()[1]-0.03*(plt.xlim()[1]-plt.xlim()[0]),
                 plt.ylim()[1]-0.03*(plt.ylim()[1]-plt.ylim()[0]),
                 '# of unique minima=%d / grid size=%d'%(len(allMin),  N*N),
                 ha='right', va='top')

    # -- is the grid to wide?
    Nest = max(np.sqrt(2*len(allMin)), rmax/np.median([f['dist'] for f in allfits])*np.sqrt(2))
    if len(allMin)/float(N*N)>0.5 or\
         2*rmax/float(N)*np.sqrt(2)/2>np.median([f['dist'] for f in allfits]):
         print '\033[41mWARNING, grid is too wide!!!',
         print '--> try N=%d\033[0m'%(np.ceil(Nest))
    elif N>1.2*Nest:
        print '\033[43mWARNING, grid may be to fine!!!',
        print '--> N=%d should be enough\033[0m'%(np.ceil(Nest))
    else:
        print '\033[42mGrid has the correct steps of %4.2fmas, optimimum step size found to be %4.2fmas\033[0m'%(
                2*rmax/float(N), 2*rmax/Nest)

    # == plot chi2 min map:
    _X, _Y = np.meshgrid(np.linspace(X[0]-np.diff(X)[0]/2.,
                                     X[-1]+np.diff(X)[0]/2.,
                                     2*len(X)+1),
                         np.linspace(Y[0]-np.diff(Y)[0]/2.,
                                     Y[-1]+np.diff(Y)[0]/2.,
                                     2*len(Y)+1))

    rbf = scipy.interpolate.Rbf([x['best']['x'] for x in allfits if x['best']['x']**2+x['best']['y']**2>rmin**2],
                                [x['best']['y'] for x in allfits if x['best']['x']**2+x['best']['y']**2>rmin**2],
                                [x['chi2'] for x in allfits if x['best']['x']**2+x['best']['y']**2>rmin**2],
                                function='linear')
    _Z = rbf(_X, _Y)

    plt.close(fig)
    plt.figure(fig, figsize=(12,5))
    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.92,
                        wspace=0.2, hspace=0.2)

    ax1 = plt.subplot(1,2,1)
    plt.title('$\chi^2_r$ best fit ($\chi^2_{UD}$=%4.2f)'%chi2_0)
    plt.pcolormesh(_X-0.5*rmax/float(N), _Y-0.5*rmax/float(N), _Z, cmap=cmap+'_r')
    plt.colorbar()
    plt.xlabel('X (mas)')
    plt.ylabel('Y (mas)')
    if rmin>0:
        c = plt.Circle((0,0), rmin, color='k', alpha=0.33, hatch='x')
        ax1.add_patch(c)

    for i, f in enumerate(allfits):
        #plt.plot(params[i]['x'], params[i]['y'], '+c',
        #         markersize=6, alpha=0.5)
        #plt.plot(f['best']['x'], f['best']['y'], ',c',
        #         markersize=8, alpha=0.3)
        plt.plot([f['best']['x'], params[i]['x']],
                 [f['best']['y'], params[i]['y']], '-y',
                 alpha=0.3, linewidth=2)
    ax1.set_aspect('equal', 'datalim')
    plt.xlim(-rmax, rmax)
    plt.ylim(-rmax, rmax)

    # -- http://www.aanda.org/articles/aa/pdf/2011/11/aa17719-11.pdf section 3.2
    n_sigma = nSigmas(chi2_0, _Z, res['NDATA']-1)

    ax2 = plt.subplot(1,2,2, sharex=ax1, sharey=ax1)
    plt.title('n$\sigma$ of detection')
    plt.pcolormesh(_X-0.5*rmax/float(N), _Y-0.5*rmax/float(N), n_sigma, cmap=cmap, vmin=0)
    plt.colorbar()
    plt.xlabel('X (mas)')
    plt.ylabel('Y (mas)')
    if rmin>0:
        c = plt.Circle((0,0), rmin, color='k', alpha=0.33, hatch='x')
        ax2.add_patch(c)
    ax2.set_aspect('equal', 'datalim')
    plt.xlim(-rmax, rmax)
    plt.ylim(-rmax, rmax)

    # --
    print 'FIT grid of size', N*N, 'in', round(time.time()-t0,1), 's'
    nsmax = np.max([a['nsigma'] for a in allMin])
    # -- keep nSigma higher than half the max
    allMin2 = filter(lambda x: x['nsigma']>nsmax/2. and a['best']['x']**2+a['best']['y']**2 > rmin**2, allMin)
    # -- keep 5 highest nSigma
    allMin2 = [allMin[i] for i in np.argsort([c['chi2'] for c in allMin])[:5]]

    for ii, i in enumerate(np.argsort([x['chi2'] for x in allMin2])):
        print '-'*32
        print 'BEST FIT %d: chi2=%5.2f'%(ii, allMin2[i]['chi2'])
        for s in ['x', 'y', 'f', 'diam*']:
            print '%5s='%s, '%5.2e +- %5.2e'%(allMin2[i]['best'][s], allMin2[i]['uncer'][s])

        # -- http://www.aanda.org/articles/aa/pdf/2011/11/aa17719-11.pdf section 3.2
        print 'chi2r_UD=%5.2f, chi2r_BIN=%5.2f, NDOF=%d'%(chi2_0, allMin2[i]['chi2'], res['NDATA']-1),
        print '  -> n sigma: %5.2f'%allMin2[i]['nsigma']

        ax1.plot(allMin2[i]['best']['x']+rmax/float(N)*np.array([-1,0,1,0,-1]),
                 allMin2[i]['best']['y']+rmax/float(N)*np.array([0,1,0,-1,0]),
                 '-', linewidth=min(allMin2[i]['nsigma'], 2),
                 alpha=min(max(allMin2[i]['nsigma']/5., 0.2), .8),
                 color='0.5' if ii>0 else 'r')
        ax2.plot(allMin2[i]['best']['x']+rmax/float(N)*np.array([-1,0,1,0,-1]),
                allMin2[i]['best']['y']+rmax/float(N)*np.array([0,1,0,-1,0]),
                '-', linewidth=min(allMin2[i]['nsigma'], 2),
                alpha=min(max(allMin2[i]['nsigma']/5., 0.2), .8),
                color='0.5' if ii>0 else 'r')
        ax2.text(allMin2[i]['best']['x'], allMin2[i]['best']['y'],
                r'%4.1f$\sigma$'%allMin2[i]['nsigma'], ha='left', va='bottom',
                color='0.5' if ii>0 else 'r')

    # -- best fit parameters:
    j = np.argmin([x['chi2'] for x in allfits if x['best']['x']**2+x['best']['y']**2>rmin**2])
    ret = [x for x in allfits if x['best']['x']**2+x['best']['y']**2>rmin**2][j]
    # -- compare with injected companion
    if 'X' in _dataheader.keys() and 'Y' in _dataheader.keys() and 'F' in _dataheader.keys():
        x0 = [x['best']['x'] for x in allfits if x['best']['x']**2+x['best']['y']**2>rmin**2][j]
        ex0 = [x['uncer']['x'] for x in allfits if x['best']['x']**2+x['best']['y']**2>rmin**2][j]
        y0 = [x['best']['y'] for x in allfits if x['best']['x']**2+x['best']['y']**2>rmin**2][j]
        ey0 = [x['uncer']['y'] for x in allfits if x['best']['x']**2+x['best']['y']**2>rmin**2][j]
        f0 = [x['best']['f'] for x in allfits if x['best']['x']**2+x['best']['y']**2>rmin**2][j]
        ef0 = [x['uncer']['f'] for x in allfits if x['best']['x']**2+x['best']['y']**2>rmin**2][j]
        print 'injected X:', _dataheader['X'], 'found at %3.1f sigmas'%((x0-_dataheader['X'])/ex0)
        print 'injected Y:', _dataheader['Y'], 'found at %3.1f sigmas'%((y0-_dataheader['Y'])/ey0)
        print 'injected F:', _dataheader['F'], 'found at %3.1f sigmas'%((f0-_dataheader['F'])/ef0)
        ax1.plot(_dataheader['X'], _dataheader['Y'], 'py', markersize=12, alpha=0.3)

    if addfits:
        print ' > updating OIFITS file:', filename
        # === add result to fits file:
        f = fits.open(filename, mode='update')
        #print len(f), f.info()
        #f.close()
        # make new HDU, a binary table will all the fits
        cols = []
        cols.append(fits.Column(name='CHI2', format='E',
                    array=[x['chi2'] for x in allfits]))
        cols.append(fits.Column(name='X', format='E',
                    array=[x['best']['x'] for x in allfits], unit='mas'))
        cols.append(fits.Column(name='E_X', format='E',
                    array=[x['uncer']['x'] for x in allfits], unit='mas'))
        cols.append(fits.Column(name='Y', format='E',
                    array=[x['best']['y'] for x in allfits], unit='mas'))
        cols.append(fits.Column(name='E_Y', format='E',
                    array=[x['uncer']['y'] for x in allfits], unit='mas'))
        cols.append(fits.Column(name='F', format='E',
                    array=[x['best']['f'] for x in allfits]))
        cols.append(fits.Column(name='E_F', format='E',
                    array=[x['uncer']['f'] for x in allfits]))
        cols.append(fits.Column(name='DIAM', format='E',
                    array=[x['best']['diam*'] for x in allfits], unit='mas'))
        cols.append(fits.Column(name='E_DIAM', format='E',
                    array=[x['uncer']['diam*'] for x in allfits], unit='mas'))
        hducols = fits.ColDefs(cols)
        hdu = fits.new_table(hducols)
        for k in res.keys():
            hdu.header[k] = res[k]
        hdu.header['EXTNAME'] = 'CANDID_FITMAP'
        hdu.header['CANDID DATE'] = time.asctime()
        hdu.header['CANDID N'] = N
        hdu.header['CANDID RMIN'] = rmin
        hdu.header['CANDID RMAX'] = rmax
        hdu.header['CANDID DIAM'] = diam
        hdu.header['CANDID OBSERVABLES'] = ', '.join(observables)
        # -- remove gridfit if already present:
        for i in range(len(f)-1):
            if f[i+1].header['EXTNAME'] == 'CANDID_FITMAP':
                f.pop(i+1)
        f.append(hdu)
        f.close()
    print 'total time %4.1fs'%(time.time()-t0)
    #return ret

def detectionLimit(filename, observables=['cp'], diam=0.7, N=100, rmax=30, removeCompanion=None, dwavel=None, fig=0, ylim=None, plotMap=True):
    """
    method='Absil':
    ---------------
    for each position (x,y), chi2_binary(x,y,f)/Chi2_UD with increasing (f) until this ratio reaches a 3 sigma level.


    method=None:
    ------------
    for each position (x,y), inject a companion (f) to the data and compare the chi2_UD/chi2_binary(x,y,f) until this ratio reaches a 3 sigma level.

    """
    global _chi2Data, _rawData, _delta

    if isinstance(filename, list):
        _rawData, _delta = [], []
        for f in filename:
            tmp = loadOifits2chi2Data(f, observables)
            _rawData.extend(tmp[0])
            _delta.extend(tmp[0])
    else:
        _rawData, _delta = loadOifits2chi2Data(filename, observables)

    if not removeCompanion is None:
        # -- 'f' < 0 to actually remove a companion !!!
        _rawData = injectCompanionData(_rawData, _delta, removeCompanion)

    _chi2Data = _rawData

    # -- if V2 is fitted, find first best fit diameter:
    if 'v2' in observables:
        res, err = np.array([]), np.array([])
        for c in _chi2Data:
            res = np.append(res, c[-2].flatten())
            err = np.append(err, c[-1].flatten())
        err += err==0
        fit = dpfit.leastsqFit(observable, _chi2Data, {'x':0, 'y':0, 'f':0, 'diam*':0.5},
                                res, err, fitOnly = ['diam*'], verbose=0)
        print 'fitted diam: %5.3f +/- %5.3f mas'%(fit['best']['diam*'], fit['uncer']['diam*'])
        diam = fit['best']['diam*']

    # -- compute ndata
    ndata = 0
    for c in _rawData:
        ndata += len(c[-1].flatten())

    allX = np.linspace(-rmax, rmax, N)
    allY = np.linspace(-rmax, rmax, N)
    absil, compinj = [], []

    t0 = time.time()

    tmp = {'x':0.0, 'y':0.0, 'f':0.0, 'diam*':diam}
    if not dwavel is None:
        tmp['dwavel'] = dwavel
    _chi2_0 = chi2Func(tmp)

    _mult = np.sqrt(2)
    _stop = False
    fratio = [0.002] # start with 0.1%

    while not _stop:
        f = fratio[-1]
        print 'fratio=%4.2f%% '%(100*f),
        allP = []
        for j,y in enumerate(allY):
            for i,x in enumerate(allX):
                if dwavel is None:
                    allP.append({'x':x, 'y':y, 'f':f, 'diam*':diam})
                else:
                    allP.append({'x':x, 'y':y, 'f':f, 'diam*':diam, 'dwavel':dwavel})
        # -- parallel, leave one CPU for common tasks
        """
        Absil test: compare chi2UD and chi2Bin for increasing binary contrast.
        The chi2UD is the same, whatever the position of the putative companion,
        so we do ne recompute it every time.
        """
        p = multiprocessing.Pool(multiprocessing.cpu_count()-1)
        allChi2 = p.map_async(chi2Func, allP)
        p.close()
        p.join()
        allChi2 = np.array(allChi2.get())
        # -- result
        allChi2.resize((len(allY), len(allX)))
        absil.append(nSigmas(allChi2, _chi2_0, ndata))

        """
        Injection test: compare chi2UD and chi2Bin for increasing binary contrast.
        A companion in injected at (x,y,f) and the ratio of chi2 is computed.
        """
        p = multiprocessing.Pool(multiprocessing.cpu_count()-1)
        allChi2 = p.map_async(chi2_f1, allP)
        p.close()
        p.join()
        allChi2 = np.array(allChi2.get())
        # -- result
        allChi2.resize((len(allY), len(allX)))
        compinj.append(nSigmas(allChi2, 1, ndata))

        # -- check the 1% min nsigma in the map:
        crit1 = [np.percentile(absil[-1].flatten(), 1.),
                np.percentile(compinj[-1].flatten(), 1.)]
        crit99 = [np.percentile(absil[-1].flatten(), 99.),
                np.percentile(compinj[-1].flatten(), 99.)]

        print 'Nsigma (1%%->99%%) = %3.1f->%3.1f, %3.1f->%3.1f '%(crit1[0], crit99[0], crit1[1], crit99[1])
        # if np.max(crit99[0]) < 3.:
        #     fratio.append(fratio[-1]*_mult**2)
        if np.min(crit1[0]) < 4:
            fratio.append(fratio[-1]*_mult)
        else:
            _stop = True

    # -- 3 sigma detection level, interpolation
    det_a = np.zeros((len(allY), len(allX)))
    det_i = np.zeros((len(allY), len(allX)))
    for i in range(len(allX)):
        for j in range(len(allY)):
            det_a[j,i] = np.interp(3.0, [r[j,i] for r in absil], fratio)
            det_i[j,i] = np.interp(3.0, [r[j,i] for r in compinj], fratio)

    print 'done in %3.1f s'%(time.time()-t0)

    _x, _y = np.meshgrid(allX, allY)

    r = np.sqrt(_x**2+_y**2).flatten()
    d_a = 100.*det_a.flatten()
    d_i = 100.*det_i.flatten()
    d_a = d_a[np.argsort(r)]
    d_i = d_i[np.argsort(r)]
    r = r[np.argsort(r)]
    d_a, d_i, r = d_a[r<allX.max()], d_i[r<allX.max()], r[r<allX.max()]
    _chi2Data, _rawData = [], []
    if not fig is None:
        plt.close(fig)
        if plotMap:
            plt.figure(fig, figsize=(12,5))
            plt.subplots_adjust(left=0.1, right=0.96)
            ax = plt.subplot(121)
        else:
            plt.figure(fig, figsize=(10,5))
            plt.subplots_adjust(left=0.1, right=0.96)
            ax = plt.subplot(111)
        title = os.path.basename(filename)+'\nObservables: '+', '.join(observables)
        plt.title(title)

        plt.plot(r, sliding_percentile(r, d_a, 1., 50),
                 '-', linewidth=2, label='Absil 50%', color='r')
        plt.plot(r, sliding_percentile(r, d_a, 1., 90),
                 '-', linewidth=2, label='Absil 90%', linestyle='dashed',
                 color='r')
        plt.plot(r, sliding_percentile(r, d_i, 1., 50),
                 '-', linewidth=2, label='Comp.Inj. 50%', color='b')
        plt.plot(r, sliding_percentile(r, d_i, 1., 90),
                 '-', linewidth=2, label='Comp.Inj. 90%', linestyle='dashed',
                 color='b')

        plt.xlabel('radial distance (mas)')
        plt.ylabel('detection level (contrast, %)')
        plt.xlim(0, allX.max())
        if not ylim is None:
            plt.ylim(ylim[0], ylim[1])
        plt.legend(loc='upper center', ncol=2)
        plt.grid()
        if plotMap:
            plt.subplot(122)
            plt.pcolormesh(_x, _y, 100*det_i, cmap='jet')
            plt.colorbar()
            plt.title('flux ratio 3$\sigma$ detection (%)')
            plt.xlabel('x (mas)')
            plt.ylabel('y (mas)')


    else:
        return {'r':r, 'd_a':d_a, 'd_i':d_i}
    # ----

def chi2_f1(param):
    """
    compare chi2Bin and chi2UD for increasing binary contrast while injecting a companion at this particula position
    """
    global _chi2Data, _rawData, _delta
    _chi2Data = injectCompanionData(_rawData, _delta, param)
    # -- UD:
    tmp = {'x':0.0, 'y':0.0, 'f':0.0, 'diam*':param['diam*']}
    if 'dwavel' in param.keys():
        tmp['dwavel'] = param['dwavel']
    # -- ratio of chi2_UD / chi2_BIN
    return chi2Func(tmp)/chi2Func(param)

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

