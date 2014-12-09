import candid

"""
import candidTest
reload(candidTest.candid); reload(candidTest)

"""

def AXCir(fast=False):
    """
    checks the basic functionalities of the library.

    'fast=True' uses grids 9 times bigger, so calculations are 9 times faster
    """
    global axcir # <- to check things afterwards

    candid.plt.close('all')

    # -- define what to test, each can be 'none', 'simple' or 'full':
    tests = {'chi2Map':'full', 'fitMap':'full', 'detect':'full'}
    #tests = {'chi2Map':'none', 'fitMap':'none', 'detect':'full'}

    if fast:
        c=2.5
    else:
        c=1.

    filename = 'AXCir.oifits'
    print '\nLOADING:', filename, '(rmin=2 mas, rmax=35 mas)'
    axcir = candid.Open(filename, rmin=2, rmax=35)

    print '\nFITUD with v2 only'
    axcir.observables = ['v2']; axcir.fitUD()
    print '\nFITUD with CP only'
    axcir.observables = ['cp']; axcir.fitUD()

    # -- PIONIER data have no T3 reduced, restrincting to V2 and CP
    axcir.observables = ['v2', 'cp']

    fig=1

    if tests['chi2Map'] == 'simple' or\
        tests['chi2Map'] == 'full':
        print '\nFIG%d - CHI2MAP with 0.5 mas step, fitted diameter and fratio=1%%'%fig
        axcir.chi2Map(.7*c, fig=fig, fratio=0.01)
        fig+=1

    if tests['chi2Map'] == 'full':
        print '\nFIG%d - CHI2MAP with 0.5 mas step and known parameters: diam=0.82 mas, fratio=0.9%%'%fig
        axcir.chi2Map(.7*c, fig=fig, diam=0.82, fratio=0.009)
        fig+=1

    if tests['fitMap'] == 'simple' or\
        tests['fitMap'] == 'full':
        print '\nFIG%d - FITMAP with 3.5 mas step '%fig
        axcir.fitMap(3.5*c, fig=fig)
        fig+=1

    if tests['fitMap'] == 'full':
        print '\nFIG%d - FITMAP with 3.5 mas step, after removing companion'%fig
        #p = {'x':6.23, 'y':-28.5, 'f':0.0089}
        print 'best parameters according to previous step:', axcir.compParam
        axcir.fitMap(3.5*c, fig=fig, removeCompanion=axcir.compParam)
        fig+=1

    if tests['detect'] == 'simple' or\
        tests['detect'] == 'full':
        print '\nFIG%d - DETECTION LIMIT'%fig
        # -- where companion is
        axcir.detectionLimit(1*c, fig=fig)
        fig+=1

    if tests['detect'] == 'full':
        print '\nFIG%d - DETECTION LIMIT after removing companion'%fig
        # -- where companion is
        p = {'x':6.23, 'y':-28.5, 'f':0.0089}
        axcir.detectionLimit(1*c, fig=fig, removeCompanion=p)
        fig+=1

    print '\nCLOSING:', filename
    axcir.close()
    return