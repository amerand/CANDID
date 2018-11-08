import candid
from matplotlib import pyplot as plt

def runAll(savefig=False):
    global o
    o = candid.Open('AXCir.oifits')
    o.chi2Map(fig=1, fratio=1.0)
    if savefig:
        plt.savefig('doc/figure_1.png')

    o.fitMap(fig=2)
    if savefig:
        plt.savefig('doc/figure_2.png')

    o.fitBoot(fig=3)
    if savefig:
        plt.savefig('doc/figure_3.png')

    p = o.bestFit['best']
    o.fitMap(fig=4, removeCompanion=p)
    if savefig:
        plt.savefig('doc/figure_4.png')

    o.detectionLimit(fig=5, step=1.5, removeCompanion=p)
    if savefig:
        plt.savefig('doc/figure_5.png')
    return
