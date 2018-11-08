import candid
from matplotlib import pyplot as plt

def runAll(savefig=False):
    global o
    o = candid.Open('AXCir.oifits')

    o.chi2Map(fig=1, fratio=1.0)

    o.fitMap(fig=2)

    o.fitBoot(fig=3)

    p = o.bestFit['best']
    o.fitMap(fig=4, removeCompanion=p)

    o.detectionLimit(fig=5, step=1.5, removeCompanion=p)

    if savefig:
        for f in [1,2,3,4,5]:
            plt.figure(f)
            plt.savefig('doc/figure_%d.png'%f)
    return
