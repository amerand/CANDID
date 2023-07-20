import candid
from matplotlib import pyplot as plt
import os

def runAll(savefig=False):
    global o
    directory = os.path.join(os.path.dirname(candid.__file__), 'demo')
    o = candid.Open(os.path.join(directory, 'AXCir.oifits'))

    o.chi2Map(fig=1, fratio=1.0)

    o.fitMap(fig=2)

    o.fitBoot(fig=3)

    p = o.bestFit['best']
    o.fitMap(fig=4, removeCompanion=p)

    o.detectionLimit(fig=5, step=1.5, removeCompanion=p)

    plt.show()

    if savefig:
        for f in [1,2,3,4,5]:
            plt.figure(f)
            plt.savefig('doc/figure_%d.png'%f)
    return
