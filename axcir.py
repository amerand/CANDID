import os, platform, re, subprocess
from matplotlib import pyplot as plt
import time

import candid
candid.CONFIG['longExecWarning'] = None
if 'chapman.sc.eso.org' in platform.uname()[1]:
    # -- limit CPU usage on shared machine
    candid.CONFIG['Ncores'] = 8

def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        command ="/usr/sbin/sysctl -n machdep.cpu.brand_string"
        return os.popen(command).read().strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = os.popen(command).read().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
    return None

# -- example in https://github.com/amerand/CANDID/blob/master/README.md

savePlots = False
times = []

print '#'*80
t0 = time.time()
axcir = candid.Open('AXCir.oifits')
times.append(('Open', time.time()-t0))

print '#'*80
t0 = time.time()
axcir.chi2Map(fig=1, fratio=1.0)
times.append(('chi2Map', time.time()-t0))
if savePlots:
    plt.figure(1)
    plt.savefig('doc/figure_1.png')

print '#'*80
t0 = time.time()
axcir.fitMap(fig=2)
times.append(('fitMap', time.time()-t0))
if savePlots:
    plt.figure(2)
    plt.savefig('doc/figure_2.png')

print '#'*80
p = axcir.bestFit['best']
axcir.fitMap(fig=3, removeCompanion=p)
if savePlots:
    plt.figure(3)
    plt.savefig('doc/figure_3.png')

print '#'*80
t0 = time.time()
axcir.fitBoot(fig=4, param=p)
times.append(('fitBoot', time.time()-t0))
if savePlots:
    plt.figure(4)
    plt.savefig('doc/figure_4.png')

print '#'*80
t0 = time.time()
axcir.detectionLimit(fig=5, removeCompanion=p)
times.append(('detectionLimit', time.time()-t0))
if savePlots:
    plt.figure(5)
    plt.savefig('doc/figure_5.png')

print '#'*80
try:
    processor = get_processor_name()
except:
    processor = '???'
if candid.CONFIG['Ncores'] is None:
    import multiprocessing
    Nc = multiprocessing.cpu_count() - 1
else:
    Nc = candid.CONFIG['Ncores']
processor += ', using %d Cores'%Nc

print 'Execution times [', processor, ']'
for t in times:
    print '%14s in [%5.1fs]'%t
print '#'*80
