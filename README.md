# [C]ompanion [A]nalysis and [N]on-[D]etection in [I]nterferometric [D]ata

This is a suite of Python2.7 tools to find faint companion around star in interferometric data in the [OIFITS format](http://www.mrao.cam.ac.uk/research/optical-interferometry/oifits/). This tool allows to systematically search for faint companions in OIFITS data, and if not found, estimates the detection limit.

## What does it do for you?

### Companion Search

The tool is based on model fitting and Chi2 minimization ([scipy.optimize.leastsq](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.leastsq.html)), with a grid for the starting points of the companion position. It ensures that all positions are explored by estimating a-posteriori if the grid was dense enough, and provide an estimate of the optimum gride density (see example).

### Detection limit
It uses Chi2 statistics to estimate the level of detection in "number of sigmas".

### Non-Detection Limit
There are 2 approachs inplemented: [Absil et al. 2011](http://adsabs.harvard.edu/abs/2011A%26A...535A..68A) and CANDID's Analytical Companion Injection (Gallenne et al. 2015, in preparation).

## Known limitations

The code has *not* been deeply error proofed. If you encounter problems or bugs, do not hesitate to contact the developpers.

* The code works only with a single OIFITS file. You will need an external tool to combine OIFITS files
* The works only with simple OIFITS files: all observations should be with the same instrument (same OI_WAVELENGTH) and all data will be taken, assuming a single target is present.
* The code has been tested of OIFITS files form CHARA/MIRC and VLTI/PIONIER. If you use other instruments and encounter problems, please contact the developpers!
* The code is not particularly fast, but uses [multiprocessing](https://docs.python.org/2/library/multiprocessing.html): our experience on Macs is that it leads to several problems:
** does not work properly with IPython Notebooks. [It seems to be known](https://github.com/ipython/ipython/issues/6109).
** It works better with the IPython console, though it sometimes seem unresponsive (it makes sometimes the estimation of running time unreliable). Morover ctrl+C does not stop the code.
** The smoothest behavior is obtained by running in a python shell.
* The code can take lots of memory because it storse lots of intermediate reuslts, so using a 64bit python is advisable

## Examples:

see "candidTest.py", function "AXCir". on a Core i7 2.2GHz, It runs in about 50 seconds with the 'fast' option (fast indeed, but not accurate!), and almost 6 minutes in normal mode.
* **chi2 Maps**. These are useful because fast, but dangerous because it is easy to miss a companion just using those. On FIG1 and FIG2, we show to runs for different diamaters and flux ratio: either the diameter is fitted to the V2 data ([FIG1](https://github.com/amerand/CANDID/blob/master/doc/figure_1.png)). The chi2 map shows a minimum only if the grid if fine enough (the structure in the map should be clear, not pixelated) but also if the parameters (inc. the flux ratio) are very close to the actual ones.
* **fit Maps**. These are better, but much slower than chi2 maps. If V2 are present, the diameter will be fitted ([FIG2](https://github.com/amerand/CANDID/blob/master/doc/figure_2.png)). Note that once a companion is found, it can be removed analytically from the data and the fit map ran again ([FIG3](https://github.com/amerand/CANDID/blob/master/doc/figure_3.png)): this demontrates that, in the case of our example, the secondary "detections" are only artefact from the main companion.
* **detection limits**. We imlemented 2 methods; Absil's and our companion injection ([FIG4](https://github.com/amerand/CANDID/blob/master/doc/figure_4.png)). Note that they give different results. We argue that our method is more robust to correlated noise. As an example, one can see the effects of running thge detection limits algorithms to the raw data set ([FIG4](https://github.com/amerand/CANDID/blob/master/doc/figure_4.png)) and after the companion is removed ([FIG5](https://github.com/amerand/CANDID/blob/master/doc/figure_5.png)): Absil's method gives different results, while ours remain barely changed.


Open OIFITS file with CANDID, and restrict search from 2 to 35 mas. Also, only consider 'v2' and 'cp' observables:
```python
import candid
from matplotlib import pyplot as plt
axcir = candid.Open('AXCir.oifits', rmin=2, rmax=35)
axcir.observables=['v2','cp']
```

### FIG1 - CHI2MAP with 0.5 mas step, fitted diameter and fratio=1%:
```python
axcir.chi2Map(0.5, fig=1, fratio=0.01)
```
```
 > observables: ['v2', 'cp']
 > UD Fit
 | best fit diameter: 0.932 +- 0.006 mas
 | chi2 UD = 0.975
 | Computing Map 140x140 ... it should take about 23 seconds
 |================================================== | 99%   0 s remainingg
 | chi2 Min:  0.90
 | at X,Y  :   6.29, -28.45 mas
 | Nsigma  :  2.64
```
![Figure 1](https://github.com/amerand/CANDID/blob/master/doc/figure_1.png)


### FIG2 - FITMAP with 3.5 mas step:
```python
axcir.fitMap(3.5, fig=2)
```
```
 > observables: ['v2', 'cp']
 > Preliminary analysis
 > UD Fit
 | best fit diameter: 0.932 +- 0.006 mas
 | chi2 UD = 0.975
 > Grid Fitting 20x20:
 ... it should take about 35 seconds
 |================================================== | 99%   0 s remaining
 | Computing map of interpolated Chi2 minima
 | 115 individual minima for 275 fits
 | 10, 50, 90 percentiles for fit displacement: 1.4, 2.6, 5.1 mas
 | Grid has the correct steps of 3.50mas, optimimum step size found to be 3.50mas
 > BEST FIT 0: chi2= 0.73
 |     x=  6.24e+00 +- 5.76e-02
 |     y= -2.85e+01 +- 8.36e-02
 |     f=  9.38e-03 +- 4.26e-04
 | diam*=  8.34e-01 +- 7.46e-03
 | chi2r_UD=0.98, chi2r_BIN=0.73, NDOF=1499 -> n sigma: 14.17
 > fixed bandwidth parameters for the map (in um):
 | dwavel;PIONIER_Pnat(1.6135391/1.7698610) =  0.156322
```
![Figure 2](https://github.com/amerand/CANDID/blob/master/doc/figure_2.png)

### FIG3 - FITMAP with 3.5 mas step, after removing companion
```python
p = {'x':6.24, 'y':-28.5, 'f':0.0094}
axcir.fitMap(3.5, fig=3, removeCompanion=p)
```
```
> observables: ['v2', 'cp']
 > Preliminary analysis
 > UD Fit
 | best fit diameter: 0.815 +- 0.006 mas
 | chi2 UD = 0.753
 > Grid Fitting 20x20: ... it should take about 24 seconds
 |================================================== | 99%   0 s remaining
 | Computing map of interpolated Chi2 minima
 | 133 individual minima for 275 fits
 | 10, 50, 90 percentiles for fit displacement: 1.2, 2.9, 5.9 mas
 | Grid has the correct steps of 3.50mas, optimimum step size found to be 3.89mas
 > BEST FIT 0: chi2= 0.71
 |     x=  6.60e+00 +- 1.32e-01
 |     y=  2.27e+00 +- 1.80e-01
 |     f=  3.55e-03 +- 3.70e-04
 | diam*=  7.52e-01 +- 9.42e-03
 | chi2r_UD=0.75, chi2r_BIN=0.71, NDOF=1499 -> n sigma:  2.05
 > fixed bandwidth parameters for the map (in um):
 | dwavel;PIONIER_Pnat(1.6135391/1.7698610) =  0.156322
```

![Figure 3](https://github.com/amerand/CANDID/blob/master/doc/figure_3.png)

### FIG4 - DETECTION LIMIT, 1mas step
This leads to a result which is not relevant: we already know there is a companion in the data, which means the assumptions of the statistical tests are not met. Interestingly, the 2 methods diverege, which means there are not sensitive in the same way to the correlated "noise" introduced by the companion.

```python
axcir.detectionLimit(1.0, fig=4); plt.subplot(212);  plt.ylim(0.2, 1.0)
```
```
 > observables: ['v2', 'cp']
 > UD Fit
 | best fit diameter: 0.932 +- 0.006 mas
 | chi2 UD = 0.975
 > Detection Limit Map 70x70 ... it should take about 109 seconds
 > Method: Absil
 |================================================== | 99%   0 s remaining
 > Method: injection
 |================================================== | 99%   0 s remaining
```

![Figure 4](https://github.com/amerand/CANDID/blob/master/doc/figure_4.png)

### FIG5 - DETECTION LIMIT, 1mas step, after removing companion
We here remove the companion analyticaly (using a high contrast hypothesis) from the V2 and CP data. In that case, the two methods (Absil and companion injection) should give more similar results, assuming the remaining noise is uncorrelated (which we know is probably not true). Comparing this to the previous result, we can see that the injection method is more robust (less change compared to data with companion). The two methods still diverge, but less.
```python
p = {'x':6.24, 'y':-28.5, 'f':0.0094}
axcir.detectionLimit(1.0, fig=5, removeCompanion=p); plt.subplot(212); plt.ylim(0.2, 1.0)
```
```
 > observables: ['v2', 'cp']
 > UD Fit
 | best fit diameter: 0.815 +- 0.006 mas
 | chi2 UD = 0.753
 > Detection Limit Map 70x70 ... it should take about 114 seconds
 > Method: Absil
 |================================================== | 99%   0 s remaining
 > Method: injection
 |================================================== | 99%   0 s remaining
```
![Figure 5](https://github.com/amerand/CANDID/blob/master/doc/figure_5.png)


### CONFIG parameters

A global dictionnary CONFIG to set parameters. The list of default parameters is shown each time the library is loaded. To modidy the parameters, you should do it as such:
```python
candid.CONFIG['longExecWarning'] = 300
```
This will, for example, set the maximum computing time to 300s (instead of the default 180s). Note that this will have to be done every time you import or reload the library.

## Informations

### Link
https://github.com/amerand/CANDID

### Developpers
[Antoine Mérand](mailto:amerand-at-eso.org) and [Alexandre Gallenne](mailto:agallenne-at-astro-udec.cl)

### Python dependencies
python2.7, numpy, scipy, matplotlib and pyfits (or astropy)

### LICENCE
*---TBD---*