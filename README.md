# [C]ompanion [A]nalysis and [N]on-[D]etection in [I]nterferometric [D]ata

This is a suite of tools to find faint companion around star in interferometric data in the [OIFITS format](http://www.mrao.cam.ac.uk/research/optical-interferometry/oifits/). This tool allows to systematically search for faint companions in OIFITS data, and if not found, estimates the detection limit.

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
* the "flags" in OIFITS are not yet taken into account. Will come soon
* The works only with simple OIFITS files: all observations should be with the same instrument (same OI_WAVELENGTH) and all data will be taken, assuming a single target is present.
* The code has been tested of OIFITS files form CHARA/MIRC and VLTI/PIONIER. If you use other instruments and encounter problems, please contact the developpers!
* The code is not particularly fast, but uses [multiprocessing](https://docs.python.org/2/library/multiprocessing.html): our experience on Macs is that it leads to several problems:
** does not work properly with IPython Notebooks. [It seems to be known](https://github.com/ipython/ipython/issues/6109).
** It works better with the IPython console, though it sometimes seem unresponsive (it makes sometimes the estimation of running time unreliable). Morover ctrl+C does not stop the code.
** The smoothest behavior is obtained by running in a python shell.
* The code can take lots of memory because it storse lots of intermediate reuslts, so using a 64bit python is advisable

## Examples:

see "candidTest.py", function "AXCir". on a Core i7 2.2GHz, It runs in about 50 seconds with the 'fast' option (fast indeed, but not accurate!), and almost 6 minutes in normal mode.
* **chi2 Maps**. These are useful because fast, but dangerous because it is easy to miss a companion just using those. On FIG1 and FIG2, we show to runs for different diamaters and flux ratio: either the diameter is fitted to the V2 data ([FIG1](https://github.com/amerand/CANDID/blob/master/doc/figure_1.png)) or fixed ([FIG2](https://github.com/amerand/CANDID/blob/master/doc/figure_2.png)). The fit sounds better but actually it is biased if a companion is present. The chi2 map shows a deep minimum only if the grid if fine enough (the structure in the map should be clear, not pixelated) but also if the parameters (inc the flux ratio) are very close to the actual ones.
* **fit Maps**. These are better, but much slower than chi2 maps. If V2 are present, the diameter will be fitted ([FIG3](https://github.com/amerand/CANDID/blob/master/doc/figure_3.png)). Note that once a companion is found, it can be removed analytically from the data and the fit map ran again ([FIG4](https://github.com/amerand/CANDID/blob/master/doc/figure_4.png)): this demontrates that, in the case of our example, the secondary "detections" are only artefact from the main companion.
* **detection limits**. We imlemented 2 methods; Absil's and our companion injection ([FIG5](https://github.com/amerand/CANDID/blob/master/doc/figure_5.png)). Note that they give different results. We argue that our method is more robust to correlated noise. As an example, one can see the effects of running thge detection limits algorithms to the raw data set ([FIG5](https://github.com/amerand/CANDID/blob/master/doc/figure_5.png)) and after the companion is removed ([FIG6](https://github.com/amerand/CANDID/blob/master/doc/figure_6.png)): Absil's method gives different results, while ours remain barely changed.


Open OIFITS file with CANDID, and restrict search from 2 to 35 mas. Also, only consider 'v2' and 'cp' observables:
```python
import candid
axcir = candid.Open('AXCir.oifits', rmin=2, rmax=35)
axcir.observables=['v2','cp']
```

### FIG1 - CHI2MAP with 0.7 mas step, fitted diameter and fratio=1%:
```python
axcir.chi2Map(0.7, fig=1, fratio=0.01)
```
```
 > observables: ['v2', 'cp']
 > UD Fit
 | best fit diameter: 0.932 +- 0.006 mas
 | chi2 UD = 0.975
 | Computing Map 100x100 ... it should take about 23 seconds
 |================================================== | 99%   0 s remainingg
 | chi2 Min:  0.96
 | at X,Y =   6.01, -28.64 mas
 | Nsigma:    0.86
```
![Figure 1](https://github.com/amerand/CANDID/blob/master/doc/figure_1.png)

### FIG2 - CHI2MAP with 0.7 mas step, and known parameters: diam=0.82 mas, fratio=0.9%:
```python
axcir.chi2Map(0.7, fig=2, diam=0.82, fratio=0.009)
```
```
 | Chi2 UD for diam=0.820mas
 |  chi2 UD = 1.168
 | Computing Map 100x100 ... it should take about 22 seconds
 |================================================== | 99%   0 s remainingg
 | chi2 Min:  0.73
 | at X,Y =   6.01, -28.64 mas
 | Nsigma:   22.07
```
![Figure 2](https://github.com/amerand/CANDID/blob/master/doc/figure_2.png)

### FIG3 - FITMAP with 3.5 mas step:
```python
axcir.fitMap(3.5, fig=3)
```
```
 > observables: ['v2', 'cp']
 > Preliminary analysis
 > UD Fit
 | best fit diameter: 0.932 +- 0.006 mas
 | chi2 UD = 0.975
 > Grid Fitting 20x20: ... it should take about 79 seconds
 |================================================== | 99%   0 s remaining
 | Computing map of interpolated Chi2 minima
 | 118 individual minima for 275 fits
 | 10, 50, 90 percentiles for fit displacement: 1.3, 2.6, 5.5 mas
 | Grid has the correct steps of 3.50mas, optimimum step size found to be 3.00mas
 > BEST FIT 0: chi2= 0.73
 |     x= 6.23e+00 +- 5.82e-02
 |     y= -2.85e+01 +- 7.99e-02
 |     f= -8.90e-03 +- 4.04e-04
 | diam*= 8.22e-01 +- 7.90e-03
 | chi2r_UD=0.98, chi2r_BIN=0.73, NDOF=1499 -> n sigma: 14.16
```
![Figure 3](https://github.com/amerand/CANDID/blob/master/doc/figure_3.png)

### FIG4 - FITMAP with 3.5 mas step, after removing companion
```python
p = {'x':6.23, 'y':-28.5, 'f':0.0089}
axcir.fitMap(3.5, fig=4, removeCompanion=p)
```
```
 > observables: ['v2', 'cp']
 > Preliminary analysis
 > UD Fit
 | best fit diameter: 0.822 +- 0.006 mas
 | chi2 UD = 0.750
 > Grid Fitting 20x20: ... it should take about 143 seconds
 |================================================== | 99%   0 s remainingg
 | Computing map of interpolated Chi2 minima
 | 138 individual minima for 275 fits
 | 10, 50, 90 percentiles for fit displacement: 1.2, 2.9, 5.5 mas
 | Grid has the correct steps of 3.50mas, optimimum step size found to be 3.00mas
 > BEST FIT 0: chi2= 0.70
 |     x= 6.56e+00 +- 1.28e-01
 |     y= 2.28e+00 +- 1.77e-01
 |     f= 3.60e-03 +- 3.67e-04
 | diam*= 7.58e-01 +- 9.33e-03
 | chi2r_UD=0.75, chi2r_BIN=0.70, NDOF=1499 -> n sigma:  2.12
```

![Figure 4](https://github.com/amerand/CANDID/blob/master/doc/figure_4.png)

### FIG5 - DETECTION LIMIT, 1mas step
```python
axcir.detectionLimit(1.0, fig=5)
```
```
 > observables: ['v2', 'cp']
 > UD Fit
 | best fit diameter: 0.932 +- 0.006 mas
 | chi2 UD = 0.975
 > Detection Limit Map 70x70 ... it should take about 112 seconds
 > Method: Absil
 |================================================== | 99%   0 s remaining
 > Method: injection
 |================================================== | 99%   0 s remaining
```

![Figure 5](https://github.com/amerand/CANDID/blob/master/doc/figure_5.png)

### FIG6 - DETECTION LIMIT, 1mas step, after removing companion
```python
p = {'x':6.23, 'y':-28.5, 'f':0.0089}
axcir.detectionLimit(1.0, fig=6, removeCompanion=p)
```
```
 > observables: ['v2', 'cp']
 > UD Fit
 | best fit diameter: 0.822 +- 0.006 mas
 | chi2 UD = 0.750
 > Detection Limit Map 70x70 ... it should take about 109 seconds
 > Method: Absil
 |================================================== | 99%   0 s remaining
 > Method: injection
 |================================================== | 99%   0 s remainingg
```
![Figure 6](https://github.com/amerand/CANDID/blob/master/doc/figure_6.png)

## Informations

### Link
https://github.com/amerand/CANDID

### Developpers
[Antoine Mérand](mailto:amerand@eso.org) and Alexandre Gallenne

### Python dependencies
numpy, scipy, matplotlib and pyfits (or astropy)

### LICENCE
*---TBD---*