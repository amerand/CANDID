# [C]ompanion [A]nalysis and [N]on-[D]etection in [I]nterferometric [D]ata

This is a suite of Python2.7 tools to find faint companion around star in interferometric data in the [OIFITS format](http://www.mrao.cam.ac.uk/research/optical-interferometry/oifits/). This tool allows to systematically search for faint companions in OIFITS data, and if not found, estimates the detection limit.

## What does it do for you?

### Companion Search

The tool is based on model fitting and Chi2 minimization ([scipy.optimize.leastsq](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.leastsq.html)), with a grid for the starting points of the companion position. It ensures that all positions are explored by estimating a-posteriori if the grid was dense enough, and provide an estimate of the optimum gride density (see example).

### Detection limit
It uses Chi2 statistics to estimate the level of detection in "number of sigmas".

### Non-Detection Limit
There are 2 approachs inplemented: [Absil et al. 2011](http://adsabs.harvard.edu/abs/2011A%26A...535A..68A) and CANDID's Analytical Companion Injection [Gallenne et al. 2015](http://arxiv.org/abs/1505.02715).

## Known limitations

The code has *not* been deeply error proofed. If you encounter problems or bugs, do not hesitate to contact the developpers.

* **The (non-)detection levels are given assuming the errors bars in the data are uncorrelated**. This is of course not the case for real data, in particular when a lot of spectral channels are present.
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
* **detection limits**. We imlemented 2 methods; Absil's and our companion injection. Note that they give slightly different results: we argue that our method is more robust to correlated noise (read our paper!). When you have detected a companion and wish to estimate the detection limit, it is important to first remove analytically the companion ([FIG5](https://github.com/amerand/CANDID/blob/master/doc/figure_5.png)).


**We strongly recommand you use plain python2.7, instead of iPython, because of the bad interactions between iPython and the multiprocessing library, which makes the estimation of the running time very unreliable.**

Open OIFITS file with CANDID:

```
>>> import candid
>>> from matplotlib import pyplot as plt
>>> axcir = candid.Open('AXCir.oifits')
 > loading file AXCir.oifits
 | WARNING: no valid T3AMP values in this HDU
 | WARNING: no valid T3AMP values in this HDU
 | Smallest spatial scale:       2.68 mas
 | Diffraction Field of view:  221.88 mas
 | WL Smearing Field of view:   55.24 mas
 | observables available: [ 'v2', 'cp']
 | instruments: [ 'VLTI-PIONIER_Pnat(1.6135391/1.7698610)']
 | rmin= not given, set to smallest spatial scale: rmin= 2.68 mas
 | rmax= not given, set to Field of View: rmax=55.24 mas
```

### FIG1 - CHI2MAP: fitted diameter and fixed flux ratio=1%:
The easiest thing to try is a chi2 map, assuming a certain flux ratio for the companion. This is quite inefficient but CANDID allows to do it. If no parametrization is given (step size 'step=', maximum radius for search 'rmax'), CANDID will guess some values based on the angular resolution and the wavelength smearing field of view. The flux ratio is given in percent.

```
>>> axcir.chi2Map(fig=1, fratio=1.0)
 > step= not given, using 1/6 X smallest spatial scale = 0.45 mas
 > observables: ['v2', 'cp']
 > UD Fit
 | best fit diameter: 0.932 +- 0.006 mas
 | chi2 UD = 0.975
 | Computing Map 248x248 ... it should take about 70 seconds
 |================================================== | 99%   0 s remainingng
 | chi2 Min:  0.97
 | at X,Y  : -12.30,   3.35 mas
 | Nsigma  :  0.83
```
![Figure 1](https://github.com/amerand/CANDID/blob/master/doc/figure_1.png)


### FIG2 - FITMAP:
Doing a grid of fit is much more efficient than doing a simple Chi2 Map (like for ([FIG1](https://github.com/amerand/CANDID/blob/master/doc/figure_1.png))). In a FITMAP, a set of binary fits are performed starting from a 2D grid of companion position. The plot displays the interpolated map of the chi2 minima (left), with the path of the fit, from start to finish (yellow lines). FITMAP will compute, a posteriori, what was the correct step size 'step='. In our example below, we let CANDID chose the step size, based on the angular resoultion of the data (1.2 wavelength/baseline). The companion is detected at the same position as for the previous example, with a much better dynamic range.
```
>>> axcir.fitMap(fig=2)
 > step= not given, using sqrt(2) x smallest spatial scale = 3.78 mas
 > observables: ['v2', 'cp']
 > Preliminary analysis
 > UD Fit
 | best fit diameter: 0.932 +- 0.006 mas
 | chi2 UD = 0.975
 > Grid Fitting 30x30: ... it should take about 54 seconds
 |================================================== |  99%   0 s remaining
 | Computing map of interpolated Chi2 minima
 | 255 individual minima for 643 fits
 | 10, 50, 90 percentiles for fit displacement: 1.1, 2.8, 5.3 mas
 | Grid has the correct steps of 3.78mas, optimimum step size found to be 3.81mas
 > BEST FIT 0: chi2= 0.73
 |     x=   6.2403 +- 0.0575 mas
 |     y= -28.5270 +- 0.0835 mas
 |     f=   0.9395 +- 0.0423 %
 | diam*=   0.8156 +- 0.0081 mas
 | chi2r_UD=0.98, chi2r_BIN=0.73, NDOF=1499 -> n sigma: 14.35 (assumes uncorr data)
```
![Figure 2](https://github.com/amerand/CANDID/blob/master/doc/figure_2.png)

### FIG3 - FITMAP, after removing companion
CANDID offers the possibility, once a companion has been detected, to remove it analytically from the data and rerun a FITMAP. This allows to estimate the dynamic range of the data set, but also to detect fainter tertiary compents. fitMap stores the best fit in the dictionnary "bestFit", which key "best" contains the dictionnary containing the parameters. Note that axcir.bestFit['uncer'] contains the corresponding error bars.

```
>>> p = axcir.bestFit['best']
>>> axcir.fitMap(fig=3, removeCompanion=p)
 > step= not given, using sqrt(2) x smallest spatial scale = 3.78 mas
 > observables: ['v2', 'cp']
 > Preliminary analysis
 > UD Fit
 | best fit diameter: 0.814 +- 0.006 mas
 | chi2 UD = 0.738
 > Grid Fitting 30x30: ... it should take about 32 seconds
 |================================================== |  99%   0 s remaining
 | Computing map of interpolated Chi2 minima
 | 308 individual minima for 643 fits
 | 10, 50, 90 percentiles for fit displacement: 1.2, 3.1, 6.4 mas
 | Grid has the correct steps of 3.78mas, optimimum step size found to be 4.25mas
 > BEST FIT 0: chi2= 0.69
 |     x=   6.6173 +- 0.1284 mas
 |     y=   2.4005 +- 0.1745 mas
 |     f=   0.3648 +- 0.0373 %
 | diam*=   0.7491 +- 0.0094 mas
 | chi2r_UD=0.74, chi2r_BIN=0.69, NDOF=1499 -> n sigma:  2.10 (assumes uncorr data)
```

![Figure 3](https://github.com/amerand/CANDID/blob/master/doc/figure_3.png)

### FIG4 - ERROR BARS:
In order to better estimate the uncertainties on the companion we found, we can use bootstraping to estimate the incertainties around the best fit parameters. The default starting is the best fitted position: since we made a fit with an analytical removal of the companion, the currently stored companion is not the one found on Fig2; thankfully we stored it in the variable 'p'.

On the correlations plots, the red dot with error bars is the fitted position; the blue ellipses are derived from the bootstrap (using a principal component analysis); the values given for each parameters are the median; the uncertainties are the 16% and 84% percentile (one sigma).

```
>>> axcir.fitBoot(fig=4, param=p)
 > 'N=' not given, setting it to Ndata/2
 |================================================== |  99%   0 s remaining
 > sigma clipping in position and flux ratio for nSigmaClip= 3.5
 | 3 fits ignored
 |    diam* =   0.8144 + 0.0118 - 0.0158 mas
 |        f =   0.9505 + 0.0984 - 0.1062 %
 |        x =   6.2286 + 0.0950 - 0.1467 mas
 |        y = -28.5341 + 0.1343 - 0.2424 mas
```

![Figure 4](https://github.com/amerand/CANDID/blob/master/doc/figure_4.png)


### FIG5 - DETECTION LIMIT, after analytically removing companion:
We here remove the companion analytically (using a high contrast hypothesis) from the V2 and CP data. This is mandatory in order to estimate the detection limit: the statistical hypothesis of the test is that the data are best described by a uniform disk.

```
>>> axcir.detectionLimit(fig=5, step=1.5, removeCompanion=p)
 > observables: ['v2', 'cp']
 > UD Fit
 | best fit diameter: 0.814 +- 0.006 mas
 | chi2 UD = 0.738
 > Detection Limit Map 74x74 ... it should take about 139 seconds
 > Method: Absil
 |================================================== | 99%   0 s remaining
 > Method: injection
 |================================================== | 99%   0 s remainingg
```
![Figure 5](https://github.com/amerand/CANDID/blob/master/doc/figure_5.png)

### CONFIG parameters

A global dictionnary CONFIG to set parameters. The list of default parameters is shown each time the library is loaded. To modidy the parameters, you should do it as such:
```
>>> candid.CONFIG['longExecWarning'] = 300
```
This will, for example, set the maximum computing time to 300s (instead of the default 180s). Note that this will have to be done every time you import or reload the library.

## Informations

### Link
https://github.com/amerand/CANDID

### Developpers
[Antoine Mérand](mailto:amerand-at-eso.org) and [Alexandre Gallenne](mailto:agallenne-at-astro-udec.cl)

### Python dependencies
python2.7, numpy, scipy, matplotlib and pyfits (or astropy)

### LICENCE (BSD)

Copyright (c) 2015, Antoine Mérand
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.