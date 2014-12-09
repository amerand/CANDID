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

see "candidTest.py", function "AXCir". on a Core i7 2.2GHz, It runs in about 50 seconds with the 'fast' option (fasr indeed, but not accurate!), and almost 6 minutes in normal mode.

Open OIFITS file with CANDID, and restrict search from 2 to 35 mas. Also, only consider 'v2' and 'cp' observables:
```python
axcir = candid.Open('AXCir.oifits', rmin=2, rmax=35)
axcir.observables=['v2','cp']
```

FIG1 - CHI2MAP with 0.7 mas step, fitted diameter and fratio=1%:
```python
axcir.chi2Map(0.7, fig=1, fratio=0.01)
```
![Figure 1](https://github.com/amerand/CANDID/blob/master/doc/figure_1.png)

FIG2 - CHI2MAP with 0.7 mas step, and known parameters: diam=0.82 mas, fratio=0.9%:
```python
axcir.chi2Map(0.7, fig=2, diam=0.82, fratio=0.009)
```
![Figure 2](https://github.com/amerand/CANDID/blob/master/doc/figure_2.png)

FIG3 - FITMAP with 3.5 mas step:
```python
axcir.fitMap(3.5, fig=3)
```
![Figure 3](https://github.com/amerand/CANDID/blob/master/doc/figure_3.png)

FIG4 - FITMAP with 3.5 mas step, after removing companion
```python
p = {'x':6.23, 'y':-28.5, 'f':0.0089}
axcir.fitMap(3.5, fig=4, removeCompanion=p)
```
![Figure 4](https://github.com/amerand/CANDID/blob/master/doc/figure_4.png)

FIG5 - DETECTION LIMIT, 1mas step
```python
axcir.detectionLimit(1.0, fig=5)
```
![Figure 5](https://github.com/amerand/CANDID/blob/master/doc/figure_5.png)


FIG6 - DETECTION LIMIT, 1mas step, after removing companion
```python
p = {'x':6.23, 'y':-28.5, 'f':0.0089}
axcir.detectionLimit(1.0, fig=6, removeCompanion=p)
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