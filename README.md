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

* works only with a single OIFITS file. You will need an external tool to combine OIFITS files
* works only with simple OIFITS files: all observations should be with the same instrument (same OI_WAVELENGTH) and all data will be taken, assuming a single target is present.
* the code has been tested of OIFITS files form CHARA/MIRC and VLTI/PIONIER. If you use other instruments and encounter problems, please contact the developpers!
* the code use [multiprocessing](https://docs.python.org/2/library/multiprocessing.html): our experience is that it does not work properly with IPython Notebooks. [It seems to be known](https://github.com/ipython/ipython/issues/6109). However, it works fine with IPython.

## Examples:

This whole example takes about 7 minutes to compute on a quadcore 2.2GHz Core i7.

```python
import candid

# -- open the MIRC OIFITS file with CANDID:
c = candid.open('AXCir.oifits')

# -- make a coarse grid fit (fast but unreliable):
c.fitMap(N=10, rmax=35, rmin=3, observables=['cp','v2','t3'], fig=1)
```
![Figure 1](https://github.com/amerand/CANDID/blob/master/doc/AXCir_fig1.png)
```python
# -- make an optimum grid fit based on previous run (slow):
c.fitMap(N=c.Nest, rmax=35, rmin=3, observables=['cp','v2','t3'], fig=2)
```
![Figure 2](https://github.com/amerand/CANDID/blob/master/doc/AXCir_fig2.png)
```python
# -- store the best fit companion
companion = c.compParam

# -- analytically remove companion and search again:
c.fitMap(N=c.Nest, rmax=35, rmin=3, observables=['cp','v2','t3'], fig=3, removeCompanion=companion)
```
![Figure 3](https://github.com/amerand/CANDID/blob/master/doc/AXCir_fig3.png)
```python
# -- compute the detection limit, after analytically removing the companion
# -- grid should be finer too (takes a few minutes again)
c.detectionLimit(N=4*c.Nest, rmax=35, rmin=3, observables=['cp','v2','t3'], fig=4, removeCompanion=companion)
```
![Figure 4](https://github.com/amerand/CANDID/blob/master/doc/AXCir_fig4.png)

## Informations

### Link
https://github.com/amerand/CANDID

### Developpers
[Antoine Mérand](mailto:amerand@eso.org) and Alexandre Gallenne

### Python dependencies
numpy, scipy, matplotlib and pyfits (or astropy)

### LICENCE
*---TBD---*