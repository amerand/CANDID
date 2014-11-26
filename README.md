# [C]ompanion [A]nalysis and [N]on-[D]etection in [I]nterferometric [D]ata

This is a suite of tools to find faint companion around star in
interferometric data in the [OIFITS format](http://www.mrao.cam.ac.uk/research
/optical-interferometry/oifits/). This tool allows to systematically search
for faint companions in OIFITS data, and if not found, estimates the detection
limit.

## What does it do for you?

### Companion Search

The tool is based on model fitting (scipy.optimize.leastsqFit) with a grid for
the starting point of the companion position. It unsures that all positions
are explored by estimating a-posteriori if the grid was dense enough, and
provide an estimate of the optimum gride density.

### Detection limit
It uses Chi2 statistics to estimate the level of detection in "number of
sigmas".

### Non-Detection Limit
There are 2 approachs inplemented: [Absil et al. 2011](http://adsabs.harvard.edu/abs/2011A%26A...535A..68A) and CANDID's Analytical Companion Injection (Gallenne et al. 2015, in preparation).

## Examples:

This whole example takes about 7 minutes to compute on a quadcore 2.2GHz Core i7.

```python
import candid

# -- open the MIRC OIFITS file with CANDID:
c = candid.open('AXCir.oifits')

# -- make a coarse grid fit (fast but unreliable):
c.fitMap(N=10, rmax=35, rmin=3, observables=['cp','v2','t3'], fig=1)

# -- make an optimum grid fit based on previous run (slow):
c.fitMap(N=c.Nest, rmax=35, rmin=3, observables=['cp','v2','t3'], fig=2)

# -- store the best fit companion
companion = c.compParam

# -- analytically remove companion and search again:
c.fitMap(N=c.Nest, rmax=35, rmin=3, observables=['cp','v2','t3'], fig=3, removeCompanion=companion)

# -- compute the detection limit, after analytically removing the companion
# -- grid should be finer too (takes a few minutes again)
c.detectionLimit(N=4*c.Nest, rmax=35, rmin=3, observables=['cp','v2','t3'], fig=4, removeCompanion=companion)
```

## Information

### Link
https://github.com/amerand/CANDID

### Developpers
[Antoine Mérand](mailto:amerand@eso.org) and Alexandre Gallenne

### Python dependencies
numpy, scipy, matplotlib and pyfits (or astropy)

### LICENCE
*---TBD---*