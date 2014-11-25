**C**ompanion **A**nalysis and **N**on-**D**etection in **I**nterferometric **D**ata

This is a suite of tools to find faint companion around star in interferometric
data in the OIFITS format.

# What does it do for you?
## Companion Search

It is based on model fitting (scipy.optimize.leastsqFit) with a grid for the
starting point of the companion position. It unsures that all positions are
explored by estimating a-posteriori if the grid was dense enough, and provide
an estimate of the optimum gride density.

## Detection limit
It uses Chi2 statistics to estimate the level of detection in "number of
sigmas".

## Non-Detection Limit
There are 2 approachs inplemented: Absil (---ref---) and CANDID's analytical
companion injection (---ref---).

# Examples:
'''python
import candid
c = candid.open(filename)
c.fitMap()
c.detectionLimit()
'''

# Information

## Link
https://github.com/amerand/CANDID

## Developpers
Antoine Mérand and Alexandre Gallenne

## Python depencedences
numpy, scipy, matplotlib and pyfits (or astropy)

## LICENCE
