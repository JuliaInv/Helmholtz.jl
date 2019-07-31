[![Build Status](https://travis-ci.org/JuliaInv/Helmholtz.jl.svg?branch=master)](https://travis-ci.org/JuliaInv/Helmholtz.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaInv/Helmholtz.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaInv/Helmholtz.jl?branch=master)

# Helmholtz.jl
A package for defining and solving the Helmholtz equation using the shifted Laplacian multigrid solver. 
This is done using the geometric MG version in Multigrid.jl.

This package is used in the following paper for solving the Helmholtz equation for multiple right-hand-sides (see Section 4 and 5.3):

Eran Treister and Eldad Haber, Full waveform inversion guided by travel time tomography, SIAM Journal on Scientific Computing, 39 (5), S587-S609, 2017.

# Requirements

This package is intended to use with julia versions 0.7 and later

This package is an add-on for [`jInv`](https://github.com/JuliaInv/jInv.jl), which needs to be installed. 

# Installation

```
Pkg.clone("https://github.com/JuliaInv/jInv.jl","jInv")
Pkg.clone("https://github.com/JuliaInv/Multigrid.jl","Multigrid")
Pkg.clone("https://github.com/JuliaInv/ForwardHelmholtz.jl","ForwardHelmholtz")
Pkg.clone("https://github.com/JuliaInv/ParSpMatVec.jl","ParSpMatVec")
Pkg.build("ParSpMatVec");

Pkg.test("Helmholtz")
```
# Examples
See //examples.

# PointSourceADR sub-package

Also available in this package is the sub-package `ForwardHelmholtz.PointSourceADR` for the solution of the Helmholtz equation for a point source, as described in the paper:

Eran Treister, Eldad Haber, A multigrid solver to the Helmholtz equation with a point source based on travel time and amplitude.

To use this sub-package type `using Helmholtz.PointSorceADR`. This sub-package also requires the FactoredEikonalFastMarching package, which can be installed by `Pkg.clone("https://github.com/JuliaInv/FactoredEikonalFastMarching.jl","FactoredEikonalFastMarching")`


