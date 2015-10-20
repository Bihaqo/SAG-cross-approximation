# What is this repository for?

*rect_maxvol* is a tool to find good square or rectangular submatrices.
Many computational problems require to find submatrix with certain extreme properties.
For example, it is necessary to use pivoting when computing LU factorization to avoid division by close to zero numbers.
Technique, used in *rect_maxvol* module, is based on greedy optimization of 1- and 2-volume of submatrix.

# Documentation

Documentation and examples are available at http://rect-maxvol.readthedocs.org/
Also, there is standalone ipython notebook example *examples/example.ipynb*

# Requirements

You need only *numpy* and *scipy* to run example, since there are pure python functions, but code will run much faster if compiled with *cython* files.

# Installation

If you want to install package into system, just run:
`python setup.py install`
Package can be built inplace with the following command:
`python setup.py build_ext --inplace`

# Main Contributor

Alexander Mikhalev <muxasizhevsk@gmail.com>
