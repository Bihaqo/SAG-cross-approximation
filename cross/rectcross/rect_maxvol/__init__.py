""".. moduleauthor:: Alexander Mikhalev <muxasizhevsk@gmail.com>

:mod:`rect_maxvol` module contains routines to find good submatrices.
How good matrix is depends on special extreme properties of the matrix.
Two of this properties are 1-volume and 2-volume with the following formulas:

:math:`vol_1(A) = \\left|\\det(A)\\right|,\\, vol_2(A) = \\sqrt{\\max(\\det(A^TA), \\det(AA^T))}`

:math:`vol_1` can be applied only to square matrices.
Routines :py:func:`rect_maxvol() <rect_maxvol.rect_maxvol>` and :py:func:`maxvol() <rect_maxvol.maxvol>` and their derivatives :py:func:`rect_maxvol_svd() <rect_maxvol.rect_maxvol_svd>`, :py:func:`maxvol_svd() <rect_maxvol.maxvol_svd>`, :py:func:`rect_maxvol_qr() <rect_maxvol.rect_maxvol_qr()>`, :py:func:`maxvol_qr() <rect_maxvol.maxvol_qr>` are greedy optimizations of :math:`vol_1` and :math:`vol_2` for submatrices.

Functions
=========
"""

__all__ = ['rect_maxvol', 'maxvol', 'rect_maxvol_svd', 'maxvol_svd', 'rect_maxvol_qr', 'maxvol_qr']

from rect_maxvol import *

