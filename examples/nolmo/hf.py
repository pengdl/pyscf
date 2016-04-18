#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, dft, scf
from pyscf.scf import nolmo

'''
A simple example to run DFT calculation.

See pyscf/dft/vxc.py for the complete list of available XC functional
'''

mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = 'sto3g',
#    symmetry = True,
#    verbose = 9,
)

mydft = dft.RKS(mol)
#mydft.xc = 'lda,vwn'
#mydft.xc = 'lda,vwn_rpa'
#mydft.xc = 'b88,p86'
mydft.xc = 'b88,lyp'
#mydft.xc = 'b97,pw91'
#mydft.xc = 'b3p86'
#mydft.xc = 'o3lyp'
#mydft.xc = 'b3lyp'
mydft.init_guess = '1e'
mydft.conv_tol = 1.e-1
mydft.kernel()
mydft.conv_tol = 1.e-12
nolmo.scf(mydft)
mydft.kernel()

#mydft.analyze()
#scf = scf.RHF(mol)
#scf.kernel()

