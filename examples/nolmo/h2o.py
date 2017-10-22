#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, dft, scf
from pyscf.scf import nolmo
from pyscf.future import tddft

'''
A simple example to run DFT calculation.

See pyscf/dft/vxc.py for the complete list of available XC functional
'''

mol = gto.Mole()
mol.build(
    atom = '''H    0.0   0.   0.
              O    0.96   0.   0.
              H   1.200364804   0.929421734   0.''',
#    basis = '3-21g',
    basis = 'sto-3g',
#    symmetry = True,
#    verbose = 9,
)

mydft = dft.RKS(mol)
#mydft.xc = 'lda,vwn'
#mydft.xc = 'lda,vwn_rpa'
#mydft.xc = 'b88,p86'
#mydft.xc = 'b88,lyp'
#mydft.xc = 'b97,pw91'
#mydft.xc = 'b3p86'
#mydft.xc = 'o3lyp'
mydft.xc = 'b3lyp'
#mydft.init_guess = '1e'
#mydft.conv_tol = 1.e-1
#mydft.kernel()
#mydft.conv_tol = 1.e-12
#mydft.nolmo()
#mydft.kernel()
#mydft.analyze()

scf = scf.RHF(mol)
scf.init_guess = '1e'
scf.conv_tol = 1.e-1
scf.kernel()
scf.conv_tol = 1.e-12
scf.nolmo()

scf.kernel()

