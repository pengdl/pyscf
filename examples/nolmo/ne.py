#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, dft, scf
from pyscf.scf import x2c

'''
A simple example to run DFT calculation.

See pyscf/dft/vxc.py for the complete list of available XC functional
'''

mol = gto.Mole()
mol.build(
    atom = 'Ne 0 0 0',  # in Angstrom
    basis = 'unc-ccpvdz',
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
#mydft.init_guess = '1e'
#mydft.kernel()

# Orbital energies, Mulliken population etc.
#mydft.analyze()

myscf = scf.RHF(mol)
myscf.conv_tol = 1.e-12
myscf.kernel()
sx2c = x2c.sfx2c1e(myscf)
sx2c.kernel()
#myscf.analyze()
X2C=x2c.UHF(mol)
X2C.kernel()
dhf = scf.DHF(mol)
dhf.kernel()
dhf.with_gaunt = True
dhf.kernel()
dhf.with_breit = True
dhf.kernel()
