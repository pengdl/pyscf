#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, dft, scf

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
mydft.kernel()

# Orbital energies, Mulliken population etc.
mydft.analyze()

#myscf = scf.RHF(mol)
#myscf.kernel()
#myscf.analyze()


