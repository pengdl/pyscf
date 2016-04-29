#!/usr/bin/env python

import time
import ctypes
import _ctypes
from functools import reduce
import numpy
import scipy.linalg
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import dhf
from pyscf.scf import _vhf
from pyscf.scf.x2c import _uncontract_mol,_get_hcore_fw

def mt(a):
    x = []
    for i in a:
        x.append(''.join([ '%7.3f'%k.real  for k in i ]))
    return '\n'+'\n'.join(x)

def mtr(a):
    x = []
    for i in a:
        x.append(''.join([ '%7.3f'%k.real  for k in i ]))
    return '\n'+'\n'.join(x)

def mti(a):
    x = []
    for i in a:
        x.append(''.join([ '%7.3f'%k.imag  for k in i ]))
    return '\n'+'\n'.join(x)

def get_hcore(mol):
    xmol, contr_coeff = _uncontract_mol(mol, True)

    c = mol.light_speed
    t = xmol.intor_symmetric('cint1e_kin_sph')
    v = xmol.intor_symmetric('cint1e_nuc_sph')
    s = xmol.intor_symmetric('cint1e_ovlp_sph')
    w = xmol.intor_symmetric('cint1e_pnucp_sph')
    m = xmol.intor('cint1e_pnucxp_sph',3)
    wx = m[0]
    wy = m[1]
    wz = m[2]
    w0 = w

    nao = t.shape[0]
    n0 = 0
    n1 = nao
    n2 = nao * 2
    n3 = nao * 3
    n4 = nao * 4
    h = numpy.zeros((n4,n4),dtype=numpy.complex)
    m = numpy.zeros((n4,n4),dtype=numpy.complex)
    ss = numpy.zeros((n2,n2),dtype=numpy.complex)
    h[n0:n1,n0:n1] = v
    h[n1:n2,n1:n2] = v
    h[n2:n3,n0:n1] = t
    h[n3:n4,n1:n2] = t
    h[n1:n2,n3:n4] = t
    h[n0:n1,n2:n3] = t
    h[n2:n3,n2:n3] -= t
    h[n3:n4,n3:n4] -= t

    h[n2:n3,n2:n3] += (.25/c**2) * (  w0 + wz*1.0j )
    h[n3:n4,n3:n4] += (.25/c**2) * (  w0 - wz*1.0j )
    h[n2:n3,n3:n4] += (.25/c**2) * (  wy + wx*1.0j )
    h[n3:n4,n2:n3] += (.25/c**2) * ( -wy + wx*1.0j )

    m[n0:n1,n0:n1] = s
    m[n1:n2,n1:n2] = s
    ss += m[:n2,:n2] 
    m[n2:n3,n2:n3] = t * (.5/c**2)
    m[n3:n4,n3:n4] = t * (.5/c**2)

    e, a = scipy.linalg.eigh(h, m)

    cl = a[:n2,n2:]
    cs = a[n2:,n2:]
    w, u = numpy.linalg.eigh(reduce(numpy.dot, (cl.T.conj(), ss, cl)))
    idx = w > 1e-14
    r = reduce(numpy.dot, (u[:,idx]/numpy.sqrt(w[idx]), u[:,idx].T.conj(), cl.T.conj(), ss))
    h1 = reduce(numpy.dot, (r.T.conj()*e[n2:], r))
    d1, d2 = contr_coeff.shape
    cc = numpy.zeros((d1*2,d2*2),dtype=numpy.complex)
    cc[:d1,:d2] = contr_coeff
    cc[d1:,d2:] = contr_coeff
    h1 = reduce(numpy.dot, (cc.T.conj(), h1, cc))
    return h1

def make_rdm1(mo_coeff, mo_occ):
    return numpy.dot(mo_coeff*mo_occ, mo_coeff.T.conj())

def get_jk(mol, dm, hermi=0, mf_opt=None):
    #print 'dmR:',mtr(dm)
    #print 'dmI:',mti(dm)
    n2 = dm.shape[0]
    n = n2 / 2
    aar = dm[:n,:n].real
    aai = dm[:n,:n].imag
    abr = dm[:n,n:].real
    abi = dm[:n,n:].imag
    #bbr = dm[n:,n:].real
    #bbi = dm[n:,n:].imag
    #bar = dm[n:,:n].real
    #bai = dm[n:,:n].imag
    dmx = numpy.array([aar,aai,abr,abi])

    vj, vk = _vhf.direct(numpy.array(dmx, copy=False),
                         mol._atm, mol._bas, mol._env,
                         vhfopt=mf_opt, hermi=hermi)
    vjx = numpy.zeros( (n2,n2),dtype=numpy.complex)
    vkx = numpy.zeros( (n2,n2),dtype=numpy.complex)

    vjx[:n,:n] =  vj[0]*2.0
    vjx[n:,n:] =  vj[0]*2.0
    vkx[:n,:n] =  vk[0] + vk[1]*1.0j
    vkx[:n,n:] =  vk[2] + vk[3]*1.0j
    vkx[n:,:n] = -vk[2] + vk[3]*1.0j
    vkx[n:,n:] =  vk[0] - vk[1]*1.0j

    return vjx, vkx


class R2C(hf.SCF):
    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        self.xuncontract = False
        self.xequation = '1e'
        self._keys = self._keys.union(['xequation', 'xuncontract'])

    def dump_flags(self):
        hf.SCF.dump_flags(self)
        logger.info(self, 'X equation %s', self.xequation)
        return self

    def build_(self, mol=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.direct_scf:
            self.opt = self.init_direct_scf(self.mol)

    def eig(self, h, s):
        e, c = scipy.linalg.eigh(h, s)
        idx = numpy.argmax(abs(c.real), axis=0)
        c[:,c[idx,range(len(e))].real<0] *= -1
        return e, c

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol)

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        s = mol.intor_symmetric('cint1e_ovlp_sph')
        n = s.shape[0]
        n2 = n*2
        ss = numpy.zeros( (n2,n2),dtype=numpy.complex)
        ss[:n,:n] = s
        ss[n:,n:] = s
        return ss

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[:mol.nelectron] = 1
        if mol.nelectron < len(mo_energy):
            logger.info(self, 'nocc = %d  HOMO = %.12g  LUMO = %.12g', \
                        mol.nelectron, mo_energy[mol.nelectron-1],
                        mo_energy[mol.nelectron])
        else:
            logger.info(self, 'nocc = %d  HOMO = %.12g  no LUMO', \
                        mol.nelectron, mo_energy[mol.nelectron-1])
        logger.debug(self, '  mo_energy = %s', mo_energy)
        return mo_occ

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        dm = make_rdm1(mo_coeff, mo_occ)
        return dm

    def get_jk_(self, mol=None, dm=None, hermi=0):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        t0 = (time.clock(), time.time())
        if self.direct_scf and self.opt is None:
            self.opt = self.init_direct_scf(mol)
        vj, vk = get_jk(mol, dm, hermi, self.opt)
        logger.timer(self, 'vj and vk', *t0)
        return vj, vk

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=0):
        '''Dirac-Coulomb'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self.direct_scf:
            ddm = numpy.array(dm, copy=False) - numpy.array(dm_last, copy=False)
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            return numpy.array(vhf_last, copy=False) + vj - vk
        else:
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk

