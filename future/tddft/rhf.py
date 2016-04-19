#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
#

from functools import reduce
import numpy
import scipy
import pyscf.lib
from pyscf.tddft import davidson
from pyscf.ao2mo import _ao2mo
from pyscf.scf.nolmo import mt

class TDA(pyscf.lib.StreamObject):
    def __init__(self, mf):
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.mol = mf.mol
        self.chkfile = mf.chkfile
        self._scf = mf

        self.conv_tol = 1e-9
        self.nstates = 3
        self.singlet = True
        self.lindep = 1e-12
        self.level_shift = 0
        self.max_space = 40
        self.max_cycle = 100
        self.max_memory = mf.max_memory
        self.chkfile = mf.chkfile

        # xy = (X,Y), normlized to 1/2: 2(XX-YY) = 1
        # In TDA or TDHF, Y = 0
        self.e = None
        self.xy = None
        self.nomo = False
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('\n')
        log.info('******** %s for %s ********',
                 self.__class__, self._scf.__class__)
        log.info('nstates = %d', self.nstates)
        if self.singlet:
            log.info('Singlet')
        else:
            log.info('Triplet')
        log.info('conv_tol = %g', self.conv_tol)
        log.info('eigh lindep = %g', self.lindep)
        log.info('eigh level_shift = %g', self.level_shift)
        log.info('eigh max_space = %d', self.max_space)
        log.info('eigh max_cycle = %d', self.max_cycle)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, pyscf.lib.current_memory()[0])
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        log.info('\n')

    def get_vind(self, zs):
        '''Compute Ax'''
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (self._scf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        nz = len(zs)
        dmvo = numpy.empty((nz,nao,nao))
        for i, z in enumerate(zs):
            dmvo[i] = reduce(numpy.dot, (orbv, z.reshape(nvir,nocc), orbo.T))
        vj, vk = self._scf.get_jk(self.mol, dmvo, hermi=0)

        if self.singlet:
            vhf = vj*2 - vk
        else:
            vhf = -vk

        #v1vo = numpy.asarray([reduce(numpy.dot, (orbv.T, v, orbo)) for v in vhf])
        v1vo = _ao2mo.nr_e2_(vhf, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir*nocc)
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        eai = eai.ravel()
        for i, z in enumerate(zs):
            v1vo[i] += eai * z
        return v1vo.reshape(nz,-1)

    def abop(self, zs):
        '''Compute Ax and Bx'''
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (self._scf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        nz = len(zs)
        dmvo = numpy.empty((nz,nao,nao))
        for i, z in enumerate(zs):
            dmvo[i] = reduce(numpy.dot, (orbv, z.reshape(nvir,nocc), orbo.T))
        vj, vk = self._scf.get_jk(self.mol, dmvo, hermi=0)

        if self.singlet:
            vhf = vj*2 - vk
        else:
            vhf = -vk

        #v1vo = numpy.asarray([reduce(numpy.dot, (orbv.T, v, orbo)) for v in vhf])
        v1vo = _ao2mo.nr_e2_(vhf, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir*nocc)
#        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
#        eai = eai.ravel()
#        for i, z in enumerate(zs):
#            v1vo[i] += eai * z
        fv = reduce(numpy.dot, (orbv.T, self._scf.sfock, orbv))
        fo = reduce(numpy.dot, (orbo.T, self._scf.sfock, orbo))
        sv = reduce(numpy.dot, (orbv.T, self._scf.ss1e, orbv))
        so = reduce(numpy.dot, (orbo.T, self._scf.ss1e, orbo))
        ss = numpy.empty_like(zs)
        for i, z in enumerate(zs):
            zm = z.reshape(nvir,nocc)
            p1 = numpy.einsum('ij,kl,jl->ik',fv,so,zm)
            p2 = numpy.einsum('ij,kl,jl->ik',sv,fo,zm)
            ps = numpy.einsum('ij,kl,jl->ik',sv,so,zm)
            v1vo[i] += (p1-p2).ravel()
            ss[i] = ps.ravel()
        return v1vo.reshape(nz,-1), ss.reshape(nz,-1)

    def ddiag(self):
        '''diag Ax = eSx'''
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (self._scf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        zs = numpy.eye(nocc*nvir)
        nz = len(zs)
        dmvo = numpy.empty((nz,nao,nao))
        for i, z in enumerate(zs):
            dmvo[i] = reduce(numpy.dot, (orbv, z.reshape(nvir,nocc), orbo.T))
        vj, vk = self._scf.get_jk(self.mol, dmvo, hermi=0)

        if self.singlet:
            vhf = vj*2 - vk
        else:
            vhf = -vk

        #v1vo = numpy.asarray([reduce(numpy.dot, (orbv.T, v, orbo)) for v in vhf])
        v1vo = _ao2mo.nr_e2_(vhf, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir*nocc)
#        v1vo = numpy.zeros_like(v1vo)
#        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
#        eai = eai.ravel()
#        for i, z in enumerate(zs):
#            v1vo[i] += eai * z
        amat = v1vo.reshape(nz,-1)
#        print 'diag',mt(amat)
        #print 'Fock:',mt(self._scf.sfock)
        #print 'S1e:',mt(self._scf.ss1e)
        fv = reduce(numpy.dot, (orbv.T, self._scf.sfock, orbv))
        fo = reduce(numpy.dot, (orbo.T, self._scf.sfock, orbo))
        #print 'fv:',mt(fv)
        #print 'fo:',mt(fo)
        sv = reduce(numpy.dot, (orbv.T, self._scf.ss1e, orbv))
        so = reduce(numpy.dot, (orbo.T, self._scf.ss1e, orbo))
        #print 'sv:',mt(sv)
        #print 'so:',mt(so)
        fvi = numpy.einsum('ij,kl->ikjl',fv,so).reshape((nvir*nocc,-1))
        fio = numpy.einsum('ij,kl->ikjl',sv,fo).reshape((nvir*nocc,-1))
        ff = fvi - fio
        #print 'ff',mt(ff)
        amat += ff
        #print 'Amat:',mt(amat)
        ss = numpy.einsum('ij,kl->ikjl',sv,so).reshape((nvir*nocc,-1))
        #print 'Smat:',mt(ss)

        w, v = scipy.linalg.eigh(amat,ss)
        print w*27.21139

    def get_precond(self, hdiag):
        def precond(x, e, x0):
            diagd = hdiag - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            return x/diagd
        return precond

    def init_guess(self, eai, nstates=None):
        if nstates is None: nstates = self.nstates
        nov = eai.size
        nroot = min(nstates, nov)
        x0 = numpy.zeros((nroot, nov))
        idx = numpy.argsort(eai.ravel())
        for i in range(nroot):
            x0[i,idx[i]] = 1  # lowest excitations
        return x0

    def kernel(self, x0=None, nomo=False):
        '''TDA diagonalization solver
        '''
        self.check_sanity()

        mo_energy = self._scf.mo_energy
        nocc = (self._scf.mo_occ>0).sum()
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])

        if x0 is None:
            x0 = self.init_guess(eai, self.nstates)

        precond = self.get_precond(eai.ravel())

        if nomo or self.nomo:
            self.e, x1 = davidson.dgeev(self.abop, x0, precond, tol=self.conv_tol,
                                    type=1, max_cycle=100, max_space=self.max_space,
                                    lindep=self.lindep, verbose=self.verbose, nroots=self.nstates)
        else:
            self.e, x1 = pyscf.lib.davidson1(self.get_vind, x0, precond,
                                         tol=self.conv_tol,
                                         nroots=self.nstates, lindep=self.lindep,
                                         max_space=self.max_space,
                                         verbose=self.verbose)
# 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
        self.xy = [(xi.reshape(eai.shape)*numpy.sqrt(.5),0) for xi in x1]
        return self.e, self.xy
CIS = TDA


class TDHF(TDA):
    def get_vind(self, xys):
        '''
        [ A  B][X]
        [-B -A][Y]
        '''
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (self._scf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        nz = len(xys)
        dms = numpy.empty((nz*2,nao,nao))
        for i in range(nz):
            x, y = xys[i].reshape(2,nvir,nocc)
            dmx = reduce(numpy.dot, (orbv, x, orbo.T))
            dmy = reduce(numpy.dot, (orbv, y, orbo.T))
            dms[i   ] = dmx + dmy.T  # AX + BY
            dms[i+nz] = dms[i].T # = dmy + dmx.T  # AY + BX
        vj, vk = self._scf.get_jk(self.mol, dms, hermi=0)

        if self.singlet:
            vhf = vj*2 - vk
        else:
            vhf = -vk
        #vhf = numpy.asarray([reduce(numpy.dot, (orbv.T, v, orbo)) for v in vhf])
        vhf = _ao2mo.nr_e2_(vhf, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir*nocc)
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        eai = eai.ravel()
        for i, z in enumerate(xys):
            x, y = z.reshape(2,-1)
            vhf[i   ] += eai * x  # AX
            vhf[i+nz] += eai * y  # AY
        hx = numpy.hstack((vhf[:nz], -vhf[nz:]))
        return hx.reshape(nz,-1)

    def ddiag(self):
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (self._scf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        xys = numpy.eye(2*nvir*nocc)
        nz = len(xys)
        dms = numpy.empty((nz*2,nao,nao))
        for i in range(nz):
            x, y = xys[i].reshape(2,nvir,nocc)
            dmx = reduce(numpy.dot, (orbv, x, orbo.T))
            dmy = reduce(numpy.dot, (orbv, y, orbo.T))
            dms[i   ] = dmx + dmy.T  # AX + BY
            dms[i+nz] = dms[i].T # = dmy + dmx.T  # AY + BX
        vj, vk = self._scf.get_jk(self.mol, dms, hermi=0)

        if self.singlet:
            vhf = vj*2 - vk
        else:
            vhf = -vk
        #vhf = numpy.asarray([reduce(numpy.dot, (orbv.T, v, orbo)) for v in vhf])
        vhf = _ao2mo.nr_e2_(vhf, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir*nocc)
#        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
#        eai = eai.ravel()
#        for i, z in enumerate(xys):
#            x, y = z.reshape(2,-1)
#            vhf[i   ] += eai * x  # AX
#            vhf[i+nz] += eai * y  # AY

        fv = reduce(numpy.dot, (orbv.T, self._scf.sfock, orbv))
        fo = reduce(numpy.dot, (orbo.T, self._scf.sfock, orbo))
        sv = reduce(numpy.dot, (orbv.T, self._scf.ss1e, orbv))
        so = reduce(numpy.dot, (orbo.T, self._scf.ss1e, orbo))
        ss = numpy.empty_like(vhf)
        for i, z in enumerate(xys):
            x, y = z.reshape(2,-1)
            xm = x.reshape(nvir,nocc)
            ym = y.reshape(nvir,nocc)
            px1 = numpy.einsum('ij,kl,jl->ik',fv,so,xm)
            px2 = numpy.einsum('ij,kl,jl->ik',sv,fo,xm)
            pxs = numpy.einsum('ij,kl,jl->ik',sv,so,xm)
            py1 = numpy.einsum('ij,kl,jl->ik',fv,so,ym)
            py2 = numpy.einsum('ij,kl,jl->ik',sv,fo,ym)
            pys = numpy.einsum('ij,kl,jl->ik',sv,so,ym)
            vhf[i   ] += (px1-px2).ravel()
            vhf[i+nz] += (py1-py2).ravel()
            ss[i   ] = pxs.ravel()
            ss[i+nz] = pys.ravel()

        hx = numpy.hstack((vhf[:nz], -vhf[nz:]))
        sx = numpy.hstack((ss[:nz], ss[nz:]))
        
        w, v = scipy.linalg.eig(hx,sx)
        print w*27.21139

    def abop(self, xys):
        '''
        [ A B][X]   [ S  0 ] [X]
        [ B A][Y] + [ 0 -S ] [Y]
        '''
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (self._scf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        nz = len(xys)
        dms = numpy.empty((nz*2,nao,nao))
        for i in range(nz):
            x, y = xys[i].reshape(2,nvir,nocc)
            dmx = reduce(numpy.dot, (orbv, x, orbo.T))
            dmy = reduce(numpy.dot, (orbv, y, orbo.T))
            dms[i   ] = dmx + dmy.T  # AX + BY
            dms[i+nz] = dms[i].T # = dmy + dmx.T  # AY + BX
        vj, vk = self._scf.get_jk(self.mol, dms, hermi=0)

        if self.singlet:
            vhf = vj*2 - vk
        else:
            vhf = -vk
        #vhf = numpy.asarray([reduce(numpy.dot, (orbv.T, v, orbo)) for v in vhf])
        vhf = _ao2mo.nr_e2_(vhf, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir*nocc)
#        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
#        eai = eai.ravel()
#        for i, z in enumerate(xys):
#            x, y = z.reshape(2,-1)
#            vhf[i   ] += eai * x  # AX
#            vhf[i+nz] += eai * y  # AY

        fv = reduce(numpy.dot, (orbv.T, self._scf.sfock, orbv))
        fo = reduce(numpy.dot, (orbo.T, self._scf.sfock, orbo))
        sv = reduce(numpy.dot, (orbv.T, self._scf.ss1e, orbv))
        so = reduce(numpy.dot, (orbo.T, self._scf.ss1e, orbo))
        ss = numpy.empty_like(vhf)
        for i, z in enumerate(xys):
            x, y = z.reshape(2,-1)
            xm = x.reshape(nvir,nocc)
            ym = y.reshape(nvir,nocc)
            px1 = numpy.einsum('ij,kl,jl->ik',fv,so,xm)
            px2 = numpy.einsum('ij,kl,jl->ik',sv,fo,xm)
            pxs = numpy.einsum('ij,kl,jl->ik',sv,so,xm)
            py1 = numpy.einsum('ij,kl,jl->ik',fv,so,ym)
            py2 = numpy.einsum('ij,kl,jl->ik',sv,fo,ym)
            pys = numpy.einsum('ij,kl,jl->ik',sv,so,ym)
            vhf[i   ] += (px1-px2).ravel()
            vhf[i+nz] += (py1-py2).ravel()
            ss[i   ] = pxs.ravel()
            ss[i+nz] = pys.ravel()

        hx = numpy.hstack((vhf[:nz], -vhf[nz:]))
        sx = numpy.hstack((ss[:nz], ss[nz:]))

        return hx.reshape(nz,-1), sx.reshape(nz,-1)

    def get_precond(self, hdiag):
        def precond(x, e, x0):
            diagd = hdiag - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            y = x.reshape(2,-1)/diagd
            return y.reshape(-1)
        return precond

    def init_guess(self, eai, nstates=None):
        if nstates is None: nstates = self.nstates
        nov = eai.size
        nroot = min(nstates, nov)
        x0 = numpy.zeros((nroot, nov*2))
        idx = numpy.argsort(eai.ravel())
        for i in range(nroot):
            x0[i,idx[i]] = 1  # lowest excitations
        return x0

    def kernel(self, x0=None, nomo=False) :
        '''TDHF diagonalization with non-Hermitian eigenvalue solver
        '''
        self.check_sanity()

        mo_energy = self._scf.mo_energy
        nocc = (self._scf.mo_occ>0).sum()
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        nvir = eai.shape[0]

        if x0 is None:
            x0 = self.init_guess(eai, self.nstates)

        precond = self.get_precond(eai.ravel())

        # We only need positive eigenvalues
        def pickeig(w, v, nroots):
            realidx = numpy.where((w.imag == 0) & (w.real > 0))[0]
            return realidx[w[realidx].real.argsort()[:nroots]]

        if nomo or self.nomo:
            w, x1 = davidson.dgeev(self.abop, x0, precond,
                             tol=self.conv_tol, type=1,
                             nroots=self.nstates, lindep=self.lindep,
                             max_space=self.max_space,
                             verbose=self.verbose)
        else:
            w, x1 = davidson.eig(self.get_vind, x0, precond,
                             tol=self.conv_tol,
                             nroots=self.nstates, lindep=self.lindep,
                             max_space=self.max_space, pick=pickeig,
                             verbose=self.verbose)
        self.e = w
        def norm_xy(z):
            x, y = z.reshape(2,nvir,nocc)
            norm = 2*(pyscf.lib.norm(x)**2 - pyscf.lib.norm(y)**2)
            norm = 1/numpy.sqrt(norm)
            return x*norm, y*norm
        self.xy = [norm_xy(z) for z in x1]

        return self.e, self.xy
RPA = TDHF


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '631g'
    mol.build()

    mf = scf.RHF(mol)
    mf.scf()
    td = TDA(mf)
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [ 11.90276464  11.90276464  16.86036434]

    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [ 11.01747918  11.01747918  13.16955056]

    td = TDHF(mf)
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [ 11.83487199  11.83487199  16.66309285]

    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [ 10.8919234   10.8919234   12.63440705]

