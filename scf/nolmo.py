#!/usr/bin/env python

'''
NonOrthogonal Local Molecular Orbital based HF/DFT SCF solver
'''

import sys
import tempfile
import time
from functools import reduce
import numpy
import scipy.linalg
import scipy.optimize
import pyscf.gto
import pyscf.lib
import pyscf.gto.ecp
from pyscf.lib import logger
from pyscf.scf import diis
from pyscf.scf import _vhf
import pyscf.scf.chkfile


def runscf(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, **kwargs):
    '''nolmo: the NOLMO-SCF driver.
    '''
    cput0 = (time.clock(), time.time())
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)
    adiis = mf.DIIS(mf, mf.diis_file)
    adiis.space = mf.diis_space
    adiis.rollback = mf.diis_space_rollback

    mol = mf.mol
    h1e = mf.get_hcore(mol)
    s1e = mf.get_ovlp(mol)
    eta =  -1.0

#    mo_energy, mo_coeff = mf.eig(h1e, s1e)
    mo_energy, mo_coeff = mf.mo_energy, mf.mo_coeff

    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    
    nocc = sum( mo_occ > 0 )
    nao  = len( mo_occ )
    nele = nocc * 2
    xmat = numpy.eye( nocc ) 
    mocc = mo_coeff[:,mo_occ>0]
    smat = reduce( numpy.dot, (mocc.T, s1e, mocc) )
    fmat = h1e - eta * s1e
    hmat = reduce( numpy.dot, (mocc.T, fmat, mocc) )
    xmat = update_xmat(hmat, smat, xmat, nocc, eta)
    dm = make_rdmx(mocc, xmat, smat)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    ns = numpy.sum( dm * s1e )
    e_tot += eta * ( nele - ns )

    logger.note(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    cycle = 0
    cput1 = logger.timer(mf, 'initialize nolmo-scf', *cput0)
    while not scf_conv and cycle < 99 : # need more cycles
        dm_last = dm
        last_hf_e = e_tot

        #f = h1e + vhf
        #f = adiis.update(s1e, dm, f)
        #fmat = f - eta * s1e
        fmat = h1e + vhf - eta * s1e

        hmat = reduce( numpy.dot, (mocc.T, fmat, mocc) )
        smat = reduce( numpy.dot, (mocc.T, s1e, mocc) )
        amat = 2*xmat - reduce(numpy.dot,(xmat, smat, xmat))
        grd1 = 4*( reduce(numpy.dot, (fmat, mocc, amat)) - reduce(numpy.dot, (s1e, mocc, xmat, hmat, xmat)) )

        if cycle == 0 :
            dec = -grd1
            gold = grd1
        else:
            a1 = numpy.sum(grd1*(grd1-gold))
            a2 = numpy.sum(grd1*grd1)
            b1 = numpy.sum(dold*(grd1-gold))
            bhs = a1/b1
            bdy = a2/b1
            beta = max(min(bhs,bdy),0.)
            dec = -grd1 + beta * dold
            gold = grd1
        dold = dec

        step = eval_step_c(fmat,s1e,mocc,dec,hmat,smat,xmat)
        #print 'step:',step
        #print 'dec:',mt(dec)
        mocc += step * dec
        #print "mocc",mt(mocc)
        mocc = renorm(s1e, mocc)
        hmat = reduce(numpy.dot, (mocc.T, fmat, mocc) )
        smat = reduce(numpy.dot, (mocc.T, s1e, mocc) )
        #print 'SMAT:',mt(smat)
        xmat = update_xmat(-eta*smat, smat, xmat, nocc, eta)

        dm = make_rdmx(mocc, xmat, smat)
        vhf = mf.get_veff(mol, dm )
        e_tot = mf.energy_tot(dm, h1e, vhf)
        ns = numpy.sum(dm * s1e)
        #print '(nele-ns):',(nele-ns)
        e_tot += eta * (nele - ns)

        norm_gorb = numpy.sum(grd1**2)
        norm_ddm = numpy.linalg.norm(dm-dm_last)
        logger.note(mf, 'cycle= %2d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if (abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad):
            scf_conv = True

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)
        cycle += 1

    logger.timer(mf, 'scf_cycle', *cput0)
    #print 'SMAT:',mt(smat)
    
    movv = get_vir(mocc, s1e)
    movv = renorm(s1e, movv)
    #print 'Cocc:',mt(mocc)
    #print 'Cvir:',mt(movv)
    fock = h1e + vhf
    mf.sfock = fock
    mf.ss1e = s1e
    #a1 = reduce(numpy.dot, (mocc.T, s1e, movv))
    #a2 = reduce(numpy.dot, (movv.T, s1e, mocc))
    #print 'Test1:',mt(a1)
    #print 'Test2:',mt(a2)
    mo_coeff = numpy.column_stack((mocc, movv))
    mo_fock = reduce(numpy.dot, (mo_coeff.T, fock, mo_coeff))
    mo_energy = mo_fock.diagonal()
    #print 'MO_coeff',mt(mo_coeff)

    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

def get_vir(c,s):
    u, w, vt = scipy.linalg.svd(s.dot(c))
    return u[:,c.shape[1]:]

def get_vir_ao(c,s):
    csc = reduce(numpy.dot, (c.T, s, c))
    w, v = numpy.linalg.eigh(csc)
    idx = w > 0.
    u = v[:,idx] / w[idx]
    t = reduce(numpy.dot, (c, u, v.T, c.T, s))
    o = numpy.eye(s.shape[0]) - t
    return o

def renorm(s,c):
    #from math import sqrt
    #smo = reduce(numpy.dot, (c.T, s, c))
    #for i in xrange(c.shape[0]):
    #    for j in xrange(c.shape[1]):
    #        c[i][j] /= sqrt(smo[j][j])
    #c = c / numpy.sqrt(smo.diagonal())
    #return c
    return c / numpy.sqrt(reduce(numpy.dot, (c.T, s, c)).diagonal())

def eval_step_c(f,sao,c,dc,h,s,x):
    hcc = reduce(numpy.dot, (dc.T,f,dc) )
    hc  = ( reduce(numpy.dot, (dc.T,f,c)) + reduce(numpy.dot, (c.T,f,dc)) )*0.5
    scc = reduce(numpy.dot, (dc.T,sao,dc) )
    sc  = ( reduce(numpy.dot, (dc.T,sao,c)) + reduce(numpy.dot, (c.T,sao,dc)) )*0.5
    hx = numpy.dot(h, x)
    sx = numpy.dot(s, x)
    hcx = numpy.dot(hc, x)
    scx = numpy.dot(sc, x)
    hccx = numpy.dot(hcc, x)
    sccx = numpy.dot(scc, x)
    a3 = 2 * numpy.sum(hccx*sccx)
    a2 = 3 * ( numpy.sum(hcx*sccx) + numpy.sum(hccx*scx) )
    a1 = numpy.sum(hccx*sx) - 2*numpy.sum(hcc*x) + numpy.sum(hx*sccx) + 4*numpy.sum(hcx*scx)
    a0 = numpy.sum(hcx*sx) - 2*numpy.sum(hc*x) + numpy.sum(hx*scx)
    #print 'a3-0:',a3,a2,a1,a0
    f = lambda x : a3*x**3+a2*x**2+a1*x+a0
    r = find_root(a3,a2,a1,a0)
#    r = scipy.optimize.fsolve(f,0.0)[0]
    #print 'f(x):',f(r)
    return r

def find_root(a,b,c,d):
    from math import sin,cos,acos,sqrt
    if abs(a) > 1.0 :
        aa, bb, cc, dd = 1.0, b/a, c/a, d/a
    else:
        aa, bb, cc, dd = a, b, c, d
    aaa=bb*bb-3.0*aa*cc
    bbb=bb*cc-9.0*aa*dd
    ccc=cc*cc-3.0*bb*dd
    deta=bbb*bbb-4.0*aaa*ccc
    if deta < 0.0 :
        if aaa <= 0.0:
            return -dd/cc
        tt = (2.0*aaa*bb-3.0*aa*bbb)/2.0/aaa**1.50
        if  tt <= -1.0 or tt >= 1.0:
            return -dd/cc
        sita=acos(tt)
        x1=(-bb-2.0*(aaa**0.50)*cos(sita/3.0))/3.0/aa
        x2=(-bb+(aaa**0.50)*(cos(sita/3.0)+sqrt(3.0)*sin(sita/3.0)))/3.0/aa
        x3=(-bb+(aaa**0.50)*(cos(sita/3.0)-sqrt(3.0)*sin(sita/3.0)))/3.0/aa
        lambdac=x1
        deta=abs(lambdac+dd/cc)
        if abs(x2+dd/cc) <= deta:
            lambdac=x2
        deta=abs(lambdac+dd/cc)
        if abs(x3+dd/cc) <= deta:
          lambdac=x3
        if x1 > 0.0 : lambdac=min(lambdac,x1)
        if x2 > 0.0 : lambdac=min(lambdac,x2)
        if x3 > 0.0 : lambdac=min(lambdac,x3)
    else :
        y1=aaa*bb+3.0*aa*(-bbb+sqrt(deta))/2.0
        y2=aaa*bb+3.0*aa*(-bbb-sqrt(deta))/2.0
        if y1 > 0.0:
            y1=y1**(1.0/3.0)
        else:
            y1=-(-y1)**(1.0/3.0)
        if y2 > 0.0:
            y2=y2**(1.0/3.0)
        else :
            y2=-(-y2)**(1.0/3.0)
        lambdac=(-bb-y1-y2)/3.0/aa
        if lambdac <= 0 : lambdac=-dd/cc
    return lambdac
     
def update_xmat(h,s,x,n,eta):
    a = 2*x - reduce(numpy.dot,(x, s, x))
    e = 2*numpy.sum( h*a ) + 2*n*eta
    it = 0
    while it < 50 :
        eold = e
        grd1 = 2 * ( 2*h - reduce(numpy.dot,(s,x,h)) - reduce(numpy.dot,(h,x,s)) )
        if it == 0 :
            dec = -grd1
            gold = grd1
        else:
            a1 = numpy.sum( grd1*(grd1-gold) )
            a2 = numpy.sum( grd1*grd1 )
            b1 = numpy.sum( dold*(grd1-gold) )
            bhs = a1/b1
            bdy = a2/b1
            beta = max(min(bhs,bdy),0.)
            dec = -grd1 + beta * dold
            gold = grd1
        dold = dec
        hdx = numpy.dot(h,dec)
        sdx = numpy.dot(s,dec)
        tmp = numpy.eye(n) - numpy.dot(s,x)
        step = numpy.sum(hdx*tmp) / numpy.sum(hdx*sdx)
        x += step * dec
        it += 1
        a = 2*x - reduce(numpy.dot,(x, s, x))
        e = 2*numpy.sum( h*a ) + 2*n*eta
        #print it,e,eold
        if abs(e-eold)<1.e-12 :
            break
    else:
        print "update_xmat didnot converge"
        raise ValueError
    c = numpy.dot(x,s)
    #print "opt x, x*s",it,mt(c)
    return x

def make_rdmx(mocc, xmat, smat):
    amat = 2*xmat - reduce( numpy.dot, (xmat,smat,xmat) )
    return 2 * reduce( numpy.dot, (mocc, amat, mocc.T) )

def mt(a):
    '''
    Debug matrix printer
    '''
    b = ""
    for i in a:
        b += '\n'+''.join( "%15.9f"%j for j in i )
    return b
        
if __name__ == '__main__':
    pass
