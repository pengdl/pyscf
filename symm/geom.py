#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import re
import numpy
import scipy.linalg
import pyscf.lib
from pyscf.gto import mole
from pyscf.lib import logger
import pyscf.symm.param

TOLERANCE = 1e-5

def parallel_vectors(v1, v2, tol=TOLERANCE):
    if numpy.allclose(v1, 0, atol=tol) or numpy.allclose(v2, 0, atol=tol):
        return True
    else:
        v3 = numpy.cross(v1/numpy.linalg.norm(v1), v2/numpy.linalg.norm(v2))
        return numpy.linalg.norm(v3) < TOLERANCE

def argsort_coords(coords, decimals=None):
    if decimals is None:
        decimals = int(-numpy.log10(TOLERANCE)) - 1
    coords = numpy.around(coords, decimals=decimals)
    idx = numpy.lexsort((coords[:,2], coords[:,1], coords[:,0]))
    return idx

def sort_coords(coords, decimals=None):
    if decimals is None:
        decimals = int(-numpy.log10(TOLERANCE)) - 1
    coords = numpy.asarray(coords)
    idx = argsort_coords(coords, decimals=decimals)
    return coords[idx]

# ref. http://en.wikipedia.org/wiki/Rotation_matrix
def rotation_mat(vec, theta):
    '''rotate angle theta along vec
    new(x,y,z) = R * old(x,y,z)'''
    vec = vec / numpy.linalg.norm(vec)
    uu = vec.reshape(-1,1) * vec.reshape(1,-1)
    ux = numpy.array((
        ( 0     ,-vec[2], vec[1]),
        ( vec[2], 0     ,-vec[0]),
        (-vec[1], vec[0], 0     )))
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    r = c * numpy.eye(3) + s * ux + (1-c) * uu
    return r

# reflection operation with householder
def householder(vec):
    vec = numpy.array(vec) / numpy.linalg.norm(vec)
    return numpy.eye(3) - vec[:,None]*vec*2

def closest_axes(axes, ref):
    xcomp, ycomp, zcomp = numpy.einsum('ix,jx->ji', axes, ref)
    z_id = numpy.argmax(abs(zcomp))
    xcomp[z_id] = ycomp[z_id] = 0       # remove z
    x_id = numpy.argmax(abs(xcomp))
    ycomp[x_id] = 0                     # remove x
    y_id = numpy.argmax(abs(ycomp))
    return x_id, y_id, z_id

def alias_axes(axes, ref):
    '''Rename axes, make it as close as possible to the ref axes
    '''
    x_id, y_id, z_id = closest_axes(axes, ref)
    new_axes = axes[[x_id,y_id,z_id]]
    if numpy.linalg.det(new_axes) < 0:
        new_axes = axes[[y_id,x_id,z_id]]
    return new_axes

def detect_symm(atoms, basis=None, verbose=logger.WARN):
    '''Detect the point group symmetry for given molecule.

    Return group name, charge center, and nex_axis (three rows for x,y,z)
    '''
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)
# a tight threshold for classifying the main class of group.  Because if the
# main group class is incorrectly assigned, the following search _search_toi
# and search_c_highest is very likely to give wrong type of symmetry
    tol = TOLERANCE / numpy.sqrt(1+len(atoms))
    log.debug('geometry tol = %g', tol)

    rawsys = SymmSys(atoms, basis)
    w, axes = scipy.linalg.eigh(rawsys.im)
    axes = axes.T

# Make sure the axes can be rotated from continuous unitary transformation
    x_id, y_id, z_id = closest_axes(axes, numpy.eye(3))
    if axes[z_id,2] < 0:
        axes[z_id] *= -1
    if axes[x_id,0] < 0:
        axes[x_id] *= -1
    if numpy.linalg.det(axes) < 0:
        axes[y_id] *= -1
    log.debug('principal inertia moments %s', w)
    log.debug('new axes %s', axes)

    if numpy.allclose(w, 0, atol=tol):
        gpname = 'SO3'
        return gpname, rawsys.charge_center, axes

    elif numpy.allclose(w[:2], 0, atol=tol): # linear molecule
        if rawsys.detect_icenter():
            gpname = 'Dooh'
        else:
            gpname = 'Coov'
        return gpname, rawsys.charge_center, axes

    else:
        try:
            if numpy.allclose(w, w[0], atol=tol): # T, O, I
                # Because rotation vectors Rx Ry Rz are 3-degenerated T representation
                # See http://www.webqc.org/symmetrypointgroup-td.html
                gpname, axes = _search_toi(rawsys)
                return gpname, rawsys.charge_center, axes

            elif numpy.allclose(w[1], w[2], atol=tol):
                axes = axes[[1,2,0]]
                n, c2x, mirrorx = rawsys.search_c_highest(axes[2])
            elif numpy.allclose(w[0], w[1], atol=tol):
                n, c2x, mirrorx = rawsys.search_c_highest(axes[2])
            else:
                n = 1
        except RotationAxisNotFound:
# FIXME:
# Some quasi symmetric system may cheat the inertia momentum.  It's common
# when two high symmetric clusters eg Td and I sit on the same charge center
# and have different orientation.  The eniter system of the two cluster has
# low symmetry.  We temporarily label this system with highest rotation axis
# C1.
            sys.stderr.write('High rotation axis is not found. Inertia momentum %s\n' % w)
            n = 1

        #print('Highest C_n = C%d' % n)
        if n >= 2:
            if c2x is not None:
                if all(rawsys.symmetric_for(householder(axes[2]))):
                    gpname = 'D%dh' % n
                elif rawsys.detect_icenter():
                    gpname = 'D%dd' % n
                else:
                    gpname = 'D%d' % n
                yaxis = numpy.cross(axes[2], c2x)
                axes = numpy.array((c2x, yaxis, axes[2]))
            elif mirrorx is not None:
                gpname = 'C%dv' % n
                yaxis = numpy.cross(axes[2], mirrorx)
                axes = numpy.array((mirrorx, yaxis, axes[2]))
            elif all(rawsys.symmetric_for(householder(axes[2]))): # xy-mirror
                gpname = 'C%dh' % n
            elif all(rawsys.symmetric_for(numpy.dot(rotation_mat(axes[2], numpy.pi/n),
                                                    householder(axes[2])))): # improper rotation
                gpname = 'S%d' % (n*2)
            else:
                gpname = 'C%d' % n
            return gpname, rawsys.charge_center, axes

        else:
            is_c2x = all(rawsys.symmetric_for(rotation_mat(axes[0], numpy.pi)))
            is_c2y = all(rawsys.symmetric_for(rotation_mat(axes[1], numpy.pi)))
            is_c2z = all(rawsys.symmetric_for(rotation_mat(axes[2], numpy.pi)))
# rotate to old axes, as close as possible?
            if is_c2z and is_c2x and is_c2y:
                if rawsys.detect_icenter():
                    gpname = 'D2h'
                else:
                    gpname = 'D2'
                axes = alias_axes(axes, numpy.eye(3))
            elif is_c2z or is_c2x or is_c2y:
                if is_c2x:
                    axes = axes[[1,2,0]]
                if is_c2y:
                    axes = axes[[2,0,1]]
                if all(rawsys.symmetric_for(householder(axes[2]))):
                    gpname = 'C2h'
                elif all(rawsys.symmetric_for(householder(axes[0]))):
                    gpname = 'C2v'
                else:
                    gpname = 'C2'
            else:
                if rawsys.detect_icenter():
                    gpname = 'Ci'
                elif all(rawsys.symmetric_for(householder(axes[0]))):
                    gpname = 'Cs'
                    axes = axes[[1,2,0]]
                elif all(rawsys.symmetric_for(householder(axes[1]))):
                    gpname = 'Cs'
                    axes = axes[[2,0,1]]
                elif all(rawsys.symmetric_for(householder(axes[2]))):
                    gpname = 'Cs'
                else:
                    gpname = 'C1'
#    charge_center = mole.charge_center(atoms)
#    if not numpy.allclose(charge_center, rawsys.charge_center, atol=tol):
#        assert(parallel_vectors(charge_center-rawsys.charge_center, axes[2]))
    return gpname, rawsys.charge_center, axes


# reduce to D2h and its subgroups
# FIXME, CPL, 209, 506
def subgroup(gpname, axes):
    if gpname in ('D2h', 'D2' , 'C2h', 'C2v', 'C2' , 'Ci' , 'Cs' , 'C1'):
        return gpname, axes
    elif gpname in ('SO3',):
        #return 'D2h', alias_axes(axes, numpy.eye(3))
        return 'Dooh', axes
    elif gpname in ('Dooh',):
        #return 'D2h', alias_axes(axes, numpy.eye(3))
        return 'Dooh', axes
    elif gpname in ('Coov',):
        #return 'C2v', axes
        return 'Coov', axes
    elif gpname in ('Oh',):
        return 'D2h', alias_axes(axes, numpy.eye(3))
    elif gpname in ('O',):
        return 'D2', alias_axes(axes, numpy.eye(3))
    elif gpname in ('Ih',):
        return 'Ci', alias_axes(axes, numpy.eye(3))
    elif gpname in ('I',):
        return 'C1', axes
    elif gpname in ('Td', 'T', 'Th'):
        #x,y,z = axes
        #x = (x+y) / numpy.linalg.norm(x+y)
        #y = numpy.cross(z, x)
        #return 'C2v', numpy.array((x,y,z))
        return 'D2', alias_axes(axes, numpy.eye(3))
    elif re.search(r'S\d+', gpname):
        n = int(re.search(r'\d+', gpname).group(0))
        if n % 2 == 0:
            return 'C%d'%(n//2), axes
        else:
            return 'Ci', axes
    else:
        n = int(re.search(r'\d+', gpname).group(0))
        if n % 2 == 0:
            if re.search(r'D\d+d', gpname):
                subname = 'D2'
            elif re.search(r'D\d+h', gpname):
                subname = 'D2h'
            elif re.search(r'D\d+', gpname):
                subname = 'D2'
            elif re.search(r'C\d+h', gpname):
                subname = 'C2h'
            elif re.search(r'C\d+v', gpname):
                subname = 'C2v'
            else:
                subname = 'C2'
        else:
            # rotate axes and
            # Dnh -> C2v
            # Dn  -> C2
            # Dnd -> Ci
            # Cnh -> Cs
            # Cnv -> Cs
            if re.search(r'D\d+h', gpname):
                subname = 'C2v'
                axes = axes[[1,2,0]]
            elif re.search(r'D\d+d', gpname):
                subname = 'C2h'
                axes = axes[[1,2,0]]
            elif re.search(r'D\d+', gpname):
                subname = 'C2'
                axes = axes[[1,2,0]]
            elif re.search(r'C\d+h', gpname):
                subname = 'Cs'
            elif re.search(r'C\d+v', gpname):
                subname = 'Cs'
                axes = axes[[1,2,0]]
            else:
                subname = 'C1'
        return subname, axes


def symm_ops(gpname, axes=None):
    if axes is not None:
        raise RuntimeError('TODO: non-standard orientation')
    op1 = numpy.eye(3)
    opi = -1

    opc2z = -numpy.eye(3)
    opc2z[2,2] = 1
    opc2x = -numpy.eye(3)
    opc2x[0,0] = 1
    opc2y = -numpy.eye(3)
    opc2y[1,1] = 1

    opcsz = numpy.dot(opc2z, opi)
    opcsx = numpy.dot(opc2x, opi)
    opcsy = numpy.dot(opc2y, opi)
    opdic = {'E'  : op1,
             'C2z': opc2z,
             'C2x': opc2x,
             'C2y': opc2y,
             'i'  : opi,
             'sz' : opcsz,
             'sx' : opcsx,
             'sy' : opcsy,}
    return opdic

def symm_identical_atoms(gpname, atoms):
    ''' Requires '''
    # Dooh Coov for linear molecule
    if gpname == 'Dooh':
        coords = numpy.array([a[1] for a in atoms], dtype=float)
        idx0 = argsort_coords(coords)
        coords0 = coords[idx0]
        opdic = symm_ops(gpname)
        newc = numpy.dot(coords, opdic['sz'])
        idx1 = argsort_coords(newc)
        dup_atom_ids = numpy.sort((idx0,idx1), axis=0).T
        uniq_idx = numpy.unique(dup_atom_ids[:,0], return_index=True)[1]
        eql_atom_ids = dup_atom_ids[uniq_idx]
        eql_atom_ids = [list(sorted(set(i))) for i in eql_atom_ids]
        return eql_atom_ids
    elif gpname == 'Coov':
        eql_atom_ids = [[i] for i,a in enumerate(atoms)]
        return eql_atom_ids

    center = mole.charge_center(atoms)
#    if not numpy.allclose(center, 0, atol=TOLERANCE):
#        sys.stderr.write('WARN: Molecular charge center %s is not on (0,0,0)\n'
#                        % center)
    opdic = symm_ops(gpname)
    ops = [opdic[op] for op in pyscf.symm.param.OPERATOR_TABLE[gpname]]
    coords = numpy.array([a[1] for a in atoms], dtype=float)
    idx = argsort_coords(coords)
    coords0 = coords[idx]

    dup_atom_ids = []
    for op in ops:
        newc = numpy.dot(coords, op)
        idx = argsort_coords(newc)
        if not numpy.allclose(coords0, newc[idx], atol=TOLERANCE):
            raise RuntimeError('Symmetry identical atoms not found')
        dup_atom_ids.append(idx)

    dup_atom_ids = numpy.sort(dup_atom_ids, axis=0).T
    uniq_idx = numpy.unique(dup_atom_ids[:,0], return_index=True)[1]
    eql_atom_ids = dup_atom_ids[uniq_idx]
    eql_atom_ids = [list(sorted(set(i))) for i in eql_atom_ids]
    return eql_atom_ids

def check_given_symm(gpname, atoms, basis=None):
# more strict than symm_identical_atoms, we required not only the coordinates
# match, but also the symbols and basis functions

#FIXME: compare the basis set when basis is given
    if gpname == 'Dooh':
        coords = numpy.array([a[1] for a in atoms], dtype=float)
        if numpy.allclose(coords[:,:2], 0, atol=TOLERANCE):
            opdic = symm_ops(gpname)
            rawsys = SymmSys(atoms, basis)
            return rawsys.detect_icenter()
        else:
            return False
    elif gpname == 'Coov':
        coords = numpy.array([a[1] for a in atoms], dtype=float)
        return numpy.allclose(coords[:,:2], 0, atol=TOLERANCE)

    opdic = symm_ops(gpname)
    ops = [opdic[op] for op in pyscf.symm.param.OPERATOR_TABLE[gpname]]
    rawsys = SymmSys(atoms, basis)
    for lst in rawsys.atomtypes.values():
        coords = rawsys.atoms[lst,1:]
        idx = argsort_coords(coords)
        coords0 = coords[idx]

        for op in ops:
            newc = numpy.dot(coords, op)
            idx = argsort_coords(newc)
            if not numpy.allclose(coords0, newc[idx], atol=TOLERANCE):
                return False
    return True

def shift_atom(atoms, orig, axis):
    c = numpy.array([a[1] for a in atoms])
    c = numpy.dot(c - orig, numpy.array(axis).T)
    return [[atoms[i][0], c[i]] for i in range(len(atoms))]

class RotationAxisNotFound(Exception):
    pass

class SymmSys(object):
    def __init__(self, atoms, basis=None):
        self.atomtypes = mole.atom_types(atoms, basis)
        # fake systems, which treates the atoms of different basis as different atoms.
        # the fake systems do not have the same symmetry as the potential
        # it's only used to determine the main (Z-)axis
        chg1 = numpy.pi - 2
        coords = []
        fake_chgs = []
        idx = []
        for k, lst in self.atomtypes.items():
            idx.append(lst)
            coords.append([atoms[i][1] for i in lst])
            ksymb = mole._rm_digit(k)
            if ksymb != k or ksymb == 'GHOST':
                # Put random charges on the decorated atoms
                fake_chgs.append([chg1] * len(lst))
                chg1 *= numpy.pi-2
            else:
                fake_chgs.append([mole._charge(ksymb)] * len(lst))
        coords = numpy.array(numpy.vstack(coords), dtype=float)
        fake_chgs = numpy.hstack(fake_chgs)
        self.charge_center = numpy.einsum('i,ij->j', fake_chgs, coords)/fake_chgs.sum()
        coords = coords - self.charge_center
        self.im = numpy.einsum('i,ij,ik->jk', fake_chgs, coords, coords)/fake_chgs.sum()

        idx = numpy.argsort(numpy.hstack(idx))
        self.atoms = numpy.hstack((fake_chgs.reshape(-1,1), coords))[idx]


    def group_atoms_by_distance(self, index):
        c = self.atoms[index,1:]
        r = numpy.sqrt(numpy.einsum('ij,ij->i', c, c))
        lst = numpy.argsort(r)
        groups = [[index[lst[0]]]]
        for i in range(len(lst)-1):
            if numpy.allclose(r[lst[i]], r[lst[i+1]], atol=TOLERANCE):
                groups[-1].append(index[lst[i+1]])
            else:
                groups.append([index[lst[i+1]]])
        return groups

    def detect_icenter(self):
        return all(self.symmetric_for(-1))

    def symmetric_for(self, op, decimals=None):
        if decimals is None:
            decimals = int(-numpy.log10(TOLERANCE)) - 1
        for lst in self.atomtypes.values():
            r0 = self.atoms[lst,1:]
            r1 = numpy.dot(r0, op)
            yield numpy.allclose(sort_coords(r0, decimals),
                                 sort_coords(r1, decimals), atol=TOLERANCE)

    def search_c_highest(self, zaxis):
        decimals = int(-numpy.log10(TOLERANCE)) - 1
        has_c2x = True
        has_mirrorx = True
        maybe_cn = []
        maybe_c2x = []
        maybe_mirrorx = []
        for atype in self.atomtypes.values():
            groups = self.group_atoms_by_distance(atype)
            for lst in groups:
                r0 = self.atoms[lst,1:]
                zcos = numpy.around(numpy.einsum('ij,j->i', r0, zaxis),
                                    decimals=decimals)
                uniq_zcos = numpy.unique(zcos)
                for d in uniq_zcos:
                    cn = (zcos==d).sum()
                    if (cn == 1):
                        if not parallel_vectors(zaxis, r0[zcos==d][0]):
                            raise RotationAxisNotFound
                    else:
                        maybe_cn.append(cn)

                # The possible C2x are composed by those vectors, whose
                # distance to xy-mirror are identical
                if has_c2x:
                    for d in uniq_zcos:
                        if numpy.allclose(d, 0, atol=TOLERANCE): # plane which crosses the orig
                            r1 = r0[zcos==d]
                            maybe_c2x.extend([r1[i1]+r1[i2]
                                              for i1 in range(len(r1))
                                              for i2 in range(i1+1)])
                        elif d > TOLERANCE:
                            mirrord = abs(zcos+d)<TOLERANCE
                            if mirrord.sum() == (zcos==d).sum():
                                above = r0[zcos==d]
                                below = r0[mirrord]
                                nelem = len(above)
                                maybe_c2x.extend([above[i1] + below[i2]
                                                  for i1 in range(nelem)
                                                  for i2 in range(nelem)])
                            else:
                                # if the number of mirrored vectors are diff,
                                # it's impossible to have c2x
                                has_c2x = False
                                break

                if has_mirrorx:
                    for d in uniq_zcos:
                        r1 = r0[zcos==d]
                        maybe_mirrorx.extend([numpy.cross(zaxis, r1[i1]+r1[i2])
                                              for i1 in range(len(r1))
                                              for i2 in range(i1+1)])

        # C_{n/m} is also possible highest Cn if n is not prime number
        possible_cn = []
        for n in sorted(set(maybe_cn)):
            for i in range(2, n+1):
                if n % i == 0:
                    possible_cn.append(i)
        possible_cn = set(possible_cn)

        for i in sorted(possible_cn, reverse=True):
            if all(self.symmetric_for(rotation_mat(zaxis, numpy.pi*2/i))):
                cn = i
                break
        else:
            raise RotationAxisNotFound

        #
        # Search for C2 perp to Cn and mirros on Cn
        #

        def pick_vectors(maybe_vec):
            maybe_vec = numpy.vstack(maybe_vec)
            # remove zero-vectors and duplicated vectors
            d = numpy.einsum('ij,ij->i', maybe_vec, maybe_vec)
            maybe_vec /= numpy.sqrt(d + 1e-200).reshape(-1,1)
            maybe_vec = maybe_vec[d>TOLERANCE**2]
            maybe_vec = _remove_dupvec(maybe_vec) # also transfer to pseudo-vector

            # remove the C2x which can be related by Cn rotation along z axis
            seen = numpy.zeros(len(maybe_vec), dtype=bool)
            for k, r1 in enumerate(maybe_vec):
                if not seen[k]:
                    cos2r1 = numpy.einsum('j,ij->i', r1, maybe_vec[k+1:])
                    for i in range(1,cn):
                        c = numpy.cos(numpy.pi*i/cn) # no 2pi because of pseudo-vector
                        seen[k+1:][abs(cos2r1-c) < TOLERANCE] = True

            possible_vec = maybe_vec[~seen]
            return possible_vec

        c2x = None
        if has_c2x:
            possible_c2x = pick_vectors(maybe_c2x)
            for c in possible_c2x:
                if all(self.symmetric_for(rotation_mat(c, numpy.pi))):
                    c2x = c
                    break

        mirrorx = None
        if has_mirrorx:
            possible_mirrorx = pick_vectors(maybe_mirrorx)
            for c in possible_mirrorx:
                if all(self.symmetric_for(householder(c))):
                    mirrorx = c
                    break

        return cn, c2x, mirrorx


# T/Td/Th/O/Oh/I/Ih
def _search_toi(rawsys):
    maybe = []
    maybe_cn = []
    for atype in rawsys.atomtypes.values():
        groups = rawsys.group_atoms_by_distance(atype)
        for lst in groups:
            if len(lst) == 2:
                r0 = coords[lst[0],1:]
                r1 = coords[lst[1],1:]
                maybe_cn.append(r1-r0)
            elif len(lst) > 2:
                coords = rawsys.atoms[lst,1:]
                r0 = coords - coords[0]
                distance_to_ref = pyscf.lib.norm(r0, axis=1)
                equal_distance = abs(distance_to_ref[:,None] - distance_to_ref) < TOLERANCE
# atoms of equal distances may be associated with rotation axis.
                for i in range(2, len(coords)):
                    for j in range(1,i):
                        if equal_distance[i,j]:
                            maybe_cn.append(numpy.cross(r0[i],r0[j]))

#
# Determine the Highest C
#
    decimals = int(-numpy.log10(TOLERANCE)) - 1
    def has_rotation(zaxis, n):
        op = rotation_mat(zaxis, numpy.pi*2/n)
        return all(rawsys.symmetric_for(op, decimals))

    maybe_cn = numpy.vstack(maybe_cn)
    # remove zero-vectors and duplicated vectors
    d = numpy.einsum('ij,ij->i', maybe_cn, maybe_cn)
    maybe_cn /= numpy.sqrt(d + 1e-200).reshape(-1,1)
    maybe_cn = maybe_cn[d>TOLERANCE**2]
    maybe_cn = _remove_dupvec(maybe_cn) # also transfer to pseudo-vector

    def search_c5_c4_c3(maybe_cn):
        for n in (5, 4, 3):
            for zaxis in maybe_cn:
                if has_rotation(zaxis, n):
                    return n, zaxis
        else:
            raise RotationAxisNotFound

    cn, zaxis = search_c5_c4_c3(maybe_cn)

    def make_axes(z, x):
        y = numpy.cross(z, x)
        x = numpy.cross(y, z) # because x might not perp to z
        x /= numpy.linalg.norm(x)
        y /= numpy.linalg.norm(y)
        z /= numpy.linalg.norm(z)
        return numpy.array((x,y,z))

    if cn == 3:
# There are more C3 axes associated to one atom, search for other C3 axes.
# The angular between two C3 axes is arccos(-1/3)
        cos = numpy.dot(maybe_cn, zaxis)
        maybe_c3 = maybe_cn[(abs(cos+1./3) < TOLERANCE) |
                            (abs(cos-1./3) < TOLERANCE)]
        for c3 in maybe_c3:
            if has_rotation(c3, 3):
                break
        else:
            raise RotationAxisNotFound('Only find one C3 axis')

        if rawsys.detect_icenter():
            gpname = 'Th'
# Because C3 axes are on the mirror of Td, two C3 can determine a mirror.
        elif all(rawsys.symmetric_for(householder(numpy.cross(zaxis, c3)))):
            gpname = 'Td'
        else:
            gpname = 'T'

        c3a = c3
        c3b = numpy.dot(c3, rotation_mat(zaxis, numpy.pi*2/3))
        c3c = numpy.dot(c3, rotation_mat(zaxis,-numpy.pi*2/3))
        zaxis, xaxis = c3c+c3b, c3a+c3b

    elif cn == 4:
        if rawsys.detect_icenter():
            gpname = 'Oh'
        else:
            gpname = 'O'

        maybe_c4 = maybe_cn[(abs(numpy.dot(maybe_cn, zaxis)) < TOLERANCE)]
        for c4 in maybe_c4:
            if has_rotation(c4, 4):
                xaxis = c4
                break
        else:
            raise RotationAxisNotFound('Only find one C4 axis')

    else:  # cn == 5
        if rawsys.detect_icenter():
            gpname = 'Ih'
        else:
            gpname = 'I'

# The angular between two C5 axes is arccos(1/numpy.sqrt(5))
        cos_c5 = 1/numpy.sqrt(5)
        cos = numpy.dot(maybe_cn, zaxis)
        maybe_c5 = maybe_cn[(abs(cos+cos_c5) < TOLERANCE) |
                            (abs(cos-cos_c5) < TOLERANCE)]
        for c5 in maybe_c5:
            if has_rotation(c5, 5):
                break
        else:
            raise RotationAxisNotFound('Only find one C5 axis')

        if abs(numpy.dot(c5, zaxis)+cos_c5) < TOLERANCE:
            c5 = -c5
        c5s = numpy.array([c5,
                           numpy.dot(c5, rotation_mat(zaxis, numpy.pi*2/5)),
                           numpy.dot(c5, rotation_mat(zaxis, numpy.pi*4/5)),
                           numpy.dot(c5, rotation_mat(zaxis, numpy.pi*6/5)),
                           numpy.dot(c5, rotation_mat(zaxis, numpy.pi*8/5)),])
        xaxis = c5s[0] + c5s[1]
        maybe_zaxis = zaxis + c5s
        where_zaxis = numpy.argmin(abs(numpy.dot(maybe_zaxis, xaxis)))
        zaxis = maybe_zaxis[where_zaxis]

    return gpname, make_axes(zaxis, xaxis)


def _pesudo_vectors(vs):
    idy0 = abs(vs[:,1])<TOLERANCE
    idz0 = abs(vs[:,2])<TOLERANCE
    vs = vs.copy()
    # ensure z component > 0
    vs[vs[:,2]<0] *= -1
    # if z component == 0, ensure y component > 0
    vs[(vs[:,1]<0) & idz0] *= -1
    # if y and z component == 0, ensure x component > 0
    vs[(vs[:,0]<0) & idy0 & idz0] *= -1
    return vs

def _remove_dupvec(vs):
    def rm_iter(vs):
        if len(vs) <= 1:
            return vs
        else:
            x = numpy.sum(abs(vs[1:]-vs[0]), axis=1)
            rest = rm_iter(vs[1:][x>TOLERANCE])
            return numpy.vstack((vs[0], rest))
    return rm_iter(_pesudo_vectors(vs))


if __name__ == "__main__":
    atom = [["O" , (1. , 0.    , 0.   ,)],
            ['H' , (0. , -.757 , 0.587,)],
            ['H' , (0. , 0.757 , 0.587,)] ]
    gpname, orig, axes = detect_symm(atom)
    atom = shift_atom(atom, orig, axes)
    print(gpname, symm_identical_atoms(gpname, atom))

    atom = [['H', (0,0,0)], ['H', (0,0,-1)], ['H', (0,0,1)]]
    gpname, orig, axes = detect_symm(atom)
    print(gpname, orig, axes)
    atom = shift_atom(atom, orig, axes)
    print(gpname, symm_identical_atoms(gpname, atom))

    atom = [['H', (0., 0., 0.)],
            ['H', (0., 0., 1.)],
            ['H', (0., 1., 0.)],
            ['H', (1., 0., 0.)],
            ['H', (-1, 0., 0.)],
            ['H', (0.,-1., 0.)],
            ['H', (0., 0.,-1.)]]
    gpname, orig, axes = detect_symm(atom)
    print(gpname, orig, axes)
    atom = shift_atom(atom, orig, axes)
    print(gpname, symm_identical_atoms(subgroup(gpname, axes)[0], atom))
