#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import _ctypes
import numpy
from pyscf.lib import misc

'''
Extension to numpy module
'''

_np_helper = misc.load_library('libnp_helper')

BLOCK_DIM = 192
PLAIN = 0
HERMITIAN = 1
ANTIHERMI = 2
SYMMETRIC = 3


# 2d -> 1d
def pack_tril(mat, axis=-1, out=None):
    '''flatten the lower triangular part of a matrix.
    Given mat, it returns mat[numpy.tril_indices(mat.shape[0])]

    Examples:

    >>> pack_tril(numpy.arange(9).reshape(3,3))
    [0 3 4 6 7 8]
    '''
    if mat.ndim == 2:
        count, nd = 1, mat.shape[0]
        shape = nd*(nd+1)//2
    else:
        count, nd = mat.shape[:2]
        shape = (count, nd*(nd+1)//2)

    if mat.ndim == 2 or axis == -1:
        mat = numpy.asarray(mat, order='C')
        out = numpy.ndarray(shape, mat.dtype, buffer=out)
        if numpy.iscomplexobj(mat):
            fn = _np_helper.NPzpack_tril_2d
        else:
            fn = _np_helper.NPdpack_tril_2d
        fn(ctypes.c_int(count), ctypes.c_int(nd),
           out.ctypes.data_as(ctypes.c_void_p),
           mat.ctypes.data_as(ctypes.c_void_p))
        return out

    else:  # pack the leading two dimension
        assert(axis == 0)
        out = mat[numpy.tril_indices(nd)]
        return out

# 1d -> 2d, write hermitian lower triangle to upper triangle
def unpack_tril(tril, filltriu=HERMITIAN, axis=-1, out=None):
    '''Reverse operation of pack_tril.  Put a vector in the lower triangular
    part of a matrix.

    Kwargs:
        filltriu : int

            | 0           Do not fill the upper triangular part, random number may appear
                          in the upper triangular part
            | 1 (default) Transpose the lower triangular part to fill the upper triangular part
            | 2           Similar to filltriu=1, negative of the lower triangular part is assign
                          to the upper triangular part to make the matrix anti-hermitian

    Examples:

    >>> unpack_tril(numpy.arange(6.))
    [[ 0. 1. 3.]
     [ 1. 2. 4.]
     [ 3. 4. 5.]]
    >>> unpack_tril(numpy.arange(6.), 0)
    [[ 0. 0. 0.]
     [ 1. 2. 0.]
     [ 3. 4. 5.]]
    >>> unpack_tril(numpy.arange(6.), 2)
    [[ 0. -1. -3.]
     [ 1.  2. -4.]
     [ 3.  4.  5.]]
    '''
    tril = numpy.asarray(tril, order='C')
    if tril.ndim == 1:
        count, nd = 1, tril.size
        nd = int(numpy.sqrt(nd*2))
        shape = (nd,nd)
    else:
        nd = tril.shape[axis]
        count = int(tril.size // nd)
        nd = int(numpy.sqrt(nd*2))
        shape = (count,nd,nd)

    if tril.ndim == 1 or axis == -1 or axis == tril.ndim-1:
        out = numpy.ndarray(shape, tril.dtype, buffer=out)
        if numpy.iscomplexobj(tril):
            fn = _np_helper.NPzunpack_tril_2d
        else:
            fn = _np_helper.NPdunpack_tril_2d
        fn(ctypes.c_int(count), ctypes.c_int(nd),
           tril.ctypes.data_as(ctypes.c_void_p),
           out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(filltriu))
        return out

    else:  # unpack the leading dimension
        assert(axis == 0)
        shape = (nd,nd) + tril.shape[1:]
        out = numpy.ndarray(shape, tril.dtype, buffer=out)
        idx = numpy.tril_indices(nd)
        if filltriu == HERMITIAN:
            for ij,(i,j) in enumerate(zip(*idx)):
                out[i,j] = out[j,i] = tril[ij]
        elif filltriu == ANTIHERMI:
            raise KeyError('filltriu == ANTIHERMI')
        else:
            out[idx] = tril
        return out

# extract a row from a tril-packed matrix
def unpack_row(tril, row_id):
    '''Extract one row of the lower triangular part of a matrix.
    It is equivalent to unpack_tril(a)[row_id]

    Examples:

    >>> unpack_row(numpy.arange(6.), 0)
    [ 0. 1. 3.]
    >>> unpack_tril(numpy.arange(6.))[0]
    [ 0. 1. 3.]
    '''
    tril = numpy.ascontiguousarray(tril)
    nd = int(numpy.sqrt(tril.size*2))
    mat = numpy.empty(nd, tril.dtype)
    if numpy.iscomplexobj(tril):
        fn = _np_helper.NPzunpack_row
    else:
        fn = _np_helper.NPdunpack_row
    fn.restype = ctypes.c_void_p
    fn(ctypes.c_int(nd), ctypes.c_int(row_id),
       tril.ctypes.data_as(ctypes.c_void_p),
       mat.ctypes.data_as(ctypes.c_void_p))
    return mat

# for i > j of 2d mat, mat[j,i] = mat[i,j]
def hermi_triu(mat, hermi=HERMITIAN, inplace=True):
    '''Use the elements of the lower triangular part to fill the upper triangular part.

    Kwargs:
        filltriu : int

            | 1 (default) return a hermitian matrix
            | 2           return an anti-hermitian matrix

    Examples:

    >>> unpack_row(numpy.arange(9.).reshape(3,3), 1)
    [[ 0.  3.  6.]
     [ 3.  4.  7.]
     [ 6.  7.  8.]]
    >>> unpack_row(numpy.arange(9.).reshape(3,3), 2)
    [[ 0. -3. -6.]
     [ 3.  4. -7.]
     [ 6.  7.  8.]]
    '''
    assert(hermi == HERMITIAN or hermi == ANTIHERMI)
    if not mat.flags.c_contiguous:
        assert(not inplace)
        mat = mat.copy(order='C')
    nd = mat.shape[0]
    if numpy.iscomplexobj(mat):
        fn = _np_helper.NPzhermi_triu
    else:
        fn = _np_helper.NPdsymm_triu
    fn.restype = ctypes.c_void_p
    fn(ctypes.c_int(nd), mat.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(hermi))
    return mat


LINEAR_DEP_THRESHOLD = 1e-10
def solve_lineq_by_SVD(a, b):
    ''' a * x = b '''
    t, w, vH = numpy.linalg.svd(a)
    idx = []
    for i,wi in enumerate(w):
        if wi > LINEAR_DEP_THRESHOLD:
            idx.append(i)
    if idx:
        idx = numpy.array(idx)
        tb = numpy.dot(numpy.array(t[:,idx]).T.conj(), numpy.array(b))
        x = numpy.dot(numpy.array(vH[idx,:]).T.conj(), tb / w[idx])
    else:
        x = numpy.zeros_like(b)
    return x

def take_2d(a, idx, idy, out=None):
    '''a(idx,idy)

    Examples:

    >>> out = numpy.arange(9.).reshape(3,3)
    >>> take_2d(a, [0,2], [0,2])
    [[ 0.  2.]
     [ 6.  8.]]
    '''
    a = numpy.asarray(a, order='C')
    if out is None:
        out = numpy.zeros((len(idx),len(idy)), dtype=a.dtype)
    else:
        out = numpy.ndarray((len(idx),len(idy)), dtype=a.dtype, buffer=out)
    if numpy.iscomplexobj(a):
        out += a.take(idx, axis=0).take(idy, axis=1)
    else:
        idx = numpy.asarray(idx, dtype=numpy.int32)
        idy = numpy.asarray(idy, dtype=numpy.int32)
        _np_helper.NPdtake_2d(out.ctypes.data_as(ctypes.c_void_p),
                              a.ctypes.data_as(ctypes.c_void_p),
                              idx.ctypes.data_as(ctypes.c_void_p),
                              idy.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_int(out.shape[1]),
                              ctypes.c_int(a.shape[1]),
                              ctypes.c_int(idx.size),
                              ctypes.c_int(idy.size))
    return out

def takebak_2d(out, a, idx, idy):
    '''Reverse operation of take_2d.  out(idx,idy) = a

    Examples:

    >>> out = numpy.zeros((3,3))
    >>> takebak_2d(out, numpy.ones((2,2)), [0,2], [0,2])
    [[ 1.  0.  1.]
     [ 0.  0.  0.]
     [ 1.  0.  1.]]
    '''
    assert(out.flags.c_contiguous)
    a = numpy.asarray(a, order='C')
    if numpy.iscomplexobj(a):
        out[idx[:,None],idy] += a
    else:
        idx = numpy.asarray(idx, dtype=numpy.int32)
        idy = numpy.asarray(idy, dtype=numpy.int32)
        _np_helper.NPdtakebak_2d(out.ctypes.data_as(ctypes.c_void_p),
                                 a.ctypes.data_as(ctypes.c_void_p),
                                 idx.ctypes.data_as(ctypes.c_void_p),
                                 idy.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(out.shape[1]),
                                 ctypes.c_int(a.shape[1]),
                                 ctypes.c_int(idx.size),
                                 ctypes.c_int(idy.size))
    return out

def transpose(a, inplace=False, out=None):
    '''Transpose array for better memory efficiency

    Examples:

    >>> transpose(numpy.ones((3,2)))
    [[ 1.  1.  1.]
     [ 1.  1.  1.]]
    '''
    arow, acol = a.shape
    if inplace:
        assert(arow == acol)
        tmp = numpy.empty((BLOCK_DIM,BLOCK_DIM))
        for c0, c1 in misc.prange(0, acol, BLOCK_DIM):
            for r0, r1 in misc.prange(0, c0, BLOCK_DIM):
                tmp[:c1-c0,:r1-r0] = a[c0:c1,r0:r1]
                a[c0:c1,r0:r1] = a[r0:r1,c0:c1].T
                a[r0:r1,c0:c1] = tmp[:c1-c0,:r1-r0].T
            # diagonal blocks
            a[c0:c1,c0:c1] = a[c0:c1,c0:c1].T
        return a
    else:
        if out is None:
            out = numpy.empty((acol,arow), a.dtype)
        else:
            out = numpy.ndarray((acol,arow), a.dtype, buffer=out)
# C code is ~5% faster for acol=arow=10000
# Note: when the input a is a submatrix of another array, cannot call NPd(z)transpose
# since NPd(z)transpose assumes data continuity
        if a.flags.c_contiguous:
            if numpy.iscomplexobj(a):
                fn = _np_helper.NPztranspose
            else:
                fn = _np_helper.NPdtranspose
            fn.restype = ctypes.c_void_p
            fn(ctypes.c_int(arow), ctypes.c_int(acol),
               a.ctypes.data_as(ctypes.c_void_p),
               out.ctypes.data_as(ctypes.c_void_p),
               ctypes.c_int(BLOCK_DIM))
        else:
            r1 = c1 = 0
            for c0 in range(0, acol-BLOCK_DIM, BLOCK_DIM):
                c1 = c0 + BLOCK_DIM
                for r0 in range(0, arow-BLOCK_DIM, BLOCK_DIM):
                    r1 = r0 + BLOCK_DIM
                    out[c0:c1,r0:r1] = a[r0:r1,c0:c1].T
                out[c0:c1,r1:arow] = a[r1:arow,c0:c1].T
            for r0 in range(0, arow-BLOCK_DIM, BLOCK_DIM):
                r1 = r0 + BLOCK_DIM
                out[c1:acol,r0:r1] = a[r0:r1,c1:acol].T
            out[c1:acol,r1:arow] = a[r1:arow,c1:acol].T
        return out

def transpose_sum(a, inplace=False, out=None):
    '''a + a.T for better memory efficiency

    Examples:

    >>> transpose_sum(numpy.arange(4.).reshape(2,2))
    [[ 0.  3.]
     [ 3.  6.]]
    '''
    assert(a.shape[0] == a.shape[1])
    na = a.shape[0]
    if inplace:
        out = a
    elif out is None:
        out = numpy.empty_like(a)
    else:
        out = numpy.ndarray(a.shape, a.dtype, buffer=out)
    for c0, c1 in misc.prange(0, na, BLOCK_DIM):
        for r0, r1 in misc.prange(0, c0, BLOCK_DIM):
            tmp = a[r0:r1,c0:c1] + a[c0:c1,r0:r1].T
            out[c0:c1,r0:r1] = tmp.T
            out[r0:r1,c0:c1] = tmp
        # diagonal blocks
        tmp = a[c0:c1,c0:c1] + a[c0:c1,c0:c1].T
        out[c0:c1,c0:c1] = tmp
    return out

# NOTE: NOT assume array a, b to be C-contiguous, since a and b are two
# pointers we want to pass in.
# numpy.dot might not call optimized blas
def ddot(a, b, alpha=1, c=None, beta=0):
    '''Matrix-matrix multiplication for double precision arrays
    '''
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    if a.flags.c_contiguous:
        trans_a = 'N'
    elif a.flags.f_contiguous:
        trans_a = 'T'
        a = a.T
    else:
        raise ValueError('a.flags: %s' % str(a.flags))

    assert(k == b.shape[0])
    if b.flags.c_contiguous:
        trans_b = 'N'
    elif b.flags.f_contiguous:
        trans_b = 'T'
        b = b.T
    else:
        raise ValueError('b.flags: %s' % str(b.flags))

    if c is None:
        c = numpy.empty((m,n))
        beta = 0

    return _dgemm(trans_a, trans_b, m, n, k, a, b, c, alpha, beta)

def zdot(a, b, alpha=1, c=None, beta=0):
    '''Matrix-matrix multiplication for double complex arrays using Gauss's
    complex multiplication algorithm
    '''
    atype = a.dtype
    btype = b.dtype

    if atype == numpy.float64 and btype == numpy.float64:
        c = ddot(a, b, alpha, c, beta)

    elif atype == numpy.float64 and btype == numpy.complex128:
        br = b.real.copy()
        bi = b.imag.copy()
        cr = ddot(a, br, alpha)
        ci = ddot(a, bi, alpha)
        if c is None:
            c = cr + ci*1j
        else:
            c *= beta
            c += cr + ci*1j

    elif atype == numpy.complex128 and btype == numpy.float64:
        ar = a.real.copy()
        ai = a.imag.copy()
        cr = ddot(ar, b, alpha)
        ci = ddot(ai, b, alpha)
        if c is None:
            c = cr + ci*1j
        else:
            c *= beta
            c += cr + ci*1j

    elif atype == numpy.complex128 and btype == numpy.complex128:
        k1 = ddot(a.real+a.imag, b.real.copy(), alpha)
        k2 = ddot(a.real.copy(), b.imag-b.real, alpha)
        k3 = ddot(a.imag.copy(), b.real+b.imag, alpha)
        if c is None:
            c = k1-k3 + (k1+k2)*1j
        else:
            c *= beta
            c += k1-k3 + (k1+k2)*1j

    else:
        if c is None:
            c = numpy.dot(a, b) * alpha
        else:
            c *= beta
            c += numpy.dot(a, b) * alpha
    return c
dot = zdot

# a, b, c in C-order
def _dgemm(trans_a, trans_b, m, n, k, a, b, c, alpha=1, beta=0,
           offseta=0, offsetb=0, offsetc=0):
    assert(a.flags.c_contiguous)
    assert(b.flags.c_contiguous)
    assert(c.flags.c_contiguous)

    _np_helper.NPdgemm(ctypes.c_char(trans_b.encode('ascii')),
                       ctypes.c_char(trans_a.encode('ascii')),
                       ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k),
                       ctypes.c_int(b.shape[1]), ctypes.c_int(a.shape[1]),
                       ctypes.c_int(c.shape[1]),
                       ctypes.c_int(offsetb), ctypes.c_int(offseta),
                       ctypes.c_int(offsetc),
                       b.ctypes.data_as(ctypes.c_void_p),
                       a.ctypes.data_as(ctypes.c_void_p),
                       c.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_double(alpha), ctypes.c_double(beta))
    return c

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def asarray(a, dtype=None, order=None):
    '''Convert a list of N-dim arrays to a (N+1) dim array.  It is equivalent to
    numpy.asarray function but more efficient.
    '''
    if not isinstance(a, numpy.ndarray):
        a = numpy.vstack(a).reshape(-1, *(a[0].shape))
    return numpy.asarray(a, dtype, order)

def norm(x, ord=None, axis=None):
    '''numpy.linalg.norm for numpy 1.6.*
    '''
    if axis is None:
        return numpy.linalg.norm(x, ord)
    elif axis == 0:
        xx = numpy.einsum('ij,ij->j', x, x)
        return numpy.sqrt(xx)
    elif axis == 1:
        xx = numpy.einsum('ij,ij->i', x, x)
        return numpy.sqrt(xx)
    else:
        return numpy.linalg.norm(x, ord, axis)
        #raise RuntimeError('Not support for axis = %d' % axis)

# numpy.linalg.cond has a bug, where it
# does not correctly generalize
# condition number if s1e is not a matrix
def cond(x, p=None):
    '''Compute the condition number'''
    if p is None:
        sigma = numpy.linalg.svd(numpy.asarray(x), compute_uv=False)
        c = sigma.T[0]/sigma.T[-1] # values are along last dimension, so
                                   # so must transpose. This transpose
                                   # is omitted in numpy.linalg
        return c
    else:
        return numpy.linalg.cond(x, p)

def cartesian_prod(arrays, out=None):
    '''
    Generate a cartesian product of input arrays.
    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

    Args:
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.

    Returns:
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.

    Examples:

    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    '''
    arrays = [numpy.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    nd = len(arrays)
    dims = [nd] + [len(x) for x in arrays]

    if out is None:
        out = numpy.empty(dims, dtype)
    else:
        out = numpy.ndarray(dims, dtype, buffer=out)
    tout = out.reshape(dims)

    shape = [-1] + [1] * nd
    for i, arr in enumerate(arrays):
        tout[i] = arr.reshape(shape[:nd-i])

    return tout.reshape(nd,-1).T

def direct_sum(subscripts, *operands):
    '''Apply the summation over many operands with the einsum fashion.

    Examples:

    >>> a = numpy.ones((6,5))
    >>> b = numpy.ones((4,3,2))
    >>> direct_sum('ij,klm->ijklm', a, b).shape
    (6, 5, 4, 3, 2)
    >>> direct_sum('ij,klm', a, b).shape
    (6, 5, 4, 3, 2)
    >>> direct_sum('i,j,klm->mjlik', a[0], a[:,0], b).shape
    (2, 6, 3, 5, 4)
    >>> direct_sum('ij-klm->ijklm', a, b).shape
    (6, 5, 4, 3, 2)
    >>> direct_sum('ij+klm', a, b).shape
    (6, 5, 4, 3, 2)
    >>> direct_sum('-i-j+klm->mjlik', a[0], a[:,0], b).shape
    (2, 6, 3, 5, 4)
    '''

    def sign_and_symbs(subscript):
        ''' sign list and notation list'''
        subscript = subscript.replace(' ', '').replace(',', '+')

        if subscript[0] not in '+-':
            subscript = '+' + subscript
        sign = [x for x in subscript if x in '+-']

        symbs = subscript[1:].replace('-', '+').split('+')
        s = ''.join(symbs)
        assert(len(set(s)) == len(s))  # make sure no duplicated symbols
        return sign, symbs

    if '->' in subscripts:
        src, dest = subscripts.split('->')
        sign, src = sign_and_symbs(src)
        dest = dest.replace(' ', '')
    else:
        sign, src = sign_and_symbs(subscripts)
        dest = ''.join(src)
    assert(len(src) == len(operands))

    for i, symb in enumerate(src):
        op = numpy.asarray(operands[i])
        assert(len(symb) == op.ndim)
        if i == 0:
            if sign[i] is '+':
                out = op
            else:
                out = -op
        elif sign[i] == '+':
            out = out.reshape(out.shape+(1,)*op.ndim) + op
        else:
            out = out.reshape(out.shape+(1,)*op.ndim) - op

    return numpy.einsum('->'.join((''.join(src), dest)), out)

def condense(opname, a, locs):
    '''
    .. code-block:: python

        nd = loc[-1]
        out = numpy.empty((nd,nd))
        for i,i0 in enumerate(loc):
            i1 = loc[i+1]
            for j,j0 in enumerate(loc):
                j1 = loc[j+1]
                out[i,j] = op(a[i0:i1,j0:j1])
        return out
    '''
    assert(a.flags.c_contiguous)
    assert(a.dtype == numpy.double)
    if not opname.startswith('NP_'):
        opname = 'NP_' + opname
    op = ctypes.c_void_p(_ctypes.dlsym(_np_helper._handle, opname))
    locs = numpy.asarray(locs, numpy.int32)
    nloc = locs.size - 1
    out = numpy.empty((nloc,nloc))
    _np_helper.NPcondense(op, out.ctypes.data_as(ctypes.c_void_p),
                          a.ctypes.data_as(ctypes.c_void_p),
                          locs.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(nloc))
    return out


if __name__ == '__main__':
    a = numpy.random.random((400,900))
    print(abs(a.T - transpose(a)).sum())
    b = a[:400,:400]
    c = numpy.copy(b)
    print(abs(b.T - transpose(c,inplace=True)).sum())

    a = numpy.random.random((400,400))
    b = a + a.T.conj()
    c = transpose_sum(a)
    print(abs(b-c).sum())

    a = a+a*.5j
    for i in range(400):
        a[i,i] = a[i,i].real
    b = a-a.T.conj()
    b = numpy.array((b,b))
    x = hermi_triu(b[0], hermi=2, inplace=0)
    print(abs(b[0]-x).sum())
    x = hermi_triu(b[1], hermi=2, inplace=0)
    print(abs(b[1]-x).sum())
    x = hermi_triu(a, hermi=1, inplace=0)
    print(abs(x-x.T.conj()).sum())

    a = numpy.random.random((400,400))
    b = numpy.random.random((400,400))
    print(abs(dot(a  ,b  )-numpy.dot(a  ,b  )).sum())
    print(abs(dot(a  ,b.T)-numpy.dot(a  ,b.T)).sum())
    print(abs(dot(a.T,b  )-numpy.dot(a.T,b  )).sum())
    print(abs(dot(a.T,b.T)-numpy.dot(a.T,b.T)).sum())

    a = numpy.random.random((400,40))
    b = numpy.random.random((40,400))
    print(abs(dot(a  ,b  )-numpy.dot(a  ,b  )).sum())
    print(abs(dot(b  ,a  )-numpy.dot(b  ,a  )).sum())
    print(abs(dot(a.T,b.T)-numpy.dot(a.T,b.T)).sum())
    print(abs(dot(b.T,a.T)-numpy.dot(b.T,a.T)).sum())
    a = numpy.random.random((400,40))
    b = numpy.random.random((400,40))
    print(abs(dot(a  ,b.T)-numpy.dot(a  ,b.T)).sum())
    print(abs(dot(b  ,a.T)-numpy.dot(b  ,a.T)).sum())
    print(abs(dot(a.T,b  )-numpy.dot(a.T,b  )).sum())
    print(abs(dot(b.T,a  )-numpy.dot(b.T,a  )).sum())

    a = numpy.random.random((400,400))
    b = numpy.random.random((400,400))
    c = numpy.random.random((400,400))
    d = numpy.random.random((400,400))
    print(numpy.allclose(numpy.dot(a+b*1j, c+d*1j), dot(a+b*1j, c+d*1j)))
    print(numpy.allclose(numpy.dot(a, c+d*1j), dot(a, c+d*1j)))
    print(numpy.allclose(numpy.dot(a+b*1j, c), dot(a+b*1j, c)))

    import itertools
    arrs = (range(3,9), range(4))
    cp = cartesian_prod(arrs)
    for i,x in enumerate(itertools.product(*arrs)):
        assert(numpy.allclose(x,cp[i]))

    locs = numpy.arange(5)
    a = numpy.random.random((locs[-1],locs[-1])) - .5
    print(numpy.allclose(a, condense('sum', a, locs)))
    print(numpy.allclose(a, condense('max', a, locs)))
    print(numpy.allclose(a, condense('min', a, locs)))
    print(numpy.allclose(abs(a), condense('abssum', a, locs)))
    print(numpy.allclose(abs(a), condense('absmax', a, locs)))
    print(numpy.allclose(abs(a), condense('absmin', a, locs)))
    print(numpy.allclose(abs(a), condense('norm', a, locs)))
