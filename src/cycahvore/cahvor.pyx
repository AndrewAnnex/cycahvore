# cython: language_level=3, boundscheck=False, emit_code_comments=True, embedsignature=True, initializedcheck=False

from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np
np.import_array()
from .cimport cahvor

def cahvor_2d_to_3d(
       double[:] pos2,
       double[:] c,
       double[:] a,
       double[:] h,
       double[:] v,
       double[:] o,
       double[:] r,
       cmod_bool_t approx):
    """

    Args:
        pos2: input 2D position
        c: input model center position vector   C
        a: input model orthog. axis unit vector A
        h: input model horizontal vector        H
        v: input model vertical vector          V
        o: input model optical axis unit vector O
        r: input model radial-distortion terms  R 
        approx: input flag to use fast approximation

    Returns:
        pos3:  output 3D origin of projection
        uvec3: output unit vector ray of projection
        par:   output partial derivative of uvec3 to pos2
    """
    cdef np.ndarray[double, ndim=1] pos3 = np.empty(3, dtype=np.double, order='C')
    cdef np.ndarray[double, ndim=1] uvec3 = np.empty(3, dtype=np.double, order='C')
    cdef np.ndarray[double, ndim=2] par = np.empty((3,2), dtype=np.double, order='C')
    cdef double[:,::1] _par = par
    cdef cmod_float_t[3][2] _tmppar
    cahvor.cmod_cahvor_2d_to_3d(&pos2[0], &c[0], &a[0], &h[0], &v[0], &o[0], &r[0], approx, &pos3[0], &uvec3[0], _tmppar)
    _par[0][0] = _tmppar[0][0]
    _par[1][0] = _tmppar[1][0]
    _par[2][0] = _tmppar[2][0]
    _par[0][1] = _tmppar[0][1]
    _par[1][1] = _tmppar[1][1]
    _par[2][1] = _tmppar[2][1]
    return pos3, uvec3, par

def cahvor_3d_to_2d(
        double[:] pos3,
        double[:] c,
        double[:] a,
        double[:] h,
        double[:] v,
        double[:] o,
        double[:] r,
        cmod_bool_t approx):
    """

    Args:
        pos3: input 3D position
        c: input model center position vector   C
        a: input model orthog. axis unit vector A
        h: input model horizontal vector        H
        v: input model vertical vector          V
        o: input model optical axis unit vector O
        r: input model radial-distortion terms  R
        approx: input flag to use fast approximation

    Returns:
        range: output range along A (same units as C)
        pos2:  output 2D image-plane projection 
        par:   output partial derivative of pos2 to pos3 
    """
    cdef np.ndarray[double, ndim=1] pos2 = np.empty(2, dtype=np.double, order='C')
    cdef np.ndarray[double, ndim=2] par = np.empty((2,3), dtype=np.double, order='C')
    cdef double[:,::1] _par = par
    cdef cmod_float_t[2][3] _tmppar
    cdef double _range = 0.0
    cdef cmod_float_t * ptr_pos3 = &pos3[0]
    cahvor.cmod_cahvor_3d_to_2d(ptr_pos3, &c[0], &a[0], &h[0], &v[0], &o[0], &r[0], approx, &_range, &pos2[0], _tmppar)
    _par[0][0] = _tmppar[0][0]
    _par[0][1] = _tmppar[0][1]
    _par[0][2] = _tmppar[0][2]
    _par[1][0] = _tmppar[1][0]
    _par[1][1] = _tmppar[1][1]
    _par[1][2] = _tmppar[1][2]
    return _range, pos2, par

@boundscheck(False)
@wraparound(False)
def cahvor_3d_to_2d_v(
    double[:,::1] pos3s,
    double[:] c,
    double[:] a,
    double[:] h,
    double[:] v,
    double[:] o,
    double[:] r,
    cmod_bool_t approx):
    """

    Args:
        pos3s: input 3D positions
        c: input model center position vector   C
        a: input model orthog. axis unit vector A
        h: input model horizontal vector        H
        v: input model vertical vector          V
        o: input model optical axis unit vector O
        r: input model radial-distortion terms  R
        approx: input flag to use fast approximation

    Returns:
        ranges: output range along A (same units as C)
        pos2s:  output 2D image-plane projection 
        pars:   output partial derivative of pos2 to pos3 

    """
    cdef int i, n
    cdef cmod_float_t _tmp_pos3[3]
    cdef cmod_float_t _tmp_range
    cdef cmod_float_t _tmp_p2[2]
    cdef cmod_float_t[2][3] _tmppar
    n = pos3s.shape[0]
    cdef np.ndarray[double, ndim=1] ranges = np.empty(n, dtype=np.double, order='C')
    cdef np.ndarray[double, ndim=2] pos2s = np.empty((n, 2), dtype=np.double, order='C')
    cdef np.ndarray[double, ndim=3] pars = np.empty((n, 2, 3), dtype=np.double, order='C')
    # stash the cahv models into c arrays
    cdef cmod_float_t * p_c = &c[0]
    cdef cmod_float_t * p_a = &a[0]
    cdef cmod_float_t * p_h = &h[0]
    cdef cmod_float_t * p_v = &v[0]
    cdef cmod_float_t * p_o = &o[0]
    cdef cmod_float_t * p_r = &r[0]
    for i in range(n):
        _tmp_pos3[0] = pos3s[i, 0]
        _tmp_pos3[1] = pos3s[i, 1]
        _tmp_pos3[2] = pos3s[i, 2]
        cahvor.cmod_cahvor_3d_to_2d(_tmp_pos3, p_c, p_a, p_h, p_v, p_o, p_r, approx, &_tmp_range, _tmp_p2, _tmppar)
        # update ranges
        ranges[i] = _tmp_range
        # update pos2s
        pos2s[i, 0] = _tmp_p2[0]
        pos2s[i, 1] = _tmp_p2[1]
        # update pars
        pars[i, 0, 0] = _tmppar[0][0]
        pars[i, 0, 1] = _tmppar[0][1]
        pars[i, 0, 2] = _tmppar[0][2]
        pars[i, 1, 0] = _tmppar[1][0]
        pars[i, 1, 1] = _tmppar[1][1]
        pars[i, 1, 2] = _tmppar[1][2]
    return ranges, pos2s, pars


@boundscheck(False)
@wraparound(False)
def cahvor_warp_to_cahvor(
    double[:] c1,
    double[:] a1,
    double[:] h1,
    double[:] v1,
    double[:] o1,
    double[:] r1,
    int approx,
    double[:] c2,
    double[:] a2,
    double[:] h2,
    double[:] v2,
    double[:] o2,
    double[:] r2,
    const double[:,::1] pos1s):
    """

    Args:
        c1: input model center position vector   C
        a1: input model orthog. axis unit vector A
        h1: input model horizontal vector        H
        v1: input model vertical vector          V
        o1: input model optical axis unit vector O
        r1: input model radial-distortion terms  R 
        approx: input flag to use fast approximation
        c2: input final model center position vector   C
        a2: input final model orthog. axis unit vector A
        h2: input final model horizontal vector        H
        v2: input final model vertical vector          V
        o1: input final model optical axis unit vector O
        r1: input final model radial-distortion terms  R 
        pos1s: input 2D positions from the first camera model 

    Returns:
        pos2s: output 2D positions in the coordinates of the second camera model
    """
    cdef int i, n
    cdef cmod_float_t _tmp_inpt[2]
    cdef cmod_float_t _tmp_p2[2]
    n = pos1s.shape[0]
    cdef np.ndarray[double, ndim=2] pos2s = np.empty((n, 2), dtype=np.double, order='C')
    # stash the cahv models into c arrays
    cdef cmod_float_t * p_c1 = &c1[0]
    cdef cmod_float_t * p_a1 = &a1[0]
    cdef cmod_float_t * p_h1 = &h1[0]
    cdef cmod_float_t * p_v1 = &v1[0]
    cdef cmod_float_t * p_o1 = &o1[0]
    cdef cmod_float_t * p_r1 = &r1[0]
    cdef cmod_float_t * p_c2 = &c2[0]
    cdef cmod_float_t * p_a2 = &a2[0]
    cdef cmod_float_t * p_h2 = &h2[0]
    cdef cmod_float_t * p_v2 = &v2[0]
    cdef cmod_float_t * p_o2 = &o2[0]
    cdef cmod_float_t * p_r2 = &r2[0]
    for i in range(n):
        _tmp_inpt[0] = pos1s[i,0]
        _tmp_inpt[1] = pos1s[i,1]
        cahvor.cmod_cahvor_warp_to_cahvor(p_c1, p_a1, p_h1, p_v1, p_o1, p_r1, _tmp_inpt, approx, p_c2, p_a2, p_h2, p_v2, p_o2, p_r2, _tmp_p2)
        pos2s[i, 0] = _tmp_p2[0]
        pos2s[i, 1] = _tmp_p2[1]
    return pos2s