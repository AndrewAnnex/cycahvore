# cython: language_level=3, boundscheck=False, emit_code_comments=True, embedsignature=True, initializedcheck=False

from cython cimport boundscheck, wraparound, nogil
import numpy as np
cimport numpy as np
np.import_array()
from . cimport cahv

cpdef cahv_2d_to_3d(
    double[:] pos2,
    double[:] c,
    double[:] a,
    double[:] h,
    double[:] v):
    """
    
    Args:
        pos2: input 2D position
        c: input model center position vector   C
        a: input model orthog. axis unit vector A
        h: input model horizontal vector        H
        v: input model vertical vector          V

    Returns:
        pos3:  output 3D origin of projection
        uvec3: output unit vector ray of projection
        par:   output partial derivative of uvec3 to pos2
    """
    cdef np.ndarray[double, ndim=1] pos3 = np.zeros(3, dtype=np.double)
    cdef np.ndarray[double, ndim=1] uvec3 = np.zeros(3, dtype=np.double)
    cdef np.ndarray[double, ndim=2] par = np.zeros((3,2), dtype=np.double)
    cdef double[:,::1] _par = par
    cdef cmod_float_t[3][2] _tmppar
    cahv.cmod_cahv_2d_to_3d(&pos2[0], &c[0], &a[0], &h[0], &v[0], &pos3[0], &uvec3[0], _tmppar)
    _par[0][0] = _tmppar[0][0]
    _par[1][0] = _tmppar[1][0]
    _par[2][0] = _tmppar[2][0]
    _par[0][1] = _tmppar[0][1]
    _par[1][1] = _tmppar[1][1]
    _par[2][1] = _tmppar[2][1]
    return pos3, uvec3, par

cpdef cahv_3d_to_2d(
    double[:] pos3,
    double[:] c,
    double[:] a,
    double[:] h,
    double[:] v):
    """

    Args:
        pos3: input 3D position 
        c: input model center position vector   C
        a: input model orthog. axis unit vector A
        h: input model horizontal vector        H
        v: input model vertical vector          V

    Returns:
        range: output range along A (same units as C)
        pos2:  output 2D image-plane projection 
        par:   output partial derivative of pos2 to pos3 

    """
    cdef np.ndarray[double, ndim=1] pos2 = np.zeros(2, dtype=np.double, order='C')
    cdef np.ndarray[double, ndim=2] par = np.zeros((2,3), dtype=np.double, order='C')
    cdef double[:,::1] _par = par
    cdef cmod_float_t[2][3] _tmppar
    cdef double _range = 0.0
    cahv.cmod_cahv_3d_to_2d(&pos3[0], &c[0], &a[0], &h[0], &v[0], &_range, &pos2[0], _tmppar)
    _par[0][0] = _tmppar[0][0]
    _par[0][1] = _tmppar[0][1]
    _par[0][2] = _tmppar[0][2]
    _par[1][0] = _tmppar[1][0]
    _par[1][1] = _tmppar[1][1]
    _par[1][2] = _tmppar[1][2]

    return _range, pos2, par

@boundscheck(False)
@wraparound(False)
cpdef cahv_3d_to_2d_v(
    double[:,::1] pos3s,
    double[:] c,
    double[:] a,
    double[:] h,
    double[:] v):
    """

    Args:
        pos3s: input 3D positions
        c: input model center vector C
        a: input model axis   vector A
        h: input model horiz. vector H
        v: input model vert.  vector V

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
    for i in range(n):
        _tmp_pos3[0] = pos3s[i, 0]
        _tmp_pos3[1] = pos3s[i, 1]
        _tmp_pos3[2] = pos3s[i, 2]
        cahv.cmod_cahv_3d_to_2d(_tmp_pos3, p_c, p_a, p_h, p_v, &_tmp_range, _tmp_p2, _tmppar)
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

cpdef cahv_3d_to_2d_ray(
    double[:] c,
    double[:] a,
    double[:] h,
    double[:] v,
    double[:] pos3,
    double[:] uvec3):
    """

    Args:
        c: input model center position vector   C
        a: input model orthog. axis unit vector A
        h: input model horizontal vector        H
        v: input model vertical vector          V
        pos3: input 3D position of line
        uvec3: input 3D unit vector of line

    Returns:
        pos2:  output 2D image-plane projection 
        uvec2: output 2D unit vector back-projected line
        par:   output derivative of pos2,uvec2 to uvec3
    """
    cdef int i, j
    cdef np.ndarray[double, ndim=1] pos2 = np.zeros(2, dtype=np.double, order='C')
    cdef np.ndarray[double, ndim=1] uvec2 = np.zeros(2, dtype=np.double, order='C')
    cdef np.ndarray[double, ndim=2] par = np.zeros((4,3), dtype=np.double, order='C')
    cdef double[:, ::1] _par = par
    cdef cmod_float_t[4][3] _tmp_par
    cahv.cmod_cahv_3d_to_2d_ray(&c[0], &a[0], &h[0], &v[0], &pos3[0], &uvec3[0], &pos2[0], &uvec2[0], _tmp_par)
    for i in range(4):
        for j in range(3):
            _par[i][j] = _tmp_par[i][j]
    return pos2, uvec2, par


cpdef cahv_internal(
    double[:] c,
    double[:] a,
    double[:] h,
    double[:] v,
    np.ndarray[np.float64_t, ndim=2] s):
    """
    
    Args:
        c: input model center position vector   C
        a: input model orthog. axis unit vector A
        h: input model horizontal vector        H
        v: input model vertical vector          V
        s: input 12x12 covariance of CAHV (optional, you can pass in empty array)

    Returns:
        hs: output horizontal scale factor 
        hc: output horizontal center 
        vs: output vertical scale factor 
        vc: output vertical center 
        theta: output angle between axes 
        s_int: output covariance matrix 

    """
    cdef int i, j
    cdef double hs, hc, vs, vc, theta = 0.0
    cdef np.ndarray[double, ndim=2] s_int = np.zeros((5,5), dtype=np.double, order='C')
    cdef double[:, ::1] _s_int = s_int
    cdef cmod_float_t[5][5] _tmp_s_int
    cdef cmod_float_t[12][12] _s
    for i in range(12):
        for j in range(12):
            _s[i][j] = s[i, j]
    cahv.cmod_cahv_internal(&c[0], &a[0], &h[0], &v[0], _s, &hs, &hc, &vs, &vc, &theta, _tmp_s_int)
    for i in range(5):
        for j in range(5):
            _s_int[i][j] = _tmp_s_int[i][j]
    return hs, hc, vs, vc, theta, s_int

@boundscheck(False)
@wraparound(False)
cpdef cahv_warp_to_cahv(
    double[:] c1,
    double[:] a1,
    double[:] h1,
    double[:] v1,
    double[:] c2,
    double[:] a2,
    double[:] h2,
    double[:] v2,
    const double[:,::1] pos1s):
    """
    
    Args:
        c1: input model center position vector   C
        a1: input model orthog. axis unit vector A
        h1: input model horizontal vector        H
        v1: input model vertical vector          V
        c2: input final model center position vector   C
        a2: input final model orthog. axis unit vector A
        h2: input final model horizontal vector        H
        v2: input final model vertical vector          V
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
    cdef cmod_float_t * p_c2 = &c2[0]
    cdef cmod_float_t * p_a2 = &a2[0]
    cdef cmod_float_t * p_h2 = &h2[0]
    cdef cmod_float_t * p_v2 = &v2[0]
    for i in range(n):
        _tmp_inpt[0] = pos1s[i,0]
        _tmp_inpt[1] = pos1s[i,1]
        cahv.cmod_cahv_warp_to_cahv(p_c1, p_a1, p_h1, p_v1, _tmp_inpt, p_c2, p_a2, p_h2, p_v2, _tmp_p2)
        pos2s[i, 0] = _tmp_p2[0]
        pos2s[i, 1] = _tmp_p2[1]
    return pos2s