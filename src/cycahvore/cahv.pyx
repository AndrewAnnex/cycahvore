from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np
np.import_array()
from . cimport *

cpdef cahv_2d_to_3d(
    double[:] pos2,
    double[:] c,
    double[:] a,
    double[:] h,
    double[:] v):
    """
    
    Args:
        pos2: 
        c: 
        a: 
        h: 
        v: 

    Returns:

    """
    cdef np.ndarray[double, ndim=1] pos3 = np.zeros(3, dtype=np.double)
    cdef np.ndarray[double, ndim=1] uvec3 = np.zeros(3, dtype=np.double)
    cdef np.ndarray[double, ndim=2] par = np.zeros((3,2), dtype=np.double)
    cdef double[:,::1] _par = par
    cdef cmod_float_t[3][2] _tmppar
    cmod_cahv_2d_to_3d(&pos2[0], &c[0], &a[0], &h[0], &v[0], &pos3[0], &uvec3[0], _tmppar)
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
        pos3:
        c:
        a:
        h:
        v:

    Returns:

    """
    cdef np.ndarray[double, ndim=1] pos2 = np.zeros(2, dtype=np.double)
    cdef np.ndarray[double, ndim=2] par = np.zeros((2,3), dtype=np.double)
    cdef double[:,::1] _par = par
    cdef cmod_float_t[2][3] _tmppar
    cdef double _range = 0.0
    cmod_cahv_3d_to_2d(&pos2[0], &c[0], &a[0], &h[0], &v[0], &_range, &pos2[0], _tmppar)
    _par[0][0] = _tmppar[0][0]
    _par[0][1] = _tmppar[0][1]
    _par[0][2] = _tmppar[0][2]
    _par[1][0] = _tmppar[1][0]
    _par[1][1] = _tmppar[1][1]
    _par[1][2] = _tmppar[1][2]

    return _range, pos2, par


cpdef cahv_internal(
    double[:] c,
    double[:] a,
    double[:] h,
    double[:] v,
    double[:,::1] s):
    """
    
    Args:
        c: 
        a: 
        h: 
        v: 
        s: 

    Returns:

    """
    cdef int i, j
    cdef double hs, hc, vs, vc, theta = 0.0
    cdef np.ndarray[double, ndim=2] s_int = np.zeros((5,5), dtype=np.double)
    cdef double[:, ::1] _s_int = s_int
    cdef cmod_float_t[5][5] _tmp_s_int
    cmod_cahv_internal(&c[0], &a[0], &h[0], &v[0], &(s[0][0]), &hs, &hc, &vs, &vc, &theta, _tmp_s_int)
    for i in range(5):
        for j in range(5):
            _s_int[i][j] = _tmp_s_int[i][j]
    return hs, hc, vs, vc, theta, s_int