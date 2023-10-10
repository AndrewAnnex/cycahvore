from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np
np.import_array()
from .cimport *

cpdef cahvore_2d_to_3d(
    double[:] pos2,
    int mtype,
    double mparm,
    double[:] c,
    double[:] a,
    double[:] h,
    double[:] v,
    double[:] o,
    double[:] r,
    double[:] e,
    int approx):
    """

    Args:
        pos2: input 2D position
        c: input model center vector C
        a: input model axis   vector A
        h: input model horiz. vector H
        v: input model vert.  vector V
        o: input model optical axis  O 
        r: input model radial-distortion terms R 
        e: input model entrance-pupil    terms E
        approx: input flag to use fast approximation

    Returns:
        pos3:  output 3D origin of projection
        uvec3: output unit vector ray of projection
        ppar:  output partial derivative of pos3  to pos2
        upar:  output partial derivative of uvec3 to pos2
    """
    cdef np.ndarray[double, ndim=1] pos3 = np.zeros(3, dtype=np.double, order='C')
    cdef np.ndarray[double, ndim=1] uvec3 = np.zeros(3, dtype=np.double, order='C')
    cdef np.ndarray[double, ndim=2] ppar = np.zeros((3,2), dtype=np.double, order='C')
    cdef double[:,::1] _ppar = ppar
    cdef cmod_float_t[3][2] _tmp_ppar
    cdef np.ndarray[double, ndim=2] upar = np.zeros((3,2), dtype=np.double, order='C')
    cdef double[:,::1] _upar = upar
    cdef cmod_float_t[3][2] _tmp_upar
    cmod_cahvore_2d_to_3d(&pos2[0], mtype, mparm, &c[0], &a[0], &h[0], &v[0], &o[0], &r[0], &e[0], approx, &pos3[0], &uvec3[0], _tmp_ppar, _tmp_upar)
    _ppar[0][0] = _tmp_ppar[0][0]
    _ppar[1][0] = _tmp_ppar[1][0]
    _ppar[2][0] = _tmp_ppar[2][0]
    _ppar[0][1] = _tmp_ppar[0][1]
    _ppar[1][1] = _tmp_ppar[1][1]
    _ppar[2][1] = _tmp_ppar[2][1]
    _upar[0][0] = _tmp_upar[0][0]
    _upar[1][0] = _tmp_upar[1][0]
    _upar[2][0] = _tmp_upar[2][0]
    _upar[0][1] = _tmp_upar[0][1]
    _upar[1][1] = _tmp_upar[1][1]
    _upar[2][1] = _tmp_upar[2][1]
    return pos3, uvec3, ppar, upar

cpdef cahvore_3d_to_2d(
        double[:] pos3,
        int mtype,
        double mparm,
        double[:] c,
        double[:] a,
        double[:] h,
        double[:] v,
        double[:] o,
        double[:] r,
        double[:] e,
        int approx):
    """

    Args:
        pos3: input 3D position
        c: input model center vector C
        a: input model axis   vector A
        h: input model horiz. vector H
        v: input model vert.  vector V
        o: input model optical axis  O 
        r: input model radial-distortion terms R 
        e: input model entrance-pupil    terms E
        approx: input flag to use fast approximation

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
    cmod_cahvore_3d_to_2d(&pos2[0], mtype, mparm, &c[0], &a[0], &h[0], &v[0], &o[0], &r[0],  &e[0], approx, &_range, &pos2[0], _tmppar)
    _par[0][0] = _tmppar[0][0]
    _par[0][1] = _tmppar[0][1]
    _par[0][2] = _tmppar[0][2]
    _par[1][0] = _tmppar[1][0]
    _par[1][1] = _tmppar[1][1]
    _par[1][2] = _tmppar[1][2]

    return _range, pos2, par