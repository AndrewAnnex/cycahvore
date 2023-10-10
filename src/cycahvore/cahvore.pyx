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
        mtype: input type of model 
        mparm: input model parameter
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
        mtype: input type of model 
        mparm: input model parameter
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

@boundscheck(False)
@wraparound(False)
cpdef cahvore_warp_to_cahvore(
    int mtype,
    double mparm,
    double[:] c1,
    double[:] a1,
    double[:] h1,
    double[:] v1,
    double[:] o1,
    double[:] r1,
    double[:] e1,
    double rdist,
    int approx,
    double[:] c2,
    double[:] a2,
    double[:] h2,
    double[:] v2,
    double[:] o2,
    double[:] r2,
    double[:] e2,
    const double[:,::1] pos1s):
    """

    Args:
        mtype: input type of model 
        mparm: input model parameter
        c1: input initial model center vector C 
        a1: input initial model axis   vector A 
        h1: input initial model horiz. vector H 
        v1: input initial model vert.  vector V 
        o1: input model optical axis  O 
        r1: input model radial-distortion terms R 
        e1: input model entrance-pupil    terms E
        rdist: input radial distance to project
        approx: input flag to use fast approximation
        c2: input final model center vector C
        a2: input final model axis   vector A
        h2: input final model horiz. vector H
        v2: input final model vert.  vector V
        o1: input final model optical axis  O 
        r1: input final model radial-distortion terms R 
        e2: input final model entrance-pupil    terms E
        pos1s: input 2D positions from the first camera model 

    Returns:
        pos2s: output 2D positions in the coordinates of the second camera model
    """
    cdef int i, j, n
    cdef cmod_float_t _tmp_inpt[3]
    cdef cmod_float_t _tmp_p3[3]
    n = pos1s.shape[0]
    cdef np.ndarray[double, ndim=2] pos2s = np.empty((n, 2), dtype=np.double, order='C')
    # stash the cahv models into c arrays
    cdef cmod_float_t * ptr_c1 = &c1[0]
    cdef cmod_float_t * ptr_a1 = &a1[0]
    cdef cmod_float_t * ptr_h1 = &h1[0]
    cdef cmod_float_t * ptr_v1 = &v1[0]
    cdef cmod_float_t * ptr_o1 = &o1[0]
    cdef cmod_float_t * ptr_r1 = &r1[0]
    cdef cmod_float_t * ptr_e1 = &e1[0]
    cdef cmod_float_t * ptr_c2 = &c2[0]
    cdef cmod_float_t * ptr_a2 = &a2[0]
    cdef cmod_float_t * ptr_h2 = &h2[0]
    cdef cmod_float_t * ptr_v2 = &v2[0]
    cdef cmod_float_t * ptr_o2 = &o2[0]
    cdef cmod_float_t * ptr_r2 = &r2[0]
    cdef cmod_float_t * ptr_e2 = &e2[0]
    for i in range(n):
        _tmp_inpt[0] = pos1s[i,0]
        _tmp_inpt[1] = pos1s[i,1]
        _tmp_inpt[2] = pos1s[i,2]
        cmod_cahvore_warp_to_cahvore(
            mtype, mparm,
            ptr_c1, ptr_a1, ptr_h1, ptr_v1, ptr_o1, ptr_r1, ptr_e1, _tmp_inpt,
            rdist, approx,
            ptr_c2, ptr_a2, ptr_h2, ptr_v2, ptr_o2, ptr_r2, ptr_e2,
            _tmp_p3)
        pos2s[i, 0] = _tmp_p3[0]
        pos2s[i, 1] = _tmp_p3[1]
        pos2s[i, 2] = _tmp_p3[2]
    return pos2s