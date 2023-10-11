from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np
np.import_array()
from .cimport cahvore

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
    cmod_bool_t approx):
    """

    Args:
        pos2: input 2D position
        mtype: input type of model 
        mparm: input model parameter
        c: input model center position vector   C
        a: input model orthog. axis unit vector A
        h: input model horizontal vector        H
        v: input model vertical vector          V
        o: input model optical axis unit vector O
        r: input model radial-distortion terms  R
        e: input model entrance-pupil    terms  E
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
    cahvore.cmod_cahvore_2d_to_3d(&pos2[0], mtype, mparm, &c[0], &a[0], &h[0], &v[0], &o[0], &r[0], &e[0], approx, &pos3[0], &uvec3[0], _tmp_ppar, _tmp_upar)
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
        cmod_bool_t approx):
    """

    Args:
        pos3: input 3D position
        mtype: input type of model 
        mparm: input model parameter
        c: input model center position vector   C
        a: input model orthog. axis unit vector A
        h: input model horizontal vector        H
        v: input model vertical vector          V
        o: input model optical axis unit vector O
        r: input model radial-distortion terms  R 
        e: input model entrance-pupil    terms  E
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
    cahvore.cmod_cahvore_3d_to_2d(&pos3[0], mtype, mparm, &c[0], &a[0], &h[0], &v[0], &o[0], &r[0],  &e[0], approx, &_range, &pos2[0], _tmppar)
    _par[0][0] = _tmppar[0][0]
    _par[0][1] = _tmppar[0][1]
    _par[0][2] = _tmppar[0][2]
    _par[1][0] = _tmppar[1][0]
    _par[1][1] = _tmppar[1][1]
    _par[1][2] = _tmppar[1][2]

    return _range, pos2, par

@boundscheck(False)
@wraparound(False)
cpdef cahvore_3d_to_2d_v(
    double[:,::1] pos3s,
    int mtype,
    double mparm,
    double[:] c,
    double[:] a,
    double[:] h,
    double[:] v,
    double[:] o,
    double[:] r,
    double[:] e,
    cmod_bool_t approx):
    """

    Args:
        pos3s: input 3D positions
        mtype: input type of model 
        mparm: input model parameter
        c: input model center position vector   C
        a: input model orthog. axis unit vector A
        h: input model horizontal vector        H
        v: input model vertical vector          V
        o: input model optical axis unit vector O
        r: input model radial-distortion terms  R
        e: input model entrance-pupil    terms  E
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
    cdef cmod_float_t * p_e = &e[0]
    for i in range(n):
        _tmp_pos3[0] = pos3s[i, 0]
        _tmp_pos3[1] = pos3s[i, 1]
        _tmp_pos3[2] = pos3s[i, 2]
        cahvore.cmod_cahvore_3d_to_2d(_tmp_pos3, mtype, mparm, p_c, p_a, p_h, p_v, p_o, p_r, p_e, approx, &_tmp_range, _tmp_p2, _tmppar)
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
    cmod_bool_t approx,
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
        c1: input model center position vector   C
        a1: input model orthog. axis unit vector A
        h1: input model horizontal vector        H
        v1: input model vertical vector          V
        o1: input model optical axis unit vector O
        r1: input model radial-distortion terms  R 
        e1: input model entrance-pupil    terms  E 
        rdist: input radial distance to project
        approx: input flag to use fast approximation
        c2: input final model center position vector   C
        a2: input final model orthog. axis unit vector A
        h2: input final model horizontal vector        H
        v2: input final model vertical vector          V
        o1: input final model optical axis unit vector O
        r1: input final model radial-distortion terms  R 
        e2: input final model entrance-pupil    terms  E 
        pos1s: input 2D positions from the first camera model 

    Returns:
        pos2s: output 2D positions in the coordinates of the second camera model
    """
    cdef int i, j, n
    cdef cmod_float_t _tmp_inpt[3]
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
    cdef cmod_float_t * p_e1 = &e1[0]
    cdef cmod_float_t * p_c2 = &c2[0]
    cdef cmod_float_t * p_a2 = &a2[0]
    cdef cmod_float_t * p_h2 = &h2[0]
    cdef cmod_float_t * p_v2 = &v2[0]
    cdef cmod_float_t * p_o2 = &o2[0]
    cdef cmod_float_t * p_r2 = &r2[0]
    cdef cmod_float_t * p_e2 = &e2[0]
    for i in range(n):
        _tmp_inpt[0] = pos1s[i,0]
        _tmp_inpt[1] = pos1s[i,1]
        _tmp_inpt[2] = pos1s[i,2]
        cahvore.cmod_cahvore_warp_to_cahvore(
            mtype, mparm,
            p_c1, p_a1, p_h1, p_v1, p_o1, p_r1, p_e1, _tmp_inpt,
            rdist, approx,
            p_c2, p_a2, p_h2, p_v2, p_o2, p_r2, p_e2,
            _tmp_p2)
        pos2s[i, 0] = _tmp_p2[0]
        pos2s[i, 1] = _tmp_p2[1]
    return pos2s