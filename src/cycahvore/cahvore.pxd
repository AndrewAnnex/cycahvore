cdef extern from "cmod_cahvore.h":
    ctypedef double cmod_float_t
    ctypedef long cmod_int_t
    ctypedef cmod_int_t cmod_bool_t
    ctypedef cmod_int_t cmod_stat_t
    ctypedef enum cmod_cahvore_type_t:
        CMOD_CAHVORE_TYPE_NONE,
        CMOD_CAHVORE_TYPE_PERSPECTIVE,
        CMOD_CAHVORE_TYPE_FISHEYE,
        CMOD_CAHVORE_TYPE_GENERAL


    cdef void cmod_cahvore_2d_to_3d(
        const cmod_float_t pos2[2], # input 2D position
        cmod_int_t mtype,           # input type of model
        cmod_float_t mparm,         # input model parameter
        const cmod_float_t c[3],    # input model center position vector   C
        const cmod_float_t a[3],    # input model orthog. axis unit vector A
        const cmod_float_t h[3],    # input model horizontal vector        H
        const cmod_float_t v[3],    # input model vertical vector          V
        const cmod_float_t o[3],    # input model optical axis unit vector O
        const cmod_float_t r[3],    # input model radial-distortion terms  R
        const cmod_float_t e[3],    # input model entrance-pupil    terms  E
        cmod_bool_t approx,         # input flag to use fast approximation
        cmod_float_t pos3[3],       # output 3D origin of projection
        cmod_float_t uvec3[3],      # output unit vector ray of projection
        cmod_float_t ppar[3][2],    # output partial derivative of pos3  to pos2
        cmod_float_t upar[3][2])    # output partial derivative of uvec3 to pos2


    cdef void cmod_cahvore_3d_to_2d(
        const cmod_float_t pos3[3], # input 3D position
        cmod_int_t mtype,           # input type of model
        cmod_float_t mparm,         # input model parameter
        const cmod_float_t c[3],    # input model center position vector   C
        const cmod_float_t a[3],    # input model orthog. axis unit vector A
        const cmod_float_t h[3],    # input model horizontal vector        H
        const cmod_float_t v[3],    # input model vertical vector          V
        const cmod_float_t o[3],    # input model optical axis unit vector O
        const cmod_float_t r[3],    # input model radial-distortion terms  R
        const cmod_float_t e[3],    # input model entrance-pupil    terms  E
        cmod_bool_t approx,         # input flag to use fast approximation
        cmod_float_t *range,        # output range along A (same units as C)
        cmod_float_t pos2[2],       # output 2D image-plane projection
        cmod_float_t par[2][3])     # output partial derivative of pos2 to pos3

    cdef void cmod_cahvore_warp_to_cahvore(
        cmod_int_t mtype,           # input type of model
        cmod_float_t mparm,         # input model parameter
        const cmod_float_t c1[3],   # input model center position vector   C
        const cmod_float_t a1[3],   # input model orthog. axis unit vector A
        const cmod_float_t h1[3],   # input model horizontal vector        H
        const cmod_float_t v1[3],   # input model vertical vector          V
        const cmod_float_t o1[3],   # input model optical axis unit vector O
        const cmod_float_t r1[3],   # input model radial-distortion terms  R
        const cmod_float_t e1[3],   # input model entrance-pupil    terms  E 
        const cmod_float_t pos1[2], # input 2D position from CAHV
        cmod_float_t rdist,         # input radial distance to project
        cmod_bool_t approx,         # input flag to use fast approximation
        const cmod_float_t c2[3],   # input final model center position vector   C
        const cmod_float_t a2[3],   # input final model orthog. axis unit vector A
        const cmod_float_t h2[3],   # input final model horizontal vector        H
        const cmod_float_t v2[3],   # input final model vertical vector          V
        const cmod_float_t o2[3],   # input final model optical axis unit vector O
        const cmod_float_t r2[3],   # input final model radial-distortion terms  R
        const cmod_float_t e2[3],   # input final model entrance-pupil    terms  E
        cmod_float_t pos2[2])       # output 2D position for CAHV