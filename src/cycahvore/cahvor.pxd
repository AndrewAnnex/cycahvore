cdef extern from "cmod_cahvor.h":
    ctypedef double cmod_float_t
    ctypedef long cmod_int_t
    ctypedef cmod_int_t cmod_bool_t
    ctypedef cmod_int_t cmod_stat_t
    
    cdef void cmod_cahvor_2d_to_3d(
        const cmod_float_t pos2[2],	# input 2D position 
        const cmod_float_t c[3],	# input model center position vector   C 
        const cmod_float_t a[3],	# input model orthog. axis unit vector A 
        const cmod_float_t h[3],	# input model horizontal vector        H 
        const cmod_float_t v[3],	# input model vertical vector          V 
        const cmod_float_t o[3],	# input model optical axis unit vector O 
        const cmod_float_t r[3],	# input model radial-distortion terms  R 
        cmod_bool_t approx,		    # input flag to use fast approximation 
        cmod_float_t pos3[3],	    # output 3D origin of projection 
        cmod_float_t uvec3[3],	    # output unit vector ray of projection 
        cmod_float_t par[3][2])	    # output partial derivative of uvec3 to pos2 


    cdef void cmod_cahvor_3d_to_2d(
        const cmod_float_t pos3[3],	# input 3D position 
        const cmod_float_t c[3],	# input model center position vector   C
        const cmod_float_t a[3],	# input model orthog. axis unit vector A
        const cmod_float_t h[3],	# input model horizontal vector        H
        const cmod_float_t v[3],	# input model vertical vector          V
        const cmod_float_t o[3],	# input model optical axis unit vector O
        const cmod_float_t r[3],	# input model radial-distortion terms  R
        cmod_bool_t approx,		    # input flag to use fast approximation 
        cmod_float_t *range,	    # output range along A (same units as C) 
        cmod_float_t pos2[2],	    # output 2D image-plane projection 
        cmod_float_t par[2][3])	    # output partial derivative of pos2 to pos3 

    cdef void cmod_cahvor_warp_to_cahvor(
        const cmod_float_t c1[3],   # input model center position vector   C
        const cmod_float_t a1[3],   # input model orthog. axis unit vector A
        const cmod_float_t h1[3],   # input model horizontal vector        H
        const cmod_float_t v1[3],   # input model vertical vector          V
        const cmod_float_t o1[3],   # input model optical axis unit vector O
        const cmod_float_t r1[3],   # input model radial-distortion terms  R
        const cmod_float_t pos1[2], # input 2D position from CAHV
        cmod_bool_t approx,         # input flag to use fast approximation
        const cmod_float_t c2[3],   # input final model center position vector   C
        const cmod_float_t a2[3],   # input final model orthog. axis unit vector A
        const cmod_float_t h2[3],   # input final model horizontal vector        H
        const cmod_float_t v2[3],   # input final model vertical vector          V
        const cmod_float_t o2[3],   # input final model optical axis unit vector O
        const cmod_float_t r2[3],   # input final model radial-distortion terms  R
        cmod_float_t pos2[2])       # output 2D position for CAHV