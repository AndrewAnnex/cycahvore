cdef extern from "cmod_cahv.h":
    ctypedef double cmod_float_t
    ctypedef long cmod_int_t
    ctypedef cmod_int_t cmod_bool_t
    ctypedef cmod_int_t cmod_stat_t

    cdef void cmod_cahv_2d_to_3d(
            const cmod_float_t pos2[2], # input 2D position 
            const cmod_float_t c[3],    # input model center vector C 
            const cmod_float_t a[3],    # input model axis   vector A 
            const cmod_float_t h[3],    # input model horiz. vector H 
            const cmod_float_t v[3],    # input model vert.  vector V 
            cmod_float_t pos3[3],	    # output 3D origin of projection 
            cmod_float_t uvec3[3],	    # output unit vector ray of projection 
            cmod_float_t par[3][2])	    # output partial derivative of uvec3 to pos2 

    cdef void cmod_cahv_3d_to_2d(
            const cmod_float_t pos3[3],  # input 3D position 
            const cmod_float_t c[3],     # input model center vector C 
            const cmod_float_t a[3],     # input model axis   vector A 
            const cmod_float_t h[3],     # input model horiz. vector H 
            const cmod_float_t v[3],     # input model vert.  vector V 
            cmod_float_t * range,	     # output range along A (same units as C) 
            cmod_float_t pos2[2],	     # output 2D image-plane projection 
            cmod_float_t par[2][3])	     # output partial derivative of pos2 to pos3 

    cdef void cmod_cahv_3d_to_2d_ray(
            const cmod_float_t c[3],     # input model center vector C
            const cmod_float_t a[3],     # input model axis   vector A
            const cmod_float_t h[3],     # input model horiz. vector H
            const cmod_float_t v[3],     # input model vert.  vector V
            const cmod_float_t pos3[3],  # input 3D position of line
            const cmod_float_t uvec3[3], # input 3D unit vector of line
            cmod_float_t pos2[2],        # output 2D image-plane projection
            cmod_float_t uvec2[2],       # output 2D unit vector back-projected line
            cmod_float_t par[4][3])      # output derivative of pos2,uvec2 to uvec3

    cdef void cmod_cahv_internal(
            const cmod_float_t c[3],	# input model center vector C 
            const cmod_float_t a[3],	# input model axis   vector A 
            const cmod_float_t h[3],	# input model horiz. vector H 
            const cmod_float_t v[3],	# input model vert.  vector V 
            cmod_float_t s[12][12],	    # input covariance of CAHV 
            cmod_float_t *hs,		    # output horizontal scale factor 
            cmod_float_t *hc,		    # output horizontal center 
            cmod_float_t *vs,		    # output vertical scale factor 
            cmod_float_t *vc,		    # output vertical center 
            cmod_float_t *theta,	    # output angle between axes 
            cmod_float_t s_int[5][5])	# output covariance matrix 

    cdef void cmod_cahv_warp_to_cahv(
        const cmod_float_t c1[3],      # input initial model center vector C
        const cmod_float_t a1[3],      # input initial model axis   vector A
        const cmod_float_t h1[3],      # input initial model horiz. vector H
        const cmod_float_t v1[3],      # input initial model vert.  vector V
        const cmod_float_t pos1[2],    # input 2D position from CAHV
        const cmod_float_t c2[3],      # input final model center vector C
        const cmod_float_t a2[3],      # input final model axis   vector A
        const cmod_float_t h2[3],      # input final model horiz. vector H
        const cmod_float_t v2[3],      # input final model vert.  vector V
        cmod_float_t pos2[2])          #   output 2D position for CAHV
