cdef extern from "cmod_cahv.h":
    ctypedef double cmod_float_t
    ctypedef long cmod_int_t
    ctypedef cmod_int_t cmod_bool_t
    ctypedef cmod_int_t cmod_stat_t

    cdef void cmod_cahv_2d_to_3d(
            const cmod_float_t pos2[2], #/* input 2D position */
            const cmod_float_t c[3],    #/* input model center vector C */
            const cmod_float_t a[3],    #/* input model axis   vector A */
            const cmod_float_t h[3],    #/* input model horiz. vector H */
            const cmod_float_t v[3],    #/* input model vert.  vector V */
            cmod_float_t pos3[3],	    #/* output 3D origin of projection */
            cmod_float_t uvec3[3],	    #/* output unit vector ray of projection */
            cmod_float_t par[3][2])	    #/* output partial derivative of uvec3 to pos2 */)

    cdef void cmod_cahv_3d_to_2d(
            const cmod_float_t pos3[3],  # /* input 3D position */
            const cmod_float_t c[3],     # /* input model center vector C */
            const cmod_float_t a[3],     # /* input model axis   vector A */
            const cmod_float_t h[3],     # /* input model horiz. vector H */
            const cmod_float_t v[3],     # /* input model vert.  vector V */
            cmod_float_t * range,	     #/* output range along A (same units as C) */
            cmod_float_t pos2[2],	     #/* output 2D image-plane projection */
            cmod_float_t par[2][3])	     #/* output partial derivative of pos2 to pos3 */