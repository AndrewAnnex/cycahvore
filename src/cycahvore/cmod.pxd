cdef extern from "cmod.h":
    ctypedef double cmod_float_t
    ctypedef long cmod_int_t
    ctypedef cmod_int_t cmod_bool_t
    ctypedef cmod_int_t cmod_stat_t

    ctypedef enum cmod_class_t:
        CMOD_CLASS_NONE,
        CMOD_CLASS_CAHV,
        CMOD_CLASS_CAHVOR,
        CMOD_CLASS_CAHVORE,
        CMOD_CLASS_PSPH

    cdef struct cmod_cahv_t:
        cmod_float_t c[3]		# model center vector C */
        cmod_float_t a[3]		# model axis   vector A */
        cmod_float_t h[3]		# model horiz. vector H */
        cmod_float_t v[3]		# model vert.  vector V */

    cdef struct cmod_cahv_ext_t:
        cmod_float_t s[12][12];	    # covariance of CAHV */
        cmod_float_t hs;		    # horizontal scale factor */
        cmod_float_t hc;		    # horizontal center */
        cmod_float_t vs;		    # vertical scale factor */
        cmod_float_t vc;		    # vertical center */
        cmod_float_t theta;		    # angle between axes */
        cmod_float_t s_int[5][5];	# covariance matrix */

    cdef struct cmod_cahvor_t:
        cmod_float_t c[3]		# model center vector C */
        cmod_float_t a[3]		# model axis   vector A */
        cmod_float_t h[3]		# model horiz. vector H */
        cmod_float_t v[3]		# model vert.  vector V */
        cmod_float_t o[3];		# model optical axis unit vector O */
        cmod_float_t r[3];		# model radial-distortion terms  R */

    cdef struct cmod_cahvor_ext_t:
        cmod_float_t s[18][18];	    # covariance of CAHVOR */
        cmod_float_t hs;		    # horizontal scale factor */
        cmod_float_t hc;		    # horizontal center */
        cmod_float_t vs;		    # vertical scale factor */
        cmod_float_t vc;		    # vertical center */
        cmod_float_t theta;		    # angle between axes */
        cmod_float_t s_int[5][5];	# covariance matrix */

    cdef struct cmod_cahvore_t:
        cmod_int_t mtype;		# type of model */
        cmod_float_t mparm;		# model parameter */
        cmod_float_t c[3]		# model center vector C */
        cmod_float_t a[3]		# model axis   vector A */
        cmod_float_t h[3]		# model horiz. vector H */
        cmod_float_t v[3]		# model vert.  vector V */
        cmod_float_t o[3];		# model optical axis unit vector O */
        cmod_float_t r[3];		# model radial-distortion terms  R */
        cmod_float_t e[3];		# model entrance-pupil    terms  E */

    cdef struct cmod_cahvore_ext_t:
        cmod_float_t s[21][21];	    # covariance of CAHVORE */
        cmod_float_t hs;		    # horizontal scale factor */
        cmod_float_t hc;		    # horizontal center */
        cmod_float_t vs;		    # vertical scale factor */
        cmod_float_t vc;		    # vertical center */
        cmod_float_t theta;		    # angle between axes */
        cmod_float_t s_int[5][5];	# covariance matrix */

    cdef struct cmod_psph_t:
        cmod_float_t c[3];		# sphere center */
        cmod_float_t ax[3];		# column rotation axis */
        cmod_float_t ay[3];		# row    rotation axis */
        cmod_float_t nx[3];		# column-plane normal vector at column zero */
        cmod_float_t ny[3];		# row   -plane normal vector at row    zero */
        cmod_float_t sx;		# column scale factor (rad/pixel) */
        cmod_float_t sy;		# row    scale factor (rad/pixel) */

    cdef struct cmod_psph_ext_t:
        cmod_float_t theta;		# angle between axes */

    cdef union cmod_kind:
        cmod_cahv_t cahv
        cmod_cahvor_t cahvor
        cmod_cahvore_t cahvore
        cmod_psph_t psph

    cdef struct cmod_t:
        cmod_int_t xdim;		# number of image columns */
        cmod_int_t ydim;		# number of image rows */
        cmod_class_t mclass;	# model class */
        cmod_kind u;

    cdef union cmod_ext_kind:
        cmod_cahv_ext_t cahv
        cmod_cahvor_ext_t cahvor
        cmod_cahvore_ext_t cahvore
        cmod_psph_ext_t psph

    cdef struct cmod_ext_t:
        cmod_t core # core model
        cmod_ext_kind ext # union of extended model elements