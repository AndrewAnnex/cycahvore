

cdef extern from "cmod_psph.h":
    ctypedef double cmod_float_t
    ctypedef long cmod_int_t
    ctypedef cmod_int_t cmod_bool_t
    ctypedef cmod_int_t cmod_stat_t

    ctypedef struct cmod_psph_t:
        cmod_float_t c[3]		# sphere center
        cmod_float_t ax[3]  	# column rotation axis
        cmod_float_t ay[3]		# row    rotation axis
        cmod_float_t nx[3]		# column-plane normal vector at column zero
        cmod_float_t ny[3]		# row   -plane normal vector at row    zero
        cmod_float_t sx		# column scale factor (rad/pixel)
        cmod_float_t sy		# row    scale factor (rad/pixel)
    
    cdef void cmod_psph_2d_to_3d(
        const cmod_float_t pos2[2],	# input 2D position
        const cmod_psph_t *psph,	# input camera model
        cmod_float_t pos3[3],	# output 3D origin of projection
        cmod_float_t uvec3[3],	# output unit vector ray of projection
        cmod_float_t par[3][2])	# output partial-derivative uvec3/pos2

    cdef void cmod_psph_3d_to_2d(
        const cmod_float_t pos3[3],	# input 3D position
        const cmod_psph_t *psph,	# input camera model
        cmod_float_t pos2[2],	# output 2D image-plane projection
        cmod_float_t par[2][3])	# output partial-derivative pos2/pos3

    cdef cmod_stat_t cmod_psph_create(
        const cmod_float_t pos[3],	# input 3D position
        const cmod_float_t xa[3],	# input rotation axis for the columns
        const cmod_float_t xv1[3],	# input X projection vector #1
        const cmod_float_t xv2[3],	# input X projection vector #2
        cmod_float_t xc1,		# input X image coord to match vector #1
        cmod_float_t xc2,		# input X image coord to match vector #2
        const cmod_float_t ya[3],	# input rotation axis for the rows
        const cmod_float_t yv1[3],	# input Y projection vector #1
        const cmod_float_t yv2[3],	# input Y projection vector #2
        cmod_float_t yc1,		# input Y image coord to match vector #1
        cmod_float_t yc2,		# input Y image coord to match vector #2
        cmod_psph_t *psph)		# output camera model

    cdef cmod_stat_t cmod_psph_create2(
        const cmod_float_t pos[3],	# input 3D projection point
        const cmod_float_t fwd[3],	# input forward-pointing vector
        const cmod_float_t up[3],	# input upward-pointing vector
        const cmod_float_t rt[3],	# input right-pointing vector
        cmod_float_t xfov,	  	# input X field of view (rad)
        cmod_float_t yfov,		# input Y field of view (rad)
        cmod_float_t xdim,		# input X image dimension
        cmod_float_t ydim,		# input Y image dimension
        cmod_float_t xc,		# input X coordinate of image center
        cmod_float_t yc,		# input Y coordinate of image center
        cmod_psph_t *psph)		# output camera model

    cdef void cmod_psph_internal(
        const cmod_psph_t *psph,	# input camera model
        cmod_float_t *theta)	# output angle between axes

    cdef void cmod_psph_iplane(
        const cmod_psph_t *psph,	# input camera model
        cmod_float_t xc,		# input X coordinate of image center
        cmod_float_t yc,		# input Y coordinate of image center
        cmod_float_t ppnt[3],	# output projection point
        cmod_float_t ndir[3],	# output normal direction
        cmod_float_t xdir[3],	# output column direction
        cmod_float_t ydir[3])	# output row direction

    cdef void cmod_psph_move(
        const cmod_float_t p_i[3],	# input initial pos of camera ref pt
        const cmod_float_t q_i[4],	# input initial orientation of camera ref pt
        const cmod_psph_t *psph_i,	# input camera model
        const cmod_float_t p_f[3],	# input final pos of camera ref pt
        const cmod_float_t q_f[4],	# input final orientation of camera ref pt
        cmod_psph_t *psph_f)	# output camera model


    cdef void cmod_psph_pose(
        const cmod_psph_t *psph,	# input camera model
        cmod_float_t xc,		# input X coordinate of image center
        cmod_float_t yc,		# input Y coordinate of image center
        cmod_float_t p[3],		# output position vector
        cmod_float_t r[3][3])	# output rotation matrix

    cdef void cmod_psph_reflect(
        const cmod_psph_t *psph_i,	# input camera model
        cmod_float_t xc,		# input X coordinate of image center
        cmod_float_t yc,		# input Y coordinate of image center
        const cmod_float_t p[3],	# input point on the reflecting plane
        const cmod_float_t n[3],	# input normal to the reflecting plane
        cmod_psph_t *psph_f,	# output camera model
        cmod_bool_t *parallel,	# output if camera view & plane are parallel
        cmod_bool_t *behind)	# output if camera behind reflecting plane


    cdef void cmod_psph_scale(
        cmod_float_t hscale,	# input horizontal scale factor
        cmod_float_t vscale,	# input vertical   scale factor
        const cmod_psph_t *psph_i,	# input camera model
        cmod_psph_t *psph_f)	# output camera model


    cdef void cmod_psph_shift(
        cmod_float_t dx,		# input horizontal shift
        cmod_float_t dy,		# input vertical   shift
        const cmod_psph_t *psph_i,	# input camera model
        cmod_psph_t *psph_f)	# output camera model