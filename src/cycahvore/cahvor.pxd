
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
        cmod_bool_t approx,		# input flag to use fast approximation
        cmod_float_t pos3[3],	# output 3D origin of projection
        cmod_float_t uvec3[3],	# output unit vector ray of projection
        cmod_float_t par[3][2])	# output partial derivative of uvec3 to pos2

    cdef void cmod_cahvor_3d_to_2d(
        const cmod_float_t pos3[3],	# input 3D position
        const cmod_float_t c[3],	# input model center vector C
        const cmod_float_t a[3],	# input model axis   vector A
        const cmod_float_t h[3],	# input model horiz. vector H
        const cmod_float_t v[3],	# input model vert.  vector V
        const cmod_float_t o[3],	# input model optical axis  O
        const cmod_float_t r[3],	# input model radial-distortion terms R
        cmod_bool_t approx,		# input flag to use fast approximation
        cmod_float_t *range,	# output range along A (same units as C)
        cmod_float_t pos2[2],	# output 2D image-plane projection
        cmod_float_t par[2][3])	# output partial derivative of pos2 to pos3

    cdef void cmod_cahvor_3d_to_2d_point(
        const cmod_float_t c[3],	# input model center vector C
        const cmod_float_t a[3],	# input model axis   vector A
        const cmod_float_t h[3],	# input model horiz. vector H
        const cmod_float_t v[3],	# input model vert.  vector V
        const cmod_float_t o[3],	# input model optical axis  O
        const cmod_float_t r[3],	# input model radial-distortion terms R
        cmod_bool_t approx,		# input flag to use fast approximation
        const cmod_float_t pos3[3],	# input 3D position of line
        const cmod_float_t uvec3[3],# input 3D unit vector of line
        cmod_float_t pos2[2],	# output 2D image-plane projection
        cmod_float_t par[2][3])	# output derivative matrix of pos2 to uvec3


    cdef void cmod_cahvor_move(
        const cmod_float_t p_i[3],	# input initial pos of camera ref pt
        const cmod_float_t q_i[4],	# input initial orientation of camera ref pt
        const cmod_float_t c_i[3],	# input initial model center vector C
        const cmod_float_t a_i[3],	# input initial model axis   vector A
        const cmod_float_t h_i[3],	# input initial model horiz. vector H
        const cmod_float_t v_i[3],	# input initial model vert.  vector V
        const cmod_float_t o_i[3],	# input initial model optical axis  O
        const cmod_float_t r_i[3],	# input initial model radial terms  R
        const cmod_float_t p_f[3],	# input final pos of camera ref pt
        const cmod_float_t q_f[4],	# input final orientation of camera ref pt
        cmod_float_t c_f[3],	# output final model center vector C
        cmod_float_t a_f[3],	# output final model axis   vector A
        cmod_float_t h_f[3],	# output final model horiz. vector H
        cmod_float_t v_f[3],	# output final model vert.  vector V
        cmod_float_t o_f[3],	# output final model optical axis  O
        cmod_float_t r_f[3])	# output final model radial terms  R

    cdef cmod_stat_t cmod_cahvor_read(
        const char *filename,	# input filename
        cmod_float_t c[3],		# output model center vector C
        cmod_float_t a[3],		# output model axis   vector A
        cmod_float_t h[3],		# output model horiz. vector H
        cmod_float_t v[3],		# output model vert.  vector V
        cmod_float_t o[3],		# output model optical axis unit vector O
        cmod_float_t r[3],		# output model radial-distortion terms  R
        cmod_float_t s[18][18],	# output covariance of CAHVOR, or NULL
        cmod_float_t *hs,		# output horizontal scale factor
        cmod_float_t *hc,		# output horizontal center
        cmod_float_t *vs,		# output vertical scale factor
        cmod_float_t *vc,		# output vertical center
        cmod_float_t *theta,	# output angle between axes
        cmod_float_t s_int[5][5])	# output covariance matrix, or NULL


    cdef cmod_stat_t cmod_cahvor_read2(
        const char *filename,	# input filename
        cmod_int_t *xdim,		# output number of columns
        cmod_int_t *ydim,		# output number of rows
        cmod_float_t c[3],		# output model center vector C
        cmod_float_t a[3],		# output model axis   vector A
        cmod_float_t h[3],		# output model horiz. vector H
        cmod_float_t v[3],		# output model vert.  vector V
        cmod_float_t o[3],		# output model optical axis unit vector O
        cmod_float_t r[3],		# output model radial-distortion terms  R
        cmod_float_t s[18][18],	# output covariance of CAHVOR, or NULL
        cmod_float_t *hs,		# output horizontal scale factor
        cmod_float_t *hc,		# output horizontal center
        cmod_float_t *vs,		# output vertical scale factor
        cmod_float_t *vc,		# output vertical center
        cmod_float_t *theta,	# output angle between axes
        cmod_float_t s_int[5][5])	# output covariance matrix, or NULL

    cdef void cmod_cahvor_reflect(
        const cmod_float_t c_i[3],	# input initial model center vector C
        const cmod_float_t a_i[3],	# input initial model axis   vector A
        const cmod_float_t h_i[3],	# input initial model horiz. vector H
        const cmod_float_t v_i[3],	# input initial model vert.  vector V
        const cmod_float_t o_i[3],	# input initial model optical axis  O
        const cmod_float_t r_i[3],	# input initial model radial terms  R
        const cmod_float_t p[3],	# input point on the reflecting plane
        const cmod_float_t n[3],	# input normal to the reflecting plane
        cmod_float_t c_f[3],	# output final model center vector C
        cmod_float_t a_f[3],	# output final model axis   vector A
        cmod_float_t h_f[3],	# output final model horiz. vector H
        cmod_float_t v_f[3],	# output final model vert.  vector V
        cmod_float_t o_f[3],	# output final model optical axis  O
        cmod_float_t r_f[3],	# output final model radial terms  R
        cmod_bool_t *parallel,	# output if camera view & plane are parallel
        cmod_bool_t *behind)	# output if camera behind reflecting plane

    cdef void cmod_cahvor_reflect_cov(
        cmod_float_t s_i[18][18],	# input initial covariance
        const cmod_float_t n[3],	# input normal to the reflecting plane
        cmod_float_t s_f[18][18])	# output final covariance


    cdef void cmod_cahvor_rot_cov(
        cmod_float_t r_i[3][3],	# input initial orientation of camera ref pt
        cmod_float_t s_i[18][18],	# input initial covariance
        cmod_float_t r_f[3][3],	# input final orientation of camera ref pt
        cmod_float_t s_f[18][18])	# output final covariance

    cdef void cmod_cahvor_rotate_cov(
        const cmod_float_t q_i[4],	# input initial orientation of camera ref pt
        cmod_float_t s_i[18][18],	# input initial covariance
        const cmod_float_t q_f[4],	# input final orientation of camera ref pt
        cmod_float_t s_f[18][18])	# output final covariance

    cdef void cmod_cahvor_scale(
        cmod_float_t hscale,	# input horizontal scale factor
        cmod_float_t vscale,	# input vertical   scale factor
        const cmod_float_t h1[3],	# input  model horiz. vector H
        const cmod_float_t v1[3],	# input  model vert.  vector V
        cmod_float_t s1[18][18],	# input  covariance matrix, or NULL
        cmod_float_t h2[3],		# output model horiz. vector H
        cmod_float_t v2[3],		# output model vert.  vector V
        cmod_float_t s2[18][18])	# output covariance matrix, or NULL

    cdef void cmod_cahvor_shift(
        cmod_float_t dx,		# input horizontal shift
        cmod_float_t dy,		# input vertical   shift
        const cmod_float_t a1[3],	# input  model axis   vector A
        const cmod_float_t h1[3],	# input  model horiz. vector H
        const cmod_float_t v1[3],	# input  model vert.  vector V
        cmod_float_t s1[18][18],	# input  covariance matrix, or NULL
        cmod_float_t h2[3],		# output model horiz. vector H
        cmod_float_t v2[3],		# output model vert.  vector V
        cmod_float_t s2[18][18])	# output covariance matrix, or NULL


    cdef void cmod_cahvor_transform_cov(
        cmod_float_t s_i[18][18],	# input initial covariance
        cmod_float_t r[3][3],	# input transform matrix of camera ref pt
        cmod_float_t s_f[18][18])	# output final covariance


    cdef cmod_stat_t cmod_cahvor_validate(
        const cmod_float_t c[3],	# input model center vector C
        const cmod_float_t a[3],	# input model axis   vector A
        const cmod_float_t h[3],	# input model horiz. vector H
        const cmod_float_t v[3],	# input model vert.  vector V
        const cmod_float_t o[3],	# input model optical axis unit vector O
        const cmod_float_t r[3])	# input model radial-distortion terms  R


    cdef void cmod_cahvor_warp_from_cahv(
        const cmod_float_t c1[3],	# input initial model center vector C
        const cmod_float_t a1[3],	# input initial model axis   vector A
        const cmod_float_t h1[3],	# input initial model horiz. vector H
        const cmod_float_t v1[3],	# input initial model vert.  vector V
        const cmod_float_t pos1[2],	# input 2D position from CAHV
        cmod_bool_t approx,		# input flag to use fast approximation
        const cmod_float_t c2[3],	# input final model center vector C
        const cmod_float_t a2[3],	# input final model axis   vector A
        const cmod_float_t h2[3],	# input final model horiz. vector H
        const cmod_float_t v2[3],	# input final model vert.  vector V
        const cmod_float_t o2[3],	# input final model optical axis  O
        const cmod_float_t r2[3],	# input final model radial  terms R
        cmod_float_t pos2[2])	# output 2D position for CAHVOR


    cdef void cmod_cahvor_warp_model(
        const cmod_float_t c[3],	# input model center vector C
        const cmod_float_t a[3],	# input model axis   vector A
        const cmod_float_t h[3],	# input model horiz. vector H
        const cmod_float_t v[3],	# input model vert.  vector V
        const cmod_float_t o[3],	# input model optical axis  O
        const cmod_float_t r[3],	# input model radial terms  R
        cmod_bool_t minfov,		# input if to minimize to common FOV
        const cmod_int_t idims[2],	# input image dimensions of input  model
        const cmod_int_t odims[2],	# input image dimensions of output model
        cmod_float_t a2[3],		# output virtual model axis   vector A
        cmod_float_t h2[3],		# output virtual model horiz. vector H
        cmod_float_t v2[3],		# output virtual model vert.  vector V
        cmod_float_t *hs,		# output horizontal scale factor
        cmod_float_t *hc,		# output horizontal center
        cmod_float_t *vs,		# output vertical scale factor
        cmod_float_t *vc,		# output vertical center
        cmod_float_t *theta)	# output angle between axes


    cdef void cmod_cahvor_warp_models(
        const cmod_float_t c1[3],	# input model 1 center vector C
        const cmod_float_t a1[3],	# input model 1 axis   vector A
        const cmod_float_t h1[3],	# input model 1 horiz. vector H
        const cmod_float_t v1[3],	# input model 1 vert.  vector V
        const cmod_float_t o1[3],	# input model 1 axis   vector O
        const cmod_float_t r1[3],	# input model 1 dist.  terms  R
        const cmod_float_t c2[3],	# input model 2 center vector C
        const cmod_float_t a2[3],	# input model 2 axis   vector A
        const cmod_float_t h2[3],	# input model 2 horiz. vector H
        const cmod_float_t v2[3],	# input model 2 vert.  vector V
        const cmod_float_t o2[3],	# input model 2 axis   vector O
        const cmod_float_t r2[3],	# input model 2 dist.  terms  R
        cmod_bool_t minfov,		# input if to minimize to common FOV
        const cmod_int_t idims[2],	# input image dimensions of input  models
        const cmod_int_t odims[2],	# input image dimensions of output models
        cmod_float_t a[3],		# output virtual model axis   vector A
        cmod_float_t h[3],		# output virtual model horiz. vector H
        cmod_float_t v[3],		# output virtual model vert.  vector V
        cmod_float_t *hs,		# output horizontal scale factor
        cmod_float_t *hc,		# output horizontal center
        cmod_float_t *vs,		# output vertical scale factor
        cmod_float_t *vc,		# output vertical center
        cmod_float_t *theta)	# output angle between axes



    cdef void cmod_cahvor_warp_models_nodims(
        const cmod_float_t c1[3],	# input model 1 center vector C
        const cmod_float_t a1[3],	# input model 1 axis   vector A
        const cmod_float_t h1[3],	# input model 1 horiz. vector H
        const cmod_float_t v1[3],	# input model 1 vert.  vector V
        const cmod_float_t o1[3],	# input model 1 axis   vector O
        const cmod_float_t r1[3],	# input model 1 dist.  terms  R
        const cmod_float_t c2[3],	# input model 2 center vector C
        const cmod_float_t a2[3],	# input model 2 axis   vector A
        const cmod_float_t h2[3],	# input model 2 horiz. vector H
        const cmod_float_t v2[3],	# input model 2 vert.  vector V
        const cmod_float_t o2[3],	# input model 2 axis   vector O
        const cmod_float_t r2[3],	# input model 2 dist.  terms  R
        cmod_bool_t minfov,		# input if to minimize to common FOV
        cmod_float_t a[3],		# output virtual model axis   vector A
        cmod_float_t h[3],		# output virtual model horiz. vector H
        cmod_float_t v[3],		# output virtual model vert.  vector V
        cmod_float_t *hs,		# output horizontal scale factor
        cmod_float_t *hc,		# output horizontal center
        cmod_float_t *vs,		# output vertical scale factor
        cmod_float_t *vc,		# output vertical center
        cmod_float_t *theta)	# output angle between axes


    cdef void cmod_cahvor_warp_models2(
        const cmod_float_t c1[3],	# input model 1 center vector C
        const cmod_float_t a1[3],	# input model 1 axis   vector A
        const cmod_float_t h1[3],	# input model 1 horiz. vector H
        const cmod_float_t v1[3],	# input model 1 vert.  vector V
        const cmod_float_t o1[3],	# input model 1 axis   vector O
        const cmod_float_t r1[3],	# input model 1 dist.  terms  R
        const cmod_float_t c2[3],	# input model 2 center vector C
        const cmod_float_t a2[3],	# input model 2 axis   vector A
        const cmod_float_t h2[3],	# input model 2 horiz. vector H
        const cmod_float_t v2[3],	# input model 2 vert.  vector V
        const cmod_float_t o2[3],	# input model 2 axis   vector O
        const cmod_float_t r2[3],	# input model 2 dist.  terms  R
        cmod_bool_t minfov,		# input if to minimize to common FOV
        const cmod_int_t idims[2],	# input image dimensions of input  models
        const cmod_int_t odims[2],	# input image dimensions of output models
        cmod_float_t a[3],		# output virtual model axis   vector A
        cmod_float_t h[3],		# output virtual model horiz. vector H
        cmod_float_t v[3],		# output virtual model vert.  vector V
        cmod_float_t *hs,		# output horizontal scale factor
        cmod_float_t *hc,		# output horizontal center
        cmod_float_t *vs,		# output vertical scale factor
        cmod_float_t *vc,		# output vertical center
        cmod_float_t *theta)	# output angle between axes



    cdef void cmod_cahvor_warp_to_cahv(
        const cmod_float_t c1[3],	# input initial model center vector C
        const cmod_float_t a1[3],	# input initial model axis   vector A
        const cmod_float_t h1[3],	# input initial model horiz. vector H
        const cmod_float_t v1[3],	# input initial model vert.  vector V
        const cmod_float_t o1[3],	# input initial model optical axis  O
        const cmod_float_t r1[3],	# input initial model radial  terms R
        const cmod_float_t pos1[2],	# input 2D position from CAHVOR
        cmod_bool_t approx,		# input flag to use fast approximation
        const cmod_float_t c2[3],	# input final model center vector C
        const cmod_float_t a2[3],	# input final model axis   vector A
        const cmod_float_t h2[3],	# input final model horiz. vector H
        const cmod_float_t v2[3],	# input final model vert.  vector V
        cmod_float_t pos2[2])	# output 2D position for CAHV


    cdef void cmod_cahvor_warp_to_cahvor(
        const cmod_float_t c1[3],	# input initial model center vector C
        const cmod_float_t a1[3],	# input initial model axis   vector A
        const cmod_float_t h1[3],	# input initial model horiz. vector H
        const cmod_float_t v1[3],	# input initial model vert.  vector V
        const cmod_float_t o1[3],	# input initial model optical axis  O
        const cmod_float_t r1[3],	# input initial model radial terms  R
        const cmod_float_t pos1[2],	# input 2D position from CAHVOR
        cmod_bool_t approx,		# input flag to use fast approximation
        const cmod_float_t c2[3],	# input final model center vector C
        const cmod_float_t a2[3],	# input final model axis   vector A
        const cmod_float_t h2[3],	# input final model horiz. vector H
        const cmod_float_t v2[3],	# input final model vert.  vector V
        const cmod_float_t o2[3],	# input final model optical axis  O
        const cmod_float_t r2[3],	# input final model radial terms  R
        cmod_float_t pos2[2])	# output 2D position for CAHV
    