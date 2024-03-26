
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
        const cmod_float_t pos2[2],	# input 2D position
        cmod_int_t mtype,		# input type of model
        cmod_float_t mparm,		# input model parameter
        const cmod_float_t c[3],	# input model center position vector   C
        const cmod_float_t a[3],	# input model orthog. axis unit vector A
        const cmod_float_t h[3],	# input model horizontal vector        H
        const cmod_float_t v[3],	# input model vertical vector          V
        const cmod_float_t o[3],	# input model optical axis unit vector O
        const cmod_float_t r[3],	# input model radial-distortion terms  R
        const cmod_float_t e[3],	# input model entrance-pupil    terms  E
        cmod_bool_t approx,		# input flag to use fast approximation
        cmod_float_t pos3[3],	# output 3D origin of projection
        cmod_float_t uvec3[3],	# output unit vector ray of projection
        cmod_float_t ppar[3][2],	# output partial derivative of pos3  to pos2
        cmod_float_t upar[3][2])	# output partial derivative of uvec3 to pos2

    cdef void cmod_cahvore_3d_to_2d(
        const cmod_float_t pos3[3],	# input 3D position
        cmod_int_t mtype,		# input type of model
        cmod_float_t mparm,		# input model parameter
        const cmod_float_t c[3],	# input model center vector C
        const cmod_float_t a[3],	# input model axis   vector A
        const cmod_float_t h[3],	# input model horiz. vector H
        const cmod_float_t v[3],	# input model vert.  vector V
        const cmod_float_t o[3],	# input model optical axis  O
        const cmod_float_t r[3],	# input model radial-distortion terms R
        const cmod_float_t e[3],	# input model entrance-pupil    terms E
        cmod_bool_t approx,		# input flag to use fast approximation
        cmod_float_t *range,	# output range along A (same units as C)
        cmod_float_t pos2[2],	# output 2D image-plane projection
        cmod_float_t par[2][3])	# output partial derivative of pos2 to pos3

    cdef void cmod_cahvore_3d_to_2d_point(
        cmod_int_t mtype,		# input type of model
        cmod_float_t mparm,		# input model parameter
        const cmod_float_t c[3],	# input model center vector C
        const cmod_float_t a[3],	# input model axis   vector A
        const cmod_float_t h[3],	# input model horiz. vector H
        const cmod_float_t v[3],	# input model vert.  vector V
        const cmod_float_t o[3],	# input model optical axis  O
        const cmod_float_t r[3],	# input model radial-distortion terms R
        const cmod_float_t e[3],	# input model entrance-pupil    terms E
        cmod_bool_t approx,		# input flag to use fast approximation
        const cmod_float_t pos3[3],	# input 3D position of line
        const cmod_float_t uvec3[3],# input 3D unit vector of line
        cmod_float_t pos2[2],	# output 2D image-plane projection
        cmod_float_t par[2][3])	# output derivative matrix of pos2 to uvec3


    cdef void cmod_cahvore_align_models(
        cmod_int_t xdim1,		# input number of columns
        cmod_int_t ydim1,		# input number of rows
        cmod_int_t mtype1,		# input type of model
        cmod_float_t mparm1,	# input model parameter
        const cmod_float_t c1[3],	# input model 1 center vector C
        const cmod_float_t a1[3],	# input model 1 axis   vector A
        const cmod_float_t h1[3],	# input model 1 horiz. vector H
        const cmod_float_t v1[3],	# input model 1 vert.  vector V
        const cmod_float_t o1[3],	# input model 1 axis   vector O
        const cmod_float_t r1[3],	# input model 1 dist.  terms  R
        const cmod_float_t e1[3],	# input model 1 pupil  terms  E
        cmod_int_t xdim2,		# input number of columns
        cmod_int_t ydim2,		# input number of rows
        cmod_int_t mtype2,		# input type of model
        cmod_float_t mparm2,	# input model parameter
        const cmod_float_t c2[3],	# input model 2 center vector C
        const cmod_float_t a2[3],	# input model 2 axis   vector A
        const cmod_float_t h2[3],	# input model 2 horiz. vector H
        const cmod_float_t v2[3],	# input model 2 vert.  vector V
        const cmod_float_t o2[3],	# input model 2 axis   vector O
        const cmod_float_t r2[3],	# input model 2 dist.  terms  R
        const cmod_float_t e2[3],	# input model 2 pupil  terms  E
        cmod_float_t a[3],		# output virtual model axis   vector A
        cmod_float_t h[3],		# output virtual model horiz. vector H
        cmod_float_t v[3],		# output virtual model vert.  vector V
        cmod_float_t o[3],		# output virtual model axis   vector O
        cmod_float_t r[3],		# output virtual model dist.  terms  R
        cmod_float_t e[3],		# output virtual model pupil  terms  E
        cmod_float_t *hs,		# output horizontal scale factor
        cmod_float_t *hc,		# output horizontal center
        cmod_float_t *vs,		# output vertical scale factor
        cmod_float_t *vc,		# output vertical center
        cmod_float_t *theta)	# output angle between axes

    cdef void cmod_cahvore_move(
        const cmod_float_t p_i[3],	# input initial pos of camera ref pt
        const cmod_float_t q_i[4],	# input initial orientation of camera ref pt
        const cmod_float_t c_i[3],	# input initial model center vector C
        const cmod_float_t a_i[3],	# input initial model axis   vector A
        const cmod_float_t h_i[3],	# input initial model horiz. vector H
        const cmod_float_t v_i[3],	# input initial model vert.  vector V
        const cmod_float_t o_i[3],	# input initial model axis   vector O
        const cmod_float_t r_i[3],	# input initial model dist.   terms R
        const cmod_float_t e_i[3],	# input initial model pupil   terms E
        const cmod_float_t p_f[3],	# input final pos of camera ref pt
        const cmod_float_t q_f[4],	# input final orientation of camera ref pt
        cmod_float_t c_f[3],	# output final model center vector C
        cmod_float_t a_f[3],	# output final model axis   vector A
        cmod_float_t h_f[3],	# output final model horiz. vector H
        cmod_float_t v_f[3],	# output final model vert.  vector V
        cmod_float_t o_f[3],	# output final model axis   vector O
        cmod_float_t r_f[3],	# output final model dist.  terms  R
        cmod_float_t e_f[3])	# output final model pupil  terms  E

    cdef void cmod_cahvore_reflect(
        const cmod_float_t c_i[3],	# input initial model center vector C
        const cmod_float_t a_i[3],	# input initial model axis   vector A
        const cmod_float_t h_i[3],	# input initial model horiz. vector H
        const cmod_float_t v_i[3],	# input initial model vert.  vector V
        const cmod_float_t o_i[3],	# input initial model axis   vector O
        const cmod_float_t r_i[3],	# input initial model dist.  terms  R
        const cmod_float_t e_i[3],	# input initial model pupil  terms  E
        const cmod_float_t p[3],	# input point on the reflecting plane
        const cmod_float_t n[3],	# input normal to the reflecting plane
        cmod_float_t c_f[3],	# output final model center vector C
        cmod_float_t a_f[3],	# output final model axis   vector A
        cmod_float_t h_f[3],	# output final model horiz. vector H
        cmod_float_t v_f[3],	# output final model vert.  vector V
        cmod_float_t o_f[3],	# output final model axis   vector O
        cmod_float_t r_f[3],	# output final model dist.  terms  R
        cmod_float_t e_f[3],	# output final model pupil  terms  E
        cmod_bool_t *parallel,	# output if camera view & plane are parallel
        cmod_bool_t *behind)	# output if camera behind reflecting plane

    cdef void cmod_cahvore_reflect_cov(
        cmod_float_t s_i[21][21],	# input initial covariance
        const cmod_float_t n[3],	# input normal to the reflecting plane
        cmod_float_t s_f[21][21])	# output final covariance

    cdef void cmod_cahvore_rot_cov(
        cmod_float_t r_i[3][3],	# input initial orientation of camera ref pt
        cmod_float_t s_i[21][21],	# input initial covariance
        cmod_float_t r_f[3][3],	# input final orientation of camera ref pt
        cmod_float_t s_f[21][21])	# output final covariance

    cdef void cmod_cahvore_rotate_cov(
        const cmod_float_t q_i[4],	# input initial orientation of camera ref pt
        cmod_float_t s_i[21][21],	# input initial covariance
        const cmod_float_t q_f[4],	# input final orientation of camera ref pt
        cmod_float_t s_f[21][21])	# output final covariance

    cdef void cmod_cahvore_scale(
        cmod_float_t hscale,	# input horizontal scale factor
        cmod_float_t vscale,	# input vertical   scale factor
        const cmod_float_t h1[3],	# input  model horiz. vector H
        const cmod_float_t v1[3],	# input  model vert.  vector V
        cmod_float_t s1[21][21],	# input  covariance matrix, or NULL
        cmod_float_t h2[3],		# output model horiz. vector H
        cmod_float_t v2[3],		# output model vert.  vector V
        cmod_float_t s2[21][21])	# output covariance matrix, or NULL

    cdef void cmod_cahvore_shift(
        cmod_float_t dx,		# input horizontal shift
        cmod_float_t dy,		# input vertical   shift
        const cmod_float_t a1[3],	# input  model axis   vector A
        const cmod_float_t h1[3],	# input  model horiz. vector H
        const cmod_float_t v1[3],	# input  model vert.  vector V
        cmod_float_t s1[21][21],	# input  covariance matrix, or NULL
        cmod_float_t h2[3],		# output model horiz. vector H
        cmod_float_t v2[3],		# output model vert.  vector V
        cmod_float_t s2[21][21])	# output covariance matrix, or NULL

    cdef void cmod_cahvore_transform_cov(
        cmod_float_t s_i[21][21],	# input initial covariance
        cmod_float_t r[3][3],	# input transform matrix of camera ref pt
        cmod_float_t s_f[21][21])	# output final covariance

    cdef void cmod_cahvore_warp_from_cahv(
        const cmod_float_t c1[3],	# input initial model center vector C
        const cmod_float_t a1[3],	# input initial model axis   vector A
        const cmod_float_t h1[3],	# input initial model horiz. vector H
        const cmod_float_t v1[3],	# input initial model vert.  vector V
        const cmod_float_t pos1[2],	# input 2D position from CAHV
        cmod_float_t rdist,		# input radial distance to project
        cmod_bool_t approx,		# input flag to use fast approximation
        cmod_int_t mtype2,		# input final model type
        cmod_float_t mparm2,	# input model parameter
        const cmod_float_t c2[3],	# input final model center vector C
        const cmod_float_t a2[3],	# input final model axis   vector A
        const cmod_float_t h2[3],	# input final model horiz. vector H
        const cmod_float_t v2[3],	# input final model vert.  vector V
        const cmod_float_t o2[3],	# input final model axis   vector O
        const cmod_float_t r2[3],	# input final model dist.  terms  R
        const cmod_float_t e2[3],	# input final model pupil  terms  E
        cmod_float_t pos2[2])	# output 2D position for CAHVORE

    cdef void cmod_cahvore_warp_model(
        cmod_int_t xdim,		# input number of columns
        cmod_int_t ydim,		# input number of rows
        cmod_int_t mtype,		# input type of model
        cmod_float_t mparm,		# input model parameter
        const cmod_float_t c[3],	# input model center vector C
        const cmod_float_t a[3],	# input model axis   vector A
        const cmod_float_t h[3],	# input model horiz. vector H
        const cmod_float_t v[3],	# input model vert.  vector V
        const cmod_float_t o[3],	# input model axis   vector O
        const cmod_float_t r[3],	# input model dist.  terms  R
        const cmod_float_t e[3],	# input model pupil  terms  E
        cmod_float_t limfov,	# input limit field of view: < Pi rad
        cmod_bool_t minfov,		# input if to minimize to common FOV
        cmod_int_t xdim2,		# input number of columns of output model
        cmod_int_t ydim2,		# input number of rows    of output model
        cmod_float_t a2[3],		# output virtual model axis   vector A
        cmod_float_t h2[3],		# output virtual model horiz. vector H
        cmod_float_t v2[3],		# output virtual model vert.  vector V
        cmod_float_t *hs,		# output horizontal scale factor
        cmod_float_t *hc,		# output horizontal center
        cmod_float_t *vs,		# output vertical scale factor
        cmod_float_t *vc,		# output vertical center
        cmod_float_t *theta)	# output angle between axes

    cdef void cmod_cahvore_warp_models(
        cmod_int_t xdim1,		# input number of columns
        cmod_int_t ydim1,		# input number of rows
        cmod_int_t mtype1,		# input type of model
        cmod_float_t mparm1,	# input model parameter
        const cmod_float_t c1[3],	# input model 1 center vector C
        const cmod_float_t a1[3],	# input model 1 axis   vector A
        const cmod_float_t h1[3],	# input model 1 horiz. vector H
        const cmod_float_t v1[3],	# input model 1 vert.  vector V
        const cmod_float_t o1[3],	# input model 1 axis   vector O
        const cmod_float_t r1[3],	# input model 1 dist.  terms  R
        const cmod_float_t e1[3],	# input model 1 pupil  terms  E
        cmod_int_t xdim2,		# input number of columns
        cmod_int_t ydim2,		# input number of rows
        cmod_int_t mtype2,		# input type of model
        cmod_float_t mparm2,	# input model parameter
        const cmod_float_t c2[3],	# input model 2 center vector C
        const cmod_float_t a2[3],	# input model 2 axis   vector A
        const cmod_float_t h2[3],	# input model 2 horiz. vector H
        const cmod_float_t v2[3],	# input model 2 vert.  vector V
        const cmod_float_t o2[3],	# input model 2 axis   vector O
        const cmod_float_t r2[3],	# input model 2 dist.  terms  R
        const cmod_float_t e2[3],	# input model 2 pupil  terms  E
        cmod_float_t limfov,	# input limit field of view: < Pi rad
        cmod_bool_t minfov,		# input if to minimize to common FOV
        cmod_int_t xdim,		# input number of columns of output model
        cmod_int_t ydim,		# input number of rows    of output model
        cmod_float_t a[3],		# output virtual model axis   vector A
        cmod_float_t h[3],		# output virtual model horiz. vector H
        cmod_float_t v[3],		# output virtual model vert.  vector V
        cmod_float_t *hs,		# output horizontal scale factor
        cmod_float_t *hc,		# output horizontal center
        cmod_float_t *vs,		# output vertical scale factor
        cmod_float_t *vc,		# output vertical center
        cmod_float_t *theta)	# output angle between axes

    cdef void cmod_cahvore_warp_models2(
        cmod_int_t xdim1,		# input number of columns
        cmod_int_t ydim1,		# input number of rows
        cmod_int_t mtype1,		# input type of model
        cmod_float_t mparm1,	# input model parameter
        const cmod_float_t c1[3],	# input model 1 center vector C
        const cmod_float_t a1[3],	# input model 1 axis   vector A
        const cmod_float_t h1[3],	# input model 1 horiz. vector H
        const cmod_float_t v1[3],	# input model 1 vert.  vector V
        const cmod_float_t o1[3],	# input model 1 axis   vector O
        const cmod_float_t r1[3],	# input model 1 dist.  terms  R
        const cmod_float_t e1[3],	# input model 1 pupil  terms  E
        cmod_int_t xdim2,		# input number of columns
        cmod_int_t ydim2,		# input number of rows
        cmod_int_t mtype2,		# input type of model
        cmod_float_t mparm2,	# input model parameter
        const cmod_float_t c2[3],	# input model 2 center vector C
        const cmod_float_t a2[3],	# input model 2 axis   vector A
        const cmod_float_t h2[3],	# input model 2 horiz. vector H
        const cmod_float_t v2[3],	# input model 2 vert.  vector V
        const cmod_float_t o2[3],	# input model 2 axis   vector O
        const cmod_float_t r2[3],	# input model 2 dist.  terms  R
        const cmod_float_t e2[3],	# input model 2 pupil  terms  E
        cmod_float_t limfov,	# input limit field of view: < Pi rad
        cmod_bool_t minfov,		# input if to minimize to common FOV
        cmod_int_t xdim,		# input number of columns of output model
        cmod_int_t ydim,		# input number of rows    of output model
        cmod_float_t a[3],		# output virtual model axis   vector A
        cmod_float_t h[3],		# output virtual model horiz. vector H
        cmod_float_t v[3],		# output virtual model vert.  vector V
        cmod_float_t *hs,		# output horizontal scale factor
        cmod_float_t *hc,		# output horizontal center
        cmod_float_t *vs,		# output vertical scale factor
        cmod_float_t *vc,		# output vertical center
        cmod_float_t *theta)	# output angle between axes

    cdef void cmod_cahvore_warp_to_cahv(
        cmod_int_t mtype,		# input type of model
        cmod_float_t mparm,		# input model parameter
        const cmod_float_t c1[3],	# input initial model center vector C
        const cmod_float_t a1[3],	# input initial model axis   vector A
        const cmod_float_t h1[3],	# input initial model horiz. vector H
        const cmod_float_t v1[3],	# input initial model vert.  vector V
        const cmod_float_t o1[3],	# input initial model axis   vector O
        const cmod_float_t r1[3],	# input initial model dist.  terms  R
        const cmod_float_t e1[3],	# input initial model pupil  terms  E
        const cmod_float_t pos1[2],	# input 2D position from CAHVORE
        cmod_float_t rdist,		# input radial distance to project
        cmod_bool_t approx,		# input flag to use fast approximation
        const cmod_float_t c2[3],	# input final model center vector C
        const cmod_float_t a2[3],	# input final model axis   vector A
        const cmod_float_t h2[3],	# input final model horiz. vector H
        const cmod_float_t v2[3],	# input final model vert.  vector V
        cmod_float_t pos2[2])	# output 2D position for CAHV

    cdef void cmod_cahvore_warp_to_cahvore(
        cmod_int_t mtype,		# input type of model
        cmod_float_t mparm,		# input model parameter
        const cmod_float_t c1[3],	# input initial model center vector C
        const cmod_float_t a1[3],	# input initial model axis   vector A
        const cmod_float_t h1[3],	# input initial model horiz. vector H
        const cmod_float_t v1[3],	# input initial model vert.  vector V
        const cmod_float_t o1[3],	# input initial model axis   vector O
        const cmod_float_t r1[3],	# input initial model dist   terms  R
        const cmod_float_t e1[3],	# input initial model pupil  terms  E
        const cmod_float_t pos1[2],	# input 2D position from CAHVORE
        cmod_float_t rdist,		# input radial distance to project
        cmod_bool_t approx,		# input flag to use fast approximation
        const cmod_float_t c2[3],	# input final model center vector C
        const cmod_float_t a2[3],	# input final model axis   vector A
        const cmod_float_t h2[3],	# input final model horiz. vector H
        const cmod_float_t v2[3],	# input final model vert.  vector V
        const cmod_float_t o2[3],	# input final model axis   vector O
        const cmod_float_t r2[3],	# input final model dist.  terms  R
        const cmod_float_t e2[3],	# input final model pupil  terms  E
        cmod_float_t pos2[2])	# output 2D position for CAHV