

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

    cdef cmod_stat_t cmod_cahv_create(
        const cmod_float_t pos[3],	# input 3D position */
        const cmod_float_t x[3],	# input dir of increasing X image coordinate */
        const cmod_float_t xv1[3],	# input X projection vector #1 */
        const cmod_float_t xv2[3],	# input X projection vector #2 */
        cmod_float_t xc1,		    # input X image coord to match vector #1 */
        cmod_float_t xc2,		    # input X image coord to match vector #2 */
        const cmod_float_t y[3],	# input dir of increasing Y image coordinate */
        const cmod_float_t yv1[3],	# input Y projection vector #1 */
        const cmod_float_t yv2[3],	# input Y projection vector #2 */
        cmod_float_t yc1,		    # input Y image coord to match vector #1 */
        cmod_float_t yc2,		    # input Y image coord to match vector #2 */
        cmod_float_t c[3],		    # output model center vector C */
        cmod_float_t a[3],		    # output model axis   vector A */
        cmod_float_t h[3],		    # output model horiz. vector H */
        cmod_float_t v[3])		    # output model vert.  vector V */

    cdef cmod_stat_t cmod_cahv_create2(
        const cmod_float_t pos[3],	# input 3D position */
        const cmod_float_t fwd[3],	# input forward-pointing vector: FOV center */
        const cmod_float_t x[3],	# input dir of increasing X image coordinate */
        const cmod_float_t y[3],	# input dir of increasing Y image coordinate */
        cmod_float_t xfov,		    # input X field of view (rad) */
        cmod_float_t yfov,		    # input Y field of view (rad) */
        cmod_float_t xdim,		    # input X image dimension */
        cmod_float_t ydim,		    # input Y image dimension */
        cmod_float_t xc,		    # input X coordinate of image center */
        cmod_float_t yc,		    # input Y coordinate of image center */
        cmod_float_t c[3],		    # output model center vector C */
        cmod_float_t a[3],		    # output model axis   vector A */
        cmod_float_t h[3],		    # output model horiz. vector H */
        cmod_float_t v[3])		    # output model vert.  vector V */

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

    cdef void cmod_cahv_iplane(
        const cmod_float_t c[3],	# input model center vector C
        const cmod_float_t a[3],	# input model axis   vector A
        const cmod_float_t h[3],	# input model horiz. vector H
        const cmod_float_t v[3],	# input model vert.  vector V
        cmod_float_t ppnt[3],	    # output projection point
        cmod_float_t ndir[3],	    # output normal direction
        cmod_float_t hdir[3],	    # output horizontal direction
        cmod_float_t vdir[3],	    # output vertical direction
        cmod_float_t *hc,		    # output horizontal center
        cmod_float_t *vc)		    # output vertical center

    cdef void cmod_cahv_move(
        const cmod_float_t p_i[3],	# input initial pos of camera ref pt */
        const cmod_float_t q_i[4],	# input initial orientation of camera ref pt */
        const cmod_float_t c_i[3],	# input initial model center vector C */
        const cmod_float_t a_i[3],	# input initial model axis   vector A */
        const cmod_float_t h_i[3],	# input initial model horiz. vector H */
        const cmod_float_t v_i[3],	# input initial model vert.  vector V */
        const cmod_float_t p_f[3],	# input final pos of camera ref pt */
        const cmod_float_t q_f[4],	# input final orientation of camera ref pt */
        cmod_float_t c_f[3],	    # output final model center vector C */
        cmod_float_t a_f[3],	    # output final model axis   vector A */
        cmod_float_t h_f[3],	    # output final model horiz. vector H */
        cmod_float_t v_f[3])	    # output final model vert.  vector V */

    cdef void cmod_cahv_pose(
        const cmod_float_t c[3],	# input model center vector C
        const cmod_float_t a[3],	# input model axis   vector A
        const cmod_float_t h[3],	# input model horiz. vector H
        const cmod_float_t v[3],	# input model vert.  vector V
        cmod_float_t p[3],		    # output position vector
        cmod_float_t r[3][3])	    # output rotation matrix

    cdef void cmod_cahv_posture(
        const cmod_float_t a[3],	# input model axis   vector A
        const cmod_float_t h[3],	# input model horiz. vector H
        const cmod_float_t v[3],	# input model vert.  vector V
        cmod_float_t r[3][3])	    # output rotation matrix

    cdef void cmod_cahv_reflect(
        const cmod_float_t c_i[3],	# input initial model center vector C
        const cmod_float_t a_i[3],	# input initial model axis   vector A
        const cmod_float_t h_i[3],	# input initial model horiz. vector H
        const cmod_float_t v_i[3],	# input initial model vert.  vector V
        const cmod_float_t p[3],	# input point on the reflecting plane
        const cmod_float_t n[3],	# input normal to the reflecting plane
        cmod_float_t c_f[3],	    # output final model center vector C
        cmod_float_t a_f[3],	    # output final model axis   vector A
        cmod_float_t h_f[3],	    # output final model horiz. vector H
        cmod_float_t v_f[3],	    # output final model vert.  vector V
        cmod_bool_t *parallel,	    # output if camera view & plane are parallel
        cmod_bool_t *behind)	    # output if camera behind reflecting plane

    cdef void cmod_cahv_reflect_cov(
        cmod_float_t s_i[12][12],	# input initial covariance
        const cmod_float_t n[3],	# input normal to the reflecting plane
        cmod_float_t s_f[12][12])	# output final covariance

    cdef void cmod_cahv_rot_cov(
        cmod_float_t r_i[3][3],	    # input initial orientation of camera ref pt
        cmod_float_t s_i[12][12],	# input initial covariance
        cmod_float_t r_f[3][3],	    # input final orientation of camera ref pt
        cmod_float_t s_f[12][12])	# output final covariance

    cdef void cmod_cahv_rotate_cov(
        const cmod_float_t q_i[4],	# input initial orientation of camera ref pt
        cmod_float_t s_i[12][12],	# input initial covariance
        const cmod_float_t q_f[4],	# input final orientation of camera ref pt
        cmod_float_t s_f[12][12])	# output final covariance

    cdef void cmod_cahv_scale(
        cmod_float_t hscale,	    # input horizontal scale factor
        cmod_float_t vscale,	    # input vertical   scale factor
        const cmod_float_t h1[3],	# input  model horiz. vector H
        const cmod_float_t v1[3],	# input  model vert.  vector V
        cmod_float_t s1[12][12],	# input  covariance matrix, or NULL
        cmod_float_t h2[3],		    # output model horiz. vector H
        cmod_float_t v2[3],		    # output model vert.  vector V
        cmod_float_t s2[12][12])	# output covariance matrix, or NULL

    cdef void cmod_cahv_shift(
        cmod_float_t dx,		    # input horizontal shift
        cmod_float_t dy,		    # input vertical   shift
        const cmod_float_t a1[3],	# input  model axis   vector A
        const cmod_float_t h1[3],	# input  model horiz. vector H
        const cmod_float_t v1[3],	# input  model vert.  vector V
        cmod_float_t s1[12][12],	# input  covariance matrix, or NULL
        cmod_float_t h2[3],		    # output model horiz. vector H
        cmod_float_t v2[3],		    # output model vert.  vector V
        cmod_float_t s2[12][12])	# output covariance matrix, or NULL

    cdef void cmod_cahv_transform_cov(
        cmod_float_t s_i[12][12],	# input initial covariance
        cmod_float_t r[3][3],	    # input transform matrix of camera ref pt
        cmod_float_t s_f[12][12])	# output final covariance

    cdef void cmod_cahv_warp_models(
        const cmod_float_t c1[3],	# input model 1 center vector C
        const cmod_float_t a1[3],	# input model 1 axis   vector A
        const cmod_float_t h1[3],	# input model 1 horiz. vector H
        const cmod_float_t v1[3],	# input model 1 vert.  vector V
        const cmod_float_t c2[3],	# input model 2 center vector C
        const cmod_float_t a2[3],	# input model 2 axis   vector A
        const cmod_float_t h2[3],	# input model 2 horiz. vector H
        const cmod_float_t v2[3],	# input model 2 vert.  vector V
        cmod_float_t a[3],		    # output virtual model axis   vector A
        cmod_float_t h[3],		    # output virtual model horiz. vector H
        cmod_float_t v[3],		    # output virtual model vert.  vector V
        cmod_float_t *hs,		    # output horizontal scale factor
        cmod_float_t *hc,		    # output horizontal center
        cmod_float_t *vs,		    # output vertical scale factor
        cmod_float_t *vc,		    # output vertical center
        cmod_float_t *theta)	    # output angle between axes

    cdef void cmod_cahv_warp_to_cahv(
        const cmod_float_t c1[3],      # input model center vector C
        const cmod_float_t a1[3],      # input model axis   vector A
        const cmod_float_t h1[3],      # input model horiz. vector H
        const cmod_float_t v1[3],      # input model vert.  vector V
        const cmod_float_t pos1[2],    # input 2D position from CAHV
        const cmod_float_t c2[3],      # input final model center vector C
        const cmod_float_t a2[3],      # input final model axis   vector A
        const cmod_float_t h2[3],      # input final model horiz. vector H
        const cmod_float_t v2[3],      # input final model vert.  vector V
        cmod_float_t pos2[2])          # output 2D position for CAHV
