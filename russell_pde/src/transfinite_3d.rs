use super::{FnVec1Param2, FnVec2Param2, FnVec3Param2};
use crate::StrError;
use russell_lab::Vector;

/// Implements the transfinite mapping
///
/// Maps a reference cube [-1,+1] × [-1,+1] × [-1,+1] into a curve-bounded region in 3D
///
/// ```text
///
///                                   +----------------+
///                                 ,'|              ,'|
///      t or z                   ,'  |  ___       ,'  |     B0(s,t)
///         ↑                   ,'    |,'5,'  [0],'    |     B1(s,t)
///         |                 ,'      |~~~     ,'      |     B2(r,t)
///         |               +'===============+'  ,'|   |     B3(r,t)
///         |               |   ,'|   |      |   |3|   |     B4(r,s)
///         |     s or y    |   |2|   |      |   |,'   |     B5(r,s)
///         +-------->      |   |,'   +- - - | +- - - -+
///       ,'                |       ,'       |       ,'
///     ,'                  |     ,' [1]  ___|     ,'
/// r or x                  |   ,'      ,'4,'|   ,'
///                         | ,'        ~~~  | ,'
///                         +----------------+'
/// ```
///
/// Note: The reference coordinates {r,s,t} ϵ [-1,+1]×[-1,+1]×[-1,+1] are also symbolized by `u`
pub struct Transfinite3d {
    // boundary function
    // len = 6
    boundary_functions: Vec<FnVec1Param2>,

    // 1st derivatives of boundary functions
    // len = 6
    deriv1_boundary_functions: Vec<FnVec2Param2>,

    // 2nd derivatives of boundary functions
    // len = 6
    deriv2_boundary_functions: Option<Vec<FnVec3Param2>>,

    // corner points
    p0: Vector,
    p1: Vector,
    p2: Vector,
    p3: Vector,
    p4: Vector,
    p5: Vector,
    p6: Vector,
    p7: Vector,

    // temporary vectors
    tm1: Vector,
    tm2: Vector,

    // boundary function evaluations
    b0st: Vector,
    b1st: Vector,
    b2rt: Vector,
    b3rt: Vector,
    b4rs: Vector,
    b5rs: Vector,
    b0mt: Vector,
    b0pt: Vector,
    b1mt: Vector,
    b1pt: Vector,
    b0sm: Vector,
    b0sp: Vector,
    b1sm: Vector,
    b1sp: Vector,
    b2rm: Vector,
    b2rp: Vector,
    b3rm: Vector,
    b3rp: Vector,

    // derivative evaluations
    db0st_ds: Vector,
    db0st_dt: Vector,
    db1st_ds: Vector,
    db1st_dt: Vector,
    db2rt_dr: Vector,
    db2rt_dt: Vector,
    db3rt_dr: Vector,
    db3rt_dt: Vector,
    db4rs_dr: Vector,
    db4rs_ds: Vector,
    db5rs_dr: Vector,
    db5rs_ds: Vector,

    db0sm_ds: Vector,
    db0sp_ds: Vector,
    db0mt_dt: Vector,
    db0pt_dt: Vector,
    db1sm_ds: Vector,
    db1sp_ds: Vector,
    db1mt_dt: Vector,
    db1pt_dt: Vector,
    db2rm_dr: Vector,
    db2rp_dr: Vector,
    db3rm_dr: Vector,
    db3rp_dr: Vector,

    ddb0st_dss: Vector,
    ddb0st_dtt: Vector,
    ddb0st_dst: Vector,
    ddb1st_dss: Vector,
    ddb1st_dtt: Vector,
    ddb1st_dst: Vector,
    ddb2rt_drr: Vector,
    ddb2rt_dtt: Vector,
    ddb2rt_drt: Vector,
    ddb3rt_drr: Vector,
    ddb3rt_dtt: Vector,
    ddb3rt_drt: Vector,
    ddb4rs_drr: Vector,
    ddb4rs_dss: Vector,
    ddb4rs_drs: Vector,
    ddb5rs_drr: Vector,
    ddb5rs_dss: Vector,
    ddb5rs_drs: Vector,

    ddb0sm_dss: Vector,
    ddb0sp_dss: Vector,
    ddb0mt_dtt: Vector,
    ddb0pt_dtt: Vector,
    ddb1sm_dss: Vector,
    ddb1sp_dss: Vector,
    ddb1mt_dtt: Vector,
    ddb1pt_dtt: Vector,
    ddb2rm_drr: Vector,
    ddb2rp_drr: Vector,
    ddb3rm_drr: Vector,
    ddb3rp_drr: Vector,
}

impl Transfinite3d {
    /// Allocates a new instance
    pub fn new(
        boundary_functions: Vec<FnVec1Param2>,
        deriv1_boundary_functions: Vec<FnVec2Param2>,
        deriv2_boundary_functions: Option<Vec<FnVec3Param2>>,
    ) -> Result<Self, StrError> {
        // checks
        if boundary_functions.len() != 6 {
            return Err("boundary_functions must have length 6");
        }
        if deriv1_boundary_functions.len() != 6 {
            return Err("deriv1_boundary_functions must have length 6");
        }
        if let Some(ref bdd_val) = deriv2_boundary_functions {
            if bdd_val.len() != 6 {
                return Err("deriv2_boundary_functions must have length 6");
            }
        }

        let mut map = Transfinite3d {
            boundary_functions,
            deriv1_boundary_functions,
            deriv2_boundary_functions,
            p0: Vector::new(3),
            p1: Vector::new(3),
            p2: Vector::new(3),
            p3: Vector::new(3),
            p4: Vector::new(3),
            p5: Vector::new(3),
            p6: Vector::new(3),
            p7: Vector::new(3),
            tm1: Vector::new(3),
            tm2: Vector::new(3),
            b0st: Vector::new(3),
            b1st: Vector::new(3),
            b2rt: Vector::new(3),
            b3rt: Vector::new(3),
            b4rs: Vector::new(3),
            b5rs: Vector::new(3),
            b0mt: Vector::new(3),
            b0pt: Vector::new(3),
            b1mt: Vector::new(3),
            b1pt: Vector::new(3),
            b0sm: Vector::new(3),
            b0sp: Vector::new(3),
            b1sm: Vector::new(3),
            b1sp: Vector::new(3),
            b2rm: Vector::new(3),
            b2rp: Vector::new(3),
            b3rm: Vector::new(3),
            b3rp: Vector::new(3),
            db0st_ds: Vector::new(3),
            db0st_dt: Vector::new(3),
            db1st_ds: Vector::new(3),
            db1st_dt: Vector::new(3),
            db2rt_dr: Vector::new(3),
            db2rt_dt: Vector::new(3),
            db3rt_dr: Vector::new(3),
            db3rt_dt: Vector::new(3),
            db4rs_dr: Vector::new(3),
            db4rs_ds: Vector::new(3),
            db5rs_dr: Vector::new(3),
            db5rs_ds: Vector::new(3),
            db0sm_ds: Vector::new(3),
            db0sp_ds: Vector::new(3),
            db0mt_dt: Vector::new(3),
            db0pt_dt: Vector::new(3),
            db1sm_ds: Vector::new(3),
            db1sp_ds: Vector::new(3),
            db1mt_dt: Vector::new(3),
            db1pt_dt: Vector::new(3),
            db2rm_dr: Vector::new(3),
            db2rp_dr: Vector::new(3),
            db3rm_dr: Vector::new(3),
            db3rp_dr: Vector::new(3),
            ddb0st_dss: Vector::new(3),
            ddb0st_dtt: Vector::new(3),
            ddb0st_dst: Vector::new(3),
            ddb1st_dss: Vector::new(3),
            ddb1st_dtt: Vector::new(3),
            ddb1st_dst: Vector::new(3),
            ddb2rt_drr: Vector::new(3),
            ddb2rt_dtt: Vector::new(3),
            ddb2rt_drt: Vector::new(3),
            ddb3rt_drr: Vector::new(3),
            ddb3rt_dtt: Vector::new(3),
            ddb3rt_drt: Vector::new(3),
            ddb4rs_drr: Vector::new(3),
            ddb4rs_dss: Vector::new(3),
            ddb4rs_drs: Vector::new(3),
            ddb5rs_drr: Vector::new(3),
            ddb5rs_dss: Vector::new(3),
            ddb5rs_drs: Vector::new(3),
            ddb0sm_dss: Vector::new(3),
            ddb0sp_dss: Vector::new(3),
            ddb0mt_dtt: Vector::new(3),
            ddb0pt_dtt: Vector::new(3),
            ddb1sm_dss: Vector::new(3),
            ddb1sp_dss: Vector::new(3),
            ddb1mt_dtt: Vector::new(3),
            ddb1pt_dtt: Vector::new(3),
            ddb2rm_drr: Vector::new(3),
            ddb2rp_drr: Vector::new(3),
            ddb3rm_drr: Vector::new(3),
            ddb3rp_drr: Vector::new(3),
        };

        // compute corners
        (map.boundary_functions[4])(&mut map.p0, -1.0, -1.0);
        (map.boundary_functions[4])(&mut map.p1, 1.0, -1.0);
        (map.boundary_functions[4])(&mut map.p2, 1.0, 1.0);
        (map.boundary_functions[4])(&mut map.p3, -1.0, 1.0);
        (map.boundary_functions[5])(&mut map.p4, -1.0, -1.0);
        (map.boundary_functions[5])(&mut map.p5, 1.0, -1.0);
        (map.boundary_functions[5])(&mut map.p6, 1.0, 1.0);
        (map.boundary_functions[5])(&mut map.p7, -1.0, 1.0);
        Ok(map)
    }

    /// Computes "real" position x(r,s,t)
    pub fn point(&mut self, x: &mut Vector, r: f64, s: f64, t: f64) {
        // auxiliary
        let m = -1.0;
        let p = 1.0;

        // compute boundary functions @ {r,s,t}
        (self.boundary_functions[0])(&mut self.b0st, s, t);
        (self.boundary_functions[1])(&mut self.b1st, s, t);
        (self.boundary_functions[2])(&mut self.b2rt, r, t);
        (self.boundary_functions[3])(&mut self.b3rt, r, t);
        (self.boundary_functions[4])(&mut self.b4rs, r, s);
        (self.boundary_functions[5])(&mut self.b5rs, r, s);

        // compute boundary functions @ edges
        (self.boundary_functions[0])(&mut self.b0mt, m, t);
        (self.boundary_functions[0])(&mut self.b0pt, p, t);
        (self.boundary_functions[1])(&mut self.b1mt, m, t);
        (self.boundary_functions[1])(&mut self.b1pt, p, t);

        (self.boundary_functions[0])(&mut self.b0sm, s, m);
        (self.boundary_functions[0])(&mut self.b0sp, s, p);
        (self.boundary_functions[1])(&mut self.b1sm, s, m);
        (self.boundary_functions[1])(&mut self.b1sp, s, p);

        (self.boundary_functions[2])(&mut self.b2rm, r, m);
        (self.boundary_functions[2])(&mut self.b2rp, r, p);
        (self.boundary_functions[3])(&mut self.b3rm, r, m);
        (self.boundary_functions[3])(&mut self.b3rp, r, p);

        // compute position
        for i in 0..3 {
            x[i] = 0.0
                + (1.0 - r) * self.b0st[i] / 2.0
                + (1.0 + r) * self.b1st[i] / 2.0
                + (1.0 - s) * self.b2rt[i] / 2.0
                + (1.0 + s) * self.b3rt[i] / 2.0
                + (1.0 - t) * self.b4rs[i] / 2.0
                + (1.0 + t) * self.b5rs[i] / 2.0
                - (1.0 - r) * (1.0 - s) * self.b0mt[i] / 4.0
                - (1.0 - r) * (1.0 + s) * self.b0pt[i] / 4.0
                - (1.0 + r) * (1.0 - s) * self.b1mt[i] / 4.0
                - (1.0 + r) * (1.0 + s) * self.b1pt[i] / 4.0
                - (1.0 - r) * (1.0 - t) * self.b0sm[i] / 4.0
                - (1.0 - r) * (1.0 + t) * self.b0sp[i] / 4.0
                - (1.0 + r) * (1.0 - t) * self.b1sm[i] / 4.0
                - (1.0 + r) * (1.0 + t) * self.b1sp[i] / 4.0
                - (1.0 - s) * (1.0 - t) * self.b2rm[i] / 4.0
                - (1.0 - s) * (1.0 + t) * self.b2rp[i] / 4.0
                - (1.0 + s) * (1.0 - t) * self.b3rm[i] / 4.0
                - (1.0 + s) * (1.0 + t) * self.b3rp[i] / 4.0
                + (1.0 - r) * (1.0 - s) * (1.0 - t) * self.p0[i] / 8.0
                + (1.0 + r) * (1.0 - s) * (1.0 - t) * self.p1[i] / 8.0
                + (1.0 + r) * (1.0 + s) * (1.0 - t) * self.p2[i] / 8.0
                + (1.0 - r) * (1.0 + s) * (1.0 - t) * self.p3[i] / 8.0
                + (1.0 - r) * (1.0 - s) * (1.0 + t) * self.p4[i] / 8.0
                + (1.0 + r) * (1.0 - s) * (1.0 + t) * self.p5[i] / 8.0
                + (1.0 + r) * (1.0 + s) * (1.0 + t) * self.p6[i] / 8.0
                + (1.0 - r) * (1.0 + s) * (1.0 + t) * self.p7[i] / 8.0;
        }
    }

    /// Computes position and the first and second order derivatives
    pub fn point_and_derivs(
        &mut self,
        x: &mut Vector,
        dx_dr: &mut Vector,
        dx_ds: &mut Vector,
        dx_dt: &mut Vector,
        d2x_dr2: Option<&mut Vector>,
        d2x_ds2: Option<&mut Vector>,
        d2x_dt2: Option<&mut Vector>,
        d2x_drs: Option<&mut Vector>,
        d2x_drt: Option<&mut Vector>,
        d2x_dst: Option<&mut Vector>,
        r: f64,
        s: f64,
        t: f64,
    ) {
        // auxiliary
        let second_derivs = d2x_dr2.is_some();
        let m = -1.0;
        let p = 1.0;

        // compute boundary functions @ {r,s,t}
        (self.boundary_functions[0])(&mut self.b0st, s, t);
        (self.boundary_functions[1])(&mut self.b1st, s, t);
        (self.boundary_functions[2])(&mut self.b2rt, r, t);
        (self.boundary_functions[3])(&mut self.b3rt, r, t);
        (self.boundary_functions[4])(&mut self.b4rs, r, s);
        (self.boundary_functions[5])(&mut self.b5rs, r, s);

        // compute boundary functions @ edges
        (self.boundary_functions[0])(&mut self.b0mt, m, t);
        (self.boundary_functions[0])(&mut self.b0pt, p, t);
        (self.boundary_functions[1])(&mut self.b1mt, m, t);
        (self.boundary_functions[1])(&mut self.b1pt, p, t);

        (self.boundary_functions[0])(&mut self.b0sm, s, m);
        (self.boundary_functions[0])(&mut self.b0sp, s, p);
        (self.boundary_functions[1])(&mut self.b1sm, s, m);
        (self.boundary_functions[1])(&mut self.b1sp, s, p);

        (self.boundary_functions[2])(&mut self.b2rm, r, m);
        (self.boundary_functions[2])(&mut self.b2rp, r, p);
        (self.boundary_functions[3])(&mut self.b3rm, r, m);
        (self.boundary_functions[3])(&mut self.b3rp, r, p);

        // compute derivatives @ {r,s,t}
        (self.deriv1_boundary_functions[0])(&mut self.db0st_ds, &mut self.db0st_dt, s, t);
        (self.deriv1_boundary_functions[1])(&mut self.db1st_ds, &mut self.db1st_dt, s, t);
        (self.deriv1_boundary_functions[2])(&mut self.db2rt_dr, &mut self.db2rt_dt, r, t);
        (self.deriv1_boundary_functions[3])(&mut self.db3rt_dr, &mut self.db3rt_dt, r, t);
        (self.deriv1_boundary_functions[4])(&mut self.db4rs_dr, &mut self.db4rs_ds, r, s);
        (self.deriv1_boundary_functions[5])(&mut self.db5rs_dr, &mut self.db5rs_ds, r, s);

        // compute derivatives @ edges
        (self.deriv1_boundary_functions[0])(&mut self.db0sm_ds, &mut self.tm1, s, m);
        (self.deriv1_boundary_functions[0])(&mut self.db0sp_ds, &mut self.tm1, s, p);
        (self.deriv1_boundary_functions[0])(&mut self.tm1, &mut self.db0mt_dt, m, t);
        (self.deriv1_boundary_functions[0])(&mut self.tm1, &mut self.db0pt_dt, p, t);

        (self.deriv1_boundary_functions[1])(&mut self.db1sm_ds, &mut self.tm1, s, m);
        (self.deriv1_boundary_functions[1])(&mut self.db1sp_ds, &mut self.tm1, s, p);
        (self.deriv1_boundary_functions[1])(&mut self.tm1, &mut self.db1mt_dt, m, t);
        (self.deriv1_boundary_functions[1])(&mut self.tm1, &mut self.db1pt_dt, p, t);

        (self.deriv1_boundary_functions[2])(&mut self.db2rm_dr, &mut self.tm1, r, m);
        (self.deriv1_boundary_functions[2])(&mut self.db2rp_dr, &mut self.tm1, r, p);
        (self.deriv1_boundary_functions[3])(&mut self.db3rm_dr, &mut self.tm1, r, m);
        (self.deriv1_boundary_functions[3])(&mut self.db3rp_dr, &mut self.tm1, r, p);

        // position and first order derivatives
        for i in 0..3 {
            // bilinear transfinite mapping in 3D
            x[i] = 0.0
                + (1.0 - r) * self.b0st[i] / 2.0
                + (1.0 + r) * self.b1st[i] / 2.0
                + (1.0 - s) * self.b2rt[i] / 2.0
                + (1.0 + s) * self.b3rt[i] / 2.0
                + (1.0 - t) * self.b4rs[i] / 2.0
                + (1.0 + t) * self.b5rs[i] / 2.0
                - (1.0 - r) * (1.0 - s) * self.b0mt[i] / 4.0
                - (1.0 - r) * (1.0 + s) * self.b0pt[i] / 4.0
                - (1.0 + r) * (1.0 - s) * self.b1mt[i] / 4.0
                - (1.0 + r) * (1.0 + s) * self.b1pt[i] / 4.0
                - (1.0 - r) * (1.0 - t) * self.b0sm[i] / 4.0
                - (1.0 - r) * (1.0 + t) * self.b0sp[i] / 4.0
                - (1.0 + r) * (1.0 - t) * self.b1sm[i] / 4.0
                - (1.0 + r) * (1.0 + t) * self.b1sp[i] / 4.0
                - (1.0 - s) * (1.0 - t) * self.b2rm[i] / 4.0
                - (1.0 - s) * (1.0 + t) * self.b2rp[i] / 4.0
                - (1.0 + s) * (1.0 - t) * self.b3rm[i] / 4.0
                - (1.0 + s) * (1.0 + t) * self.b3rp[i] / 4.0
                + (1.0 - r) * (1.0 - s) * (1.0 - t) * self.p0[i] / 8.0
                + (1.0 + r) * (1.0 - s) * (1.0 - t) * self.p1[i] / 8.0
                + (1.0 + r) * (1.0 + s) * (1.0 - t) * self.p2[i] / 8.0
                + (1.0 - r) * (1.0 + s) * (1.0 - t) * self.p3[i] / 8.0
                + (1.0 - r) * (1.0 - s) * (1.0 + t) * self.p4[i] / 8.0
                + (1.0 + r) * (1.0 - s) * (1.0 + t) * self.p5[i] / 8.0
                + (1.0 + r) * (1.0 + s) * (1.0 + t) * self.p6[i] / 8.0
                + (1.0 - r) * (1.0 + s) * (1.0 + t) * self.p7[i] / 8.0;

            // derivative of x with respect to r
            dx_dr[i] = 0.0 - self.b0st[i] / 2.0
                + self.b1st[i] / 2.0
                + (1.0 - s) * self.db2rt_dr[i] / 2.0
                + (1.0 + s) * self.db3rt_dr[i] / 2.0
                + (1.0 - t) * self.db4rs_dr[i] / 2.0
                + (1.0 + t) * self.db5rs_dr[i] / 2.0
                + (1.0 - s) * self.b0mt[i] / 4.0
                + (1.0 + s) * self.b0pt[i] / 4.0
                - (1.0 - s) * self.b1mt[i] / 4.0
                - (1.0 + s) * self.b1pt[i] / 4.0
                + (1.0 - t) * self.b0sm[i] / 4.0
                + (1.0 + t) * self.b0sp[i] / 4.0
                - (1.0 - t) * self.b1sm[i] / 4.0
                - (1.0 + t) * self.b1sp[i] / 4.0
                - (1.0 - s) * (1.0 - t) * self.db2rm_dr[i] / 4.0
                - (1.0 - s) * (1.0 + t) * self.db2rp_dr[i] / 4.0
                - (1.0 + s) * (1.0 - t) * self.db3rm_dr[i] / 4.0
                - (1.0 + s) * (1.0 + t) * self.db3rp_dr[i] / 4.0
                - (1.0 - s) * (1.0 - t) * self.p0[i] / 8.0
                + (1.0 - s) * (1.0 - t) * self.p1[i] / 8.0
                + (1.0 + s) * (1.0 - t) * self.p2[i] / 8.0
                - (1.0 + s) * (1.0 - t) * self.p3[i] / 8.0
                - (1.0 - s) * (1.0 + t) * self.p4[i] / 8.0
                + (1.0 - s) * (1.0 + t) * self.p5[i] / 8.0
                + (1.0 + s) * (1.0 + t) * self.p6[i] / 8.0
                - (1.0 + s) * (1.0 + t) * self.p7[i] / 8.0;

            // derivative of x with respect to s
            dx_ds[i] = 0.0 + (1.0 - r) * self.db0st_ds[i] / 2.0 + (1.0 + r) * self.db1st_ds[i] / 2.0
                - self.b2rt[i] / 2.0
                + self.b3rt[i] / 2.0
                + (1.0 - t) * self.db4rs_ds[i] / 2.0
                + (1.0 + t) * self.db5rs_ds[i] / 2.0
                + (1.0 - r) * self.b0mt[i] / 4.0
                - (1.0 - r) * self.b0pt[i] / 4.0
                + (1.0 + r) * self.b1mt[i] / 4.0
                - (1.0 + r) * self.b1pt[i] / 4.0
                - (1.0 - r) * (1.0 - t) * self.db0sm_ds[i] / 4.0
                - (1.0 - r) * (1.0 + t) * self.db0sp_ds[i] / 4.0
                - (1.0 + r) * (1.0 - t) * self.db1sm_ds[i] / 4.0
                - (1.0 + r) * (1.0 + t) * self.db1sp_ds[i] / 4.0
                + (1.0 - t) * self.b2rm[i] / 4.0
                + (1.0 + t) * self.b2rp[i] / 4.0
                - (1.0 - t) * self.b3rm[i] / 4.0
                - (1.0 + t) * self.b3rp[i] / 4.0
                - (1.0 - r) * (1.0 - t) * self.p0[i] / 8.0
                - (1.0 + r) * (1.0 - t) * self.p1[i] / 8.0
                + (1.0 + r) * (1.0 - t) * self.p2[i] / 8.0
                + (1.0 - r) * (1.0 - t) * self.p3[i] / 8.0
                - (1.0 - r) * (1.0 + t) * self.p4[i] / 8.0
                - (1.0 + r) * (1.0 + t) * self.p5[i] / 8.0
                + (1.0 + r) * (1.0 + t) * self.p6[i] / 8.0
                + (1.0 - r) * (1.0 + t) * self.p7[i] / 8.0;

            // derivative of x with respect to t
            dx_dt[i] = 0.0
                + (1.0 - r) * self.db0st_dt[i] / 2.0
                + (1.0 + r) * self.db1st_dt[i] / 2.0
                + (1.0 - s) * self.db2rt_dt[i] / 2.0
                + (1.0 + s) * self.db3rt_dt[i] / 2.0
                - self.b4rs[i] / 2.0
                + self.b5rs[i] / 2.0
                - (1.0 - r) * (1.0 - s) * self.db0mt_dt[i] / 4.0
                - (1.0 - r) * (1.0 + s) * self.db0pt_dt[i] / 4.0
                - (1.0 + r) * (1.0 - s) * self.db1mt_dt[i] / 4.0
                - (1.0 + r) * (1.0 + s) * self.db1pt_dt[i] / 4.0
                + (1.0 - r) * self.b0sm[i] / 4.0
                - (1.0 - r) * self.b0sp[i] / 4.0
                + (1.0 + r) * self.b1sm[i] / 4.0
                - (1.0 + r) * self.b1sp[i] / 4.0
                + (1.0 - s) * self.b2rm[i] / 4.0
                - (1.0 - s) * self.b2rp[i] / 4.0
                + (1.0 + s) * self.b3rm[i] / 4.0
                - (1.0 + s) * self.b3rp[i] / 4.0
                - (1.0 - r) * (1.0 - s) * self.p0[i] / 8.0
                - (1.0 + r) * (1.0 - s) * self.p1[i] / 8.0
                - (1.0 + r) * (1.0 + s) * self.p2[i] / 8.0
                - (1.0 - r) * (1.0 + s) * self.p3[i] / 8.0
                + (1.0 - r) * (1.0 - s) * self.p4[i] / 8.0
                + (1.0 + r) * (1.0 - s) * self.p5[i] / 8.0
                + (1.0 + r) * (1.0 + s) * self.p6[i] / 8.0
                + (1.0 - r) * (1.0 + s) * self.p7[i] / 8.0;
        }

        // skip second order derivatives
        if !second_derivs {
            return;
        }

        // unwrap optional arguments
        let d2x_dr2 = d2x_dr2.unwrap();
        let d2x_ds2 = d2x_ds2.unwrap();
        let d2x_dt2 = d2x_dt2.unwrap();
        let d2x_drs = d2x_drs.unwrap();
        let d2x_drt = d2x_drt.unwrap();
        let d2x_dst = d2x_dst.unwrap();

        // only 2nd cross-derivatives may be non-zero
        if self.deriv2_boundary_functions.is_none() {
            for i in 0..3 {
                d2x_dr2[i] = 0.0;
                d2x_ds2[i] = 0.0;
                d2x_dt2[i] = 0.0;

                // derivative of dx/dr with respect to s
                d2x_drs[i] = 0.0 - self.db0st_ds[i] / 2.0 + self.db1st_ds[i] / 2.0 - self.db2rt_dr[i] / 2.0
                    + self.db3rt_dr[i] / 2.0
                    - self.b0mt[i] / 4.0
                    + self.b0pt[i] / 4.0
                    + self.b1mt[i] / 4.0
                    - self.b1pt[i] / 4.0
                    + (1.0 - t) * self.db0sm_ds[i] / 4.0
                    + (1.0 + t) * self.db0sp_ds[i] / 4.0
                    - (1.0 - t) * self.db1sm_ds[i] / 4.0
                    - (1.0 + t) * self.db1sp_ds[i] / 4.0
                    + (1.0 - t) * self.db2rm_dr[i] / 4.0
                    + (1.0 + t) * self.db2rp_dr[i] / 4.0
                    - (1.0 - t) * self.db3rm_dr[i] / 4.0
                    - (1.0 + t) * self.db3rp_dr[i] / 4.0
                    + (1.0 - t) * self.p0[i] / 8.0
                    - (1.0 - t) * self.p1[i] / 8.0
                    + (1.0 - t) * self.p2[i] / 8.0
                    - (1.0 - t) * self.p3[i] / 8.0
                    + (1.0 + t) * self.p4[i] / 8.0
                    - (1.0 + t) * self.p5[i] / 8.0
                    + (1.0 + t) * self.p6[i] / 8.0
                    - (1.0 + t) * self.p7[i] / 8.0;

                // derivative of dx/dr with respect to t
                d2x_drt[i] = 0.0 - self.db0st_dt[i] / 2.0 + self.db1st_dt[i] / 2.0 - self.db4rs_dr[i] / 2.0
                    + self.db5rs_dr[i] / 2.0
                    + (1.0 - s) * self.db0mt_dt[i] / 4.0
                    + (1.0 + s) * self.db0pt_dt[i] / 4.0
                    - (1.0 - s) * self.db1mt_dt[i] / 4.0
                    - (1.0 + s) * self.db1pt_dt[i] / 4.0
                    - self.b0sm[i] / 4.0
                    + self.b0sp[i] / 4.0
                    + self.b1sm[i] / 4.0
                    - self.b1sp[i] / 4.0
                    + (1.0 - s) * self.db2rm_dr[i] / 4.0
                    - (1.0 - s) * self.db2rp_dr[i] / 4.0
                    + (1.0 + s) * self.db3rm_dr[i] / 4.0
                    - (1.0 + s) * self.db3rp_dr[i] / 4.0
                    + (1.0 - s) * self.p0[i] / 8.0
                    - (1.0 - s) * self.p1[i] / 8.0
                    - (1.0 + s) * self.p2[i] / 8.0
                    + (1.0 + s) * self.p3[i] / 8.0
                    - (1.0 - s) * self.p4[i] / 8.0
                    + (1.0 - s) * self.p5[i] / 8.0
                    + (1.0 + s) * self.p6[i] / 8.0
                    - (1.0 + s) * self.p7[i] / 8.0;

                // derivative of dx/ds with respect to t
                d2x_dst[i] = 0.0 - self.db2rt_dt[i] / 2.0 + self.db3rt_dt[i] / 2.0 - self.db4rs_ds[i] / 2.0
                    + self.db5rs_ds[i] / 2.0
                    + (1.0 - r) * self.db0mt_dt[i] / 4.0
                    - (1.0 - r) * self.db0pt_dt[i] / 4.0
                    + (1.0 + r) * self.db1mt_dt[i] / 4.0
                    - (1.0 + r) * self.db1pt_dt[i] / 4.0
                    + (1.0 - r) * self.db0sm_ds[i] / 4.0
                    - (1.0 - r) * self.db0sp_ds[i] / 4.0
                    + (1.0 + r) * self.db1sm_ds[i] / 4.0
                    - (1.0 + r) * self.db1sp_ds[i] / 4.0
                    - self.b2rm[i] / 4.0
                    + self.b2rp[i] / 4.0
                    + self.b3rm[i] / 4.0
                    - self.b3rp[i] / 4.0
                    + (1.0 - r) * self.p0[i] / 8.0
                    + (1.0 + r) * self.p1[i] / 8.0
                    - (1.0 + r) * self.p2[i] / 8.0
                    - (1.0 - r) * self.p3[i] / 8.0
                    - (1.0 - r) * self.p4[i] / 8.0
                    - (1.0 + r) * self.p5[i] / 8.0
                    + (1.0 + r) * self.p6[i] / 8.0
                    + (1.0 - r) * self.p7[i] / 8.0;
            }
            return;
        }

        // compute second derivatives @ {r,s,t}
        let deriv2_boundary_functions = self.deriv2_boundary_functions.as_ref().unwrap();
        (deriv2_boundary_functions[0])(&mut self.ddb0st_dss, &mut self.ddb0st_dtt, &mut self.ddb0st_dst, s, t);
        (deriv2_boundary_functions[1])(&mut self.ddb1st_dss, &mut self.ddb1st_dtt, &mut self.ddb1st_dst, s, t);
        (deriv2_boundary_functions[2])(&mut self.ddb2rt_drr, &mut self.ddb2rt_dtt, &mut self.ddb2rt_drt, r, t);
        (deriv2_boundary_functions[3])(&mut self.ddb3rt_drr, &mut self.ddb3rt_dtt, &mut self.ddb3rt_drt, r, t);
        (deriv2_boundary_functions[4])(&mut self.ddb4rs_drr, &mut self.ddb4rs_dss, &mut self.ddb4rs_drs, r, s);
        (deriv2_boundary_functions[5])(&mut self.ddb5rs_drr, &mut self.ddb5rs_dss, &mut self.ddb5rs_drs, r, s);

        // compute second derivatives @ edges
        (deriv2_boundary_functions[0])(&mut self.ddb0sm_dss, &mut self.tm1, &mut self.tm2, s, m);
        (deriv2_boundary_functions[0])(&mut self.ddb0sp_dss, &mut self.tm1, &mut self.tm2, s, p);
        (deriv2_boundary_functions[0])(&mut self.tm1, &mut self.ddb0mt_dtt, &mut self.tm2, m, t);
        (deriv2_boundary_functions[0])(&mut self.tm1, &mut self.ddb0pt_dtt, &mut self.tm2, p, t);

        (deriv2_boundary_functions[1])(&mut self.ddb1sm_dss, &mut self.tm1, &mut self.tm2, s, m);
        (deriv2_boundary_functions[1])(&mut self.ddb1sp_dss, &mut self.tm1, &mut self.tm2, s, p);
        (deriv2_boundary_functions[1])(&mut self.tm1, &mut self.ddb1mt_dtt, &mut self.tm2, m, t);
        (deriv2_boundary_functions[1])(&mut self.tm1, &mut self.ddb1pt_dtt, &mut self.tm2, p, t);

        (deriv2_boundary_functions[2])(&mut self.ddb2rm_drr, &mut self.tm1, &mut self.tm2, r, m);
        (deriv2_boundary_functions[2])(&mut self.ddb2rp_drr, &mut self.tm1, &mut self.tm2, r, p);
        (deriv2_boundary_functions[3])(&mut self.ddb3rm_drr, &mut self.tm1, &mut self.tm2, r, m);
        (deriv2_boundary_functions[3])(&mut self.ddb3rp_drr, &mut self.tm1, &mut self.tm2, r, p);

        // second order derivatives
        for i in 0..3 {
            // derivative of dx/dr with respect to r
            d2x_dr2[i] = 0.0
                + (1.0 - s) * self.ddb2rt_drr[i] / 2.0
                + (1.0 + s) * self.ddb3rt_drr[i] / 2.0
                + (1.0 - t) * self.ddb4rs_drr[i] / 2.0
                + (1.0 + t) * self.ddb5rs_drr[i] / 2.0
                - (1.0 - s) * (1.0 - t) * self.ddb2rm_drr[i] / 4.0
                - (1.0 - s) * (1.0 + t) * self.ddb2rp_drr[i] / 4.0
                - (1.0 + s) * (1.0 - t) * self.ddb3rm_drr[i] / 4.0
                - (1.0 + s) * (1.0 + t) * self.ddb3rp_drr[i] / 4.0;

            // derivative of dx/ds with respect to s
            d2x_ds2[i] = 0.0
                + (1.0 - r) * self.ddb0st_dss[i] / 2.0
                + (1.0 + r) * self.ddb1st_dss[i] / 2.0
                + (1.0 - t) * self.ddb4rs_dss[i] / 2.0
                + (1.0 + t) * self.ddb5rs_dss[i] / 2.0
                - (1.0 - r) * (1.0 - t) * self.ddb0sm_dss[i] / 4.0
                - (1.0 - r) * (1.0 + t) * self.ddb0sp_dss[i] / 4.0
                - (1.0 + r) * (1.0 - t) * self.ddb1sm_dss[i] / 4.0
                - (1.0 + r) * (1.0 + t) * self.ddb1sp_dss[i] / 4.0;

            // derivative of dx/dt with respect to t
            d2x_dt2[i] = 0.0
                + (1.0 - r) * self.ddb0st_dtt[i] / 2.0
                + (1.0 + r) * self.ddb1st_dtt[i] / 2.0
                + (1.0 - s) * self.ddb2rt_dtt[i] / 2.0
                + (1.0 + s) * self.ddb3rt_dtt[i] / 2.0
                - (1.0 - r) * (1.0 - s) * self.ddb0mt_dtt[i] / 4.0
                - (1.0 - r) * (1.0 + s) * self.ddb0pt_dtt[i] / 4.0
                - (1.0 + r) * (1.0 - s) * self.ddb1mt_dtt[i] / 4.0
                - (1.0 + r) * (1.0 + s) * self.ddb1pt_dtt[i] / 4.0;

            // derivative of dx/dr with respect to s
            d2x_drs[i] = 0.0 - self.db0st_ds[i] / 2.0 + self.db1st_ds[i] / 2.0 - self.db2rt_dr[i] / 2.0
                + self.db3rt_dr[i] / 2.0
                + (1.0 - t) * self.ddb4rs_drs[i] / 2.0
                + (1.0 + t) * self.ddb5rs_drs[i] / 2.0
                - self.b0mt[i] / 4.0
                + self.b0pt[i] / 4.0
                + self.b1mt[i] / 4.0
                - self.b1pt[i] / 4.0
                + (1.0 - t) * self.db0sm_ds[i] / 4.0
                + (1.0 + t) * self.db0sp_ds[i] / 4.0
                - (1.0 - t) * self.db1sm_ds[i] / 4.0
                - (1.0 + t) * self.db1sp_ds[i] / 4.0
                + (1.0 - t) * self.db2rm_dr[i] / 4.0
                + (1.0 + t) * self.db2rp_dr[i] / 4.0
                - (1.0 - t) * self.db3rm_dr[i] / 4.0
                - (1.0 + t) * self.db3rp_dr[i] / 4.0
                + (1.0 - t) * self.p0[i] / 8.0
                - (1.0 - t) * self.p1[i] / 8.0
                + (1.0 - t) * self.p2[i] / 8.0
                - (1.0 - t) * self.p3[i] / 8.0
                + (1.0 + t) * self.p4[i] / 8.0
                - (1.0 + t) * self.p5[i] / 8.0
                + (1.0 + t) * self.p6[i] / 8.0
                - (1.0 + t) * self.p7[i] / 8.0;

            // derivative of dx/dr with respect to t
            d2x_drt[i] = 0.0 - self.db0st_dt[i] / 2.0
                + self.db1st_dt[i] / 2.0
                + (1.0 - s) * self.ddb2rt_drt[i] / 2.0
                + (1.0 + s) * self.ddb3rt_drt[i] / 2.0
                - self.db4rs_dr[i] / 2.0
                + self.db5rs_dr[i] / 2.0
                + (1.0 - s) * self.db0mt_dt[i] / 4.0
                + (1.0 + s) * self.db0pt_dt[i] / 4.0
                - (1.0 - s) * self.db1mt_dt[i] / 4.0
                - (1.0 + s) * self.db1pt_dt[i] / 4.0
                - self.b0sm[i] / 4.0
                + self.b0sp[i] / 4.0
                + self.b1sm[i] / 4.0
                - self.b1sp[i] / 4.0
                + (1.0 - s) * self.db2rm_dr[i] / 4.0
                - (1.0 - s) * self.db2rp_dr[i] / 4.0
                + (1.0 + s) * self.db3rm_dr[i] / 4.0
                - (1.0 + s) * self.db3rp_dr[i] / 4.0
                + (1.0 - s) * self.p0[i] / 8.0
                - (1.0 - s) * self.p1[i] / 8.0
                - (1.0 + s) * self.p2[i] / 8.0
                + (1.0 + s) * self.p3[i] / 8.0
                - (1.0 - s) * self.p4[i] / 8.0
                + (1.0 - s) * self.p5[i] / 8.0
                + (1.0 + s) * self.p6[i] / 8.0
                - (1.0 + s) * self.p7[i] / 8.0;

            // derivative of dx/ds with respect to t
            d2x_dst[i] = 0.0 + (1.0 - r) * self.ddb0st_dst[i] / 2.0 + (1.0 + r) * self.ddb1st_dst[i] / 2.0
                - self.db2rt_dt[i] / 2.0
                + self.db3rt_dt[i] / 2.0
                - self.db4rs_ds[i] / 2.0
                + self.db5rs_ds[i] / 2.0
                + (1.0 - r) * self.db0mt_dt[i] / 4.0
                - (1.0 - r) * self.db0pt_dt[i] / 4.0
                + (1.0 + r) * self.db1mt_dt[i] / 4.0
                - (1.0 + r) * self.db1pt_dt[i] / 4.0
                + (1.0 - r) * self.db0sm_ds[i] / 4.0
                - (1.0 - r) * self.db0sp_ds[i] / 4.0
                + (1.0 + r) * self.db1sm_ds[i] / 4.0
                - (1.0 + r) * self.db1sp_ds[i] / 4.0
                - self.b2rm[i] / 4.0
                + self.b2rp[i] / 4.0
                + self.b3rm[i] / 4.0
                - self.b3rp[i] / 4.0
                + (1.0 - r) * self.p0[i] / 8.0
                + (1.0 + r) * self.p1[i] / 8.0
                - (1.0 + r) * self.p2[i] / 8.0
                - (1.0 - r) * self.p3[i] / 8.0
                - (1.0 - r) * self.p4[i] / 8.0
                - (1.0 + r) * self.p5[i] / 8.0
                + (1.0 + r) * self.p6[i] / 8.0
                + (1.0 - r) * self.p7[i] / 8.0;
        }
    }

    /// Returns the corner points
    pub fn get_corners(&self) -> (&Vector, &Vector, &Vector, &Vector, &Vector, &Vector, &Vector, &Vector) {
        (
            &self.p0, &self.p1, &self.p2, &self.p3, &self.p4, &self.p5, &self.p6, &self.p7,
        )
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Transfinite3d;
    use crate::TransfiniteSamples;
    use russell_lab::{vec_approx_eq, vec_deriv1_approx_eq, vec_deriv2_approx_eq, Vector};

    fn check_derivs(map: &mut Transfinite3d, tol_d1: f64, tol_d2: f64) {
        let mut x_ana = Vector::new(3);
        let mut dx_dr_ana = Vector::new(3);
        let mut dx_ds_ana = Vector::new(3);
        let mut dx_dt_ana = Vector::new(3);
        let mut d2x_dr2_ana = Vector::new(3);
        let mut d2x_ds2_ana = Vector::new(3);
        let mut d2x_dt2_ana = Vector::new(3);
        let mut d2x_drs_ana = Vector::new(3);
        let mut d2x_drt_ana = Vector::new(3);
        let mut d2x_dst_ana = Vector::new(3);
        let args = &mut 0_u8;
        for s_at in [-0.5, 0.0, 0.5] {
            for t_at in [-0.5, 0.0, 0.5] {
                for r_at in [-0.5, 0.0, 0.5] {
                    map.point_and_derivs(
                        &mut x_ana,
                        &mut dx_dr_ana,
                        &mut dx_ds_ana,
                        &mut dx_dt_ana,
                        Some(&mut d2x_dr2_ana),
                        Some(&mut d2x_ds2_ana),
                        Some(&mut d2x_dt2_ana),
                        Some(&mut d2x_drs_ana),
                        Some(&mut d2x_drt_ana),
                        Some(&mut d2x_dst_ana),
                        r_at,
                        s_at,
                        t_at,
                    );
                    // dx/dr
                    vec_deriv1_approx_eq(&dx_dr_ana, r_at, args, tol_d1, |x, r, _| {
                        map.point(x, r, s_at, t_at);
                        Ok(())
                    });
                    // dx/ds
                    vec_deriv1_approx_eq(&dx_ds_ana, s_at, args, tol_d1, |x, s, _| {
                        map.point(x, r_at, s, t_at);
                        Ok(())
                    });
                    // dx/dt
                    vec_deriv1_approx_eq(&dx_dt_ana, t_at, args, tol_d1, |x, t, _| {
                        map.point(x, r_at, s_at, t);
                        Ok(())
                    });
                    // d²x/dr²
                    vec_deriv2_approx_eq(&d2x_dr2_ana, r_at, args, tol_d2, |x, r, _| {
                        map.point(x, r, s_at, t_at);
                        Ok(())
                    });
                    // d²x/ds²
                    vec_deriv2_approx_eq(&d2x_ds2_ana, s_at, args, tol_d2, |x, s, _| {
                        map.point(x, r_at, s, t_at);
                        Ok(())
                    });
                    // d²x/dt²
                    vec_deriv2_approx_eq(&d2x_dt2_ana, t_at, args, tol_d2, |x, t, _| {
                        map.point(x, r_at, s_at, t);
                        Ok(())
                    });
                }
            }
        }
    }

    #[test]
    fn transfinite_3d_works_brick() {
        // allocate transfinite map
        let (lx, ly, lz) = (2.0, 3.0, 4.0);
        let mut map = TransfiniteSamples::brick_3d(lx, ly, lz);

        // check corners
        let (p0, p1, p2, p3, p4, p5, p6, p7) = map.get_corners();
        vec_approx_eq(&p0, &[0.0, 0.0, 0.0], 1e-15);
        vec_approx_eq(&p1, &[lx, 0.0, 0.0], 1e-15);
        vec_approx_eq(&p2, &[lx, ly, 0.0], 1e-15);
        vec_approx_eq(&p3, &[0.0, ly, 0.0], 1e-15);
        vec_approx_eq(&p4, &[0.0, 0.0, lz], 1e-15);
        vec_approx_eq(&p5, &[lx, 0.0, lz], 1e-15);
        vec_approx_eq(&p6, &[lx, ly, lz], 1e-15);
        vec_approx_eq(&p7, &[0.0, ly, lz], 1e-15);

        // check derivatives
        check_derivs(&mut map, 1e-11, 1e-8);
    }

    #[test]
    fn transfinite_3d_works_quarter_ring() {
        // allocate transfinite map
        let r_in = 1.0;
        let r_out = 6.0;
        let thickness = 2.0;
        let mut map = TransfiniteSamples::quarter_ring_3d(r_in, r_out, thickness);

        // check corners
        let (p0, p1, p2, p3, p4, p5, p6, p7) = map.get_corners();
        vec_approx_eq(&p0, &[0.0, r_in, 0.0], 1e-15);
        vec_approx_eq(&p1, &[thickness, r_in, 0.0], 1e-15);
        vec_approx_eq(&p2, &[thickness, r_out, 0.0], 1e-15);
        vec_approx_eq(&p3, &[0.0, r_out, 0.0], 1e-15);
        vec_approx_eq(&p4, &[0.0, 0.0, r_in], 1e-15);
        vec_approx_eq(&p5, &[thickness, 0.0, r_in], 1e-15);
        vec_approx_eq(&p6, &[thickness, 0.0, r_out], 1e-15);
        vec_approx_eq(&p7, &[0.0, 0.0, r_out], 1e-15);

        // check derivatives
        check_derivs(&mut map, 1e-9, 1e-7);
    }

    #[test]
    fn transfinite_3d_captures_errors() {
        assert_eq!(
            Transfinite3d::new(vec![], vec![], None).err(),
            Some("boundary_functions must have length 6")
        );
        assert_eq!(
            Transfinite3d::new(
                vec![
                    Box::new(|_, _, _| {}),
                    Box::new(|_, _, _| {}),
                    Box::new(|_, _, _| {}),
                    Box::new(|_, _, _| {}),
                    Box::new(|_, _, _| {}),
                    Box::new(|_, _, _| {})
                ],
                vec![Box::new(|_, _, _, _| {})],
                None
            )
            .err(),
            Some("deriv1_boundary_functions must have length 6")
        );
        assert_eq!(
            Transfinite3d::new(
                vec![
                    Box::new(|_, _, _| {}),
                    Box::new(|_, _, _| {}),
                    Box::new(|_, _, _| {}),
                    Box::new(|_, _, _| {}),
                    Box::new(|_, _, _| {}),
                    Box::new(|_, _, _| {})
                ],
                vec![
                    Box::new(|_, _, _, _| {}),
                    Box::new(|_, _, _, _| {}),
                    Box::new(|_, _, _, _| {}),
                    Box::new(|_, _, _, _| {}),
                    Box::new(|_, _, _, _| {}),
                    Box::new(|_, _, _, _| {})
                ],
                Some(vec![Box::new(|_, _, _, _, _| {})])
            )
            .err(),
            Some("deriv2_boundary_functions must have length 6")
        );
    }
}
