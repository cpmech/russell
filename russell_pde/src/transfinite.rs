use russell_lab::Vector;

/// Defines a vector function f(s) of a scalar argument s (vector scalar)
///
/// Input:
/// * `s` -- input scalar
///
/// Output:
/// * `f` -- output vector
pub type Vs = Box<dyn Fn(&mut Vector, f64) + Send + Sync>;

/// Defines a vector function f(a,b) of two scalar arguments (vector scalar scalar)
///
/// Input:
/// * `a` -- first input scalar
/// * `b` -- second input scalar
///
/// Output:
/// * `f` -- output vector
pub type Vss = Box<dyn Fn(&mut Vector, f64, f64) + Send + Sync>;

/// Defines two vector functions u(a,b) and v(a,b) of two scalar arguments
/// (vector vector scalar scalar)
///
/// Input:
/// * `a` -- first input scalar
/// * `b` -- second input scalar
///
/// Output:
/// * `u` -- first output vector
/// * `v` -- second output vector
pub type Vvss = Box<dyn Fn(&mut Vector, &mut Vector, f64, f64) + Send + Sync>;

/// Defines three vector functions u(a,b), v(a,b) and w(a,b) of two scalar arguments
/// (vector vector vector scalar scalar)
///
/// Input:
/// * `a` -- first input scalar
/// * `b` -- second input scalar
///
/// Output:
/// * `u` -- first output vector
/// * `v` -- second output vector
/// * `w` -- second output vector
pub type Vvvss = Box<dyn Fn(&mut Vector, &mut Vector, &mut Vector, f64, f64) + Send + Sync>;

/// Maps a reference square [-1,+1]×[-1,+1] into a curve-bounded quadrilateral
pub struct Transfinite {
    ndim: usize, // space dimension

    // input data for 2d
    e: Vec<Vs>,           // [4] 2D boundary functions
    ed: Vec<Vs>,          // [4] 2D 1st derivatives of boundary functions
    edd: Option<Vec<Vs>>, // [4] 2D 2nd derivatives of boundary functions

    // input data for 3d
    b: Vec<Vss>,             // [6] 3D boundary function
    bd: Vec<Vvss>,           // [6] 3D 1st derivatives of boundary functions
    bdd: Option<Vec<Vvvss>>, // [6] 3D 2nd derivatives of boundary functions

    // workspace for 2d
    p0: Vector,
    p1: Vector, // corner points
    p2: Vector,
    p3: Vector, // corner points
    e0s: Vector,
    e1s: Vector, // 2D function evaluations
    e2r: Vector,
    e3r: Vector, // 2D function evaluations

    de0s_ds: Vector,
    de1s_ds: Vector, // derivative evaluation
    de2r_dr: Vector,
    de3r_dr: Vector, // derivative evaluation
    dde0s_dss: Vector,
    dde1s_dss: Vector, // derivative evaluation
    dde2r_drr: Vector,
    dde3r_drr: Vector, // derivative evaluation

    // workspace for 3d
    p4: Vector,
    p5: Vector, // corner points
    p6: Vector,
    p7: Vector, // corner points
    tm1: Vector,
    tm2: Vector, // temporary vectors

    b0st: Vector,
    b1st: Vector,
    b2rt: Vector, // function evaluation
    b3rt: Vector,
    b4rs: Vector,
    b5rs: Vector, // function evaluation

    b0mt: Vector,
    b0pt: Vector,
    b1mt: Vector,
    b1pt: Vector, // function evaluation
    b0sm: Vector,
    b0sp: Vector,
    b1sm: Vector,
    b1sp: Vector, // function evaluation
    b2rm: Vector,
    b2rp: Vector,
    b3rm: Vector,
    b3rp: Vector, // function evaluation

    db0st_ds: Vector,
    db0st_dt: Vector, // derivative evaluation
    db1st_ds: Vector,
    db1st_dt: Vector, // derivative evaluation
    db2rt_dr: Vector,
    db2rt_dt: Vector, // derivative evaluation
    db3rt_dr: Vector,
    db3rt_dt: Vector, // derivative evaluation
    db4rs_dr: Vector,
    db4rs_ds: Vector, // derivative evaluation
    db5rs_dr: Vector,
    db5rs_ds: Vector, // derivative evaluation

    db0sm_ds: Vector,
    db0sp_ds: Vector, // derivative evaluation
    db0mt_dt: Vector,
    db0pt_dt: Vector, // derivative evaluation
    db1sm_ds: Vector,
    db1sp_ds: Vector, // derivative evaluation
    db1mt_dt: Vector,
    db1pt_dt: Vector, // derivative evaluation
    db2rm_dr: Vector,
    db2rp_dr: Vector, // derivative evaluation
    db3rm_dr: Vector,
    db3rp_dr: Vector, // derivative evaluation

    ddb0st_dss: Vector,
    ddb0st_dtt: Vector,
    ddb0st_dst: Vector, // derivative evaluation
    ddb1st_dss: Vector,
    ddb1st_dtt: Vector,
    ddb1st_dst: Vector, // derivative evaluation
    ddb2rt_drr: Vector,
    ddb2rt_dtt: Vector,
    ddb2rt_drt: Vector, // derivative evaluation
    ddb3rt_drr: Vector,
    ddb3rt_dtt: Vector,
    ddb3rt_drt: Vector, // derivative evaluation
    ddb4rs_drr: Vector,
    ddb4rs_dss: Vector,
    ddb4rs_drs: Vector, // derivative evaluation
    ddb5rs_drr: Vector,
    ddb5rs_dss: Vector,
    ddb5rs_drs: Vector, // derivative evaluation

    ddb0sm_dss: Vector,
    ddb0sp_dss: Vector, // derivative evaluation
    ddb0mt_dtt: Vector,
    ddb0pt_dtt: Vector, // derivative evaluation
    ddb1sm_dss: Vector,
    ddb1sp_dss: Vector, // derivative evaluation
    ddb1mt_dtt: Vector,
    ddb1pt_dtt: Vector, // derivative evaluation
    ddb2rm_drr: Vector,
    ddb2rp_drr: Vector, // derivative evaluation
    ddb3rm_drr: Vector,
    ddb3rp_drr: Vector, // derivative evaluation
}

impl Transfinite {
    /// Allocates a new structure for 2D
    pub fn new_2d(e: Vec<Vs>, ed: Vec<Vs>, edd: Option<Vec<Vs>>) -> Self {
        assert_eq!(e.len(), 4);
        assert_eq!(ed.len(), 4);
        if let Some(ref edd_val) = edd {
            assert_eq!(edd_val.len(), 4);
        }

        let mut o = Transfinite {
            ndim: 2,
            e,
            ed,
            edd,
            b: Vec::new(),
            bd: Vec::new(),
            bdd: None,
            p0: Vector::new(2),
            p1: Vector::new(2),
            p2: Vector::new(2),
            p3: Vector::new(2),
            e0s: Vector::new(2),
            e1s: Vector::new(2),
            e2r: Vector::new(2),
            e3r: Vector::new(2),
            de0s_ds: Vector::new(2),
            de1s_ds: Vector::new(2),
            de2r_dr: Vector::new(2),
            de3r_dr: Vector::new(2),
            dde0s_dss: Vector::new(2),
            dde1s_dss: Vector::new(2),
            dde2r_drr: Vector::new(2),
            dde3r_drr: Vector::new(2),
            // 3D workspace (unused)
            p4: Vector::new(0),
            p5: Vector::new(0),
            p6: Vector::new(0),
            p7: Vector::new(0),
            tm1: Vector::new(0),
            tm2: Vector::new(0),
            b0st: Vector::new(0),
            b1st: Vector::new(0),
            b2rt: Vector::new(0),
            b3rt: Vector::new(0),
            b4rs: Vector::new(0),
            b5rs: Vector::new(0),
            b0mt: Vector::new(0),
            b0pt: Vector::new(0),
            b1mt: Vector::new(0),
            b1pt: Vector::new(0),
            b0sm: Vector::new(0),
            b0sp: Vector::new(0),
            b1sm: Vector::new(0),
            b1sp: Vector::new(0),
            b2rm: Vector::new(0),
            b2rp: Vector::new(0),
            b3rm: Vector::new(0),
            b3rp: Vector::new(0),
            db0st_ds: Vector::new(0),
            db0st_dt: Vector::new(0),
            db1st_ds: Vector::new(0),
            db1st_dt: Vector::new(0),
            db2rt_dr: Vector::new(0),
            db2rt_dt: Vector::new(0),
            db3rt_dr: Vector::new(0),
            db3rt_dt: Vector::new(0),
            db4rs_dr: Vector::new(0),
            db4rs_ds: Vector::new(0),
            db5rs_dr: Vector::new(0),
            db5rs_ds: Vector::new(0),
            db0sm_ds: Vector::new(0),
            db0sp_ds: Vector::new(0),
            db0mt_dt: Vector::new(0),
            db0pt_dt: Vector::new(0),
            db1sm_ds: Vector::new(0),
            db1sp_ds: Vector::new(0),
            db1mt_dt: Vector::new(0),
            db1pt_dt: Vector::new(0),
            db2rm_dr: Vector::new(0),
            db2rp_dr: Vector::new(0),
            db3rm_dr: Vector::new(0),
            db3rp_dr: Vector::new(0),
            ddb0st_dss: Vector::new(0),
            ddb0st_dtt: Vector::new(0),
            ddb0st_dst: Vector::new(0),
            ddb1st_dss: Vector::new(0),
            ddb1st_dtt: Vector::new(0),
            ddb1st_dst: Vector::new(0),
            ddb2rt_drr: Vector::new(0),
            ddb2rt_dtt: Vector::new(0),
            ddb2rt_drt: Vector::new(0),
            ddb3rt_drr: Vector::new(0),
            ddb3rt_dtt: Vector::new(0),
            ddb3rt_drt: Vector::new(0),
            ddb4rs_drr: Vector::new(0),
            ddb4rs_dss: Vector::new(0),
            ddb4rs_drs: Vector::new(0),
            ddb5rs_drr: Vector::new(0),
            ddb5rs_dss: Vector::new(0),
            ddb5rs_drs: Vector::new(0),
            ddb0sm_dss: Vector::new(0),
            ddb0sp_dss: Vector::new(0),
            ddb0mt_dtt: Vector::new(0),
            ddb0pt_dtt: Vector::new(0),
            ddb1sm_dss: Vector::new(0),
            ddb1sp_dss: Vector::new(0),
            ddb1mt_dtt: Vector::new(0),
            ddb1pt_dtt: Vector::new(0),
            ddb2rm_drr: Vector::new(0),
            ddb2rp_drr: Vector::new(0),
            ddb3rm_drr: Vector::new(0),
            ddb3rp_drr: Vector::new(0),
        };

        // compute corners
        (o.e[0])(&mut o.p0, -1.0);
        (o.e[0])(&mut o.p3, 1.0);
        (o.e[1])(&mut o.p1, -1.0);
        (o.e[1])(&mut o.p2, 1.0);
        o
    }

    /// Allocates a new structure for 3D
    pub fn new_3d(b: Vec<Vss>, bd: Vec<Vvss>, bdd: Option<Vec<Vvvss>>) -> Self {
        assert_eq!(b.len(), 6);
        assert_eq!(bd.len(), 6);
        if let Some(ref bdd_val) = bdd {
            assert_eq!(bdd_val.len(), 6);
        }

        let mut o = Transfinite {
            ndim: 3,
            e: Vec::new(),
            ed: Vec::new(),
            edd: None,
            b,
            bd,
            bdd,
            // 2D workspace (unused)
            p0: Vector::new(3),
            p1: Vector::new(3),
            p2: Vector::new(3),
            p3: Vector::new(3),
            e0s: Vector::new(0),
            e1s: Vector::new(0),
            e2r: Vector::new(0),
            e3r: Vector::new(0),
            de0s_ds: Vector::new(0),
            de1s_ds: Vector::new(0),
            de2r_dr: Vector::new(0),
            de3r_dr: Vector::new(0),
            dde0s_dss: Vector::new(0),
            dde1s_dss: Vector::new(0),
            dde2r_drr: Vector::new(0),
            dde3r_drr: Vector::new(0),
            // 3D workspace
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
        (o.b[4])(&mut o.p0, -1.0, -1.0);
        (o.b[4])(&mut o.p1, 1.0, -1.0);
        (o.b[4])(&mut o.p2, 1.0, 1.0);
        (o.b[4])(&mut o.p3, -1.0, 1.0);
        (o.b[5])(&mut o.p4, -1.0, -1.0);
        (o.b[5])(&mut o.p5, 1.0, -1.0);
        (o.b[5])(&mut o.p6, 1.0, 1.0);
        (o.b[5])(&mut o.p7, -1.0, 1.0);
        o
    }

    /// Computes "real" position x(r,s,t)
    pub fn point(&mut self, x: &mut Vector, u: &Vector) {
        // 2D
        if self.ndim == 2 {
            // compute boundary functions @ {r,s}
            let r = u[0];
            let s = u[1];
            (self.e[0])(&mut self.e0s, s);
            (self.e[1])(&mut self.e1s, s);
            (self.e[2])(&mut self.e2r, r);
            (self.e[3])(&mut self.e3r, r);

            // compute position
            for i in 0..self.ndim {
                x[i] = 0.0
                    + (1.0 - r) * self.e0s[i] / 2.0
                    + (1.0 + r) * self.e1s[i] / 2.0
                    + (1.0 - s) * self.e2r[i] / 2.0
                    + (1.0 + s) * self.e3r[i] / 2.0
                    - (1.0 - r) * (1.0 - s) * self.p0[i] / 4.0
                    - (1.0 + r) * (1.0 - s) * self.p1[i] / 4.0
                    - (1.0 + r) * (1.0 + s) * self.p2[i] / 4.0
                    - (1.0 - r) * (1.0 + s) * self.p3[i] / 4.0;
            }
            return;
        }

        // 3D
        let r = u[0];
        let s = u[1];
        let t = u[2];
        let m = -1.0;
        let p = 1.0;

        // compute boundary functions @ {r,s,t}
        (self.b[0])(&mut self.b0st, s, t);
        (self.b[1])(&mut self.b1st, s, t);
        (self.b[2])(&mut self.b2rt, r, t);
        (self.b[3])(&mut self.b3rt, r, t);
        (self.b[4])(&mut self.b4rs, r, s);
        (self.b[5])(&mut self.b5rs, r, s);

        // compute boundary functions @ edges
        (self.b[0])(&mut self.b0mt, m, t);
        (self.b[0])(&mut self.b0pt, p, t);
        (self.b[1])(&mut self.b1mt, m, t);
        (self.b[1])(&mut self.b1pt, p, t);

        (self.b[0])(&mut self.b0sm, s, m);
        (self.b[0])(&mut self.b0sp, s, p);
        (self.b[1])(&mut self.b1sm, s, m);
        (self.b[1])(&mut self.b1sp, s, p);

        (self.b[2])(&mut self.b2rm, r, m);
        (self.b[2])(&mut self.b2rp, r, p);
        (self.b[3])(&mut self.b3rm, r, m);
        (self.b[3])(&mut self.b3rp, r, p);

        // compute position
        for i in 0..self.ndim {
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
        ddx_drr: Option<&mut Vector>,
        ddx_dss: Option<&mut Vector>,
        ddx_dtt: Option<&mut Vector>,
        ddx_drs: Option<&mut Vector>,
        ddx_drt: Option<&mut Vector>,
        ddx_dst: Option<&mut Vector>,
        u: &Vector,
    ) {
        // auxiliary
        let second_derivs = ddx_drr.is_some();
        let m = -1.0;
        let p = 1.0;

        // 2D
        if self.ndim == 2 {
            // auxiliary
            let r = u[0];
            let s = u[1];

            // compute boundary functions @ {r,s}
            (self.e[0])(&mut self.e0s, s);
            (self.e[1])(&mut self.e1s, s);
            (self.e[2])(&mut self.e2r, r);
            (self.e[3])(&mut self.e3r, r);

            // compute derivatives @ {r,s}
            (self.ed[0])(&mut self.de0s_ds, s);
            (self.ed[1])(&mut self.de1s_ds, s);
            (self.ed[2])(&mut self.de2r_dr, r);
            (self.ed[3])(&mut self.de3r_dr, r);

            // position and first order derivatives
            for i in 0..self.ndim {
                // bilinear transfinite mapping in 2D
                x[i] = 0.0
                    + (1.0 - r) * self.e0s[i] / 2.0
                    + (1.0 + r) * self.e1s[i] / 2.0
                    + (1.0 - s) * self.e2r[i] / 2.0
                    + (1.0 + s) * self.e3r[i] / 2.0
                    - (1.0 - r) * (1.0 - s) * self.p0[i] / 4.0
                    - (1.0 + r) * (1.0 - s) * self.p1[i] / 4.0
                    - (1.0 + r) * (1.0 + s) * self.p2[i] / 4.0
                    - (1.0 - r) * (1.0 + s) * self.p3[i] / 4.0;

                // derivative of x with respect to r
                dx_dr[i] = 0.0 - self.e0s[i] / 2.0
                    + self.e1s[i] / 2.0
                    + (1.0 - s) * self.de2r_dr[i] / 2.0
                    + (1.0 + s) * self.de3r_dr[i] / 2.0
                    + (1.0 - s) * self.p0[i] / 4.0
                    - (1.0 - s) * self.p1[i] / 4.0
                    - (1.0 + s) * self.p2[i] / 4.0
                    + (1.0 + s) * self.p3[i] / 4.0;

                // derivative of x with respect to s
                dx_ds[i] = 0.0 + (1.0 - r) * self.de0s_ds[i] / 2.0 + (1.0 + r) * self.de1s_ds[i] / 2.0
                    - self.e2r[i] / 2.0
                    + self.e3r[i] / 2.0
                    + (1.0 - r) * self.p0[i] / 4.0
                    + (1.0 + r) * self.p1[i] / 4.0
                    - (1.0 + r) * self.p2[i] / 4.0
                    - (1.0 - r) * self.p3[i] / 4.0;
            }

            // skip second order derivatives
            if !second_derivs {
                return;
            }

            // unwrap optional arguments
            let ddx_drr = ddx_drr.unwrap();
            let ddx_dss = ddx_dss.unwrap();
            let ddx_drs = ddx_drs.unwrap();

            // only 2nd cross-derivatives may be non-zero
            if self.edd.is_none() {
                for i in 0..self.ndim {
                    ddx_drr[i] = 0.0;
                    ddx_dss[i] = 0.0;
                    ddx_drs[i] = 0.0 - self.de0s_ds[i] / 2.0 + self.de1s_ds[i] / 2.0 - self.de2r_dr[i] / 2.0
                        + self.de3r_dr[i] / 2.0
                        - self.p0[i] / 4.0
                        + self.p1[i] / 4.0
                        - self.p2[i] / 4.0
                        + self.p3[i] / 4.0;
                }
                return;
            }

            // compute second derivatives @ {r,s,t}
            let edd = self.edd.as_ref().unwrap();
            (edd[0])(&mut self.dde0s_dss, s);
            (edd[1])(&mut self.dde1s_dss, s);
            (edd[2])(&mut self.dde2r_drr, r);
            (edd[3])(&mut self.dde3r_drr, r);

            // second order derivatives
            for i in 0..self.ndim {
                // derivative of dx/dr with respect to r
                ddx_drr[i] = 0.0 + (1.0 - s) * self.dde2r_drr[i] / 2.0 + (1.0 + s) * self.dde3r_drr[i] / 2.0;

                // derivative of dx/ds with respect to s
                ddx_dss[i] = 0.0 + (1.0 - r) * self.dde0s_dss[i] / 2.0 + (1.0 + r) * self.dde1s_dss[i] / 2.0;

                // derivative of dx/dr with respect to s
                ddx_drs[i] = 0.0 - self.de0s_ds[i] / 2.0 + self.de1s_ds[i] / 2.0 - self.de2r_dr[i] / 2.0
                    + self.de3r_dr[i] / 2.0
                    - self.p0[i] / 4.0
                    + self.p1[i] / 4.0
                    - self.p2[i] / 4.0
                    + self.p3[i] / 4.0;
            }
            return;
        }

        // auxiliary
        let r = u[0];
        let s = u[1];
        let t = u[2];

        // compute boundary functions @ {r,s,t}
        (self.b[0])(&mut self.b0st, s, t);
        (self.b[1])(&mut self.b1st, s, t);
        (self.b[2])(&mut self.b2rt, r, t);
        (self.b[3])(&mut self.b3rt, r, t);
        (self.b[4])(&mut self.b4rs, r, s);
        (self.b[5])(&mut self.b5rs, r, s);

        // compute boundary functions @ edges
        (self.b[0])(&mut self.b0mt, m, t);
        (self.b[0])(&mut self.b0pt, p, t);
        (self.b[1])(&mut self.b1mt, m, t);
        (self.b[1])(&mut self.b1pt, p, t);

        (self.b[0])(&mut self.b0sm, s, m);
        (self.b[0])(&mut self.b0sp, s, p);
        (self.b[1])(&mut self.b1sm, s, m);
        (self.b[1])(&mut self.b1sp, s, p);

        (self.b[2])(&mut self.b2rm, r, m);
        (self.b[2])(&mut self.b2rp, r, p);
        (self.b[3])(&mut self.b3rm, r, m);
        (self.b[3])(&mut self.b3rp, r, p);

        // compute derivatives @ {r,s,t}
        (self.bd[0])(&mut self.db0st_ds, &mut self.db0st_dt, s, t);
        (self.bd[1])(&mut self.db1st_ds, &mut self.db1st_dt, s, t);
        (self.bd[2])(&mut self.db2rt_dr, &mut self.db2rt_dt, r, t);
        (self.bd[3])(&mut self.db3rt_dr, &mut self.db3rt_dt, r, t);
        (self.bd[4])(&mut self.db4rs_dr, &mut self.db4rs_ds, r, s);
        (self.bd[5])(&mut self.db5rs_dr, &mut self.db5rs_ds, r, s);

        // compute derivatives @ edges
        (self.bd[0])(&mut self.db0sm_ds, &mut self.tm1, s, m);
        (self.bd[0])(&mut self.db0sp_ds, &mut self.tm1, s, p);
        (self.bd[0])(&mut self.tm1, &mut self.db0mt_dt, m, t);
        (self.bd[0])(&mut self.tm1, &mut self.db0pt_dt, p, t);

        (self.bd[1])(&mut self.db1sm_ds, &mut self.tm1, s, m);
        (self.bd[1])(&mut self.db1sp_ds, &mut self.tm1, s, p);
        (self.bd[1])(&mut self.tm1, &mut self.db1mt_dt, m, t);
        (self.bd[1])(&mut self.tm1, &mut self.db1pt_dt, p, t);

        (self.bd[2])(&mut self.db2rm_dr, &mut self.tm1, r, m);
        (self.bd[2])(&mut self.db2rp_dr, &mut self.tm1, r, p);
        (self.bd[3])(&mut self.db3rm_dr, &mut self.tm1, r, m);
        (self.bd[3])(&mut self.db3rp_dr, &mut self.tm1, r, p);

        // position and first order derivatives
        for i in 0..self.ndim {
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
        let ddx_drr = ddx_drr.unwrap();
        let ddx_dss = ddx_dss.unwrap();
        let ddx_dtt = ddx_dtt.unwrap();
        let ddx_drs = ddx_drs.unwrap();
        let ddx_drt = ddx_drt.unwrap();
        let ddx_dst = ddx_dst.unwrap();

        // only 2nd cross-derivatives may be non-zero
        if self.bdd.is_none() {
            for i in 0..self.ndim {
                ddx_drr[i] = 0.0;
                ddx_dss[i] = 0.0;
                ddx_dtt[i] = 0.0;

                // derivative of dx/dr with respect to s
                ddx_drs[i] = 0.0 - self.db0st_ds[i] / 2.0 + self.db1st_ds[i] / 2.0 - self.db2rt_dr[i] / 2.0
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
                ddx_drt[i] = 0.0 - self.db0st_dt[i] / 2.0 + self.db1st_dt[i] / 2.0 - self.db4rs_dr[i] / 2.0
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
                ddx_dst[i] = 0.0 - self.db2rt_dt[i] / 2.0 + self.db3rt_dt[i] / 2.0 - self.db4rs_ds[i] / 2.0
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
        let bdd = self.bdd.as_ref().unwrap();
        (bdd[0])(&mut self.ddb0st_dss, &mut self.ddb0st_dtt, &mut self.ddb0st_dst, s, t);
        (bdd[1])(&mut self.ddb1st_dss, &mut self.ddb1st_dtt, &mut self.ddb1st_dst, s, t);
        (bdd[2])(&mut self.ddb2rt_drr, &mut self.ddb2rt_dtt, &mut self.ddb2rt_drt, r, t);
        (bdd[3])(&mut self.ddb3rt_drr, &mut self.ddb3rt_dtt, &mut self.ddb3rt_drt, r, t);
        (bdd[4])(&mut self.ddb4rs_drr, &mut self.ddb4rs_dss, &mut self.ddb4rs_drs, r, s);
        (bdd[5])(&mut self.ddb5rs_drr, &mut self.ddb5rs_dss, &mut self.ddb5rs_drs, r, s);

        // compute second derivatives @ edges
        (bdd[0])(&mut self.ddb0sm_dss, &mut self.tm1, &mut self.tm2, s, m);
        (bdd[0])(&mut self.ddb0sp_dss, &mut self.tm1, &mut self.tm2, s, p);
        (bdd[0])(&mut self.tm1, &mut self.ddb0mt_dtt, &mut self.tm2, m, t);
        (bdd[0])(&mut self.tm1, &mut self.ddb0pt_dtt, &mut self.tm2, p, t);

        (bdd[1])(&mut self.ddb1sm_dss, &mut self.tm1, &mut self.tm2, s, m);
        (bdd[1])(&mut self.ddb1sp_dss, &mut self.tm1, &mut self.tm2, s, p);
        (bdd[1])(&mut self.tm1, &mut self.ddb1mt_dtt, &mut self.tm2, m, t);
        (bdd[1])(&mut self.tm1, &mut self.ddb1pt_dtt, &mut self.tm2, p, t);

        (bdd[2])(&mut self.ddb2rm_drr, &mut self.tm1, &mut self.tm2, r, m);
        (bdd[2])(&mut self.ddb2rp_drr, &mut self.tm1, &mut self.tm2, r, p);
        (bdd[3])(&mut self.ddb3rm_drr, &mut self.tm1, &mut self.tm2, r, m);
        (bdd[3])(&mut self.ddb3rp_drr, &mut self.tm1, &mut self.tm2, r, p);

        // second order derivatives
        for i in 0..self.ndim {
            // derivative of dx/dr with respect to r
            ddx_drr[i] = 0.0
                + (1.0 - s) * self.ddb2rt_drr[i] / 2.0
                + (1.0 + s) * self.ddb3rt_drr[i] / 2.0
                + (1.0 - t) * self.ddb4rs_drr[i] / 2.0
                + (1.0 + t) * self.ddb5rs_drr[i] / 2.0
                - (1.0 - s) * (1.0 - t) * self.ddb2rm_drr[i] / 4.0
                - (1.0 - s) * (1.0 + t) * self.ddb2rp_drr[i] / 4.0
                - (1.0 + s) * (1.0 - t) * self.ddb3rm_drr[i] / 4.0
                - (1.0 + s) * (1.0 + t) * self.ddb3rp_drr[i] / 4.0;

            // derivative of dx/ds with respect to s
            ddx_dss[i] = 0.0
                + (1.0 - r) * self.ddb0st_dss[i] / 2.0
                + (1.0 + r) * self.ddb1st_dss[i] / 2.0
                + (1.0 - t) * self.ddb4rs_dss[i] / 2.0
                + (1.0 + t) * self.ddb5rs_dss[i] / 2.0
                - (1.0 - r) * (1.0 - t) * self.ddb0sm_dss[i] / 4.0
                - (1.0 - r) * (1.0 + t) * self.ddb0sp_dss[i] / 4.0
                - (1.0 + r) * (1.0 - t) * self.ddb1sm_dss[i] / 4.0
                - (1.0 + r) * (1.0 + t) * self.ddb1sp_dss[i] / 4.0;

            // derivative of dx/dt with respect to t
            ddx_dtt[i] = 0.0
                + (1.0 - r) * self.ddb0st_dtt[i] / 2.0
                + (1.0 + r) * self.ddb1st_dtt[i] / 2.0
                + (1.0 - s) * self.ddb2rt_dtt[i] / 2.0
                + (1.0 + s) * self.ddb3rt_dtt[i] / 2.0
                - (1.0 - r) * (1.0 - s) * self.ddb0mt_dtt[i] / 4.0
                - (1.0 - r) * (1.0 + s) * self.ddb0pt_dtt[i] / 4.0
                - (1.0 + r) * (1.0 - s) * self.ddb1mt_dtt[i] / 4.0
                - (1.0 + r) * (1.0 + s) * self.ddb1pt_dtt[i] / 4.0;

            // derivative of dx/dr with respect to s
            ddx_drs[i] = 0.0 - self.db0st_ds[i] / 2.0 + self.db1st_ds[i] / 2.0 - self.db2rt_dr[i] / 2.0
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
            ddx_drt[i] = 0.0 - self.db0st_dt[i] / 2.0
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
            ddx_dst[i] = 0.0 + (1.0 - r) * self.ddb0st_dst[i] / 2.0 + (1.0 + r) * self.ddb1st_dst[i] / 2.0
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
}

#[cfg(test)]
mod tests {
    use super::{Transfinite, Vs};
    use russell_lab::{vec_approx_eq, Vector};

    #[test]
    fn new_2d_works() {
        // identity mapping
        // e0(s) = (-1, s)
        let e0: Vs = Box::new(|x, s| {
            x[0] = -1.0;
            x[1] = s;
        });
        // e1(s) = (+1, s)
        let e1: Vs = Box::new(|x, s| {
            x[0] = 1.0;
            x[1] = s;
        });
        // e2(r) = (r, -1)
        let e2: Vs = Box::new(|x, r| {
            x[0] = r;
            x[1] = -1.0;
        });
        // e3(r) = (r, +1)
        let e3: Vs = Box::new(|x, r| {
            x[0] = r;
            x[1] = 1.0;
        });

        // derivatives
        let ed0: Vs = Box::new(|dx, _| {
            dx[0] = 0.0;
            dx[1] = 1.0;
        });
        let ed1: Vs = Box::new(|dx, _| {
            dx[0] = 0.0;
            dx[1] = 1.0;
        });
        let ed2: Vs = Box::new(|dx, _| {
            dx[0] = 1.0;
            dx[1] = 0.0;
        });
        let ed3: Vs = Box::new(|dx, _| {
            dx[0] = 1.0;
            dx[1] = 0.0;
        });

        let e = vec![e0, e1, e2, e3];
        let ed = vec![ed0, ed1, ed2, ed3];

        let mut tr = Transfinite::new_2d(e, ed, None);

        let mut x = Vector::new(2);
        let u = Vector::from(&[0.0, 0.0]);
        tr.point(&mut x, &u);
        vec_approx_eq(&x, &[0.0, 0.0], 1e-15);

        let u = Vector::from(&[0.5, 0.5]);
        tr.point(&mut x, &u);
        vec_approx_eq(&x, &[0.5, 0.5], 1e-15);

        let mut dx_dr = Vector::new(2);
        let mut dx_ds = Vector::new(2);
        let mut dx_dt = Vector::new(0); // unused
        tr.point_and_derivs(
            &mut x, &mut dx_dr, &mut dx_ds, &mut dx_dt, None, None, None, None, None, None, &u,
        );
        vec_approx_eq(&x, &[0.5, 0.5], 1e-15);
        vec_approx_eq(&dx_dr, &[1.0, 0.0], 1e-15);
        vec_approx_eq(&dx_ds, &[0.0, 1.0], 1e-15);
    }
}
