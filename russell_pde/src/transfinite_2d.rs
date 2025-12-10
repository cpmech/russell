use super::FnVec1Param1;
use russell_lab::Vector;

/// Implements the transfinite mapping
///
/// Maps a reference square [-1,+1] × [-1,+1] into a curve-bounded region in 2D
///
/// ```text
///                                              B3(r(x,y)) _,'\
///             B3(r)                                    _,'    \ B1(s(x,y))
///            ┌───────┐                              _,'        \
///            │       │                             \            \
///       B0(s)│       │B1(s)  MAPS TO                \         _,'
///  s         │       │                    B0(s(x,y)) \     _,'
///  │         └───────┘               y                \ _,'  B2(r(x,y))
///  └──r       B2(r)                  │                 '
///                                    └──x
/// ```
///
/// Note: The reference coordinates {r,s,t} ϵ [-1,+1]×[-1,+1]×[-1,+1] are also symbolized by `u`
pub struct Transfinite2d {
    // boundary functions
    // len = 4
    boundary_functions: Vec<FnVec1Param1>,

    // 1st derivatives of boundary functions
    // len = 4
    deriv1_boundary_functions: Vec<FnVec1Param1>,

    // 2nd derivatives of boundary functions
    // len = 4
    deriv2_boundary_functions: Option<Vec<FnVec1Param1>>,

    // corner points
    p0: Vector,
    p1: Vector,
    p2: Vector,
    p3: Vector,

    // boundary function evaluations
    b0s: Vector,
    b1s: Vector,
    b2r: Vector,
    b3r: Vector,

    // derivative evaluations
    db0s_ds: Vector,
    db1s_ds: Vector,
    db2r_dr: Vector,
    db3r_dr: Vector,
    ddb0s_dss: Vector,
    ddb1s_dss: Vector,
    ddb2r_drr: Vector,
    ddb3r_drr: Vector,
}

impl Transfinite2d {
    /// Allocates a new instance
    pub fn new(
        boundary_functions: Vec<FnVec1Param1>,
        deriv1_boundary_functions: Vec<FnVec1Param1>,
        deriv2_boundary_functions: Option<Vec<FnVec1Param1>>,
    ) -> Self {
        // checks
        assert_eq!(boundary_functions.len(), 4);
        assert_eq!(deriv1_boundary_functions.len(), 4);
        if let Some(ref edd_val) = deriv2_boundary_functions {
            assert_eq!(edd_val.len(), 4);
        }

        let mut map = Transfinite2d {
            boundary_functions,
            deriv1_boundary_functions,
            deriv2_boundary_functions,
            p0: Vector::new(2),
            p1: Vector::new(2),
            p2: Vector::new(2),
            p3: Vector::new(2),
            b0s: Vector::new(2),
            b1s: Vector::new(2),
            b2r: Vector::new(2),
            b3r: Vector::new(2),
            db0s_ds: Vector::new(2),
            db1s_ds: Vector::new(2),
            db2r_dr: Vector::new(2),
            db3r_dr: Vector::new(2),
            ddb0s_dss: Vector::new(2),
            ddb1s_dss: Vector::new(2),
            ddb2r_drr: Vector::new(2),
            ddb3r_drr: Vector::new(2),
        };

        // compute corners
        (map.boundary_functions[0])(&mut map.p0, -1.0);
        (map.boundary_functions[0])(&mut map.p3, 1.0);
        (map.boundary_functions[1])(&mut map.p1, -1.0);
        (map.boundary_functions[1])(&mut map.p2, 1.0);
        map
    }

    /// Computes "real" position x(r,s,t)
    pub fn point(&mut self, x: &mut Vector, r: f64, s: f64) {
        // compute boundary functions @ {r,s}
        (self.boundary_functions[0])(&mut self.b0s, s);
        (self.boundary_functions[1])(&mut self.b1s, s);
        (self.boundary_functions[2])(&mut self.b2r, r);
        (self.boundary_functions[3])(&mut self.b3r, r);

        // compute position
        for i in 0..2 {
            x[i] = 0.0
                + (1.0 - r) * self.b0s[i] / 2.0
                + (1.0 + r) * self.b1s[i] / 2.0
                + (1.0 - s) * self.b2r[i] / 2.0
                + (1.0 + s) * self.b3r[i] / 2.0
                - (1.0 - r) * (1.0 - s) * self.p0[i] / 4.0
                - (1.0 + r) * (1.0 - s) * self.p1[i] / 4.0
                - (1.0 + r) * (1.0 + s) * self.p2[i] / 4.0
                - (1.0 - r) * (1.0 + s) * self.p3[i] / 4.0;
        }
    }

    /// Computes position and the first and second order derivatives
    pub fn point_and_derivs(
        &mut self,
        x: &mut Vector,
        dx_dr: &mut Vector,
        dx_ds: &mut Vector,
        d2x_dr2: Option<&mut Vector>,
        d2x_ds2: Option<&mut Vector>,
        d2x_drs: Option<&mut Vector>,
        r: f64,
        s: f64,
    ) {
        // auxiliary
        let second_derivs = d2x_dr2.is_some();

        // compute boundary functions @ {r,s}
        (self.boundary_functions[0])(&mut self.b0s, s);
        (self.boundary_functions[1])(&mut self.b1s, s);
        (self.boundary_functions[2])(&mut self.b2r, r);
        (self.boundary_functions[3])(&mut self.b3r, r);

        // compute derivatives @ {r,s}
        (self.deriv1_boundary_functions[0])(&mut self.db0s_ds, s);
        (self.deriv1_boundary_functions[1])(&mut self.db1s_ds, s);
        (self.deriv1_boundary_functions[2])(&mut self.db2r_dr, r);
        (self.deriv1_boundary_functions[3])(&mut self.db3r_dr, r);

        // position and first order derivatives
        for i in 0..2 {
            // bilinear transfinite mapping in 2D
            x[i] = 0.0
                + (1.0 - r) * self.b0s[i] / 2.0
                + (1.0 + r) * self.b1s[i] / 2.0
                + (1.0 - s) * self.b2r[i] / 2.0
                + (1.0 + s) * self.b3r[i] / 2.0
                - (1.0 - r) * (1.0 - s) * self.p0[i] / 4.0
                - (1.0 + r) * (1.0 - s) * self.p1[i] / 4.0
                - (1.0 + r) * (1.0 + s) * self.p2[i] / 4.0
                - (1.0 - r) * (1.0 + s) * self.p3[i] / 4.0;

            // derivative of x with respect to r
            dx_dr[i] = 0.0 - self.b0s[i] / 2.0
                + self.b1s[i] / 2.0
                + (1.0 - s) * self.db2r_dr[i] / 2.0
                + (1.0 + s) * self.db3r_dr[i] / 2.0
                + (1.0 - s) * self.p0[i] / 4.0
                - (1.0 - s) * self.p1[i] / 4.0
                - (1.0 + s) * self.p2[i] / 4.0
                + (1.0 + s) * self.p3[i] / 4.0;

            // derivative of x with respect to s
            dx_ds[i] = 0.0 + (1.0 - r) * self.db0s_ds[i] / 2.0 + (1.0 + r) * self.db1s_ds[i] / 2.0 - self.b2r[i] / 2.0
                + self.b3r[i] / 2.0
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
        let d2x_dr2 = d2x_dr2.unwrap();
        let d2x_ds2 = d2x_ds2.unwrap();
        let d2x_drs = d2x_drs.unwrap();

        // only 2nd cross-derivatives may be non-zero
        if self.deriv2_boundary_functions.is_none() {
            for i in 0..2 {
                d2x_dr2[i] = 0.0;
                d2x_ds2[i] = 0.0;
                d2x_drs[i] = 0.0 - self.db0s_ds[i] / 2.0 + self.db1s_ds[i] / 2.0 - self.db2r_dr[i] / 2.0
                    + self.db3r_dr[i] / 2.0
                    - self.p0[i] / 4.0
                    + self.p1[i] / 4.0
                    - self.p2[i] / 4.0
                    + self.p3[i] / 4.0;
            }
            return;
        }

        // compute second derivatives @ {r,s,t}
        let edd = self.deriv2_boundary_functions.as_ref().unwrap();
        (edd[0])(&mut self.ddb0s_dss, s);
        (edd[1])(&mut self.ddb1s_dss, s);
        (edd[2])(&mut self.ddb2r_drr, r);
        (edd[3])(&mut self.ddb3r_drr, r);

        // second order derivatives
        for i in 0..2 {
            // derivative of dx/dr with respect to r
            d2x_dr2[i] = 0.0 + (1.0 - s) * self.ddb2r_drr[i] / 2.0 + (1.0 + s) * self.ddb3r_drr[i] / 2.0;

            // derivative of dx/ds with respect to s
            d2x_ds2[i] = 0.0 + (1.0 - r) * self.ddb0s_dss[i] / 2.0 + (1.0 + r) * self.ddb1s_dss[i] / 2.0;

            // derivative of dx/dr with respect to s
            d2x_drs[i] = 0.0 - self.db0s_ds[i] / 2.0 + self.db1s_ds[i] / 2.0 - self.db2r_dr[i] / 2.0
                + self.db3r_dr[i] / 2.0
                - self.p0[i] / 4.0
                + self.p1[i] / 4.0
                - self.p2[i] / 4.0
                + self.p3[i] / 4.0;
        }
    }

    /// Returns the corner points
    pub fn get_corners(&self) -> (&Vector, &Vector, &Vector, &Vector) {
        (&self.p0, &self.p1, &self.p2, &self.p3)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Transfinite2d;
    use crate::TransfiniteSamples;
    use russell_lab::math::SQRT_2;
    use russell_lab::{vec_approx_eq, vec_deriv1_approx_eq, vec_deriv2_approx_eq, Vector};

    fn check_derivs(map: &mut Transfinite2d, tol_d1: f64, tol_d2: f64) {
        let mut x_ana = Vector::new(2);
        let mut dx_dr_ana = Vector::new(2);
        let mut dx_ds_ana = Vector::new(2);
        let mut d2x_dr2_ana = Vector::new(2);
        let mut d2x_ds2_ana = Vector::new(2);
        let mut d2x_drs_ana = Vector::new(2);
        let mut x_tmp = Vector::new(2);
        let mut dx_ds_tmp = Vector::new(2);
        let args = &mut 0_u8;
        for s_at in [-0.5, 0.0, 0.5] {
            for r_at in [-0.5, 0.0, 0.5] {
                map.point_and_derivs(
                    &mut x_ana,
                    &mut dx_dr_ana,
                    &mut dx_ds_ana,
                    Some(&mut d2x_dr2_ana),
                    Some(&mut d2x_ds2_ana),
                    Some(&mut d2x_drs_ana),
                    r_at,
                    s_at,
                );
                // dx/dr
                vec_deriv1_approx_eq(&dx_dr_ana, r_at, args, tol_d1, |x, r, _| {
                    map.point(x, r, s_at);
                    Ok(())
                });
                // dx/ds
                vec_deriv1_approx_eq(&dx_ds_ana, s_at, args, tol_d1, |x, s, _| {
                    map.point(x, r_at, s);
                    Ok(())
                });
                // d²x/dr²
                vec_deriv2_approx_eq(&d2x_dr2_ana, r_at, args, tol_d2, |x, r, _| {
                    map.point(x, r, s_at);
                    Ok(())
                });
                // d²x/ds²
                vec_deriv2_approx_eq(&d2x_ds2_ana, s_at, args, tol_d2, |x, s, _| {
                    map.point(x, r_at, s);
                    Ok(())
                });
                // d²x/(dr ds)
                vec_deriv1_approx_eq(&d2x_drs_ana, s_at, args, tol_d2, |dx_dr, s, _| {
                    map.point_and_derivs(&mut x_tmp, dx_dr, &mut dx_ds_tmp, None, None, None, r_at, s);
                    Ok(())
                });
            }
        }
    }

    #[test]
    fn transfinite_2d_works_1() {
        // allocate transfinite map for quarter ring
        let r_in = 2.0;
        let r_out = 6.0;
        let mut map = TransfiniteSamples::quarter_ring_2d(r_in, r_out);

        // check corners
        let (p0, p1, p2, p3) = map.get_corners();
        vec_approx_eq(&p0, &[r_in, 0.0], 1e-15);
        vec_approx_eq(&p1, &[r_out, 0.0], 1e-15);
        vec_approx_eq(&p2, &[0.0, r_out], 1e-15);
        vec_approx_eq(&p3, &[0.0, r_in], 1e-15);

        // check some points
        let mut x = Vector::new(2);
        map.point(&mut x, 0.0, -1.0);
        vec_approx_eq(&x, &[0.5 * (r_in + r_out), 0.0], 1e-15);
        map.point(&mut x, 1.0, 0.0);
        vec_approx_eq(&x, &[r_out * SQRT_2 / 2.0, r_out * SQRT_2 / 2.0], 1e-15);
        map.point(&mut x, 0.0, 1.0);
        vec_approx_eq(&x, &[0.0, 0.5 * (r_in + r_out)], 1e-15);
        map.point(&mut x, -1.0, 0.0);
        vec_approx_eq(&x, &[r_in * SQRT_2 / 2.0, r_in * SQRT_2 / 2.0], 1e-15);
        map.point(&mut x, 0.0, 0.0);
        vec_approx_eq(
            &x,
            &[(r_in + r_out) * SQRT_2 / 4.0, (r_in + r_out) * SQRT_2 / 4.0],
            1e-15,
        );

        // check derivatives
        check_derivs(&mut map, 1e-9, 1e-8);
    }
}
