use super::FnVec1Param1;
use crate::StrError;
use russell_lab::math::chebyshev_lobatto_points;
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
    ///
    /// # Arguments
    ///
    /// * `boundary_functions` - boundary functions [B0(s), B1(s), B2(r), B3(r)]
    /// * `deriv1_boundary_functions` - 1st derivatives of boundary functions
    /// * `deriv2_boundary_functions` - 2nd derivatives of boundary functions (optional)
    pub fn new(
        boundary_functions: Vec<FnVec1Param1>,
        deriv1_boundary_functions: Vec<FnVec1Param1>,
        deriv2_boundary_functions: Option<Vec<FnVec1Param1>>,
    ) -> Result<Self, StrError> {
        // checks
        if boundary_functions.len() != 4 {
            return Err("boundary_functions must have length 4");
        }
        if deriv1_boundary_functions.len() != 4 {
            return Err("deriv1_boundary_functions must have length 4");
        }
        if let Some(ref edd_val) = deriv2_boundary_functions {
            if edd_val.len() != 4 {
                return Err("deriv2_boundary_functions must have length 4");
            }
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
        Ok(map)
    }

    /// Calculates the position x(r,s) at reference location (r,s)
    ///
    /// # Arguments
    ///
    /// * `x` - position vector (output)
    /// * `r` - reference coordinate r (input)
    /// * `s` - reference coordinate s (input)
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

    /// Calculates the position x(r,s) at reference location (r,s) and its derivatives
    ///
    /// # Arguments
    ///
    /// * `x` - position vector (output)
    /// * `dx_dr` - derivative of position with respect to r (output)
    /// * `dx_ds` - derivative of position with respect to s (output)
    /// * `d2x_dr2` - 2nd derivative of position with respect to r (output, optional)
    /// * `d2x_ds2` - 2nd derivative of position with respect to s (output, optional)
    /// * `d2x_drs` - mixed 2nd derivative of position with respect to r and s (output, optional)
    /// * `r` - reference coordinate r (input)
    /// * `s` - reference coordinate s (input)
    ///
    /// # Notes
    ///
    /// If any of the second derivative arguments are `None`, then second derivatives are not computed
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
        if d2x_dr2.is_none() || d2x_ds2.is_none() || d2x_drs.is_none() {
            return;
        }

        // unwrap optional arguments
        let d2x_dr2 = d2x_dr2.unwrap();
        let d2x_ds2 = d2x_ds2.unwrap();
        let d2x_drs = d2x_drs.unwrap();

        // calculate the mixed derivative
        // Note: Even if the 2nd boundary derivatives are nil, the mixed derivative may be non-zero
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

        // compute the 2nd boundary derivatives
        let d2_bry_func = self.deriv2_boundary_functions.as_ref().unwrap();
        (d2_bry_func[0])(&mut self.ddb0s_dss, s);
        (d2_bry_func[1])(&mut self.ddb1s_dss, s);
        (d2_bry_func[2])(&mut self.ddb2r_drr, r);
        (d2_bry_func[3])(&mut self.ddb3r_drr, r);

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

    /// Generates a triangulation of the mapped region
    ///
    /// Returns `(xx, yy, triangles)` where `xx` and `yy` are the x- and y-coordinates of the points,
    /// and `triangles` is a list of triangles (connectivity).
    ///
    /// # Arguments
    ///
    /// * `nr` - number of points along the r-direction
    /// * `ns` - number of points along the s-direction
    /// * `cgl_r` - whether to use Chebyshev-Gauss-Lobatto points in the r-direction
    /// * `cgl_s` - whether to use Chebyshev-Gauss-Lobatto points in the s-direction
    pub fn triangulate(
        &mut self,
        nr: usize,
        ns: usize,
        cgl_r: bool,
        cgl_s: bool,
    ) -> (Vec<f64>, Vec<f64>, Vec<Vec<usize>>) {
        assert!(nr >= 2, "nr must be at least 2");
        assert!(ns >= 2, "ns must be at least 2");
        let np = nr * ns;
        let mut xx = Vec::with_capacity(np);
        let mut yy = Vec::with_capacity(np);
        let mut triangles = Vec::with_capacity((nr - 1) * (ns - 1) * 2);
        let ksi = if cgl_r {
            chebyshev_lobatto_points(nr - 1)
        } else {
            Vector::linspace(-1.0, 1.0, nr).unwrap()
        };
        let eta = if cgl_s {
            chebyshev_lobatto_points(ns - 1)
        } else {
            Vector::linspace(-1.0, 1.0, ns).unwrap()
        };
        let mut x = Vector::new(2);
        for j in 0..ns {
            let s = eta[j];
            for i in 0..nr {
                let r = ksi[i];
                self.point(&mut x, r, s);
                xx.push(x[0]);
                yy.push(x[1]);
                if i > 0 && j > 0 {
                    let m = i + j * nr;
                    triangles.push(vec![m - nr - 1, m - nr, m]);
                    triangles.push(vec![m - nr - 1, m, m - 1]);
                }
            }
        }
        (xx, yy, triangles)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Transfinite2d;
    use crate::TransfiniteSamples;
    use plotpy::{linspace, Canvas, Plot, PolyCode};
    use russell_lab::math::SQRT_2;
    use russell_lab::{vec_approx_eq, vec_deriv1_approx_eq, vec_deriv2_approx_eq, Vector};

    const SAVE_FIGURE: bool = false;

    fn draw_lines_2d(canvas: &mut Canvas, map: &mut Transfinite2d, np: usize, dot_size: f64) {
        canvas.set_face_color("None");
        let mut x = Vector::new(2);
        let tt = linspace(-1.0, 1.0, np);
        // lines in r-direction
        for j in 0..np {
            let s = tt[j];
            map.point(&mut x, tt[0], s);
            canvas.polycurve_begin();
            canvas.polycurve_add(x[0], x[1], PolyCode::MoveTo);
            for i in 1..np {
                let r = tt[i];
                map.point(&mut x, r, s);
                canvas.polycurve_add(x[0], x[1], PolyCode::LineTo);
            }
            canvas.polycurve_end(false);
        }
        // lines in s-direction
        for i in 0..np {
            let r = tt[i];
            map.point(&mut x, r, tt[0]);
            canvas.polycurve_begin();
            canvas.polycurve_add(x[0], x[1], PolyCode::MoveTo);
            for j in 1..np {
                let s = tt[j];
                map.point(&mut x, r, s);
                canvas.polycurve_add(x[0], x[1], PolyCode::LineTo);
            }
            canvas.polycurve_end(false);
        }
        // points at corners
        map.point(&mut x, -1.0, -1.0);
        canvas.draw_circle(x[0], x[1], dot_size);
        map.point(&mut x, 1.0, -1.0);
        canvas.draw_circle(x[0], x[1], dot_size);
        map.point(&mut x, 1.0, 1.0);
        canvas.draw_circle(x[0], x[1], dot_size);
        map.point(&mut x, -1.0, 1.0);
        canvas.draw_circle(x[0], x[1], dot_size);
        // point in the center
        map.point(&mut x, 0.0, 0.0);
        canvas.draw_circle(x[0], x[1], dot_size);
    }

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
    fn transfinite_2d_works_quadrilateral() {
        // allocate transfinite map
        let xa = &[1.0, 0.0];
        let xb = &[6.0, 4.0];
        let xc = &[1.0, 6.0];
        let xd = &[0.0, 5.0];
        let mut map = TransfiniteSamples::quadrilateral_2d(xa, xb, xc, xd);

        // check corners
        let (p0, p1, p2, p3) = map.get_corners();
        vec_approx_eq(&p0, xa, 1e-15);
        vec_approx_eq(&p1, xb, 1e-15);
        vec_approx_eq(&p2, xc, 1e-15);
        vec_approx_eq(&p3, xd, 1e-15);

        // check derivatives
        check_derivs(&mut map, 1e-10, 1e-8);
    }

    #[test]
    fn transfinite_2d_works_quarter_ring() {
        // allocate transfinite map
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

    #[test]
    fn transfinite_2d_works_half_ring() {
        // allocate transfinite map
        let r_in = 2.0;
        let r_out = 6.0;
        let mut map = TransfiniteSamples::half_ring_2d(r_in, r_out);

        // check corners
        let (p0, p1, p2, p3) = map.get_corners();
        vec_approx_eq(&p0, &[r_in, 0.0], 1e-15);
        vec_approx_eq(&p1, &[r_out, 0.0], 1e-15);
        vec_approx_eq(&p2, &[-r_out, 0.0], 1e-15);
        vec_approx_eq(&p3, &[-r_in, 0.0], 1e-15);

        // check some points
        let mut x = Vector::new(2);
        map.point(&mut x, 0.0, -1.0);
        vec_approx_eq(&x, &[0.5 * (r_in + r_out), 0.0], 1e-15);
        map.point(&mut x, 1.0, 0.0);
        vec_approx_eq(&x, &[0.0, r_out], 1e-15);
        map.point(&mut x, 0.0, 1.0);
        vec_approx_eq(&x, &[-0.5 * (r_in + r_out), 0.0], 1e-15);
        map.point(&mut x, -1.0, 0.0);
        vec_approx_eq(&x, &[0.0, r_in], 1e-15);
        map.point(&mut x, 0.0, 0.0);
        vec_approx_eq(&x, &[0.0, 0.5 * (r_in + r_out)], 1e-15);

        // check derivatives
        check_derivs(&mut map, 1e-8, 1e-8);
    }

    #[test]
    fn transfinite_2d_works_perforated_lozenge() {
        // allocate transfinite map
        let radius = 1.0;
        let diagonal = 3.0;
        let mut map = TransfiniteSamples::quarter_perforated_lozenge_2d(radius, diagonal);

        // check corners
        let (p0, p1, p2, p3) = map.get_corners();
        vec_approx_eq(&p0, &[radius, 0.0], 1e-15);
        vec_approx_eq(&p1, &[diagonal, 0.0], 1e-15);
        vec_approx_eq(&p2, &[0.0, diagonal], 1e-15);
        vec_approx_eq(&p3, &[0.0, radius], 1e-15);

        // check some points
        let mut x = Vector::new(2);
        map.point(&mut x, 0.0, -1.0);
        vec_approx_eq(&x, &[radius + (diagonal - radius) / 2.0, 0.0], 1e-15);
        map.point(&mut x, 1.0, 0.0);
        vec_approx_eq(&x, &[diagonal / 2.0, diagonal / 2.0], 1e-15);
        map.point(&mut x, 0.0, 1.0);
        vec_approx_eq(&x, &[0.0, radius + (diagonal - radius) / 2.0], 1e-15);
        map.point(&mut x, -1.0, 0.0);
        vec_approx_eq(&x, &[radius / SQRT_2, radius / SQRT_2], 1e-15);
        map.point(&mut x, 0.0, 0.0);
        vec_approx_eq(
            &x,
            &[
                (radius / SQRT_2 + diagonal / 2.0) / 2.0,
                (radius / SQRT_2 + diagonal / 2.0) / 2.0,
            ],
            1e-15,
        );

        // check derivatives
        check_derivs(&mut map, 1e-10, 1e-8);
    }

    #[test]
    fn transfinite_2d_captures_errors() {
        assert_eq!(
            Transfinite2d::new(vec![], vec![], None).err(),
            Some("boundary_functions must have length 4")
        );
        assert_eq!(
            Transfinite2d::new(
                vec![
                    Box::new(|_, _| {}),
                    Box::new(|_, _| {}),
                    Box::new(|_, _| {}),
                    Box::new(|_, _| {})
                ],
                vec![Box::new(|_, _| {})],
                None
            )
            .err(),
            Some("deriv1_boundary_functions must have length 4")
        );
        assert_eq!(
            Transfinite2d::new(
                vec![
                    Box::new(|_, _| {}),
                    Box::new(|_, _| {}),
                    Box::new(|_, _| {}),
                    Box::new(|_, _| {})
                ],
                vec![
                    Box::new(|_, _| {}),
                    Box::new(|_, _| {}),
                    Box::new(|_, _| {}),
                    Box::new(|_, _| {})
                ],
                Some(vec![Box::new(|_, _| {})])
            )
            .err(),
            Some("deriv2_boundary_functions must have length 4")
        );
    }

    #[test]
    fn triangulation_works() {
        // allocate transfinite map
        let xa = &[1.0, 0.0];
        let xb = &[6.0, 4.0];
        let xc = &[1.0, 6.0];
        let xd = &[0.0, 5.0];
        let mut map = TransfiniteSamples::quadrilateral_2d(xa, xb, xc, xd);

        // triangulate
        // let (xx, yy, triangles) = map.triangulate(4, 5, true, false);
        let (xx, yy, triangles) = map.triangulate(2, 2, true, false);

        // check sizes
        assert_eq!(&xx, &[1.0, 6.0, 0.0, 1.0]);
        assert_eq!(&yy, &[0.0, 4.0, 5.0, 6.0]);
        assert_eq!(triangles[0], vec![0, 1, 3]);
        assert_eq!(triangles[1], vec![0, 3, 2]);

        // draw
        if SAVE_FIGURE {
            let mut canvas = Canvas::new();
            let mut canvas_tri = Canvas::new();
            canvas_tri.set_edge_color("green").set_line_width(2.0);
            canvas_tri.draw_triangles(&xx, &yy, &triangles);
            draw_lines_2d(&mut canvas, &mut map, 21, 0.03);
            let mut plot = Plot::new();
            plot.add(&canvas)
                .add(&canvas_tri)
                .set_range(0.0, 6.5, 0.0, 6.5)
                .set_equal_axes(true)
                .set_figure_size_points(600.0, 600.0)
                .save("/tmp/russell_pde/test_triangulation_works.svg")
                .unwrap();
        }
    }
}
