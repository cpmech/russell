use super::{FnVec1Param1, FnVec1Param2, FnVec2Param2, FnVec3Param2, Transfinite2d, Transfinite3d};
use russell_lab::Vector;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

pub struct TransfiniteSamples {}

impl TransfiniteSamples {
    /// Generates a transfinite mapping of a quadrilateral
    ///
    /// A,B,C,D -- the four corners (counter-clockwise order)
    ///
    /// ```text
    ///             Γ3(r)
    ///           D───────C
    ///           │       │
    ///      Γ0(s)│       │Γ1(s)
    ///           │       │
    ///           A───────B
    ///             Γ2(r)
    /// ```
    ///
    /// Note that: r ϵ [-1,+1] and s ϵ [-1,+1]
    pub fn quadrilateral_2d(xa: &[f64], xb: &[f64], xc: &[f64], xd: &[f64]) -> Transfinite2d {
        let (xa_0, xa_1) = (xa[0], xa[1]);
        let (xb_0, xb_1) = (xb[0], xb[1]);
        let (xc_0, xc_1) = (xc[0], xc[1]);
        let (xd_0, xd_1) = (xd[0], xd[1]);

        let (scale0_0, scale0_1) = ((xd_0 - xa_0) / 2.0, (xd_1 - xa_1) / 2.0);
        let (scale1_0, scale1_1) = ((xc_0 - xb_0) / 2.0, (xc_1 - xb_1) / 2.0);
        let (scale2_0, scale2_1) = ((xb_0 - xa_0) / 2.0, (xb_1 - xa_1) / 2.0);
        let (scale3_0, scale3_1) = ((xc_0 - xd_0) / 2.0, (xc_1 - xd_1) / 2.0);

        let boundary_functions: Vec<FnVec1Param1> = vec![
            // Γ0(s) with s ϵ [-1,+1]
            Box::new(move |x, s| {
                x[0] = xa_0 + (1.0 + s) * scale0_0;
                x[1] = xa_1 + (1.0 + s) * scale0_1;
            }),
            // Γ1(s) with s ϵ [-1,+1]
            Box::new(move |x, s| {
                x[0] = xb_0 + (1.0 + s) * scale1_0;
                x[1] = xb_1 + (1.0 + s) * scale1_1;
            }),
            // Γ2(r) with r ϵ [-1,+1]
            Box::new(move |x, r| {
                x[0] = xa_0 + (1.0 + r) * scale2_0;
                x[1] = xa_1 + (1.0 + r) * scale2_1;
            }),
            // Γ3(r) with r ϵ [-1,+1]
            Box::new(move |x, r| {
                x[0] = xd_0 + (1.0 + r) * scale3_0;
                x[1] = xd_1 + (1.0 + r) * scale3_1;
            }),
        ];

        let deriv1_boundary_functions: Vec<FnVec1Param1> = vec![
            // dΓ0/ds
            Box::new(move |dx_ds, _| {
                dx_ds[0] = scale0_0;
                dx_ds[1] = scale0_1;
            }),
            // dΓ1/ds
            Box::new(move |dx_ds, _| {
                dx_ds[0] = scale1_0;
                dx_ds[1] = scale1_1;
            }),
            // dΓ2/dr
            Box::new(move |dx_dr, _| {
                dx_dr[0] = scale2_0;
                dx_dr[1] = scale2_1;
            }),
            // dΓ3/dr
            Box::new(move |dx_dr, _| {
                dx_dr[0] = scale3_0;
                dx_dr[1] = scale3_1;
            }),
        ];

        Transfinite2d::new(boundary_functions, deriv1_boundary_functions, None).unwrap()
    }

    /// Generates a transfinite mapping of a quarter of a ring centered @ (0,0)
    ///
    /// ```text
    ///      ,- ,
    ///    B3|    ' ,B1       B0(s)
    ///      |        ,       B1(s)
    ///      ''-.B0    ,      B2(r)
    ///          \      ,     B3(r)
    ///      .    |_B2__,
    ///      |←a →|
    ///      |←    b   →|
    /// ```
    ///
    /// a -- inner radius
    /// b -- outer radius
    pub fn quarter_ring_2d(a: f64, b: f64) -> Transfinite2d {
        let boundary_functions: Vec<FnVec1Param1> = vec![
            // B0(s)
            Box::new(move |x, s| {
                let theta = PI * (1.0 + s) / 4.0;
                x[0] = a * theta.cos();
                x[1] = a * theta.sin();
            }),
            // B1(s)
            Box::new(move |x, s| {
                let theta = PI * (1.0 + s) / 4.0;
                x[0] = b * theta.cos();
                x[1] = b * theta.sin();
            }),
            // B2(r)
            Box::new(move |x, r| {
                x[0] = a + 0.5 * (1.0 + r) * (b - a);
                x[1] = 0.0;
            }),
            // B3(r)
            Box::new(move |x, r| {
                x[0] = 0.0;
                x[1] = a + 0.5 * (1.0 + r) * (b - a);
            }),
        ];

        let deriv1_boundary_functions: Vec<FnVec1Param1> = vec![
            // dB0/ds
            Box::new(move |dx_ds, s| {
                let theta = PI * (1.0 + s) / 4.0;
                dx_ds[0] = -a * theta.sin() * PI / 4.0;
                dx_ds[1] = a * theta.cos() * PI / 4.0;
            }),
            // dB1/ds
            Box::new(move |dx_ds, s| {
                let theta = PI * (1.0 + s) / 4.0;
                dx_ds[0] = -b * theta.sin() * PI / 4.0;
                dx_ds[1] = b * theta.cos() * PI / 4.0;
            }),
            // dB2/dr
            Box::new(move |dx_dr, _| {
                dx_dr[0] = 0.5 * (b - a);
                dx_dr[1] = 0.0;
            }),
            // dB3/dr
            Box::new(move |dx_dr, _| {
                dx_dr[0] = 0.0;
                dx_dr[1] = 0.5 * (b - a);
            }),
        ];

        let deriv2_boundary_functions: Vec<FnVec1Param1> = vec![
            // d²B0/ds²
            Box::new(move |d2x_ds2, s| {
                let theta = PI * (1.0 + s) / 4.0;
                d2x_ds2[0] = -a * theta.cos() * PI * PI / 16.0;
                d2x_ds2[1] = -a * theta.sin() * PI * PI / 16.0;
            }),
            // d²B1/ds²
            Box::new(move |d2x_ds2, s| {
                let theta = PI * (1.0 + s) / 4.0;
                d2x_ds2[0] = -b * theta.cos() * PI * PI / 16.0;
                d2x_ds2[1] = -b * theta.sin() * PI * PI / 16.0;
            }),
            // d²B2/dr²
            Box::new(move |d2x_dr2, _| {
                d2x_dr2[0] = 0.0;
                d2x_dr2[1] = 0.0;
            }),
            // d²B3/dr²
            Box::new(move |d2x_dr2, _| {
                d2x_dr2[0] = 0.0;
                d2x_dr2[1] = 0.0;
            }),
        ];

        Transfinite2d::new(
            boundary_functions,
            deriv1_boundary_functions,
            Some(deriv2_boundary_functions),
        )
        .unwrap()
    }

    /// Generates a transfinite mapping of a half of a ring centered @ (0,0)
    ///
    /// ```text
    ///                 B1
    ///               , - - ,
    ///           , '         ' ,        B0(s)
    ///         ,       B0        ,      B1(s)
    ///        ,      .-'''-.      ,     B2(r)
    ///       ,      /       \      ,    B3(r)
    ///       ,_B3__|    .    |_B2__,
    ///                  |←a →|
    ///                  |←    b   →|
    /// ```
    ///
    /// a -- inner radius
    /// b -- outer radius
    pub fn half_ring_2d(a: f64, b: f64) -> Transfinite2d {
        let boundary_functions: Vec<FnVec1Param1> = vec![
            // B0(s)
            Box::new(move |x, s| {
                let theta = PI * (s + 1.0) / 2.0;
                x[0] = a * theta.cos();
                x[1] = a * theta.sin();
            }),
            // B1(s)
            Box::new(move |x, s| {
                let theta = PI * (s + 1.0) / 2.0;
                x[0] = b * theta.cos();
                x[1] = b * theta.sin();
            }),
            // B2(r)
            Box::new(move |x, r| {
                x[0] = a + (b - a) * (r + 1.0) / 2.0;
                x[1] = 0.0;
            }),
            // B3(r)
            Box::new(move |x, r| {
                x[0] = -a - (b - a) * (r + 1.0) / 2.0;
                x[1] = 0.0;
            }),
        ];

        let deriv1_boundary_functions: Vec<FnVec1Param1> = vec![
            // dB0/ds
            Box::new(move |dx_ds, s| {
                let theta = PI * (s + 1.0) / 2.0;
                dx_ds[0] = -a * theta.sin() * PI / 2.0;
                dx_ds[1] = a * theta.cos() * PI / 2.0;
            }),
            // dB1/ds
            Box::new(move |dx_ds, s| {
                let theta = PI * (s + 1.0) / 2.0;
                dx_ds[0] = -b * theta.sin() * PI / 2.0;
                dx_ds[1] = b * theta.cos() * PI / 2.0;
            }),
            // dB2/dr
            Box::new(move |dx_dr, _| {
                dx_dr[0] = (b - a) / 2.0;
                dx_dr[1] = 0.0;
            }),
            // dB3/dr
            Box::new(move |dx_dr, _| {
                dx_dr[0] = -(b - a) / 2.0;
                dx_dr[1] = 0.0;
            }),
        ];

        let deriv2_boundary_functions: Vec<FnVec1Param1> = vec![
            // d²B0/ds²
            Box::new(move |d2x_ds2, s| {
                let theta = PI * (s + 1.0) / 2.0;
                d2x_ds2[0] = -a * theta.cos() * PI * PI / 4.0;
                d2x_ds2[1] = -a * theta.sin() * PI * PI / 4.0;
            }),
            // d²B1/ds²
            Box::new(move |d2x_ds2, s| {
                let theta = PI * (s + 1.0) / 2.0;
                d2x_ds2[0] = -b * theta.cos() * PI * PI / 4.0;
                d2x_ds2[1] = -b * theta.sin() * PI * PI / 4.0;
            }),
            // d²B2/dr²
            Box::new(move |d2x_dr2, _| {
                d2x_dr2[0] = 0.0;
                d2x_dr2[1] = 0.0;
            }),
            // d²B3/dr²
            Box::new(move |d2x_dr2, _| {
                d2x_dr2[0] = 0.0;
                d2x_dr2[1] = 0.0;
            }),
        ];

        Transfinite2d::new(
            boundary_functions,
            deriv1_boundary_functions,
            Some(deriv2_boundary_functions),
        )
        .unwrap()
    }

    /// Generates a transfinite mapping of a quarter of a perforated lozenge
    /// (diamond shape) centered @ (0,0)
    ///
    /// a -- inner radius
    /// b -- diagonal of lozenge (diamond)
    pub fn quarter_perforated_lozenge_2d(a: f64, b: f64) -> Transfinite2d {
        let boundary_functions: Vec<FnVec1Param1> = vec![
            // B0(s)
            Box::new(move |x, s| {
                let theta = PI * (1.0 + s) / 4.0;
                x[0] = a * theta.cos();
                x[1] = a * theta.sin();
            }),
            // B1(s)
            Box::new(move |x, s| {
                x[0] = b * 0.5 * (1.0 - s);
                x[1] = b * 0.5 * (1.0 + s);
            }),
            // B2(r)
            Box::new(move |x, r| {
                x[0] = a + 0.5 * (1.0 + r) * (b - a);
                x[1] = 0.0;
            }),
            // B3(r)
            Box::new(move |x, r| {
                x[0] = 0.0;
                x[1] = a + 0.5 * (1.0 + r) * (b - a);
            }),
        ];

        let deriv1_boundary_functions: Vec<FnVec1Param1> = vec![
            // dB0/ds
            Box::new(move |dx_ds, s| {
                let theta = PI * (1.0 + s) / 4.0;
                dx_ds[0] = -a * theta.sin() * PI / 4.0;
                dx_ds[1] = a * theta.cos() * PI / 4.0;
            }),
            // dB1/ds
            Box::new(move |dx_ds, _| {
                dx_ds[0] = -b * 0.5;
                dx_ds[1] = b * 0.5;
            }),
            // dB2/dr
            Box::new(move |dx_dr, _| {
                dx_dr[0] = 0.5 * (b - a);
                dx_dr[1] = 0.0;
            }),
            // dB3/dr
            Box::new(move |dx_dr, _| {
                dx_dr[0] = 0.0;
                dx_dr[1] = 0.5 * (b - a);
            }),
        ];

        let deriv2_boundary_functions: Vec<FnVec1Param1> = vec![
            // d²B0/ds²
            Box::new(move |d2x_ds2, s| {
                let theta = PI * (1.0 + s) / 4.0;
                d2x_ds2[0] = -a * theta.cos() * PI * PI / 16.0;
                d2x_ds2[1] = -a * theta.sin() * PI * PI / 16.0;
            }),
            // d²B1/ds²
            Box::new(move |d2x_ds2, _| {
                d2x_ds2[0] = 0.0;
                d2x_ds2[1] = 0.0;
            }),
            // d²B2/dr²
            Box::new(move |d2x_dr2, _| {
                d2x_dr2[0] = 0.0;
                d2x_dr2[1] = 0.0;
            }),
            // d²B3/dr²
            Box::new(move |d2x_dr2, _| {
                d2x_dr2[0] = 0.0;
                d2x_dr2[1] = 0.0;
            }),
        ];

        Transfinite2d::new(
            boundary_functions,
            deriv1_boundary_functions,
            Some(deriv2_boundary_functions),
        )
        .unwrap()
    }

    /// Generates a transfinite mapping of a "brick"
    pub fn brick_3d(lx: f64, ly: f64, lz: f64) -> Transfinite3d {
        let boundary_functions: Vec<FnVec1Param2> = vec![
            // B0(s,t)
            Box::new(move |x, s, t| {
                x[0] = 0.0;
                x[1] = (1.0 + s) * ly / 2.0;
                x[2] = (1.0 + t) * lz / 2.0;
            }),
            // B1(s,t)
            Box::new(move |x, s, t| {
                x[0] = lx;
                x[1] = (1.0 + s) * ly / 2.0;
                x[2] = (1.0 + t) * lz / 2.0;
            }),
            // B2(r,t)
            Box::new(move |x, r, t| {
                x[0] = (1.0 + r) * lx / 2.0;
                x[1] = 0.0;
                x[2] = (1.0 + t) * lz / 2.0;
            }),
            // B3(r,t)
            Box::new(move |x, r, t| {
                x[0] = (1.0 + r) * lx / 2.0;
                x[1] = ly;
                x[2] = (1.0 + t) * lz / 2.0;
            }),
            // B4(r,s)
            Box::new(move |x, r, s| {
                x[0] = (1.0 + r) * lx / 2.0;
                x[1] = (1.0 + s) * ly / 2.0;
                x[2] = 0.0;
            }),
            // B5(r,s)
            Box::new(move |x, r, s| {
                x[0] = (1.0 + r) * lx / 2.0;
                x[1] = (1.0 + s) * ly / 2.0;
                x[2] = lz;
            }),
        ];

        let deriv1_boundary_functions: Vec<FnVec2Param2> = vec![
            // Bd0(s,t)
            Box::new(move |dx_ds, dx_dt, _, _| {
                dx_ds[0] = 0.0;
                dx_ds[1] = ly / 2.0;
                dx_ds[2] = 0.0;
                dx_dt[0] = 0.0;
                dx_dt[1] = 0.0;
                dx_dt[2] = lz / 2.0;
            }),
            // Bd1(s,t)
            Box::new(move |dx_ds, dx_dt, _, _| {
                dx_ds[0] = 0.0;
                dx_ds[1] = ly / 2.0;
                dx_ds[2] = 0.0;
                dx_dt[0] = 0.0;
                dx_dt[1] = 0.0;
                dx_dt[2] = lz / 2.0;
            }),
            // Bd2(r,t)
            Box::new(move |dx_dr, dx_dt, _, _| {
                dx_dr[0] = lx / 2.0;
                dx_dr[1] = 0.0;
                dx_dr[2] = 0.0;
                dx_dt[0] = 0.0;
                dx_dt[1] = 0.0;
                dx_dt[2] = lz / 2.0;
            }),
            // Bd3(r,t)
            Box::new(move |dx_dr, dx_dt, _, _| {
                dx_dr[0] = lx / 2.0;
                dx_dr[1] = 0.0;
                dx_dr[2] = 0.0;
                dx_dt[0] = 0.0;
                dx_dt[1] = 0.0;
                dx_dt[2] = lz / 2.0;
            }),
            // Bd4(r,s)
            Box::new(move |dx_dr, dx_ds, _, _| {
                dx_dr[0] = lx / 2.0;
                dx_dr[1] = 0.0;
                dx_dr[2] = 0.0;
                dx_ds[0] = 0.0;
                dx_ds[1] = ly / 2.0;
                dx_ds[2] = 0.0;
            }),
            // Bd5(r,s)
            Box::new(move |dx_dr, dx_ds, _, _| {
                dx_dr[0] = lx / 2.0;
                dx_dr[1] = 0.0;
                dx_dr[2] = 0.0;
                dx_ds[0] = 0.0;
                dx_ds[1] = ly / 2.0;
                dx_ds[2] = 0.0;
            }),
        ];

        Transfinite3d::new(boundary_functions, deriv1_boundary_functions, None).unwrap()
    }

    /// Generates a transfinite mapping of a quarter of a 3d ring centered @ (0,0)
    ///
    /// a -- inner radius
    /// b -- outer radius
    /// h -- thickness along x-direction
    pub fn quarter_ring_3d(a: f64, b: f64, h: f64) -> Transfinite3d {
        let surf = Arc::new(Mutex::new(TransfiniteSamples::quarter_ring_2d(a, b)));

        let surf_0 = surf.clone();
        let surf_1 = surf.clone();
        let boundary_functions: Vec<FnVec1Param2> = vec![
            // B0(s,t)
            Box::new(move |x, s, t| {
                let mut surf = surf_0.lock().unwrap();
                let mut x2d = Vector::new(2);
                surf.point(&mut x2d, s, t);
                x[0] = 0.0;
                x[1] = x2d[0];
                x[2] = x2d[1];
            }),
            // B1(s,t)
            Box::new(move |x, s, t| {
                let mut surf = surf_1.lock().unwrap();
                let mut x2d = Vector::new(2);
                surf.point(&mut x2d, s, t);
                x[0] = h;
                x[1] = x2d[0];
                x[2] = x2d[1];
            }),
            // B2(r,t)
            Box::new(move |x, r, t| {
                let theta = (1.0 + t) * PI / 4.0;
                x[0] = (1.0 + r) * h / 2.0;
                x[1] = a * theta.cos();
                x[2] = a * theta.sin();
            }),
            // B3(r,t)
            Box::new(move |x, r, t| {
                let theta = (1.0 + t) * PI / 4.0;
                x[0] = (1.0 + r) * h / 2.0;
                x[1] = b * theta.cos();
                x[2] = b * theta.sin();
            }),
            // B4(r,s)
            Box::new(move |x, r, s| {
                x[0] = (1.0 + r) * h / 2.0;
                x[1] = a + (1.0 + s) * (b - a) / 2.0;
                x[2] = 0.0;
            }),
            // B5(r,s)
            Box::new(move |x, r, s| {
                x[0] = (1.0 + r) * h / 2.0;
                x[1] = 0.0;
                x[2] = a + (1.0 + s) * (b - a) / 2.0;
            }),
        ];

        let surf_d0 = surf.clone();
        let surf_d1 = surf.clone();
        let deriv1_boundary_functions: Vec<FnVec2Param2> = vec![
            // Bd0(s,t)
            Box::new(move |dx_ds, dx_dt, s, t| {
                let mut surf = surf_d0.lock().unwrap();
                let mut tmp = Vector::new(2);
                let mut dx_dr_2d = Vector::new(2);
                let mut dx_ds_2d = Vector::new(2);
                surf.point_and_derivs(&mut tmp, &mut dx_dr_2d, &mut dx_ds_2d, None, None, None, s, t);
                dx_ds[0] = 0.0;
                dx_ds[1] = dx_dr_2d[0];
                dx_ds[2] = dx_dr_2d[1];
                dx_dt[0] = 0.0;
                dx_dt[1] = dx_ds_2d[0];
                dx_dt[2] = dx_ds_2d[1];
            }),
            // Bd1(s,t)
            Box::new(move |dx_ds, dx_dt, s, t| {
                let mut surf = surf_d1.lock().unwrap();
                let mut tmp = Vector::new(2);
                let mut dx_dr_2d = Vector::new(2);
                let mut dx_ds_2d = Vector::new(2);
                surf.point_and_derivs(&mut tmp, &mut dx_dr_2d, &mut dx_ds_2d, None, None, None, s, t);
                dx_ds[0] = 0.0;
                dx_ds[1] = dx_dr_2d[0];
                dx_ds[2] = dx_dr_2d[1];
                dx_dt[0] = 0.0;
                dx_dt[1] = dx_ds_2d[0];
                dx_dt[2] = dx_ds_2d[1];
            }),
            // Bd2(r,t)
            Box::new(move |dx_dr, dx_dt, _, t| {
                let theta = (1.0 + t) * PI / 4.0;
                dx_dr[0] = h / 2.0;
                dx_dr[1] = 0.0;
                dx_dr[2] = 0.0;
                dx_dt[0] = 0.0;
                dx_dt[1] = -a * theta.sin() * PI / 4.0;
                dx_dt[2] = a * theta.cos() * PI / 4.0;
            }),
            // Bd3(r,t)
            Box::new(move |dx_dr, dx_dt, _, t| {
                let theta = (1.0 + t) * PI / 4.0;
                dx_dr[0] = h / 2.0;
                dx_dr[1] = 0.0;
                dx_dr[2] = 0.0;
                dx_dt[0] = 0.0;
                dx_dt[1] = -b * theta.sin() * PI / 4.0;
                dx_dt[2] = b * theta.cos() * PI / 4.0;
            }),
            // Bd4(r,s)
            Box::new(move |dx_dr, dx_ds, _, _| {
                dx_dr[0] = h / 2.0;
                dx_dr[1] = 0.0;
                dx_dr[2] = 0.0;
                dx_ds[0] = 0.0;
                dx_ds[1] = (b - a) / 2.0;
                dx_ds[2] = 0.0;
            }),
            // Bd5(r,s)
            Box::new(move |dx_dr, dx_ds, _, _| {
                dx_dr[0] = h / 2.0;
                dx_dr[1] = 0.0;
                dx_dr[2] = 0.0;
                dx_ds[0] = 0.0;
                dx_ds[1] = 0.0;
                dx_ds[2] = (b - a) / 2.0;
            }),
        ];

        let surf_dd0 = surf.clone();
        let surf_dd1 = surf.clone();
        let deriv2_boundary_functions: Vec<FnVec3Param2> = vec![
            // Bdd0(s,t)
            Box::new(move |d2x_ds2, d2x_dt2, d2x_dst, s, t| {
                let mut surf = surf_dd0.lock().unwrap();
                let mut x_tmp = Vector::new(2);
                let mut dx_dr_tmp = Vector::new(2);
                let mut dx_ds_tmp = Vector::new(2);
                let mut d2x_dr2_2d = Vector::new(2);
                let mut d2x_ds2_2d = Vector::new(2);
                let mut d2x_drs_2d = Vector::new(2);
                surf.point_and_derivs(
                    &mut x_tmp,
                    &mut dx_dr_tmp,
                    &mut dx_ds_tmp,
                    Some(&mut d2x_dr2_2d),
                    Some(&mut d2x_ds2_2d),
                    Some(&mut d2x_drs_2d),
                    s,
                    t,
                );
                d2x_ds2[0] = 0.0;
                d2x_ds2[1] = d2x_dr2_2d[0];
                d2x_ds2[2] = d2x_dr2_2d[1];

                d2x_dt2[0] = 0.0;
                d2x_dt2[1] = d2x_ds2_2d[0];
                d2x_dt2[2] = d2x_ds2_2d[1];

                d2x_dst[0] = 0.0;
                d2x_dst[1] = d2x_drs_2d[0];
                d2x_dst[2] = d2x_drs_2d[1];
            }),
            // Bdd1(s,t)
            Box::new(move |d2x_ds2, d2x_dt2, d2x_dst, s, t| {
                let mut surf = surf_dd1.lock().unwrap();
                let mut x_tmp = Vector::new(2);
                let mut dx_dr_tmp = Vector::new(2);
                let mut dx_ds_tmp = Vector::new(2);
                let mut d2x_dr2_2d = Vector::new(2);
                let mut d2x_ds_2d = Vector::new(2);
                let mut d2x_drs_2d = Vector::new(2);
                surf.point_and_derivs(
                    &mut x_tmp,
                    &mut dx_dr_tmp,
                    &mut dx_ds_tmp,
                    Some(&mut d2x_dr2_2d),
                    Some(&mut d2x_ds_2d),
                    Some(&mut d2x_drs_2d),
                    s,
                    t,
                );
                d2x_ds2[0] = 0.0;
                d2x_ds2[1] = d2x_dr2_2d[0];
                d2x_ds2[2] = d2x_dr2_2d[1];

                d2x_dt2[0] = 0.0;
                d2x_dt2[1] = d2x_ds_2d[0];
                d2x_dt2[2] = d2x_ds_2d[1];

                d2x_dst[0] = 0.0;
                d2x_dst[1] = d2x_drs_2d[0];
                d2x_dst[2] = d2x_drs_2d[1];
            }),
            // Bdd2(r,t)
            Box::new(move |d2x_dr2, d2x_dt2, d2x_drt, _, t| {
                let theta = (1.0 + t) * PI / 4.0;
                d2x_dr2[0] = 0.0;
                d2x_dr2[1] = 0.0;
                d2x_dr2[2] = 0.0;

                d2x_dt2[0] = 0.0;
                d2x_dt2[1] = -a * theta.cos() * PI * PI / 16.0;
                d2x_dt2[2] = -a * theta.sin() * PI * PI / 16.0;

                d2x_drt[0] = 0.0;
                d2x_drt[1] = 0.0;
                d2x_drt[2] = 0.0;
            }),
            // Bdd3(r,t)
            Box::new(move |d2x_dr2, d2x_dt2, d2x_drt, _, t| {
                let theta = (1.0 + t) * PI / 4.0;
                d2x_dr2[0] = 0.0;
                d2x_dr2[1] = 0.0;
                d2x_dr2[2] = 0.0;

                d2x_dt2[0] = 0.0;
                d2x_dt2[1] = -b * theta.cos() * PI * PI / 16.0;
                d2x_dt2[2] = -b * theta.sin() * PI * PI / 16.0;

                d2x_drt[0] = 0.0;
                d2x_drt[1] = 0.0;
                d2x_drt[2] = 0.0;
            }),
            // Bdd4(r,s)
            Box::new(move |d2x_dr2, d2x_ds2, d2x_drs, _, _| {
                d2x_dr2[0] = 0.0;
                d2x_dr2[1] = 0.0;
                d2x_dr2[2] = 0.0;

                d2x_ds2[0] = 0.0;
                d2x_ds2[1] = 0.0;
                d2x_ds2[2] = 0.0;

                d2x_drs[0] = 0.0;
                d2x_drs[1] = 0.0;
                d2x_drs[2] = 0.0;
            }),
            // Bdd5(r,s)
            Box::new(move |d2x_dr2, d2x_ds2, d2x_drs, _, _| {
                d2x_dr2[0] = 0.0;
                d2x_dr2[1] = 0.0;
                d2x_dr2[2] = 0.0;

                d2x_ds2[0] = 0.0;
                d2x_ds2[1] = 0.0;
                d2x_ds2[2] = 0.0;

                d2x_drs[0] = 0.0;
                d2x_drs[1] = 0.0;
                d2x_drs[2] = 0.0;
            }),
        ];

        Transfinite3d::new(
            boundary_functions,
            deriv1_boundary_functions,
            Some(deriv2_boundary_functions),
        )
        .unwrap()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::TransfiniteSamples;
    use crate::{Transfinite2d, Transfinite3d};
    use plotpy::{linspace, Canvas, Plot, PolyCode};
    use russell_lab::{vec_approx_eq, Vector};

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

    fn draw_surface_lines_3d(canvas: &mut Canvas, map: &mut Transfinite3d, np: usize) {
        canvas.set_face_color("None");
        let mut x = Vector::new(3);
        let param = linspace(-1.0, 1.0, np);

        // surface @ r_min and r_max
        for r in [-1.0, 1.0] {
            // lines along s (varying s, fixed t)
            for j in 0..np {
                let t = param[j];
                map.point(&mut x, r, -1.0, t);
                canvas.polyline_3d_begin();
                canvas.polyline_3d_add(x[0], x[1], x[2]);
                for i in 1..np {
                    let s = param[i];
                    map.point(&mut x, r, s, t);
                    canvas.polyline_3d_add(x[0], x[1], x[2]);
                }
                canvas.polyline_3d_end();
            }
            // lines along t (varying t, fixed s)
            for i in 0..np {
                let s = param[i];
                map.point(&mut x, r, s, -1.0);
                canvas.polyline_3d_begin();
                canvas.polyline_3d_add(x[0], x[1], x[2]);
                for j in 1..np {
                    let t = param[j];
                    map.point(&mut x, r, s, t);
                    canvas.polyline_3d_add(x[0], x[1], x[2]);
                }
                canvas.polyline_3d_end();
            }
        }

        // surface @ s_min and s_max
        for s in [-1.0, 1.0] {
            // lines along r (varying r, fixed t)
            for j in 0..np {
                let t = param[j];
                map.point(&mut x, -1.0, s, t);
                canvas.polyline_3d_begin();
                canvas.polyline_3d_add(x[0], x[1], x[2]);
                for i in 1..np {
                    let r = param[i];
                    map.point(&mut x, r, s, t);
                    canvas.polyline_3d_add(x[0], x[1], x[2]);
                }
                canvas.polyline_3d_end();
            }
            // lines along t (varying t, fixed r)
            for i in 0..np {
                let r = param[i];
                map.point(&mut x, r, s, -1.0);
                canvas.polyline_3d_begin();
                canvas.polyline_3d_add(x[0], x[1], x[2]);
                for j in 1..np {
                    let t = param[j];
                    map.point(&mut x, r, s, t);
                    canvas.polyline_3d_add(x[0], x[1], x[2]);
                }
                canvas.polyline_3d_end();
            }
        }

        // surface @ t_min and t_max
        for t in [-1.0, 1.0] {
            // lines along r (varying r, fixed s)
            for j in 0..np {
                let s = param[j];
                map.point(&mut x, param[0], s, t);
                canvas.polyline_3d_begin();
                canvas.polyline_3d_add(x[0], x[1], x[2]);
                for i in 1..np {
                    let r = param[i];
                    map.point(&mut x, r, s, t);
                    canvas.polyline_3d_add(x[0], x[1], x[2]);
                }
                canvas.polyline_3d_end();
            }
            // lines along s (varying s, fixed r)
            for i in 0..np {
                let r = param[i];
                map.point(&mut x, r, param[0], t);
                canvas.polyline_3d_begin();
                canvas.polyline_3d_add(x[0], x[1], x[2]);
                for j in 1..np {
                    let s = param[j];
                    map.point(&mut x, r, s, t);
                    canvas.polyline_3d_add(x[0], x[1], x[2]);
                }
                canvas.polyline_3d_end();
            }
        }
    }

    #[test]
    fn test_quadrilateral_2d() {
        let xa = &[1.0, 0.0];
        let xb = &[6.0, 4.0];
        let xc = &[1.0, 6.0];
        let xd = &[0.0, 5.0];
        let mut map = TransfiniteSamples::quadrilateral_2d(xa, xb, xc, xd);

        let mut x = Vector::new(2);

        map.point(&mut x, -1.0, -1.0);
        vec_approx_eq(&x, xa, 1e-15);

        map.point(&mut x, 1.0, -1.0);
        vec_approx_eq(&x, xb, 1e-15);

        map.point(&mut x, 1.0, 1.0);
        vec_approx_eq(&x, xc, 1e-15);

        map.point(&mut x, -1.0, 1.0);
        vec_approx_eq(&x, xd, 1e-15);

        if SAVE_FIGURE {
            let mut canvas = Canvas::new();
            draw_lines_2d(&mut canvas, &mut map, 21, 0.03);
            let mut plot = Plot::new();
            plot.add(&canvas)
                .set_range(-0.05, 6.05, -0.05, 6.05)
                .set_equal_axes(true)
                .set_figure_size_points(600.0, 600.0)
                .save("/tmp/russell_pde/test_quadrilateral_2d.svg")
                .unwrap();
        }
    }

    #[test]
    fn test_quarter_ring_2d() {
        let r_in = 1.0;
        let r_out = 6.0;
        let mut map = TransfiniteSamples::quarter_ring_2d(r_in, r_out);

        let mut x = Vector::new(2);

        map.point(&mut x, -1.0, -1.0);
        vec_approx_eq(&x, &[r_in, 0.0], 1e-15);

        map.point(&mut x, 1.0, -1.0);
        vec_approx_eq(&x, &[r_out, 0.0], 1e-15);

        map.point(&mut x, 1.0, 1.0);
        vec_approx_eq(&x, &[0.0, r_out], 1e-15);

        map.point(&mut x, -1.0, 1.0);
        vec_approx_eq(&x, &[0.0, r_in], 1e-15);

        if SAVE_FIGURE {
            let mut circles = Canvas::new();
            circles.set_face_color("None").set_edge_color("red").set_line_width(2.0);
            circles.draw_circle(0.0, 0.0, r_in);
            circles.draw_circle(0.0, 0.0, r_out);
            let mut canvas = Canvas::new();
            draw_lines_2d(&mut canvas, &mut map, 21, 0.03);
            let mut plot = Plot::new();
            plot.add(&circles)
                .add(&canvas)
                .set_range(-1.05, 6.5, -1.05, 6.5)
                .set_equal_axes(true)
                .set_figure_size_points(600.0, 600.0)
                .save("/tmp/russell_pde/test_quarter_ring_2d.svg")
                .unwrap();
        }
    }

    #[test]
    fn test_half_ring_2d() {
        let r_in = 1.0;
        let r_out = 6.0;
        let mut map = TransfiniteSamples::half_ring_2d(r_in, r_out);

        let mut x = Vector::new(2);

        map.point(&mut x, -1.0, -1.0);
        vec_approx_eq(&x, &[r_in, 0.0], 1e-15);

        map.point(&mut x, 1.0, -1.0);
        vec_approx_eq(&x, &[r_out, 0.0], 1e-15);

        map.point(&mut x, 1.0, 1.0);
        vec_approx_eq(&x, &[-r_out, 0.0], 1e-15);

        map.point(&mut x, -1.0, 1.0);
        vec_approx_eq(&x, &[-r_in, 0.0], 1e-15);

        if SAVE_FIGURE {
            let mut circles = Canvas::new();
            circles.set_face_color("None").set_edge_color("red").set_line_width(2.0);
            circles.draw_circle(0.0, 0.0, r_in);
            circles.draw_circle(0.0, 0.0, r_out);
            let mut canvas = Canvas::new();
            draw_lines_2d(&mut canvas, &mut map, 41, 0.03);
            let mut plot = Plot::new();
            plot.add(&circles)
                .add(&canvas)
                .set_range(-6.05, 6.05, -1.05, 6.5)
                .set_equal_axes(true)
                .set_figure_size_points(800.0, 600.0)
                .save("/tmp/russell_pde/test_half_ring_2d.svg")
                .unwrap();
        }
    }

    #[test]
    fn test_quarter_perforated_lozenge_2d() {
        let radius = 1.0;
        let diagonal = 3.0;
        let mut map = TransfiniteSamples::quarter_perforated_lozenge_2d(radius, diagonal);

        let mut x = Vector::new(2);

        map.point(&mut x, -1.0, -1.0);
        vec_approx_eq(&x, &[radius, 0.0], 1e-15);

        map.point(&mut x, 1.0, -1.0);
        vec_approx_eq(&x, &[diagonal, 0.0], 1e-15);

        map.point(&mut x, 1.0, 1.0);
        vec_approx_eq(&x, &[0.0, diagonal], 1e-15);

        map.point(&mut x, -1.0, 1.0);
        vec_approx_eq(&x, &[0.0, radius], 1e-15);

        if SAVE_FIGURE {
            let mut circle = Canvas::new();
            circle.set_face_color("None").set_edge_color("red").set_line_width(2.0);
            circle.draw_circle(0.0, 0.0, radius);
            let mut canvas = Canvas::new();
            draw_lines_2d(&mut canvas, &mut map, 21, 0.02);
            let mut plot = Plot::new();
            plot.add(&circle)
                .add(&canvas)
                .set_range(-1.05, 3.05, -1.05, 3.05)
                .set_equal_axes(true)
                .set_figure_size_points(600.0, 600.0)
                .save("/tmp/russell_pde/test_quarter_perforated_lozenge_2d.svg")
                .unwrap();
        }
    }

    #[test]
    fn test_brick_3d() {
        let (lx, ly, lz) = (2.0, 3.0, 4.0);
        let mut map = TransfiniteSamples::brick_3d(lx, ly, lz);

        let mut x = Vector::new(3);

        // z = 0 //////////

        map.point(&mut x, -1.0, -1.0, -1.0);
        vec_approx_eq(&x, &[0.0, 0.0, 0.0], 1e-15);

        map.point(&mut x, 1.0, -1.0, -1.0);
        vec_approx_eq(&x, &[lx, 0.0, 0.0], 1e-15);

        map.point(&mut x, 1.0, 1.0, -1.0);
        vec_approx_eq(&x, &[lx, ly, 0.0], 1e-15);

        map.point(&mut x, -1.0, 1.0, -1.0);
        vec_approx_eq(&x, &[0.0, ly, 0.0], 1e-15);

        // z = lz //////////

        map.point(&mut x, -1.0, -1.0, 1.0);
        vec_approx_eq(&x, &[0.0, 0.0, lz], 1e-15);

        map.point(&mut x, 1.0, -1.0, 1.0);
        vec_approx_eq(&x, &[lx, 0.0, lz], 1e-15);

        map.point(&mut x, 1.0, 1.0, 1.0);
        vec_approx_eq(&x, &[lx, ly, lz], 1e-15);

        map.point(&mut x, -1.0, 1.0, 1.0);
        vec_approx_eq(&x, &[0.0, ly, lz], 1e-15);

        if SAVE_FIGURE {
            let mut canvas = Canvas::new();
            draw_surface_lines_3d(&mut canvas, &mut map, 11);
            let mut plot = Plot::new();
            canvas.draw_glyph_3d(0.0, 0.0, 4.0);
            plot.add(&canvas)
                .set_camera(30.0, 30.0)
                .set_hide_3d_grid(true)
                .set_equal_axes(true)
                .set_figure_size_points(800.0, 800.0)
                .save("/tmp/russell_pde/test_brick_3d.svg")
                .unwrap();
        }
    }

    #[test]
    fn test_quarter_ring_3d() {
        let r_in = 1.0;
        let r_out = 6.0;
        let thickness = 2.0;
        let mut map = TransfiniteSamples::quarter_ring_3d(r_in, r_out, thickness);

        let mut x = Vector::new(3);

        // t = -1.0 //////////

        map.point(&mut x, -1.0, -1.0, -1.0);
        vec_approx_eq(&x, &[0.0, r_in, 0.0], 1e-15);

        map.point(&mut x, 1.0, -1.0, -1.0);
        vec_approx_eq(&x, &[thickness, r_in, 0.0], 1e-15);

        map.point(&mut x, 1.0, 1.0, -1.0);
        vec_approx_eq(&x, &[thickness, r_out, 0.0], 1e-15);

        map.point(&mut x, -1.0, 1.0, -1.0);
        vec_approx_eq(&x, &[0.0, r_out, 0.0], 1e-15);

        // t = 1.0 //////////

        map.point(&mut x, -1.0, -1.0, 1.0);
        vec_approx_eq(&x, &[0.0, 0.0, r_in], 1e-15);

        map.point(&mut x, 1.0, -1.0, 1.0);
        vec_approx_eq(&x, &[thickness, 0.0, r_in], 1e-15);

        map.point(&mut x, 1.0, 1.0, 1.0);
        vec_approx_eq(&x, &[thickness, 0.0, r_out], 1e-15);

        map.point(&mut x, -1.0, 1.0, 1.0);
        vec_approx_eq(&x, &[0.0, 0.0, r_out], 1e-15);

        if SAVE_FIGURE {
            let mut canvas = Canvas::new();
            draw_surface_lines_3d(&mut canvas, &mut map, 21);
            let mut plot = Plot::new();
            canvas.draw_glyph_3d(0.0, 0.0, r_out);
            plot.add(&canvas)
                .set_camera(30.0, 30.0)
                .set_hide_3d_grid(true)
                .set_equal_axes(true)
                .set_figure_size_points(800.0, 800.0)
                .save("/tmp/russell_pde/test_quarter_ring_3d.svg")
                .unwrap();
        }
    }
}
