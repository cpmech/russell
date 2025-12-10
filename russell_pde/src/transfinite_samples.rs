use crate::transfinite::{Transfinite, Vs, Vss, Vvss, Vvvss};
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
    pub fn quadrilateral_2d(xa: &[f64], xb: &[f64], xc: &[f64], xd: &[f64]) -> Transfinite {
        let (xa_0, xa_1) = (xa[0], xa[1]);
        let (xb_0, xb_1) = (xb[0], xb[1]);
        let (xc_0, xc_1) = (xc[0], xc[1]);
        let (xd_0, xd_1) = (xd[0], xd[1]);

        let (scale0_0, scale0_1) = ((xd_0 - xa_0) / 2.0, (xd_1 - xa_1) / 2.0);
        let (scale1_0, scale1_1) = ((xc_0 - xb_0) / 2.0, (xc_1 - xb_1) / 2.0);
        let (scale2_0, scale2_1) = ((xb_0 - xa_0) / 2.0, (xb_1 - xa_1) / 2.0);
        let (scale3_0, scale3_1) = ((xc_0 - xd_0) / 2.0, (xc_1 - xd_1) / 2.0);

        let boundary_functions: Vec<Vs> = vec![
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

        let deriv1_boundary_functions: Vec<Vs> = vec![
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

        Transfinite::new_2d(boundary_functions, deriv1_boundary_functions, None)
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
    pub fn quarter_ring_2d(a: f64, b: f64) -> Transfinite {
        let boundary_functions: Vec<Vs> = vec![
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

        let deriv1_boundary_functions: Vec<Vs> = vec![
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

        let deriv2_boundary_functions: Vec<Vs> = vec![
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

        Transfinite::new_2d(
            boundary_functions,
            deriv1_boundary_functions,
            Some(deriv2_boundary_functions),
        )
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
    pub fn half_ring_2d(a: f64, b: f64) -> Transfinite {
        let e: Vec<Vs> = vec![
            // B[0](s)
            Box::new(move |x, s| {
                let theta = PI * (s + 1.0) / 2.0;
                x[0] = a * theta.cos();
                x[1] = a * theta.sin();
            }),
            // B[1](s)
            Box::new(move |x, s| {
                let theta = PI * (s + 1.0) / 2.0;
                x[0] = b * theta.cos();
                x[1] = b * theta.sin();
            }),
            // B[2](r)
            Box::new(move |x, r| {
                x[0] = a + (b - a) * (r + 1.0) / 2.0;
                x[1] = 0.0;
            }),
            // B[3](r)
            Box::new(move |x, r| {
                x[0] = -a - (b - a) * (r + 1.0) / 2.0;
                x[1] = 0.0;
            }),
        ];

        let ed: Vec<Vs> = vec![
            // dB[0]/ds
            Box::new(move |dxds, s| {
                let theta = PI * (s + 1.0) / 2.0;
                dxds[0] = -a * theta.sin() * PI / 2.0;
                dxds[1] = a * theta.cos() * PI / 2.0;
            }),
            // dB[1]/ds
            Box::new(move |dxds, s| {
                let theta = PI * (s + 1.0) / 2.0;
                dxds[0] = -b * theta.sin() * PI / 2.0;
                dxds[1] = b * theta.cos() * PI / 2.0;
            }),
            // dB[2]/dr
            Box::new(move |dxdr, _| {
                dxdr[0] = (b - a) / 2.0;
                dxdr[1] = 0.0;
            }),
            // dB[3]/dr
            Box::new(move |dxdr, _| {
                dxdr[0] = -(b - a) / 2.0;
                dxdr[1] = 0.0;
            }),
        ];

        let edd: Vec<Vs> = vec![
            // d²B[0]/ds²
            Box::new(move |ddxdss, s| {
                let theta = PI * (s + 1.0) / 2.0;
                ddxdss[0] = -a * theta.cos() * PI * PI / 4.0;
                ddxdss[1] = -a * theta.sin() * PI * PI / 4.0;
            }),
            // d²B[1]/ds²
            Box::new(move |ddxdss, s| {
                let theta = PI * (s + 1.0) / 2.0;
                ddxdss[0] = -b * theta.cos() * PI * PI / 4.0;
                ddxdss[1] = -b * theta.sin() * PI * PI / 4.0;
            }),
            // d²B[2]/dr²
            Box::new(move |ddxdrr, _| {
                ddxdrr[0] = 0.0;
                ddxdrr[1] = 0.0;
            }),
            // d²B[3]/dr²
            Box::new(move |ddxdrr, _| {
                ddxdrr[0] = 0.0;
                ddxdrr[1] = 0.0;
            }),
        ];

        Transfinite::new_2d(e, ed, Some(edd))
    }

    /// Generates a transfinite mapping of a quarter of a perforated lozenge
    /// (diamond shape) centered @ (0,0)
    ///
    /// a -- inner radius
    /// b -- diagonal of lozenge (diamond)
    pub fn quarter_perforated_lozenge_2d(a: f64, b: f64) -> Transfinite {
        let e: Vec<Vs> = vec![
            // B[0](s)
            Box::new(move |x, s| {
                let theta = PI * (1.0 + s) / 4.0;
                x[0] = a * theta.cos();
                x[1] = a * theta.sin();
            }),
            // B[1](s)
            Box::new(move |x, s| {
                x[0] = b * 0.5 * (1.0 - s);
                x[1] = b * 0.5 * (1.0 + s);
            }),
            // B[2](r)
            Box::new(move |x, r| {
                x[0] = a + 0.5 * (1.0 + r) * (b - a);
                x[1] = 0.0;
            }),
            // B[3](r)
            Box::new(move |x, r| {
                x[0] = 0.0;
                x[1] = a + 0.5 * (1.0 + r) * (b - a);
            }),
        ];

        let ed: Vec<Vs> = vec![
            // dB[0]/ds
            Box::new(move |dxds, s| {
                let theta = PI * (1.0 + s) / 4.0;
                dxds[0] = -a * theta.sin() * PI / 4.0;
                dxds[1] = a * theta.cos() * PI / 4.0;
            }),
            // dB[1]/ds
            Box::new(move |dxds, _| {
                dxds[0] = -b * 0.5;
                dxds[1] = b * 0.5;
            }),
            // dB[2]/dr
            Box::new(move |dxdr, _| {
                dxdr[0] = 0.5 * (b - a);
                dxdr[1] = 0.0;
            }),
            // dB[3]/dr
            Box::new(move |dxdr, _| {
                dxdr[0] = 0.0;
                dxdr[1] = 0.5 * (b - a);
            }),
        ];

        let edd: Vec<Vs> = vec![
            // d²B[0]/ds²
            Box::new(move |ddxdss, s| {
                let theta = PI * (1.0 + s) / 4.0;
                ddxdss[0] = -a * theta.cos() * PI * PI / 16.0;
                ddxdss[1] = -a * theta.sin() * PI * PI / 16.0;
            }),
            // d²B[1]/ds²
            Box::new(move |ddxdss, _| {
                ddxdss[0] = 0.0;
                ddxdss[1] = 0.0;
            }),
            // d²B[2]/dr²
            Box::new(move |ddxdrr, _| {
                ddxdrr[0] = 0.0;
                ddxdrr[1] = 0.0;
            }),
            // d²B[3]/dr²
            Box::new(move |ddxdrr, _| {
                ddxdrr[0] = 0.0;
                ddxdrr[1] = 0.0;
            }),
        ];

        Transfinite::new_2d(e, ed, Some(edd))
    }

    /// Generates a transfinite mapping of a cube
    pub fn cube_3d(lx: f64, ly: f64, lz: f64) -> Transfinite {
        let b: Vec<Vss> = vec![
            // B[0](s,t)
            Box::new(move |x, s, t| {
                x[0] = 0.0;
                x[1] = (1.0 + s) * ly / 2.0;
                x[2] = (1.0 + t) * lz / 2.0;
            }),
            // B[1](s,t)
            Box::new(move |x, s, t| {
                x[0] = lx;
                x[1] = (1.0 + s) * ly / 2.0;
                x[2] = (1.0 + t) * lz / 2.0;
            }),
            // B[2](r,t)
            Box::new(move |x, r, t| {
                x[0] = (1.0 + r) * lx / 2.0;
                x[1] = 0.0;
                x[2] = (1.0 + t) * lz / 2.0;
            }),
            // B[3](r,t)
            Box::new(move |x, r, t| {
                x[0] = (1.0 + r) * lx / 2.0;
                x[1] = ly;
                x[2] = (1.0 + t) * lz / 2.0;
            }),
            // B[4](r,s)
            Box::new(move |x, r, s| {
                x[0] = (1.0 + r) * lx / 2.0;
                x[1] = (1.0 + s) * ly / 2.0;
                x[2] = 0.0;
            }),
            // B[5](r,s)
            Box::new(move |x, r, s| {
                x[0] = (1.0 + r) * lx / 2.0;
                x[1] = (1.0 + s) * ly / 2.0;
                x[2] = lz;
            }),
        ];

        let bd: Vec<Vvss> = vec![
            // Bd[0](s,t)
            Box::new(move |dxds, dxdt, _, _| {
                dxds[0] = 0.0;
                dxds[1] = ly / 2.0;
                dxds[2] = 0.0;
                dxdt[0] = 0.0;
                dxdt[1] = 0.0;
                dxdt[2] = lz / 2.0;
            }),
            // Bd[1](s,t)
            Box::new(move |dxds, dxdt, _, _| {
                dxds[0] = 0.0;
                dxds[1] = ly / 2.0;
                dxds[2] = 0.0;
                dxdt[0] = 0.0;
                dxdt[1] = 0.0;
                dxdt[2] = lz / 2.0;
            }),
            // Bd[2](r,t)
            Box::new(move |dxdr, dxdt, _, _| {
                dxdr[0] = lx / 2.0;
                dxdr[1] = 0.0;
                dxdr[2] = 0.0;
                dxdt[0] = 0.0;
                dxdt[1] = 0.0;
                dxdt[2] = lz / 2.0;
            }),
            // Bd[3](r,t)
            Box::new(move |dxdr, dxdt, _, _| {
                dxdr[0] = lx / 2.0;
                dxdr[1] = 0.0;
                dxdr[2] = 0.0;
                dxdt[0] = 0.0;
                dxdt[1] = 0.0;
                dxdt[2] = lz / 2.0;
            }),
            // Bd[4](r,s)
            Box::new(move |dxdr, dxds, _, _| {
                dxdr[0] = lx / 2.0;
                dxdr[1] = 0.0;
                dxdr[2] = 0.0;
                dxds[0] = 0.0;
                dxds[1] = ly / 2.0;
                dxds[2] = 0.0;
            }),
            // Bd[5](r,s)
            Box::new(move |dxdr, dxds, _, _| {
                dxdr[0] = lx / 2.0;
                dxdr[1] = 0.0;
                dxdr[2] = 0.0;
                dxds[0] = 0.0;
                dxds[1] = ly / 2.0;
                dxds[2] = 0.0;
            }),
        ];

        Transfinite::new_3d(b, bd, None)
    }

    /// Generates a transfinite mapping of a quarter of a 3d ring centered @ (0,0)
    ///
    /// a -- inner radius
    /// b -- outer radius
    /// h -- thickness along x-direction
    pub fn quarter_ring_3d(a: f64, b: f64, h: f64) -> Transfinite {
        let surf = Arc::new(Mutex::new(TransfiniteSamples::quarter_ring_2d(a, b)));

        let surf_0 = surf.clone();
        let surf_1 = surf.clone();
        let b_funcs: Vec<Vss> = vec![
            // B[0](s,t)
            Box::new(move |x, s, t| {
                let mut surf = surf_0.lock().unwrap();
                let u2d = Vector::from(&[s, t]);
                let mut x2d = Vector::new(2);
                surf.point(&mut x2d, &u2d);
                x[0] = 0.0;
                x[1] = x2d[0];
                x[2] = x2d[1];
            }),
            // B[1](s,t)
            Box::new(move |x, s, t| {
                let mut surf = surf_1.lock().unwrap();
                let u2d = Vector::from(&[s, t]);
                let mut x2d = Vector::new(2);
                surf.point(&mut x2d, &u2d);
                x[0] = h;
                x[1] = x2d[0];
                x[2] = x2d[1];
            }),
            // B[2](r,t)
            Box::new(move |x, r, t| {
                let theta = (1.0 + t) * PI / 4.0;
                x[0] = (1.0 + r) * h / 2.0;
                x[1] = a * theta.cos();
                x[2] = a * theta.sin();
            }),
            // B[3](r,t)
            Box::new(move |x, r, t| {
                let theta = (1.0 + t) * PI / 4.0;
                x[0] = (1.0 + r) * h / 2.0;
                x[1] = b * theta.cos();
                x[2] = b * theta.sin();
            }),
            // B[4](r,s)
            Box::new(move |x, r, s| {
                x[0] = (1.0 + r) * h / 2.0;
                x[1] = a + (1.0 + s) * (b - a) / 2.0;
                x[2] = 0.0;
            }),
            // B[5](r,s)
            Box::new(move |x, r, s| {
                x[0] = (1.0 + r) * h / 2.0;
                x[1] = 0.0;
                x[2] = a + (1.0 + s) * (b - a) / 2.0;
            }),
        ];

        let surf_d0 = surf.clone();
        let surf_d1 = surf.clone();
        let bd: Vec<Vvss> = vec![
            // Bd[0](s,t)
            Box::new(move |dxds, dxdt, s, t| {
                let mut surf = surf_d0.lock().unwrap();
                let u2d = Vector::from(&[s, t]);
                let mut tmp = Vector::new(2);
                let mut dxdr2d = Vector::new(2);
                let mut dxds2d = Vector::new(2);
                let mut dummy = Vector::new(0);
                surf.point_and_derivs(
                    &mut tmp,
                    &mut dxdr2d,
                    &mut dxds2d,
                    &mut dummy,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    &u2d,
                );
                dxds[0] = 0.0;
                dxds[1] = dxdr2d[0];
                dxds[2] = dxdr2d[1];
                dxdt[0] = 0.0;
                dxdt[1] = dxds2d[0];
                dxdt[2] = dxds2d[1];
            }),
            // Bd[1](s,t)
            Box::new(move |dxds, dxdt, s, t| {
                let mut surf = surf_d1.lock().unwrap();
                let u2d = Vector::from(&[s, t]);
                let mut tmp = Vector::new(2);
                let mut dxdr2d = Vector::new(2);
                let mut dxds2d = Vector::new(2);
                let mut dummy = Vector::new(0);
                surf.point_and_derivs(
                    &mut tmp,
                    &mut dxdr2d,
                    &mut dxds2d,
                    &mut dummy,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    &u2d,
                );
                dxds[0] = 0.0;
                dxds[1] = dxdr2d[0];
                dxds[2] = dxdr2d[1];
                dxdt[0] = 0.0;
                dxdt[1] = dxds2d[0];
                dxdt[2] = dxds2d[1];
            }),
            // Bd[2](r,t)
            Box::new(move |dxdr, dxdt, _, t| {
                let theta = (1.0 + t) * PI / 4.0;
                dxdr[0] = h / 2.0;
                dxdr[1] = 0.0;
                dxdr[2] = 0.0;
                dxdt[0] = 0.0;
                dxdt[1] = -a * theta.sin() * PI / 4.0;
                dxdt[2] = a * theta.cos() * PI / 4.0;
            }),
            // Bd[3](r,t)
            Box::new(move |dxdr, dxdt, _, t| {
                let theta = (1.0 + t) * PI / 4.0;
                dxdr[0] = h / 2.0;
                dxdr[1] = 0.0;
                dxdr[2] = 0.0;
                dxdt[0] = 0.0;
                dxdt[1] = -b * theta.sin() * PI / 4.0;
                dxdt[2] = b * theta.cos() * PI / 4.0;
            }),
            // Bd[4](r,s)
            Box::new(move |dxdr, dxds, _, _| {
                dxdr[0] = h / 2.0;
                dxdr[1] = 0.0;
                dxdr[2] = 0.0;
                dxds[0] = 0.0;
                dxds[1] = (b - a) / 2.0;
                dxds[2] = 0.0;
            }),
            // Bd[5](r,s)
            Box::new(move |dxdr, dxds, _, _| {
                dxdr[0] = h / 2.0;
                dxdr[1] = 0.0;
                dxdr[2] = 0.0;
                dxds[0] = 0.0;
                dxds[1] = 0.0;
                dxds[2] = (b - a) / 2.0;
            }),
        ];

        let surf_dd0 = surf.clone();
        let surf_dd1 = surf.clone();
        let bdd: Vec<Vvvss> = vec![
            // Bdd[0](s,t)
            Box::new(move |ddxdss, ddxdtt, ddxdst, s, t| {
                let mut surf = surf_dd0.lock().unwrap();
                let u2d = Vector::from(&[s, t]);
                let mut x_tmp = Vector::new(2);
                let mut dx_dr_tmp = Vector::new(2);
                let mut dx_ds_tmp = Vector::new(2);
                let mut dx_dt_tmp = Vector::new(2);
                let mut ddxdrr2d = Vector::new(2);
                let mut ddxdss2d = Vector::new(2);
                let mut ddxdrs2d = Vector::new(2);
                surf.point_and_derivs(
                    &mut x_tmp,
                    &mut dx_dr_tmp,
                    &mut dx_ds_tmp,
                    &mut dx_dt_tmp,
                    Some(&mut ddxdrr2d),
                    Some(&mut ddxdss2d),
                    None,
                    Some(&mut ddxdrs2d),
                    None,
                    None,
                    &u2d,
                );
                ddxdss[0] = 0.0;
                ddxdss[1] = ddxdrr2d[0];
                ddxdss[2] = ddxdrr2d[1];

                ddxdtt[0] = 0.0;
                ddxdtt[1] = ddxdss2d[0];
                ddxdtt[2] = ddxdss2d[1];

                ddxdst[0] = 0.0;
                ddxdst[1] = ddxdrs2d[0];
                ddxdst[2] = ddxdrs2d[1];
            }),
            // Bdd[1](s,t)
            Box::new(move |ddxdss, ddxdtt, ddxdst, s, t| {
                let mut surf = surf_dd1.lock().unwrap();
                let u2d = Vector::from(&[s, t]);
                let mut x_tmp = Vector::new(2);
                let mut dx_dr_tmp = Vector::new(2);
                let mut dx_ds_tmp = Vector::new(2);
                let mut dx_dt_tmp = Vector::new(2);
                let mut ddxdrr2d = Vector::new(2);
                let mut ddxdss2d = Vector::new(2);
                let mut ddxdrs2d = Vector::new(2);
                surf.point_and_derivs(
                    &mut x_tmp,
                    &mut dx_dr_tmp,
                    &mut dx_ds_tmp,
                    &mut dx_dt_tmp,
                    Some(&mut ddxdrr2d),
                    Some(&mut ddxdss2d),
                    None,
                    Some(&mut ddxdrs2d),
                    None,
                    None,
                    &u2d,
                );
                ddxdss[0] = 0.0;
                ddxdss[1] = ddxdrr2d[0];
                ddxdss[2] = ddxdrr2d[1];

                ddxdtt[0] = 0.0;
                ddxdtt[1] = ddxdss2d[0];
                ddxdtt[2] = ddxdss2d[1];

                ddxdst[0] = 0.0;
                ddxdst[1] = ddxdrs2d[0];
                ddxdst[2] = ddxdrs2d[1];
            }),
            // Bdd[2](r,t)
            Box::new(move |ddxdrr, ddxdtt, ddxdrt, _, t| {
                let theta = (1.0 + t) * PI / 4.0;
                ddxdrr[0] = 0.0;
                ddxdrr[1] = 0.0;
                ddxdrr[2] = 0.0;

                ddxdtt[0] = 0.0;
                ddxdtt[1] = -a * theta.cos() * PI * PI / 16.0;
                ddxdtt[2] = -a * theta.sin() * PI * PI / 16.0;

                ddxdrt[0] = 0.0;
                ddxdrt[1] = 0.0;
                ddxdrt[2] = 0.0;
            }),
            // Bdd[3](r,t)
            Box::new(move |ddxdrr, ddxdtt, ddxdrt, _, t| {
                let theta = (1.0 + t) * PI / 4.0;
                ddxdrr[0] = 0.0;
                ddxdrr[1] = 0.0;
                ddxdrr[2] = 0.0;

                ddxdtt[0] = 0.0;
                ddxdtt[1] = -b * theta.cos() * PI * PI / 16.0;
                ddxdtt[2] = -b * theta.sin() * PI * PI / 16.0;

                ddxdrt[0] = 0.0;
                ddxdrt[1] = 0.0;
                ddxdrt[2] = 0.0;
            }),
            // Bdd[4](r,s)
            Box::new(move |ddxdrr, ddxdss, ddxdrs, _, _| {
                ddxdrr[0] = 0.0;
                ddxdrr[1] = 0.0;
                ddxdrr[2] = 0.0;

                ddxdss[0] = 0.0;
                ddxdss[1] = 0.0;
                ddxdss[2] = 0.0;

                ddxdrs[0] = 0.0;
                ddxdrs[1] = 0.0;
                ddxdrs[2] = 0.0;
            }),
            // Bdd[5](r,s)
            Box::new(move |ddxdrr, ddxdss, ddxdrs, _, _| {
                ddxdrr[0] = 0.0;
                ddxdrr[1] = 0.0;
                ddxdrr[2] = 0.0;

                ddxdss[0] = 0.0;
                ddxdss[1] = 0.0;
                ddxdss[2] = 0.0;

                ddxdrs[0] = 0.0;
                ddxdrs[1] = 0.0;
                ddxdrs[2] = 0.0;
            }),
        ];

        Transfinite::new_3d(b_funcs, bd, Some(bdd))
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::TransfiniteSamples;
    use crate::Transfinite;
    use plotpy::{linspace, Canvas, Plot, PolyCode};
    use russell_lab::{vec_approx_eq, Vector};

    const SAVE_FIGURE: bool = false;

    fn draw_lines_2d(canvas: &mut Canvas, map: &mut Transfinite, np: usize, dot_size: f64) {
        canvas.set_face_color("None");
        let mut x = Vector::new(2);
        let mut u = Vector::new(2);
        let rr = linspace(-1.0, 1.0, np);
        let ss = rr.clone();
        // lines in r-direction
        for j in 0..np {
            u[1] = ss[j];
            u[0] = rr[0];
            map.point(&mut x, &u);
            canvas.polycurve_begin();
            canvas.polycurve_add(x[0], x[1], PolyCode::MoveTo);
            for i in 1..np {
                u[0] = rr[i];
                map.point(&mut x, &u);
                canvas.polycurve_add(x[0], x[1], PolyCode::LineTo);
            }
            canvas.polycurve_end(false);
        }
        // lines in s-direction
        for i in 0..np {
            u[0] = rr[i];
            u[1] = ss[0];
            map.point(&mut x, &u);
            canvas.polycurve_begin();
            canvas.polycurve_add(x[0], x[1], PolyCode::MoveTo);
            for j in 1..np {
                u[1] = ss[j];
                map.point(&mut x, &u);
                canvas.polycurve_add(x[0], x[1], PolyCode::LineTo);
            }
            canvas.polycurve_end(false);
        }
        // points at corners
        u[0] = -1.0;
        u[1] = -1.0;
        map.point(&mut x, &u);
        canvas.draw_circle(x[0], x[1], dot_size);
        u[0] = 1.0;
        u[1] = -1.0;
        map.point(&mut x, &u);
        canvas.draw_circle(x[0], x[1], dot_size);
        u[0] = 1.0;
        u[1] = 1.0;
        map.point(&mut x, &u);
        canvas.draw_circle(x[0], x[1], dot_size);
        u[0] = -1.0;
        u[1] = 1.0;
        map.point(&mut x, &u);
        canvas.draw_circle(x[0], x[1], dot_size);
        // point in the center
        u[0] = 0.0;
        u[1] = 0.0;
        map.point(&mut x, &u);
        canvas.draw_circle(x[0], x[1], dot_size);
    }

    #[test]
    fn test_quadrilateral_2d() {
        let xa = &[1.0, 0.0];
        let xb = &[6.0, 4.0];
        let xc = &[1.0, 6.0];
        let xd = &[0.0, 5.0];
        let mut map = TransfiniteSamples::quadrilateral_2d(xa, xb, xc, xd);

        let mut x = Vector::new(2);
        let mut u = Vector::new(2);

        u[0] = -1.0;
        u[1] = -1.0;
        map.point(&mut x, &u);
        vec_approx_eq(&x, xa, 1e-15);

        u[0] = 1.0;
        u[1] = -1.0;
        map.point(&mut x, &u);
        vec_approx_eq(&x, xb, 1e-15);

        u[0] = 1.0;
        u[1] = 1.0;
        map.point(&mut x, &u);
        vec_approx_eq(&x, xc, 1e-15);

        u[0] = -1.0;
        u[1] = 1.0;
        map.point(&mut x, &u);
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
        let mut u = Vector::new(2);

        u[0] = -1.0;
        u[1] = -1.0;
        map.point(&mut x, &u);
        vec_approx_eq(&x, &[r_in, 0.0], 1e-15);

        u[0] = 1.0;
        u[1] = -1.0;
        map.point(&mut x, &u);
        vec_approx_eq(&x, &[r_out, 0.0], 1e-15);

        u[0] = 1.0;
        u[1] = 1.0;
        map.point(&mut x, &u);
        vec_approx_eq(&x, &[0.0, r_out], 1e-15);

        u[0] = -1.0;
        u[1] = 1.0;
        map.point(&mut x, &u);
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
}
