use russell_lab::{approx_eq, math, read_table};
use std::collections::HashMap;

#[test]
fn test_bessel_functions_sml() {
    let dat: HashMap<String, Vec<f64>> = read_table(
        "data/reference/as-9-bessel-integer-sml.cmp",
        Some(&["x", "J0", "J1", "J2", "Y0", "Y1", "Y2"]),
    )
    .unwrap();

    let xx = dat.get("x").unwrap();
    let j0 = dat.get("J0").unwrap();
    let j1 = dat.get("J1").unwrap();
    let j2 = dat.get("J2").unwrap();
    let y0 = dat.get("Y0").unwrap();
    let y1 = dat.get("Y1").unwrap();
    let y2 = dat.get("Y2").unwrap();

    for (i, x) in xx.into_iter().enumerate() {
        approx_eq(math::bessel_j0(*x), j0[i], 1e-15);
        approx_eq(math::bessel_j1(*x), j1[i], 1e-15);
        approx_eq(math::bessel_jn(2, *x), j2[i], 1e-15);
        if i == 0 {
            assert_eq!(math::bessel_y0(*x), f64::NEG_INFINITY);
            assert_eq!(math::bessel_y1(*x), f64::NEG_INFINITY);
            assert_eq!(math::bessel_yn(2, *x), f64::NEG_INFINITY);
        } else {
            approx_eq(math::bessel_y0(*x), y0[i], 1e-15);
            approx_eq(math::bessel_y1(*x), y1[i], 1e-15);
            approx_eq(math::bessel_yn(2, *x), y2[i], 1e-15);
        }
    }
}

#[test]
fn test_bessel_functions_big() {
    let dat: HashMap<String, Vec<f64>> = read_table(
        "data/reference/as-9-bessel-integer-big.cmp",
        Some(&["x", "J0", "J1", "J2", "Y0", "Y1", "Y2"]),
    )
    .unwrap();

    let xx = dat.get("x").unwrap();
    let j0 = dat.get("J0").unwrap();
    let j1 = dat.get("J1").unwrap();
    let j2 = dat.get("J2").unwrap();
    let y0 = dat.get("Y0").unwrap();
    let y1 = dat.get("Y1").unwrap();
    let y2 = dat.get("Y2").unwrap();

    for (i, x) in xx.into_iter().enumerate() {
        approx_eq(math::bessel_j0(*x), j0[i], 1e-15);
        approx_eq(math::bessel_j1(*x), j1[i], 1e-15);
        approx_eq(math::bessel_jn(2, *x), j2[i], 1e-14);
        if i == 0 {
            assert_eq!(math::bessel_y0(*x), f64::NEG_INFINITY);
            assert_eq!(math::bessel_y1(*x), f64::NEG_INFINITY);
            assert_eq!(math::bessel_yn(2, *x), f64::NEG_INFINITY);
        } else {
            approx_eq(math::bessel_y0(*x), y0[i], 1e-15);
            approx_eq(math::bessel_y1(*x), y1[i], 1e-15);
            approx_eq(math::bessel_yn(2, *x), y2[i], 1e-13);
        }
    }
}

#[test]
fn test_modified_bessel_functions_sml() {
    let dat: HashMap<String, Vec<f64>> = read_table(
        "data/reference/as-9-modbessel-integer-sml.cmp",
        Some(&["x", "I0", "I1", "I2", "I3", "K0", "K1", "K2", "K3"]),
    )
    .unwrap();

    let xx = dat.get("x").unwrap();
    let i0 = dat.get("I0").unwrap();
    let i1 = dat.get("I1").unwrap();
    let i2 = dat.get("I2").unwrap();
    let i3 = dat.get("I3").unwrap();
    let k0 = dat.get("K0").unwrap();
    let k1 = dat.get("K1").unwrap();
    let k2 = dat.get("K2").unwrap();
    let k3 = dat.get("K3").unwrap();

    for (i, x) in xx.into_iter().enumerate() {
        approx_eq(math::bessel_i0(*x), i0[i], 1e-8);
        approx_eq(math::bessel_i1(*x), i1[i], 1e-7);
        approx_eq(math::bessel_in(2, *x), i2[i], 1e-7);
        approx_eq(math::bessel_in(3, *x), i3[i], 1e-7);
        if i == 0 {
            assert_eq!(math::bessel_k0(*x), f64::INFINITY);
            assert_eq!(math::bessel_k1(*x), f64::INFINITY);
            assert_eq!(math::bessel_kn(2, *x), f64::INFINITY);
            assert_eq!(math::bessel_kn(3, *x), f64::INFINITY);
        } else {
            approx_eq(math::bessel_k0(*x), k0[i], 1e-15);
            approx_eq(math::bessel_k1(*x), k1[i], 1e-15);
            approx_eq(math::bessel_kn(2, *x), k2[i], 1e-15);
            approx_eq(math::bessel_kn(3, *x), k3[i], 1e-14);
        }
    }
}

#[test]
fn test_modified_bessel_functions_big() {
    let dat: HashMap<String, Vec<f64>> = read_table(
        "data/reference/as-9-modbessel-integer-big.cmp",
        Some(&["x", "I0", "I1", "I2", "I3", "K0", "K1", "K2", "K3"]),
    )
    .unwrap();

    let xx = dat.get("x").unwrap();
    let i0 = dat.get("I0").unwrap();
    let i1 = dat.get("I1").unwrap();
    let i2 = dat.get("I2").unwrap();
    let i3 = dat.get("I3").unwrap();
    let k0 = dat.get("K0").unwrap();
    let k1 = dat.get("K1").unwrap();
    let k2 = dat.get("K2").unwrap();
    let k3 = dat.get("K3").unwrap();

    for (i, x) in xx.into_iter().enumerate() {
        approx_eq(math::bessel_i0(*x), i0[i], 1e-6);
        approx_eq(math::bessel_i1(*x), i1[i], 1e-6);
        approx_eq(math::bessel_in(2, *x), i2[i], 1e-6);
        approx_eq(math::bessel_in(3, *x), i3[i], 1e-6);
        if i == 0 {
            assert_eq!(math::bessel_k0(*x), f64::INFINITY);
            assert_eq!(math::bessel_k1(*x), f64::INFINITY);
            assert_eq!(math::bessel_kn(2, *x), f64::INFINITY);
            assert_eq!(math::bessel_kn(3, *x), f64::INFINITY);
        } else {
            approx_eq(math::bessel_k0(*x), k0[i], 1e-15);
            approx_eq(math::bessel_k1(*x), k1[i], 1e-14);
            approx_eq(math::bessel_kn(2, *x), k2[i], 1e-13);
            approx_eq(math::bessel_kn(3, *x), k3[i], 1e-12);
        }
    }
}

#[test]
fn test_modified_bessel_functions_neg() {
    let dat: HashMap<String, Vec<f64>> = read_table(
        "data/reference/as-9-modbessel-integer-neg.cmp",
        Some(&["x", "I0", "I1", "I2", "I3"]),
    )
    .unwrap();

    let xx = dat.get("x").unwrap();
    let i0 = dat.get("I0").unwrap();
    let i1 = dat.get("I1").unwrap();
    let i2 = dat.get("I2").unwrap();
    let i3 = dat.get("I3").unwrap();

    for (i, x) in xx.into_iter().enumerate() {
        approx_eq(math::bessel_i0(*x), i0[i], 1e-12);
        approx_eq(math::bessel_i1(*x), i1[i], 1e-12);
        approx_eq(math::bessel_in(2, *x), i2[i], 1e-11);
        approx_eq(math::bessel_in(3, *x), i3[i], 1e-12);
    }
}
