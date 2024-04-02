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
