use russell_lab::math::{elliptic_e, elliptic_f, elliptic_pi, PI};
use russell_lab::{approx_eq, read_table};
use std::collections::HashMap;

#[test]
fn test_elliptic_f() {
    for fp in [
        "data/reference/as-17-elliptic-integrals-table17.5-small.cmp",
        "data/reference/as-17-elliptic-integrals-table17.5-big.cmp",
    ] {
        let dat: HashMap<String, Vec<f64>> = read_table(fp, Some(&["phi", "k", "F"])).unwrap();

        let all_phi = dat.get("phi").unwrap();
        let k = dat.get("k").unwrap();
        let ff = dat.get("F").unwrap();

        for (i, phi) in all_phi.into_iter().enumerate() {
            // println!("phi = {:?}, k = {:?}", phi, k[i],);
            let p = if f64::abs(phi - PI / 2.0) < 1e-15 {
                // handle small noise on PI/2
                PI / 2.0
            } else {
                *phi
            };
            let cond = f64::abs(f64::sin(*phi) * k[i] - 1.0);
            if cond < f64::EPSILON {
                // handle k·sin(φ) == 1
                assert!(elliptic_f(p, k[i] * k[i]).unwrap().is_infinite());
            } else {
                approx_eq(elliptic_f(p, k[i] * k[i]).unwrap(), ff[i], 1e-13);
            }
        }
    }
}

#[test]
fn test_elliptic_e() {
    for fp in [
        "data/reference/as-17-elliptic-integrals-table17.6-small.cmp",
        "data/reference/as-17-elliptic-integrals-table17.6-big.cmp",
    ] {
        let dat: HashMap<String, Vec<f64>> = read_table(fp, Some(&["phi", "k", "E"])).unwrap();

        let all_phi = dat.get("phi").unwrap();
        let k = dat.get("k").unwrap();
        let ff = dat.get("E").unwrap();

        for (i, phi) in all_phi.into_iter().enumerate() {
            // println!("phi = {:?}, k = {:?}", phi, k[i],);
            let p = if f64::abs(phi - PI / 2.0) < 1e-15 {
                // handle small noise on PI/2
                PI / 2.0
            } else {
                *phi
            };
            approx_eq(elliptic_e(p, k[i] * k[i]).unwrap(), ff[i], 1e-14);
        }
    }
}

#[test]
fn test_elliptic_pi() {
    for fp in [
        "data/reference/as-17-elliptic-integrals-table17.9-small.cmp",
        "data/reference/as-17-elliptic-integrals-table17.9-big.cmp",
    ] {
        let dat: HashMap<String, Vec<f64>> = read_table(fp, Some(&["n", "phi", "k", "PI"])).unwrap();

        let all_phi = dat.get("phi").unwrap();
        let n = dat.get("n").unwrap();
        let k = dat.get("k").unwrap();
        let ff = dat.get("PI").unwrap();

        for (i, phi) in all_phi.into_iter().enumerate() {
            // println!("n = {:?}, phi = {:?}, k = {:?}", n[i], phi, k[i],);
            let p = if f64::abs(phi - PI / 2.0) < 1e-15 {
                // handle small noise on PI/2
                PI / 2.0
            } else {
                *phi
            };
            let s = f64::sin(*phi);
            let cond1 = f64::abs(s * k[i] - 1.0);
            let cond2 = f64::abs(s * s * n[i] - 1.0);
            if cond1 < f64::EPSILON || cond2 < f64::EPSILON {
                assert!(elliptic_pi(n[i], p, k[i]).unwrap().is_infinite());
            } else {
                approx_eq(elliptic_pi(n[i], p, k[i]).unwrap(), ff[i], 1e-13);
            }
        }
    }
}
