use russell_lab::math::{elliptic_f, PI};
use russell_lab::{approx_eq, read_table};
use std::collections::HashMap;

#[test]
fn test_elliptic_functions_sml() {
    let dat: HashMap<String, Vec<f64>> = read_table(
        "data/reference/as-17-elliptic-integrals-table17.5-small.cmp",
        // "data/reference/as-17-elliptic-integrals-table17.5-big.cmp",
        Some(&["phi", "k", "F"]),
    )
    .unwrap();

    let all_phi = dat.get("phi").unwrap();
    let k = dat.get("k").unwrap();
    let ff = dat.get("F").unwrap();

    for (i, phi) in all_phi.into_iter().enumerate() {
        let cond = f64::abs(f64::sin(*phi) * k[i] - 1.0);
        // println!("phi = {:?}, k = {:?}", phi, k[i],);
        let p = if f64::abs(phi - PI / 2.0) < 1e-15 {
            // handle small noise on PI/2
            PI / 2.0
        } else {
            *phi
        };
        if cond < f64::EPSILON {
            // handle k·sin(φ) == 1
            assert!(elliptic_f(p, k[i]).unwrap().is_infinite());
        } else {
            approx_eq(elliptic_f(p, k[i]).unwrap(), ff[i], 1e-14);
        }
    }
}
