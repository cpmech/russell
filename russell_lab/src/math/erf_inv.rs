pub fn erf_inv(_x: f64) -> f64 {
    panic!("TODO: erf_inv")
}

pub fn erfc_inv(_x: f64) -> f64 {
    panic!("TODO: erfc_inv")
}

/*
const SOLUTION_ERF_INV: [f64; 10] = [
    4.746037673358033586786350696e-01,
    8.559054432692110956388764172e-01,
    -2.45427830571707336251331946e-02,
    -4.78116683518973366268905506e-01,
    1.479804430319470983648120853e+00,
    2.654485787128896161882650211e-01,
    5.027444534221520197823192493e-01,
    2.466703532707627818954585670e-01,
    1.632011465103005426240343116e-01,
    -1.06672334642196900710000389e+00,
];


const SPECIAL_CASES_ERF_INV: [f64; 6] = [1.0, -1.0, 0.0, f64::NEG_INFINITY, f64::INFINITY, f64::NAN];

const SPECIAL_CASES_SOLUTION_ERF_INV: [f64; 6] = [f64::INFINITY, f64::NEG_INFINITY, 0.0, f64::NAN, f64::NAN, f64::NAN];

const SPECIAL_CASES_ERFC_INV: [f64; 6] = [0.0, 2.0, 1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];

const SPECIAL_CASES_SOLUTION_ERFC_INV: [f64; 6] = [f64::INFINITY, f64::NEG_INFINITY, 0.0, f64::NAN, f64::NAN, f64::NAN];

// #[test]
fn test_erf_inv() {
    for i in 0..VALUES.len() {
        let a = VALUES[i] / 10.0;
        let f = math::erf_inv(a);
        if !very_close(SOLUTION_ERF_INV[i], f) {
            println!("erf_inv({}) = {}, want {}", a, f, SOLUTION_ERF_INV[i]);
            panic!("erf_inv failed");
        }
    }
    for i in 0..SPECIAL_CASES_ERF_INV.len() {
        let f = math::erf_inv(SPECIAL_CASES_ERF_INV[i]);
        if !alike(SPECIAL_CASES_SOLUTION_ERF_INV[i], f) {
            println!(
                "erf_inv({}) = {}, want {}",
                SPECIAL_CASES_ERF_INV[i], f, SPECIAL_CASES_SOLUTION_ERF_INV[i]
            );
            panic!("erf_inv special cases failed");
        }
    }
    let mut x = -0.9;
    while x <= 0.90 {
        let f = math::erf(math::erf_inv(x));
        if !close(x, f) {
            println!("erf(erf_inv({})) = {}, want {}", x, f, x);
            panic!("erf(erf_inv(x)) = x failed");
        }
        x += 1e-2;
    }
    let mut x = -0.9;
    while x <= 0.90 {
        let f = math::erf_inv(math::erf(x));
        if !close(x, f) {
            println!("erf_inv(erf({})) = {}, want {}", x, f, x);
            panic!("erf_inv(erf(x)) = x failed");
        }
        x += 1e-2;
    }
}

// #[test]
fn test_erfc_inv() {
    for i in 0..VALUES.len() {
        let a = 1.0 - (VALUES[i] / 10.0);
        let f = math::erfc_inv(a);
        if !very_close(SOLUTION_ERF_INV[i], f) {
            println!("erfc_inv({}) = {}, want {}", a, f, SOLUTION_ERF_INV[i]);
            panic!("erfc_inv failed");
        }
    }
    for i in 0..SPECIAL_CASES_ERFC_INV.len() {
        let f = math::erfc_inv(SPECIAL_CASES_ERFC_INV[i]);
        if !alike(SPECIAL_CASES_SOLUTION_ERFC_INV[i], f) {
            println!(
                "erfc_inv({}) = {}, want {}",
                SPECIAL_CASES_ERFC_INV[i], f, SPECIAL_CASES_SOLUTION_ERFC_INV[i]
            );
            panic!("erfc_inv special cases failed");
        }
    }
    let mut x = 0.1;
    while x <= 1.9 {
        let f = math::erfc(math::erfc_inv(x));
        if !close(x, f) {
            println!("erfc(erfc_inv({})) = {}, want {}", x, f, x);
            panic!("erfc(erfc_inv(x)) = x");
        }
        x += 1e-2;
    }
    let mut x = 0.1;
    while x <= 1.9 {
        let f = math::erfc_inv(math::erfc(x));
        if !close(x, f) {
            println!("erfc_inv(erfc({})) = {}, want {}", x, f, x);
            panic!("erfc_inv(erfc(x)) = x");
        }
        x += 1e-2;
    }
}

*/
