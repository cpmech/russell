//! Compares the accuracy of the available polar decomposition algorithms
//! ([PolarAlgo]) on a family of matrices with a known exact polar factor.
//!
//! The deformation gradients `F(y)` are from Higham & Noferini (2016), test
//! (5.2). Their exact rotation `R` is known, and the exact right stretch is
//! `U = Rᵀ F`. The condition number of `F` is `1/y`, so decreasing `y`
//! increases the sensitivity to ill-conditioning, exposing:
//!
//! * [PolarAlgo::Eigen] — squares the condition number (through `C = Fᵀ F`),
//!   so the error grows like `cond²`; it fails (singular) for `cond ≥ 1e8`
//! * [PolarAlgo::SVD] and [PolarAlgo::Quaternion] — robust across the entire
//!   range (direct methods)
//! * [PolarAlgo::Iterative] — accurate up to `cond ≈ 1e8` (about `1e-8`), but
//!   loses accuracy for extremely ill-conditioned `F` (`cond ≈ 1e10`)
//!
//! Reference: N. J. Higham and V. Noferini, "An algorithm to compute the polar
//! decomposition of a 3*3 matrix", Num. Algorithms, 73(2):349-369, 2016.

use russell_lab::{Matrix, mat_max_abs_diff, mat_t_mat_mul};
use russell_tensor::{PolarAlgo, StrError, Tensor2, polar_decomp_mx};

/// The polar decomposition algorithms under comparison
const ALGOS: [PolarAlgo; 4] = [
    PolarAlgo::Eigen,
    PolarAlgo::SVD,
    PolarAlgo::Iterative,
    PolarAlgo::Quaternion,
];

fn main() -> Result<(), StrError> {
    let yy = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10];

    // Exact polar factor R (the right stretch is computed as U = Rᵀ F)
    let r_exact = Matrix::from(&case52_rotation());

    // Compute the errors for all cases
    let mut results = Vec::new();
    for &y in &yy {
        let ff = Tensor2::<9>::from_std_matrix(&case52(y))?;
        let mut u_exact = Matrix::new(3, 3);
        mat_t_mat_mul(&mut u_exact, 1.0, &r_exact, &ff.as_std_matrix(), 0.0)?;

        let mut err_r = [0.0; 4];
        let mut err_u = [0.0; 4];
        for (i, algo) in ALGOS.iter().enumerate() {
            let mut rr = Tensor2::<9>::new();
            let mut uu = Tensor2::<6>::new();
            match polar_decomp_mx(&mut rr, &mut uu, None, *algo, &ff) {
                Ok(_) => {
                    err_r[i] = mat_max_abs_diff(&rr.as_std_matrix(), &r_exact)?.2;
                    err_u[i] = mat_max_abs_diff(&uu.as_std_matrix(), &u_exact)?.2;
                }
                Err(_) => {
                    err_r[i] = f64::NAN;
                    err_u[i] = f64::NAN;
                }
            }
        }
        results.push((1.0 / y, err_r, err_u));
    }

    // Rotation error
    println!("\nMax error in R (rotation)");
    print_header();
    for (cond, err_r, _) in &results {
        print_row(*cond, err_r);
    }

    // Right stretch error
    println!("\nMax error in U (right stretch)");
    print_header();
    for (cond, _, err_u) in &results {
        print_row(*cond, err_u);
    }

    println!("\nn/a = the algorithm failed (singular matrix)");

    Ok(())
}

/// Prints the table header (row is the condition number `1/y`)
///
/// The algorithm names come from `Debug`, so they cannot drift from [ALGOS].
fn print_header() {
    print!("{:>8}", "1/y");
    for algo in ALGOS {
        print!("  {:>10}", format!("{algo:?}"));
    }
    println!();
}

/// Prints one table row with the errors of the four algorithms
///
/// A `n/a` entry means that the algorithm failed (e.g. a singular matrix).
fn print_row(cond: f64, errs: &[f64; 4]) {
    print!("{:>8.0e}", cond);
    for e in errs {
        if e.is_nan() {
            print!("  {:>10}", "n/a");
        } else {
            print!("  {:>10.2e}", e);
        }
    }
    println!();
}

/// Returns the Higham & Noferini (2016) test (5.2) matrix for a given scale `y`
fn case52(y: f64) -> [[f64; 3]; 3] {
    #[rustfmt::skip]
    let a = [
        [(720.0 * y - 25.0) / 1275.0, (-650.0 * y + 300.0) / 1275.0, (710.0 * y + 300.0) / 1275.0],
        [(396.0 * y + 70.0) / 1275.0, (-145.0 * y - 840.0) / 1275.0, (178.0 * y - 840.0) / 1275.0],
        [(972.0 * y - 10.0) / 1275.0, (610.0 * y + 120.0) / 1275.0, (-529.0 * y + 120.0) / 1275.0],
    ];
    a
}

/// Returns the exact rotation of the Higham & Noferini (2016) test (5.2) matrix
fn case52_rotation() -> [[f64; 3]; 3] {
    [
        [139.0 / 255.0, -14.0 / 51.0, 202.0 / 255.0],
        [466.0 / 1275.0, -197.0 / 255.0, -662.0 / 1275.0],
        [962.0 / 1275.0, 146.0 / 255.0, -409.0 / 1275.0],
    ]
}
