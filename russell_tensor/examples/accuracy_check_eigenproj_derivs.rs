//! Compares the derivatives of the eigenprojectors computed with
//! `EigenProjDerivsT2::calc_with_inv` (Miehe's inverse-based approach) and
//! `EigenProjDerivsT2::calc_with_char_poly` (Panteghini's characteristic-polynomial
//! approach), using the Habera-Zilian eigenvalue method (the default).
//!
//! The symmetric tensors are built as `A = U ⋅ diag(d) ⋅ Uᵀ` with the orthogonal
//! transformation `U_sym` used in `accuracy_check_eigenvalues`, for diagonal cases
//! that approach the coalescent (two nearly equal eigenvalues) limit:
//!
//! * `D1 = diag(1, 1 + δ, 1 + 2δ)` — approaching a triple eigenvalue
//! * `D2 = diag(-1, 1, 1 + δ)` — approaching a double eigenvalue (the pair `1`, `1 + δ`)
//!
//! For each δ, a numerical derivative `d_num` is computed with `deriv1_central5` and
//! the following differences are reported (the relative ones are normalized by the
//! largest component magnitude of the two operands):
//!
//! * `max|ΔdP|` — maximum absolute difference between the two analytical methods
//! * `rel(inv,cp)` — relative difference between the two analytical methods
//! * `rel(inv,num)`, `rel(cp,num)` — relative difference of each analytical method
//!   with respect to the numerical derivative
//!
//! The coalescent cases rejected by both methods are labelled `failed1` (spherical
//! state) and `failed2` (two repeated eigenvalues); any other error aborts the example.

use russell_lab::deriv1_central5;
use russell_tensor::{EigenProjDerivsT2, EigenProjsT2, EigenValMethod, StrError, Tensor2, Tensor4};

/// Message returned when all eigenvalues are (numerically) equal
const ERR_SPHERICAL: &str = "Failed due to spherical state (all equal eigenvalues)";

/// Message returned when two eigenvalues are (numerically) equal
const ERR_REPEATED: &str = "Failed due to two repeated eigenvalues";

/// Per-row results of the comparison
struct Row {
    max_abs_inv_cp: Option<f64>,
    rel_inv_cp: Option<f64>,
    rel_inv_num: Option<f64>,
    rel_cp_num: Option<f64>,
    status: &'static str,
}

fn main() -> Result<(), StrError> {
    let method = EigenValMethod::AnalyticalHZ;
    let u = u_sym();
    let deltas = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8];

    for (label, is_d1) in [
        ("D1: diag(1, 1 + δ, 1 + 2δ)  (approaching a triple eigenvalue)", true),
        ("D2: diag(-1, 1, 1 + δ)      (approaching a double eigenvalue)", false),
    ] {
        println!("\n{}", label);
        println!(
            "{:>8}  {:>10}  {:>12}  {:>12}  {:>12}  status",
            "δ", "max|ΔdP|", "rel(inv,cp)", "rel(inv,num)", "rel(cp,num)"
        );
        for &delta in &deltas {
            let d = if is_d1 {
                [1.0, 1.0 + delta, 1.0 + 2.0 * delta]
            } else {
                [-1.0, 1.0, 1.0 + delta]
            };
            let a = build_a(&u, &d);
            let aa = Tensor2::<6>::from_std_matrix(&a)?;
            let row = compare(&aa, method)?;
            println!(
                "{:>8.0e}  {:>10}  {:>12}  {:>12}  {:>12}  {}",
                delta,
                fmt(row.max_abs_inv_cp),
                fmt(row.rel_inv_cp),
                fmt(row.rel_inv_num),
                fmt(row.rel_cp_num),
                row.status
            );
        }
    }

    println!("\nfailed1 = {}", ERR_SPHERICAL);
    println!("failed2 = {}", ERR_REPEATED);
    Ok(())
}

/// Formats an optional error value
fn fmt(value: Option<f64>) -> String {
    match value {
        Some(x) => format!("{:.2e}", x),
        None => "-".to_string(),
    }
}

/// Runs both analytical methods and the numerical derivative, and computes the differences
fn compare(aa: &Tensor2<6>, method: EigenValMethod) -> Result<Row, StrError> {
    let mut calc = EigenProjDerivsT2::new();
    let mut ll = [0.0; 3];
    let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
    let mut d_inv = [Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
    let mut d_cp = [Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];

    let r_inv = calc.calc_with_inv(&mut ll, &mut projs, &mut d_inv, aa, method);
    let r_cp = calc.calc_with_char_poly(&mut ll, &mut projs, &mut d_cp, aa, method);

    // classify the outcome; the two expected coalescent errors are labelled, while
    // any other error is propagated so that the example stops
    let status = match (r_inv, r_cp) {
        (Ok(()), Ok(())) => "ok",
        (Err(e1), Err(e2)) => {
            if e1 == ERR_SPHERICAL && e2 == ERR_SPHERICAL {
                "failed1"
            } else if e1 == ERR_REPEATED && e2 == ERR_REPEATED {
                "failed2"
            } else {
                return Err(e1);
            }
        }
        (Err(e), Ok(())) => return Err(e),
        (Ok(()), Err(e)) => return Err(e),
    };

    // nothing else to compute for the rejected cases
    if status != "ok" {
        return Ok(Row {
            max_abs_inv_cp: None,
            rel_inv_cp: None,
            rel_inv_num: None,
            rel_cp_num: None,
            status,
        });
    }

    let d_num = numerical_deriv(aa, method);
    Ok(Row {
        max_abs_inv_cp: Some(max_abs(&d_inv, &d_cp)),
        rel_inv_cp: Some(rel(&d_inv, &d_cp)),
        rel_inv_num: Some(rel(&d_inv, &d_num)),
        rel_cp_num: Some(rel(&d_cp, &d_num)),
        status,
    })
}

/// Returns the maximum absolute difference between two derivative sets
fn max_abs(a: &[Tensor4<6>; 3], b: &[Tensor4<6>; 3]) -> f64 {
    let mut max = 0.0;
    for k in 0..3 {
        for m in 0..6 {
            for n in 0..6 {
                max = f64::max(max, f64::abs(a[k].get(m, n) - b[k].get(m, n)));
            }
        }
    }
    max
}

/// Returns `max|a - b| / max(max|a|, max|b|)` (0 if both operands vanish)
fn rel(a: &[Tensor4<6>; 3], b: &[Tensor4<6>; 3]) -> f64 {
    let mut scale = 0.0;
    for k in 0..3 {
        for m in 0..6 {
            for n in 0..6 {
                scale = f64::max(scale, f64::abs(a[k].get(m, n)));
                scale = f64::max(scale, f64::abs(b[k].get(m, n)));
            }
        }
    }
    if scale > 0.0 { max_abs(a, b) / scale } else { 0.0 }
}

/// Arguments for the numerical differentiation of the eigenprojectors
struct NumArgs {
    method: EigenValMethod,
    calc: EigenProjsT2,
    ll: [f64; 3],
    projs: [Tensor2<6>; 3],
    aa: Tensor2<6>,
    k: usize,
    m: usize,
    n: usize,
}

/// Computes the derivatives of the eigenprojectors by numerical differentiation
fn numerical_deriv(aa: &Tensor2<6>, method: EigenValMethod) -> [Tensor4<6>; 3] {
    let mut args = NumArgs {
        method,
        calc: EigenProjsT2::new(),
        ll: [0.0; 3],
        projs: [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()],
        aa: aa.clone(),
        k: 0,
        m: 0,
        n: 0,
    };
    let mut d_num = [Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
    for k in 0..3 {
        args.k = k;
        for m in 0..6 {
            args.m = m;
            for n in 0..6 {
                args.n = n;
                let x = args.aa.get(n);
                let res = deriv1_central5(x, &mut args, |x, args| {
                    let original = args.aa.get(args.n);
                    args.aa.set(args.n, x);
                    args.calc
                        .calculate_mx(&mut args.ll, &mut args.projs, &args.aa, args.method)
                        .unwrap();
                    args.aa.set(args.n, original);
                    Ok(args.projs[args.k].get(args.m))
                })
                .unwrap();
                d_num[k].set(m, n, res);
            }
        }
    }
    d_num
}

/// Orthogonal transformation matrix from the papers
fn u_sym() -> [[f64; 3]; 3] {
    let r2 = f64::sqrt(2.0);
    [[1.0 / r2, -0.5, 0.5], [1.0 / r2, 0.5, -0.5], [0.0, 1.0 / r2, 1.0 / r2]]
}

/// Builds the symmetric matrix A = U ⋅ diag(d) ⋅ Uᵀ (and symmetrizes it)
fn build_a(u: &[[f64; 3]; 3], d: &[f64; 3]) -> [[f64; 3]; 3] {
    let mut a = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                a[i][j] += u[i][k] * d[k] * u[j][k];
            }
        }
    }
    for i in 0..3 {
        for j in (i + 1)..3 {
            let m = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = m;
            a[j][i] = m;
        }
    }
    a
}
