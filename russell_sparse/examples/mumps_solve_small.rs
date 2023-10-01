use russell_chk::vec_approx_eq;
use russell_lab::{mat_inverse, mat_norm, Norm};
use russell_lab::{Matrix, Vector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // constants
    let ndim = 5; // number of rows = number of columns
    let nnz = 13; // number of non-zero values, including duplicates

    // allocate solver
    let mut mumps = SolverMUMPS::new()?;

    // allocate the coefficient matrix
    //  2  3  .  .  .
    //  3  .  4  .  6
    //  . -1 -3  2  .
    //  .  .  1  .  .
    //  .  4  2  .  1
    let mut coo = SparseMatrix::new_coo(ndim, ndim, nnz, None, true)?;
    coo.put(0, 0, 1.0)?; // << (0, 0, a00/2) duplicate
    coo.put(0, 0, 1.0)?; // << (0, 0, a00/2) duplicate
    coo.put(1, 0, 3.0)?;
    coo.put(0, 1, 3.0)?;
    coo.put(2, 1, -1.0)?;
    coo.put(4, 1, 4.0)?;
    coo.put(1, 2, 4.0)?;
    coo.put(2, 2, -3.0)?;
    coo.put(3, 2, 1.0)?;
    coo.put(4, 2, 2.0)?;
    coo.put(2, 3, 2.0)?;
    coo.put(1, 4, 6.0)?;
    coo.put(4, 4, 1.0)?;

    // parameters
    let mut params = LinSolParams::new();
    params.compute_error_estimates = true;
    params.compute_condition_numbers = true;
    params.verbose = false;

    // call factorize
    mumps.factorize(&mut coo, Some(params))?;

    // allocate x and rhs
    let mut x = Vector::new(ndim);
    let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);

    // calculate the solution
    mumps.solve(&mut x, &coo, &rhs, false)?;
    println!("x =\n{}", x);

    // check the results
    let correct = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    vec_approx_eq(x.as_data(), &correct, 1e-14);

    // analysis
    let a = coo.as_dense();
    let mut ai = Matrix::new(5, 5);
    let det_a = mat_inverse(&mut ai, &a).unwrap();
    let norm_a = mat_norm(&a, Norm::Inf);
    let norm_ai = mat_norm(&ai, Norm::Inf);
    let cond = norm_a * norm_ai;
    let rcond = 1.0 / cond;
    let verify = VerifyLinSys::new(&coo, &x, &rhs).unwrap();
    let mut stats = StatsLinSol::new();
    mumps.update_stats(&mut stats);
    let s = &stats.mumps_stats;
    println!("\n___ ANALYSIS ________________________");
    println!("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
    println!("a =\n{}", a);
    println!("a⁻¹ =\n{:.5}", ai);
    println!("              det(a) = {:?}", det_a);
    println!("                ‖a‖∞ = {:?}", norm_a);
    println!("              ‖a⁻¹‖∞ = {:?}", norm_ai);
    println!("cond = ‖a‖∞ · ‖a⁻¹‖∞ = {:?}", cond);
    println!("    rcond = 1 / cond = {:?}", rcond);
    println!("           max_abs_a = {:?}", verify.max_abs_a);
    println!("          max_abs_ax = {:?}", verify.max_abs_ax);
    println!("        max_abs_diff = {:?}", verify.max_abs_diff);
    println!("      relative_error = {:?}", verify.relative_error);
    println!("              norm_a = {:?}", &s.inf_norm_a);
    println!("              norm_x = {:?}", &s.inf_norm_x);
    println!("            residual = {:?}", &s.scaled_residual);
    println!("              omega1 = {:?}", &s.backward_error_omega1);
    println!("              omega2 = {:?}", &s.backward_error_omega2);
    println!("               delta = {:?}", &s.normalized_delta_x);
    println!("               cond1 = {:?}", &s.condition_number1);
    println!("               cond2 = {:?}", &s.condition_number2);
    Ok(())
}
