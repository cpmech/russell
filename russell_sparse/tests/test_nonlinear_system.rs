use russell_lab::algo::num_jacobian;
use russell_lab::*;
use russell_sparse::prelude::*;
use russell_sparse::StrError;

#[cfg(feature = "with_mumps")]
use serial_test::serial;

fn calc_residual(rr: &mut Vector, uu: &Vector) {
    let (d1, d2, d3, d4) = (uu[0], uu[1], uu[2], uu[3]);
    rr[0] = 2.0 * d1 + d1 * d1 * d1 * d1 + d2 + 3.0 * d1 * d2 * d2 - 9.0 * d4 + d4 * d4 * d4 * d4 - 0.2;
    rr[1] = d1 + 3.0 * d1 * d1 * d2 + 10.0 * d2 + 4.0 * d2 * d2 + 2.0 * d2 * d3 - 8.0 * d3 + 7.0 * d4 + 0.1;
    rr[2] = -8.0 * d2 + d2 * d2 + 3.0 * d3 + d3 * d3 + 2.0 * d4;
    rr[3] = -9.0 * d1 + 4.0 * d1 * d4 * d4 * d4 + 7.0 * d2 + 2.0 * d3 + 5.0 * d4 - 0.5;
}

fn calc_jacobian(jj: &mut CooMatrix, uu: &Vector) -> Result<(), StrError> {
    let (d1, d2, d3, d4) = (uu[0], uu[1], uu[2], uu[3]);

    jj.reset();

    jj.put(0, 0, 2.0 + 4.0 * d1 * d1 * d1 + 3.0 * d2 * d2)?;
    jj.put(0, 1, 1.0 + 6.0 * d1 * d2)?;
    jj.put(0, 2, 0.0)?;
    jj.put(0, 3, -9.0 + 4.0 * d4 * d4 * d4)?;

    jj.put(1, 0, 1.0 + 6.0 * d1 * d2)?;
    jj.put(1, 1, 10.0 + 3.0 * d1 * d1 + 8.0 * d2 + 2.0 * d3)?;
    jj.put(1, 2, -8.0 + 2.0 * d2)?;
    jj.put(1, 3, 7.0)?;

    jj.put(2, 0, 0.0)?;
    jj.put(2, 1, -8.0 + 2.0 * d2)?;
    jj.put(2, 2, 3.0 + 2.0 * d3)?;
    jj.put(2, 3, 2.0)?;

    jj.put(3, 0, -9.0 + 4.0 * d4 * d4 * d4)?;
    jj.put(3, 1, 7.0)?;
    jj.put(3, 2, 2.0)?;
    jj.put(3, 3, 5.0 + 12.0 * d1 * d4 * d4)?;
    Ok(())
}

#[test]
fn check_jacobian() {
    struct Args {}
    let args = &mut Args {};
    let neq = 4;
    let uu = Vector::from(&[1.0, -3.0, 7.0, -2.5]);
    let jj_num = num_jacobian(neq, 0.0, &uu, 1.0, args, |r, _, u, _| {
        calc_residual(r, u);
        Ok(())
    })
    .unwrap();
    let nnz = neq * neq;
    let mut jj_tri = CooMatrix::new(neq, neq, nnz, Sym::No).unwrap();
    calc_jacobian(&mut jj_tri, &uu).unwrap();
    let mut jj_ana = Matrix::new(neq, neq);
    jj_tri.to_dense(&mut jj_ana).unwrap();
    mat_approx_eq(&jj_ana, &jj_num, 1e-8);
}

fn solve_nonlinear_system(genie: Genie) -> Result<(), StrError> {
    let (neq, nnz) = (4, 16);
    let mut solver = LinSolver::new(genie)?;
    let mut jj = CooMatrix::new(neq, neq, nnz, Sym::No).unwrap();
    let mut rr = Vector::new(neq);
    let mut uu = Vector::from(&[0.0, 0.0, 0.0, 0.0]);
    let mut mdu = Vector::new(neq);
    let mut norm_rr0 = 1.0;
    println!(
        "{:>4}{:>13}{:>13}{:>13}{:>13}{:>15}",
        "it", "d1", "d2", "d3", "d4", "err"
    );
    let uu_ref = &[
        vec![0.000000, 0.000000, 0.000000, 0.000000],
        vec![-0.236393, -0.106230, -0.225574, -0.086557],
        vec![-0.196773, -0.079071, -0.171604, -0.074904],
        vec![-0.194395, -0.077412, -0.168376, -0.074249],
        vec![-0.194386, -0.077406, -0.168364, -0.074246],
        vec![-0.194386, -0.077406, -0.168364, -0.074246],
    ];
    let mut it = 0;
    while it < 10 {
        calc_residual(&mut rr, &uu);
        let err = if it == 0 {
            norm_rr0 = vec_norm(&rr, Norm::Euc);
            1.0
        } else {
            vec_norm(&rr, Norm::Euc) / norm_rr0
        };
        println!(
            "{:>4}{:>13.6}{:>13.6}{:>13.6}{:>13.6}{:>15.6e}",
            it, uu[0], uu[1], uu[2], uu[3], err
        );
        vec_approx_eq(&uu, &uu_ref[it], 1e-6);
        if err < 1e-13 {
            break;
        }
        calc_jacobian(&mut jj, &uu)?;
        solver.actual.factorize(&jj, None)?;
        solver.actual.solve(&mut mdu, &rr, false)?;
        vec_update(&mut uu, -1.0, &mdu)?;
        it += 1;
    }
    if it != 5 {
        Err("number of iterations must be 5")
    } else {
        Ok(())
    }
}

#[test]
#[serial]
#[cfg(feature = "with_mumps")]
fn test_nonlinear_system_mumps() -> Result<(), StrError> {
    solve_nonlinear_system(Genie::Mumps)
}

#[test]
fn test_nonlinear_system_umfpack() -> Result<(), StrError> {
    solve_nonlinear_system(Genie::Umfpack)
}
