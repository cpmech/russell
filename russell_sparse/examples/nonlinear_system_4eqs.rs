use russell_lab::{vec_norm, vec_update, Norm, Vector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;
use structopt::StructOpt;

/// Command line options
#[derive(StructOpt, Debug)]
#[structopt(
    name = "nonlinear_system_4equations",
    about = "Solve a nonlinear system four equations."
)]
struct Options {
    #[structopt(short = "g", long, default_value = "Umfpack")]
    genie: String,
}

fn main() -> Result<(), StrError> {
    // parse options
    let opt = Options::from_args();

    // select linear solver
    let genie = match opt.genie.to_lowercase().as_str() {
        "mumps" => Genie::Mumps,
        "umfpack" => Genie::Umfpack,
        _ => Genie::Umfpack,
    };
    println!("... solving problem with {:?} ...", genie);
    let mut solver = LinSolver::new(genie)?;

    // allocate Jacobian matrix (J) as SparseMatrix
    let (neq, nnz) = (4, 16);
    let mut jj = SparseMatrix::new_coo(neq, neq, nnz, Sym::No).unwrap();

    // allocate residual (rr), vector of unknowns (uu), and minus-uu (mdu)
    let mut rr = Vector::new(neq);
    let mut uu = Vector::from(&[0.0, 0.0, 0.0, 0.0]);
    let mut mdu = Vector::new(neq);

    // message
    println!(
        "{:>4}{:>13}{:>13}{:>13}{:>13}{:>15}",
        "it", "d1", "d2", "d3", "d4", "err"
    );

    // loop until the residual is smaller than tolerance
    const TOLERANCE: f64 = 1e-13;
    let mut it = 0;
    let mut norm_rr0 = 1.0;
    while it < 10 {
        // calculate residual
        calc_residual(&mut rr, &uu);

        // first error or use the norm of residual as error,
        // normalized by the norm of the first residual
        let err = if it == 0 {
            norm_rr0 = vec_norm(&rr, Norm::Euc);
            1.0
        } else {
            vec_norm(&rr, Norm::Euc) / norm_rr0
        };

        // message
        println!(
            "{:>4}{:>13.6}{:>13.6}{:>13.6}{:>13.6}{:>15.6e}",
            it, uu[0], uu[1], uu[2], uu[3], err
        );

        // exit point
        if err < TOLERANCE {
            break;
        }

        // calculate Jacobian matrix
        calc_jacobian(jj.get_coo_mut()?, &uu)?;

        // factorize and solve linear system: J * mdu = rr
        solver.actual.factorize(&mut jj, None)?;
        solver.actual.solve(&mut mdu, &jj, &rr, false)?;

        // update the vector of unknowns: uu -= mdu
        vec_update(&mut uu, -1.0, &mdu)?;
        it += 1;
    }
    Ok(())
}

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
