use plotpy::StrError;
use russell_ode::prelude::*;
use serde::Serialize;
use std::fs::{self, File};
use std::path::Path;

const PATH_KEY: &str = "/tmp/russell_ode/brusselator_pde_radau5_2nd";

#[derive(Serialize)]
pub struct ProblemData {
    pub alpha: f64,
    pub npoint: usize,
}

fn main() -> Result<(), StrError> {
    // ODE system
    let alpha = 0.1;
    let npoint = 129;
    let second_book = true;
    let (system, t0, mut yy0, mut args) = Samples::brusselator_pde(alpha, npoint, second_book, false);

    // final t
    let t1 = 11.0;

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.set_tolerances(1e-4, 1e-4, None)?;

    // allocate the solver
    let mut solver = OdeSolver::new(params, &system)?;

    // output
    let h_out = 1.0;
    solver
        .enable_output()
        .set_dense_h_out(h_out)?
        .set_dense_file_writing(PATH_KEY)?;

    // solve the ODE system
    solver.solve(&mut yy0, t0, t1, None, &mut args)?;

    // get statistics
    let stat = solver.stats();
    println!("{}", stat);

    // save problem data
    let problem_data = ProblemData { alpha, npoint };
    problem_data.write_json(&format!("{}_problem_data.json", PATH_KEY))?;
    Ok(())
}

impl ProblemData {
    pub fn write_json(&self, full_path: &str) -> Result<(), StrError> {
        let path = Path::new(full_path).to_path_buf();
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).map_err(|_| "cannot create directory")?;
        }
        let mut file = File::create(&path).map_err(|_| "cannot create file")?;
        serde_json::to_writer(&mut file, &self).map_err(|_| "cannot write file")?;
        Ok(())
    }
}
