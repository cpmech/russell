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
    let (system, mut data, mut args) = Samples::brusselator_pde(alpha, npoint, second_book, false);

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.set_tolerances(1e-4, 1e-4, None)?;

    // output
    let mut out = Output::new();
    let h_out = 1.0;
    out.set_dense_file_writing(true, h_out, PATH_KEY)?;

    // solve the ODE system
    let mut solver = OdeSolver::new(params, &system)?;
    let x1 = 11.0;
    solver.solve(&mut data.y0, data.x0, x1, None, Some(&mut out), &mut args)?;

    // get statistics
    let stat = solver.stats();
    println!("{}", stat);

    // save problem data
    let problem_data = ProblemData { alpha, npoint };
    problem_data.write_json(format!("{}_problem_data.json", PATH_KEY).as_str())?;
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
