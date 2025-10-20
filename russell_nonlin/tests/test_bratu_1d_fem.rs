#![allow(unused)]
use gemlab::prelude::*;
use russell_lab::{Norm, Vector};
use russell_nonlin::{AutoStep, Config, IniDir, Method, NoArgs, Output, Solver, State, Status, Stop, System};

const DRAW_MESH: bool = true;

// Runs the test
fn run_test(
    bordering: bool,
    alpha: f64,
    npt: usize,
    stop: Stop,
    auto: AutoStep,
    alpha0_lam_crit_tol: f64,
    alpha02_1st_lam_crit_tol: f64,
    alpha02_2nd_lam_crit_tol: f64,
) {
    // filename stem
    let key = if auto.yes() { "auto" } else { "fixed" };
    let stem = format!(
        "/tmp/russell_nonlin/test_bratu_1d_fem_alpha{}_npt{}_{}",
        alpha, npt, key
    );

    // generate the mesh
    assert!(npt >= 2);
    let ncell = npt - 1;
    let mut mesh = Mesh::new_zero_homogeneous(2, npt, ncell, GeoKind::Lin2).unwrap();
    let dx = 1.0 / (npt - 1) as f64;
    for i in 1..npt {
        mesh.points[i].coords[0] = (i as f64) * dx;
        mesh.cells[i - 1].points[0] = i - 1;
        mesh.cells[i - 1].points[1] = i;
    }
    if DRAW_MESH {
        let mut fig = Figure::new();
        fig.show_cell_ids(true).show_point_ids(true).show_point_dots(true);
        fig.draw(&mesh, &format!("{}_mesh.svg", stem)).unwrap();
    }
    mesh.check_all().unwrap();
}

#[test]
fn test_bratu_1d_lmm_auto() {
    let bordering = false;
    let auto = AutoStep::Yes;
    for alpha in [0.0] {
        for (npt, tol1, tol2, tol3) in [(8, 0.0, 0.0, 0.0)] {
            let max_nrm_max = if alpha == 0.0 { 9.0 } else { 30.0 };
            let stop = Stop::MaxNormU(max_nrm_max, Norm::Inf, 0, npt);
            run_test(bordering, alpha, npt, stop, auto, tol1, tol2, tol3);
        }
    }
}
