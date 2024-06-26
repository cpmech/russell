use plotpy::{Plot, Surface};
use russell_lab::approx_eq;
use russell_ode::{Method, OdeSolver, Params, Samples};
use serde::Deserialize;
use std::{env, fs::File, io::BufReader, path::Path};

const SAVE_FIGURE: bool = false;

#[test]
fn test_radau5_brusselator_pde() {
    // get get ODE system
    let alpha = 2e-3;
    let npoint = 9;
    let (system, t0, mut yy0, mut args) = Samples::brusselator_pde(alpha, npoint, false, false);

    // final time
    let t1 = 0.1;

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.set_tolerances(1e-3, 1e-3, None).unwrap();

    // allocate the solver
    let mut solver = OdeSolver::new(params, system).unwrap();

    // solve the ODE system
    let yy = &mut yy0;
    solver.solve(yy, t0, t1, None, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();
    println!("{}", stat);
    assert_eq!(stat.n_function, 24);

    // check results at middle node
    let ij_mid = (npoint - 1) / 2; // i or j indices of middle node
    let m_mid = ij_mid + ij_mid * npoint; // vector index of middle node
    let s = npoint * npoint;
    let ndim = 2 * s;
    let uu = &yy.as_data()[0..s];
    let vv = &yy.as_data()[s..ndim];
    let math = ReferenceData::read("data/reference/brusselator_pde_2d_n9_mathematica.json");
    approx_eq(uu[m_mid], math.uu[ij_mid][ij_mid], 1e-7);
    approx_eq(vv[m_mid], math.vv[ij_mid][ij_mid], 1e-7);

    if SAVE_FIGURE {
        let plot = |v_field: bool| {
            let mut surf1 = Surface::new();
            let mut surf2 = Surface::new();
            surf1
                .set_with_surface(false)
                .set_with_wireframe(true)
                .set_wire_line_style("--");
            surf2
                .set_with_surface(false)
                .set_with_wireframe(true)
                .set_wire_line_style("-")
                .set_wire_line_color("red");
            let mut grid_x = vec![vec![0.0; npoint]; npoint];
            let mut grid_y = vec![vec![0.0; npoint]; npoint];
            let mut grid_z = vec![vec![0.0; npoint]; npoint];
            args.loop_over_grid_points(|m, x, y| {
                let row = m / npoint;
                let col = m % npoint;
                grid_x[col][row] = x;
                grid_y[col][row] = y;
                if v_field {
                    let s = npoint * npoint;
                    grid_z[col][row] = yy[s + m];
                } else {
                    grid_z[col][row] = yy[m];
                }
            });
            surf1.draw(&grid_x, &grid_y, &grid_z);
            if v_field {
                surf2.draw(&math.xx, &math.yy, &math.vv);
            } else {
                surf2.draw(&math.xx, &math.yy, &math.uu);
            }
            let mut plot = Plot::new();
            let field = if v_field { "v".to_string() } else { "u".to_string() };
            let path = format!("/tmp/russell_ode/test_radau5_brusselator_pde_{}.svg", field);
            plot.add(&surf1)
                .add(&surf2)
                .set_title(&format!("{} @ t = {}", field.to_uppercase(), t1))
                .set_label_z(&field.to_uppercase())
                .set_save_pad_inches(0.3)
                .set_figure_size_points(600.0, 600.0)
                .save(&path)
                .unwrap();
        };
        plot(false);
        plot(true);
    }
}

#[derive(Deserialize)]
struct ReferenceData {
    #[allow(unused)]
    pub t: f64,
    pub xx: Vec<Vec<f64>>,
    pub yy: Vec<Vec<f64>>,
    pub uu: Vec<Vec<f64>>,
    pub vv: Vec<Vec<f64>>,
}

impl ReferenceData {
    pub fn read(rel_path: &str) -> Self {
        let full_path = format!("{}/{}", env::var("CARGO_MANIFEST_DIR").unwrap(), rel_path);
        let path = Path::new(&full_path).to_path_buf();
        let input = File::open(path).map_err(|_| "cannot open file").unwrap();
        let buffered = BufReader::new(input);
        let data = serde_json::from_reader(buffered)
            .map_err(|_| "cannot parse JSON file")
            .unwrap();
        data
    }
}
