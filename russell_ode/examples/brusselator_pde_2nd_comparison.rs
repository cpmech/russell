use plotpy::{Plot, Surface};
use russell_ode::{Method, OdeSolver, Params, Samples};
use serde::Deserialize;
use std::{env, fs::File, io::BufReader, path::Path};

// This example compares Russell results with Mathematica results for
// the Brusselator PDE in 2D with periodic BCs as in the second book
// The Mathematica code is at the end of this file as comments

fn main() {
    // get get ODE system
    let alpha = 0.1;
    let npoint = 101;
    let second_book = true;
    let (system, t0, yy0, mut args) = Samples::brusselator_pde(alpha, npoint, second_book, false);

    // final t
    let t1 = 11.5;

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.set_tolerances(1e-4, 1e-4, None).unwrap();

    // solve the ODE system
    let mut solver = OdeSolver::new(params, &system).unwrap();
    let mut yy = yy0.clone();
    solver.solve(&mut yy, t0, t1, None, None, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();
    println!("{}", stat);

    // load Mathematica results
    let dat_t0 = ReferenceData::read("data/reference/brusselator_pde_2d_n9_t0_math.json");
    let dat_t1 = ReferenceData::read("data/reference/brusselator_pde_2d_n9_t1_math.json");

    // generate plots
    let mut grid_x = vec![vec![0.0; npoint]; npoint];
    let mut grid_y = vec![vec![0.0; npoint]; npoint];
    let mut grid_z = vec![vec![0.0; npoint]; npoint];
    let mut do_plot = |t_ini: bool, v_field: bool| {
        let mut surf1 = Surface::new();
        let mut surf2 = Surface::new();
        surf1
            .set_with_surface(false)
            .set_with_wireframe(true)
            .set_row_stride(5)
            .set_col_stride(5)
            .set_wire_line_style("--");
        if t_ini {
            surf2
                .set_with_surface(false)
                .set_with_points(true)
                .set_point_style("o")
                .set_point_color("red");
        } else {
            surf2
                .set_with_surface(false)
                .set_with_wireframe(true)
                .set_wire_line_style("-")
                .set_wire_line_color("red");
        }
        args.loop_over_grid_points(|m, x, y| {
            let row = m / npoint;
            let col = m % npoint;
            grid_x[col][row] = x;
            grid_y[col][row] = y;
            let sol = if t_ini { &yy0 } else { &yy };
            if v_field {
                let s = npoint * npoint;
                grid_z[col][row] = sol[s + m];
            } else {
                grid_z[col][row] = sol[m];
            }
        });
        surf1.draw(&grid_x, &grid_y, &grid_z);
        if t_ini {
            if v_field {
                surf2.draw(&dat_t0.xx, &dat_t0.yy, &dat_t0.vv);
            } else {
                surf2.draw(&dat_t0.xx, &dat_t0.yy, &dat_t0.uu);
            }
        } else {
            if v_field {
                surf2.draw(&dat_t1.xx, &dat_t1.yy, &dat_t1.vv);
            } else {
                surf2.draw(&dat_t1.xx, &dat_t1.yy, &dat_t1.uu);
            }
        }
        let mut plot = Plot::new();
        let t = if t_ini { t0 } else { t1 };
        let st = if t_ini { "t0" } else { "t1" };
        let field = if v_field { "v".to_string() } else { "u".to_string() };
        let path = format!("/tmp/russell_ode/brusselator_pde_2nd_comparison_{}_{}.svg", st, field);
        plot.add(&surf1)
            .add(&surf2)
            .set_title(format!("{} @ t = {}", field.to_uppercase(), t).as_str())
            .set_label_z(&field.to_uppercase())
            .set_save_pad_inches(0.3)
            .set_figure_size_points(600.0, 600.0)
            .save(path.as_str())
            .unwrap();
    };
    do_plot(true, true);
    do_plot(true, false);
    do_plot(false, true);
    do_plot(false, false);
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
        let path = Path::new(full_path.as_str()).to_path_buf();
        let input = File::open(path).map_err(|_| "cannot open file").unwrap();
        let buffered = BufReader::new(input);
        let data = serde_json::from_reader(buffered)
            .map_err(|_| "cannot parse JSON file")
            .unwrap();
        data
    }
}

// Mathematica Code ////////////////////////////////////////////////////////////////////////////

// (* MODEL *)
//
// A = 1; B = 3.4; \[Alpha] = 0.1;
// inhomogeneity = If[(x - 0.3)^2 + (y - 0.6)^2 <= 0.1^2 && t >= 1.1, 5, 0];
// dudt = A - (1 + B) u[t, x, y] + u[t, x, y]^2 v[t, x, y] + \[Alpha] (D[u[t, x, y], {x, 2}] + D[u[t, x, y], {y, 2}]) + inhomogeneity;
// dvdt = B u[t, x, y] - u[t, x, y]^2 v[t, x, y] + \[Alpha] (D[v[t, x, y], {x, 2}] + D[v[t, x, y], {y, 2}]);
// PDE = {D[u[t, x, y], t] == dudt, D[v[t, x, y], t] == dvdt};
// IC = {
//    u[0, x, y] == 22 y (1 - y)^(3/2),
//    v[0, x, y] == 27 x (1 - x)^(3/2)
//    };
// BC = {
//    u[t, 0, y] == u[t, 1, y],
//    u[t, x, 0] == u[t, x, 1],
//    v[t, 0, y] == v[t, 1, y],
//    v[t, x, 0] == v[t, x, 1]
//    };
//
// (* NUMERICAL SOLUTION *)
//
// npoint = 101;
// dx = 1/(npoint - 1);
// t1 = 11.5;
// sol = NDSolve[{PDE, BC, IC}, {u, v}, {t, 0, t1}, {x, 0, 1}, {y, 0, 1}, Method -> {"PDEDiscretization" -> {"MethodOfLines", "SpatialDiscretization" -> {"TensorProductGrid", "MinPoints" -> npoint - 1, "MaxPoints" -> npoint - 1}}}]
//
// (* GENERATE REFERENCE RESULTS *)
//
// generate[tIni_] := Module[{np, del, tref, xx, yy, uu, vv, mathData},
//    np = 21;
//    del = 1/(np - 1);
//    tref = If[tIni, 0, t1];
//    xx = Table[x, {x, 0, 1, del}, {y, 0, 1, del}];
//    yy = Table[y, {x, 0, 1, del}, {y, 0, 1, del}];
//    uu = Table[ First[u[tref, x, y] /. sol], {x, 0, 1, del}, {y, 0, 1, del}];
//    vv = Table[ First[v[tref, x, y] /. sol], {x, 0, 1, del}, {y, 0, 1, del}];
//    mathData = {"t" -> tref, "xx" -> xx, "yy" -> yy, "uu" -> uu, "vv" -> vv};
//    If[tIni, Export["brusselator_pde_2d_n9_t0_math.json", mathData, "JSON", "Compact" -> True],
//     Export["brusselator_pde_2d_n9_t1_math.json", mathData, "JSON", "Compact" -> True]]
//    ];
// generate[True]
// generate[False]
