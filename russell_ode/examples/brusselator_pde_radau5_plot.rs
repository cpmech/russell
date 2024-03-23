use plotpy::{Plot, StrError, Surface};
use russell_lab::Vector;
use russell_ode::{prelude::*, PdeDiscreteLaplacian2d};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

const PATH_KEY: &str = "/tmp/russell_ode/brusselator_pde_radau5";

fn main() -> Result<(), StrError> {
    println!("... generating figures ...");
    let summary = OutSummary::read_json(format!("{}_summary.json", PATH_KEY).as_str())?;
    let mut uu_plot = Graph::new(summary.count)?;
    let mut vv_plot = Graph::new(summary.count)?;
    for idx in 0..summary.count {
        let path = format!("{}_{}.json", PATH_KEY, idx);
        let res = OutData::read_json(path.as_str())?;
        uu_plot.draw(res.x, &res.y, false)?;
        vv_plot.draw(res.x, &res.y, true)?;
    }
    uu_plot.save(false)?;
    vv_plot.save(true)?;
    println!("... done ...");
    Ok(())
}

#[derive(Deserialize)]
pub struct ProblemData {
    pub alpha: f64,
    pub npoint: usize,
}

impl ProblemData {
    pub fn read_json(full_path: &str) -> Result<Self, StrError> {
        let path = Path::new(full_path).to_path_buf();
        let input = File::open(path).map_err(|_| "cannot open file")?;
        let buffered = BufReader::new(input);
        let stat = serde_json::from_reader(buffered).map_err(|_| "cannot parse JSON file")?;
        Ok(stat)
    }
}

struct Graph {
    npoint: usize,
    fdm: PdeDiscreteLaplacian2d,
    grid_x: Vec<Vec<f64>>,
    grid_y: Vec<Vec<f64>>,
    grid_z: Vec<Vec<f64>>,
    plot: Plot,
    nrow: usize,
    ncol: usize,
    index: usize,
}

impl Graph {
    pub fn new(n_files: usize) -> Result<Self, StrError> {
        let problem_data = ProblemData::read_json(format!("{}_problem_data.json", PATH_KEY).as_str())?;
        let alpha = problem_data.alpha;
        let npoint = problem_data.npoint;
        let fdm = PdeDiscreteLaplacian2d::new(alpha, alpha, 0.0, 1.0, 0.0, 1.0, npoint, npoint)?;
        let mut grid_x = vec![vec![0.0; npoint]; npoint];
        let mut grid_y = vec![vec![0.0; npoint]; npoint];
        fdm.loop_over_grid_points(|m, x, y| {
            let i = m % npoint;
            let j = m / npoint;
            grid_x[i][j] = x;
            grid_y[i][j] = y;
        });
        let plot = Plot::new();
        let ncol = 4;
        let nrow = n_files / 4;
        let index = 1;
        Ok(Graph {
            npoint,
            fdm,
            grid_x,
            grid_y,
            grid_z: vec![vec![0.0; npoint]; npoint],
            plot,
            nrow,
            ncol,
            index,
        })
    }

    pub fn draw(&mut self, t: f64, yy: &Vector, v_field: bool) -> Result<(), StrError> {
        let field = if v_field { "V" } else { "U" };
        self.fdm.loop_over_grid_points(|m, _, _| {
            let i = m % self.npoint;
            let j = m / self.npoint;
            let s = if v_field { self.npoint * self.npoint } else { 0 };
            self.grid_z[i][j] = yy[s + m];
        });
        let mut surf = Surface::new();
        self.plot.set_subplot_3d(self.nrow, self.ncol, self.index);
        surf.set_with_wireframe(true)
            .set_with_surface(false)
            .draw(&self.grid_x, &self.grid_y, &self.grid_z);
        let title = format!("{} @ t = {:?}", field, t);
        self.plot
            .add(&surf)
            .set_title(title.as_str())
            .set_camera(30.0, 210.0)
            .set_hide_xticks()
            .set_hide_yticks()
            .set_hide_zticks()
            .set_num_ticks_x(3)
            .set_num_ticks_y(3)
            .set_num_ticks_z(3)
            .set_label_x_and_pad("x", -15.0)
            .set_label_y_and_pad("y", -15.0)
            .set_label_z_and_pad(field, -15.0)
            .set_vertical_gap(0.2);
        self.index += 1;
        Ok(())
    }

    pub fn save(&mut self, v_field: bool) -> Result<(), StrError> {
        let field = if v_field { "v" } else { "u" };
        let path = format!("{}_{}.svg", PATH_KEY, field);
        let width = 1000.0;
        let height = 1.5 * width;
        self.plot.set_figure_size_points(width, height).save(path.as_str())
    }
}
