use plotpy::{Plot, StrError, Surface};
use russell_lab::Vector;
use russell_ode::{prelude::*, PdeDiscreteLaplacian2d};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

const PATH_KEY: &str = "/tmp/russell_ode/brusselator_pde_radau5";

fn main() -> Result<(), StrError> {
    let mut graph = Graph::new()?;
    let summary = OutSummary::read_json(format!("{}_summary.json", PATH_KEY).as_str())?;
    for idx in 0..summary.count {
        let path = format!("{}_{}.json", PATH_KEY, idx);
        println!("{}", path);
        let res = OutData::read_json(path.as_str())?;
        graph.snapshot(&res.y)?;
    }
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
    laplacian: PdeDiscreteLaplacian2d,
    grid_x: Vec<Vec<f64>>,
    grid_y: Vec<Vec<f64>>,
    grid_z: Vec<Vec<f64>>,
    count: usize,
}

impl Graph {
    pub fn new() -> Result<Self, StrError> {
        let problem_data = ProblemData::read_json(format!("{}_problem_data.json", PATH_KEY).as_str())?;
        let alpha = problem_data.alpha;
        let npoint = problem_data.npoint;
        let laplacian = PdeDiscreteLaplacian2d::new(alpha, alpha, 0.0, 1.0, 0.0, 1.0, npoint, npoint)?;
        let mut grid_x = vec![vec![0.0; npoint]; npoint];
        let mut grid_y = vec![vec![0.0; npoint]; npoint];
        laplacian.loop_over_grid_points(|m, x, y| {
            let row = m / npoint;
            let col = m % npoint;
            grid_x[col][row] = x;
            grid_y[col][row] = y;
        });
        Ok(Graph {
            npoint,
            laplacian,
            grid_x,
            grid_y,
            grid_z: vec![vec![0.0; npoint]; npoint],
            count: 0,
        })
    }

    pub fn snapshot(&mut self, yy: &Vector) -> Result<(), StrError> {
        // u
        self.laplacian.loop_over_grid_points(|m, _, _| {
            let row = m / self.npoint;
            let col = m % self.npoint;
            self.grid_z[col][row] = yy[m];
        });
        let mut surf = Surface::new();
        surf.set_with_wireframe(true)
            .set_with_surface(false)
            .draw(&self.grid_x, &self.grid_y, &self.grid_z);
        let mut plot = Plot::new();
        let path = format!("{}_u_{:0>3}.svg", PATH_KEY, self.count);
        plot.add(&surf).set_camera(30.0, 210.0).save(path.as_str())?;

        // v
        self.laplacian.loop_over_grid_points(|m, _, _| {
            let s = self.npoint * self.npoint;
            let row = m / self.npoint;
            let col = m % self.npoint;
            self.grid_z[col][row] = yy[s + m];
        });
        let mut surf = Surface::new();
        surf.set_with_wireframe(true)
            .set_with_surface(false)
            .draw(&self.grid_x, &self.grid_y, &self.grid_z);
        let mut plot = Plot::new();
        let path = format!("{}_v_{:0>2}.svg", PATH_KEY, self.count);
        plot.add(&surf).set_camera(30.0, 210.0).save(path.as_str())?;

        // next
        self.count += 1;
        Ok(())
    }
}
