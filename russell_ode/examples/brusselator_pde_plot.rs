use plotpy::{Plot, StrError, Surface};
use russell_lab::Vector;
use russell_ode::{prelude::*, PdeDiscreteLaplacian2d};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use structopt::StructOpt;

const OUT_DIR: &str = "/tmp/russell_ode/";

/// Command line options
#[derive(StructOpt)]
#[structopt(name = "BrusselatorPlotter", about = "Plot Brusselator PDE Results")]
struct Options {
    /// Second book
    #[structopt(long)]
    second_book: bool,
}

fn main() -> Result<(), StrError> {
    // parse options
    let opt = Options::from_args();
    let version = if opt.second_book { "_2nd" } else { "" };
    let path_key = format!("{}/brusselator_pde_radau5{}", OUT_DIR, version);

    // read summary
    println!("... generating figures ...");
    let count = OutCount::read_json(&format!("{}_count.json", path_key))?;
    let mut uu_plot = Graph::new(opt.second_book, &path_key, count.n)?;
    let mut vv_plot = Graph::new(opt.second_book, &path_key, count.n)?;

    // loop over time stations
    for idx in 0..count.n {
        let path = format!("{}_{}.json", path_key, idx);
        let res = OutData::read_json(&path)?;
        uu_plot.draw(res.x, &res.y, false)?;
        vv_plot.draw(res.x, &res.y, true)?;
    }

    // save figures
    uu_plot.save(false)?;
    vv_plot.save(true)?;
    println!("... {} ...", path_key);
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
    second_book: bool,
    path_key: String,
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
    pub fn new(second_book: bool, path_key: &String, n_files: usize) -> Result<Self, StrError> {
        let problem_data = ProblemData::read_json(&format!("{}_problem_data.json", path_key))?;
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
        let (nrow, ncol) = if second_book { (4, 3) } else { (6, 4) };
        assert_eq!(nrow * ncol, n_files);
        let index = 1;
        Ok(Graph {
            second_book,
            path_key: path_key.clone(),
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
        if self.second_book {
            surf.set_with_wireframe(false)
                .set_with_surface(true)
                .set_surf_line_color("black")
                .set_surf_line_width(0.3)
                .set_row_stride(10)
                .set_col_stride(10)
                .set_colormap_name("terrain");
        } else {
            surf.set_with_wireframe(true).set_with_surface(false);
        }
        surf.draw(&self.grid_x, &self.grid_y, &self.grid_z);
        let title = format!("{} @ t = {:?}", field, t);
        self.plot.add(&surf).set_title(&title).set_camera(30.0, 210.0);
        if self.second_book {
            self.plot
                .set_hide_xticks()
                .set_hide_yticks()
                .set_label_x_and_pad("x", -15.0)
                .set_label_y_and_pad("y", -15.0)
                .set_label_z("")
                .set_vertical_gap(0.2);
        } else {
            self.plot
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
        }
        self.index += 1;
        Ok(())
    }

    pub fn save(&mut self, v_field: bool) -> Result<(), StrError> {
        let field = if v_field { "v" } else { "u" };
        let path = format!("{}_{}.svg", self.path_key, field);
        let (width, height) = if self.second_book {
            (800.0, 1000.0)
        } else {
            (1000.0, 1500.0)
        };
        self.plot.set_figure_size_points(width, height).save(&path)
    }
}
