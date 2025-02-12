use crate::{AsArray2D, StrError};
use crate::{Matrix, NumMatrix};
use plotpy::{Canvas, Plot, Text};
use std::cmp;
use std::fmt::Write;
use std::{collections::HashMap, vec};

/// Graph defines a (directed) graph structure
///
/// Warning: The algorithms here are not as optimized as they could be.
/// A specialized graph library certainly would provide better performance.
/// Also, the matrix is not sparse, so it is not suitable for large graphs.
pub struct Graph {
    /// Holds all edges (connectivity)
    ///
    /// Size: (nedge, 2)
    edges: NumMatrix<usize>,

    /// Holds the weights of edges
    ///
    /// The default value is 1.0.
    ///
    /// Size: nedge
    weights: Vec<f64>,

    /// Maps a node index to a list of edges sharing the node
    ///
    /// Size: nnode
    shares: HashMap<usize, Vec<usize>>,

    /// Holds the distances (is the distance matrix)
    ///
    /// If the real coordinates of points are provided, the distance is the Euclidean distance
    /// multiplied by the weight of the edge. Otherwise, the distance is simply the weight of the edge.
    ///
    /// Size: (nnode, nnode)
    dist: Matrix,

    /// Holds the next tree connection (next-hop matrix)
    ///
    /// The value of usize::MAX is used to indicate "no connection".
    ///
    /// Size: (nnode, nnode)
    next: NumMatrix<usize>,

    /// Indicates whether the distance and 'next' matrices are ready (e.g., after shortest paths have been computed)
    ready_path: bool,
}

impl Graph {
    /// Creates a new graph from a series of edges
    ///
    /// Each edge must contain two nodes or more, with the first two nodes
    /// defining the edge and the others being ignored.
    pub fn new<'a, T>(edges: &'a T) -> Self
    where
        T: AsArray2D<'a, usize>,
    {
        let mut shares = HashMap::new();
        let (nedge, ncorner) = edges.size();
        assert!(ncorner >= 2, "edges must have at least two nodes");
        for e in 0..nedge {
            let (a, b) = (edges.at(e, 0), edges.at(e, 1));
            shares.entry(a).or_insert(Vec::new()).push(e);
            shares.entry(b).or_insert(Vec::new()).push(e);
        }
        let nnode = shares.len();
        Graph {
            edges: NumMatrix::from(edges),
            weights: vec![1.0; nedge],
            shares,
            dist: Matrix::new(nnode, nnode),
            next: NumMatrix::new(nnode, nnode),
            ready_path: false,
        }
    }

    /// Sets the weight of an edge
    ///
    /// **Important**: The weight must be non-negative.
    ///
    /// # Panics
    ///
    /// This function may panic if the edge is out of bounds or the weight is negative.
    pub fn set_weight(&mut self, edge: usize, non_neg_value: f64) -> &mut Self {
        assert!(non_neg_value >= 0.0, "edge weight must be ≥ 0");
        self.weights[edge] = non_neg_value;
        self.ready_path = false;
        self
    }

    /// Returns the number of edges
    pub fn get_nedge(&self) -> usize {
        self.edges.nrow()
    }

    /// Returns the number of nodes
    pub fn get_nnode(&self) -> usize {
        self.shares.len()
    }

    /// Computes the shortest paths using the Floyd-Warshall algorithm
    ///
    /// An example of a graph with weights:
    ///
    /// ```text
    ///          $10
    ///    0 ––––––––––→ 3
    ///    │      1      ↑
    ///    │             │
    /// $5 │ 0         3 │ $1
    ///    │             │
    ///    ↓      2      |
    ///    1 ––––––––––→ 2
    ///          $3
    ///
    /// the weights are indicated by the dollar sign
    /// ```
    ///
    /// The initial distance matrix is:
    ///
    /// ```text
    /// j=   0  1  2  3
    ///   ┌             ┐ i=
    ///   │  0  5  ∞ 10 │  0  ⇒  w(0→1)=5, w(0→3)=10
    ///   │  ∞  0  3  ∞ │  1  ⇒  w(1→2)=3
    ///   │  ∞  ∞  0  1 │  2  ⇒  w(2→3)=1
    ///   │  ∞  ∞  ∞  0 │  3
    ///   └             ┘
    /// ∞ means that there are no connections from i to j
    /// i,j are node indices
    /// ```
    ///
    /// The final distance matrix is:
    ///
    /// ```text
    /// ┌         ┐
    /// │ 0 5 8 9 │
    /// │ ∞ 0 3 4 │
    /// │ ∞ ∞ 0 1 │
    /// │ ∞ ∞ ∞ 0 │
    /// └         ┘
    /// ```
    ///
    /// See, e.g., <https://algorithms.discrete.ma.tum.de/graph-algorithms/spp-floyd-warshall/index_en.html>
    pub fn shortest_paths_fw(&mut self) {
        self.calc_dist_and_next();
        let nnode = self.dist.nrow();
        for k in 0..nnode {
            for i in 0..nnode {
                for j in 0..nnode {
                    let sum = self.dist.get(i, k) + self.dist.get(k, j);
                    if self.dist.get(i, j) > sum {
                        self.dist.set(i, j, sum);
                        self.next.set(i, j, self.next.get(i, k));
                    }
                }
            }
        }
        self.ready_path = true;
    }

    /// Returns a path from start to end points
    ///
    /// **Important**: The shortest paths must be computed first.
    pub fn path(&self, start_point: usize, end_point: usize) -> Result<Vec<usize>, StrError> {
        if !self.ready_path {
            return Err("a path finding algorithm (e.g., shortest_paths_fw) must be called first");
        }
        let mut path = vec![start_point];
        let mut current_point = start_point;
        while current_point != end_point {
            if current_point == usize::MAX {
                return Err("no path found");
            }
            current_point = self.next.get(current_point, end_point);
            path.push(current_point);
        }
        Ok(path)
    }

    /// Calculates the distance matrix and initializes the next-hop matrix
    ///
    /// Note: there is no need to call this function because it is called by `shortest_paths_fw` already.
    /// Nonetheless, it may be useful for debugging the initial matrices.
    pub fn calc_dist_and_next(&mut self) {
        // initialize 'dist' and 'next' matrices
        let nnode = self.shares.len();
        for i in 0..nnode {
            for j in 0..nnode {
                if i == j {
                    self.dist.set(i, j, 0.0);
                } else {
                    self.dist.set(i, j, f64::MAX);
                }
                self.next.set(i, j, usize::MAX);
            }
        }

        // compute distances and initialize next-hop matrix
        let nedge = self.edges.nrow();
        for e in 0..nedge {
            let (i, j) = (self.edges.get(e, 0), self.edges.get(e, 1));
            self.dist.set(i, j, self.weights[e]);
            self.next.set(i, j, j);
        }
    }

    /// Returns a string representation of the distance or next matrices
    ///
    /// If `next_mat` is true, the next-hop matrix is returned; otherwise, the distance matrix is returned.
    pub fn str_mat(&self, next_mat: bool) -> String {
        // handle empty matrix
        let (nrow, ncol) = self.dist.dims();
        if nrow == 0 || ncol == 0 {
            print!("[]");
        }
        // find largest width
        let mut width = 0;
        let mut buf = String::new();
        for i in 0..nrow {
            for j in 0..ncol {
                if next_mat {
                    let val = self.next.get(i, j);
                    if val == usize::MAX {
                        write!(&mut buf, "∞").unwrap();
                    } else {
                        write!(&mut buf, "{}", val).unwrap();
                    }
                } else {
                    let val = self.dist.get(i, j);
                    if val == f64::MAX {
                        write!(&mut buf, "∞").unwrap();
                    } else {
                        write!(&mut buf, "{}", val).unwrap();
                    }
                }
                width = cmp::max(buf.chars().count(), width);
                buf.clear();
            }
        }
        // draw matrix
        width += 1;
        buf.clear();
        write!(&mut buf, "┌{:1$}┐\n", " ", width * ncol + 1).unwrap();
        for i in 0..nrow {
            if i > 0 {
                write!(&mut buf, " │\n").unwrap();
            }
            for j in 0..ncol {
                if j == 0 {
                    write!(&mut buf, "│").unwrap();
                }
                if next_mat {
                    let val = self.next.get(i, j);
                    if val == usize::MAX {
                        write!(&mut buf, "{:>1$}", "∞", width).unwrap();
                    } else {
                        write!(&mut buf, "{:>1$}", val, width).unwrap();
                    }
                } else {
                    let val = self.dist.get(i, j);
                    if val == f64::MAX {
                        write!(&mut buf, "{:>1$}", "∞", width).unwrap();
                    } else {
                        write!(&mut buf, "{:>1$}", val, width).unwrap();
                    }
                }
            }
        }
        write!(&mut buf, " │\n").unwrap();
        write!(&mut buf, "└{:1$}┘", " ", width * ncol + 1).unwrap();
        buf
    }

    /// Draws the graph
    ///
    /// # Arguments
    ///
    /// * `full_path`: The full path to save the plot
    /// * `show_edge_ids`: Whether to show edge IDs or not
    /// * `show_weights`: Whether to show edge weights or not
    /// * `precision`: (optional) The precision for the edge weights
    /// * `labels_n`: (optional) A dictionary with node labels
    /// * `labels_e`: (optional) A dictionary with edge labels
    /// * `radius`: (optional) The radius of the vertex circle
    /// * `gap_arrows`: (optional) The gap between arrows
    /// * `gap_labels`: (optional) The gap between the edge arrows and the respective label
    /// * `fig_width_pt`: (optional) The figure width in points
    pub fn draw(
        &self,
        full_path: &str,
        coords: &Vec<Vec<f64>>,
        show_edge_ids: bool,
        show_weights: bool,
        precision: Option<usize>,
        labels_n: Option<HashMap<usize, String>>,
        labels_e: Option<HashMap<usize, String>>,
        radius: Option<f64>,
        gap_arrows: Option<f64>,
        gap_labels: Option<f64>,
        fig_width_pt: Option<f64>,
    ) -> Result<(), StrError> {
        // check
        let nnode = coords.len();
        if nnode != self.dist.nrow() {
            return Err("number of nodes in coordinates must match the number of nodes in the graph");
        }

        // drawing objects
        let mut text_n = Text::new();
        text_n
            .set_color("black")
            .set_fontsize(8.0)
            .set_align_vertical("center")
            .set_align_horizontal("center");
        let mut text_e = Text::new();
        text_e
            .set_color("#0074c5")
            .set_fontsize(7.0)
            .set_align_vertical("center")
            .set_align_horizontal("center");
        let mut canvas_n = Canvas::new();
        canvas_n
            .set_face_color("none")
            .set_edge_color("#b40b00")
            .set_line_width(0.8);
        let mut canvas_e = Canvas::new();
        canvas_e
            .set_edge_color("#474747")
            .set_arrow_style("->")
            .set_arrow_scale(12.0);

        // constants
        let mut gap_arrows = gap_arrows.unwrap_or(0.15);
        let gap_labels = gap_labels.unwrap_or(0.12);
        let radius = radius.unwrap_or(0.2);

        // draw nodes and find limits
        let mut xmin = coords[0][0];
        let mut ymin = coords[0][1];
        let mut xmax = xmin;
        let mut ymax = ymin;
        for i in 0..nnode {
            let (x, y) = (coords[i][0], coords[i][1]);

            // plot vertex label
            let mut lbl = &format!("{}", i);
            if let Some(labels_n) = labels_n.as_ref() {
                lbl = labels_n.get(&i).unwrap_or(&lbl);
            }
            text_n.draw(x, y, lbl);

            // plot vertex circle with partition color if available
            canvas_n.draw_circle(x, y, radius);

            // update limits
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }

        // distance between edges
        if gap_arrows > 2.0 * radius {
            gap_arrows = 1.8 * radius;
        }
        let w = gap_arrows / 2.0;
        let l = f64::sqrt(radius * radius - w * w);

        // draw edges
        let mut xa = [0.0; 2];
        let mut xb = [0.0; 2];
        let mut xc = [0.0; 2];
        let mut dx = [0.0; 2];
        let mut mu = [0.0; 2];
        let mut nu = [0.0; 2];
        let mut xi = [0.0; 2];
        let mut xj = [0.0; 2];
        let nedge = self.edges.nrow();
        for k in 0..nedge {
            let mut ll = 0.0;
            for i in 0..2 {
                xa[i] = coords[self.edges.get(k, 0)][i];
                xb[i] = coords[self.edges.get(k, 1)][i];
                xc[i] = (xa[i] + xb[i]) / 2.0;
                dx[i] = xb[i] - xa[i];
                ll += dx[i] * dx[i];
            }
            ll = ll.sqrt();
            mu[0] = dx[0] / ll;
            mu[1] = dx[1] / ll;
            nu[0] = -dx[1] / ll;
            nu[1] = dx[0] / ll;

            for i in 0..2 {
                xi[i] = xa[i] + l * mu[i] - w * nu[i];
                xj[i] = xb[i] - l * mu[i] - w * nu[i];
                xc[i] = (xi[i] + xj[i]) / 2.0 - gap_labels * nu[i];
            }

            canvas_e.draw_arrow(xi[0], xi[1], xj[0], xj[1]);

            if show_edge_ids || show_weights {
                let mut lbl = String::new();
                let mut sep = "";
                if show_edge_ids {
                    lbl.push_str(&format!("{}", k));
                    sep = "| ";
                };
                if show_weights {
                    match precision {
                        Some(p) => lbl.push_str(&format!("{}${:.2$}", sep, self.weights[k], p)),
                        None => lbl.push_str(&format!("{}${}", sep, self.weights[k])),
                    }
                }
                if let Some(labels_e) = labels_e.as_ref() {
                    lbl = labels_e.get(&k).unwrap_or(&lbl).to_string();
                }
                if show_edge_ids && show_weights && f64::abs(dx[0]) < 1e-3 {
                    text_e.set_rotation(90.0);
                } else {
                    text_e.set_rotation(0.0);
                }
                text_e.draw(xc[0], xc[1], &lbl);
            }
        }

        // update range
        xmin -= 2.0 * radius;
        xmax += 2.0 * radius;
        ymin -= 2.0 * radius;
        ymax += 2.0 * radius;

        // add objects to plot
        let width = fig_width_pt.unwrap_or(600.0);
        let mut plot = Plot::new();
        plot.add(&canvas_n).add(&canvas_e).add(&text_n);
        if show_edge_ids {
            plot.add(&text_e);
        }
        plot.set_range(xmin, xmax, ymin, ymax)
            .set_equal_axes(true)
            .add(&text_e)
            .set_hide_axes(true)
            .set_figure_size_points(width, width)
            .save(full_path)
    }
}

#[cfg(test)]
mod tests {
    use super::Graph;
    use crate::read_table;
    use std::collections::HashMap;

    const SAVE_FIGURE: bool = false;

    #[test]
    fn new_works_1() {
        //  0 ––––––––––→ 3
        //  │      1      ↑
        //  │             │
        //  │ 0         3 │
        //  │             │
        //  ↓      2      |
        //  1 ––––––––––→ 2

        // edge:       0       1       2       3
        let edges = [[0, 1], [0, 3], [1, 2], [2, 3]];
        let graph = Graph::new(&edges);

        // check edges
        assert_eq!(
            format!("{}", graph.edges),
            "┌     ┐\n\
             │ 0 1 │\n\
             │ 0 3 │\n\
             │ 1 2 │\n\
             │ 2 3 │\n\
             └     ┘"
        );

        // check weights
        assert_eq!(&graph.weights, &[1.0, 1.0, 1.0, 1.0]);

        // check shares
        let mut entries: Vec<_> = graph.shares.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            format!("{:?}", entries),
            "[(0, [0, 1]), (1, [0, 2]), (2, [2, 3]), (3, [1, 3])]"
        );

        // check 'dist'
        assert_eq!(graph.dist.dims(), (4, 4));

        // check 'next'
        assert_eq!(graph.next.dims(), (4, 4));

        // check 'ready_path'
        assert_eq!(graph.ready_path, false);

        // check getters
        assert_eq!(graph.get_nedge(), 4);
        assert_eq!(graph.get_nnode(), 4);
    }

    #[test]
    fn shortest_paths_fw_works_1() {
        //  0 ––––––––––→ 3
        //  │      1      ↑
        //  │             │
        //  │ 0         3 │
        //  │             │
        //  ↓      2      |
        //  1 ––––––––––→ 2

        // edge:       0       1       2       3
        let edges = [[0, 1], [0, 3], [1, 2], [2, 3]];
        let mut graph = Graph::new(&edges);

        // initial 'dist' matrix
        graph.calc_dist_and_next();
        assert_eq!(
            format!("{}", graph.str_mat(false)),
            "┌         ┐\n\
             │ 0 1 ∞ 1 │\n\
             │ ∞ 0 1 ∞ │\n\
             │ ∞ ∞ 0 1 │\n\
             │ ∞ ∞ ∞ 0 │\n\
             └         ┘"
        );

        // call shortest_paths_fw
        graph.shortest_paths_fw();

        // check 'dist'
        assert_eq!(
            format!("{}", graph.str_mat(false)),
            "┌         ┐\n\
             │ 0 1 2 1 │\n\
             │ ∞ 0 1 2 │\n\
             │ ∞ ∞ 0 1 │\n\
             │ ∞ ∞ ∞ 0 │\n\
             └         ┘"
        );

        // check paths
        assert_eq!(graph.path(0, 3).unwrap(), &[0, 3]);
        assert_eq!(graph.path(0, 1).unwrap(), &[0, 1]);
        assert_eq!(graph.path(1, 2).unwrap(), &[1, 2]);
        assert_eq!(graph.path(2, 3).unwrap(), &[2, 3]);
        assert_eq!(graph.path(3, 0).err(), Some("no path found"));
        assert_eq!(graph.path(1, 0).err(), Some("no path found"));
        assert_eq!(graph.path(2, 1).err(), Some("no path found"));
        assert_eq!(graph.path(3, 2).err(), Some("no path found"));
        assert_eq!(graph.path(0, 2).unwrap(), &[0, 1, 2]);
    }

    #[test]
    fn shortest_paths_fw_works_2() {
        //          $10
        //    0 ––––––––––→ 3
        //    │      1      ↑
        //    │             │
        // $5 │ 0         3 │ $1
        //    │             │
        //    ↓      2      |
        //    1 ––––––––––→ 2
        //          $3
        //
        // the weights are indicated by the dollar sign

        // edge:       0       1       2       3
        let edges = [[0, 1], [0, 3], [1, 2], [2, 3]];
        let mut graph = Graph::new(&edges);
        graph
            .set_weight(0, 5.0)
            .set_weight(1, 10.0)
            .set_weight(2, 3.0)
            .set_weight(3, 1.0);

        // initial 'dist' matrix
        graph.calc_dist_and_next();
        assert_eq!(
            format!("{}", graph.str_mat(false)),
            "┌             ┐\n\
             │  0  5  ∞ 10 │\n\
             │  ∞  0  3  ∞ │\n\
             │  ∞  ∞  0  1 │\n\
             │  ∞  ∞  ∞  0 │\n\
             └             ┘"
        );

        // call shortest_paths_fw
        graph.shortest_paths_fw();

        // check 'dist'
        assert_eq!(
            format!("{}", graph.str_mat(false)),
            "┌         ┐\n\
             │ 0 5 8 9 │\n\
             │ ∞ 0 3 4 │\n\
             │ ∞ ∞ 0 1 │\n\
             │ ∞ ∞ ∞ 0 │\n\
             └         ┘"
        );

        // check path
        assert_eq!(graph.path(0, 3).unwrap(), &[0, 1, 2, 3]); // lower cost path with 5+3+1 < 10
        assert_eq!(graph.path(0, 1).unwrap(), &[0, 1]);
        assert_eq!(graph.path(1, 2).unwrap(), &[1, 2]);
        assert_eq!(graph.path(2, 3).unwrap(), &[2, 3]);
        assert_eq!(graph.path(3, 0).err(), Some("no path found"));
        assert_eq!(graph.path(1, 0).err(), Some("no path found"));
        assert_eq!(graph.path(2, 1).err(), Some("no path found"));
        assert_eq!(graph.path(3, 2).err(), Some("no path found"));
        assert_eq!(graph.path(0, 2).unwrap(), &[0, 1, 2]);

        // call shortest_paths_fw again with different weights
        graph.set_weight(3, 13.0);
        graph.shortest_paths_fw();

        // check 'dist'
        assert_eq!(
            format!("{}", graph.str_mat(false)),
            "┌             ┐\n\
             │  0  5  8 10 │\n\
             │  ∞  0  3 16 │\n\
             │  ∞  ∞  0 13 │\n\
             │  ∞  ∞  ∞  0 │\n\
             └             ┘"
        );

        // check path
        assert_eq!(graph.path(0, 3).unwrap(), &[0, 3]);
        assert_eq!(graph.path(0, 1).unwrap(), &[0, 1]);
        assert_eq!(graph.path(1, 2).unwrap(), &[1, 2]);
        assert_eq!(graph.path(2, 3).unwrap(), &[2, 3]);
        assert_eq!(graph.path(3, 0).err(), Some("no path found"));
        assert_eq!(graph.path(1, 0).err(), Some("no path found"));
        assert_eq!(graph.path(2, 1).err(), Some("no path found"));
        assert_eq!(graph.path(3, 2).err(), Some("no path found"));
        assert_eq!(graph.path(0, 2).unwrap(), &[0, 1, 2]);

        if SAVE_FIGURE {
            let coords = vec![vec![0.0, 1.0], vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 1.0]];
            graph
                .draw(
                    "/tmp/russell/test_shortest_paths_fw_works_2.svg",
                    &coords,
                    true,
                    true,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();
        }
    }

    #[test]
    fn shortest_paths_fw_works_3() {
        //              $3
        //      4 ––––––––––––––→ 5 .  6
        //      ↑        0        │  `. $4
        //      │                 │    `.
        //      │                 │      `v
        //  $11 │ 1            $7 │ 4      3
        //      |                 │     ,^
        //      │                 │   ,' $9
        //      │    2       3    ↓ ,'  5
        //      1 ←––––– 0 –––––→ 2
        //           $6      $8
        //
        // the weights are indicated by the dollar sign

        // edge:       0       1       2       3       4       5       6
        let edges = [[4, 5], [1, 4], [0, 1], [0, 2], [5, 2], [2, 3], [5, 3]];
        let mut graph = Graph::new(&edges);
        graph
            .set_weight(0, 3.0)
            .set_weight(1, 11.0)
            .set_weight(2, 6.0)
            .set_weight(3, 8.0)
            .set_weight(4, 7.0)
            .set_weight(5, 9.0)
            .set_weight(6, 4.0);

        // check getters
        assert_eq!(graph.get_nedge(), 7);
        assert_eq!(graph.get_nnode(), 6);

        // initial 'dist' matrix
        graph.calc_dist_and_next();
        assert_eq!(
            format!("{}", graph.str_mat(false)),
            "┌                   ┐\n\
             │  0  6  8  ∞  ∞  ∞ │\n\
             │  ∞  0  ∞  ∞ 11  ∞ │\n\
             │  ∞  ∞  0  9  ∞  ∞ │\n\
             │  ∞  ∞  ∞  0  ∞  ∞ │\n\
             │  ∞  ∞  ∞  ∞  0  3 │\n\
             │  ∞  ∞  7  4  ∞  0 │\n\
             └                   ┘"
        );

        // shortest paths
        graph.shortest_paths_fw();

        // check 'dist'
        assert_eq!(
            format!("{}", graph.str_mat(false)),
            "┌                   ┐\n\
             │  0  6  8 17 17 20 │\n\
             │  ∞  0 21 18 11 14 │\n\
             │  ∞  ∞  0  9  ∞  ∞ │\n\
             │  ∞  ∞  ∞  0  ∞  ∞ │\n\
             │  ∞  ∞ 10  7  0  3 │\n\
             │  ∞  ∞  7  4  ∞  0 │\n\
             └                   ┘"
        );

        // check paths
        assert_eq!(graph.path(0, 3).unwrap(), &[0, 2, 3]);
        assert_eq!(graph.path(4, 3).unwrap(), &[4, 5, 3]);
        assert_eq!(graph.path(0, 2).unwrap(), &[0, 2]);
        assert_eq!(graph.path(0, 5).unwrap(), &[0, 1, 4, 5]);
        assert_eq!(graph.path(2, 4).err(), Some("no path found"));
        assert_eq!(graph.path(1, 0).err(), Some("no path found"));

        if SAVE_FIGURE {
            let coords = vec![
                vec![1.0, 0.0],
                vec![0.0, 0.0],
                vec![2.0, 0.0],
                vec![3.0, 1.0],
                vec![0.0, 2.0],
                vec![2.0, 2.0],
            ];
            let labels_n = HashMap::from([
                (0, "N0".to_string()),
                (1, "N1".to_string()),
                (2, "N2".to_string()),
                (3, "N3".to_string()),
                (4, "N4".to_string()),
                (5, "N5".to_string()),
            ]);
            let labels_e = HashMap::from([
                (0, "E0".to_string()),
                (1, "E1".to_string()),
                (2, "E2".to_string()),
                (3, "E3".to_string()),
                (4, "E4".to_string()),
                (5, "E5".to_string()),
                (6, "E6".to_string()),
            ]);
            graph
                .draw(
                    "/tmp/russell/test_shortest_paths_fw_works_3.svg",
                    &coords,
                    true,
                    true,
                    None,
                    Some(labels_n),
                    Some(labels_e),
                    Some(0.1),
                    Some(0.0),
                    None,
                    Some(400.0),
                )
                .unwrap();
        }
    }

    #[test]
    fn shortest_paths_fw_works_4() {
        // read the graph
        let table: HashMap<String, Vec<f64>> =
            read_table("data/tables/SiouxFalls.flow", Some(&["from", "to", "cap", "cost"])).unwrap();

        // create the graph
        let from = table.get("from").unwrap();
        let to = table.get("to").unwrap();
        let edges: Vec<_> = from
            .iter()
            .zip(to.iter())
            .map(|(a, b)| vec![*a as usize - 1, *b as usize - 1])
            .collect();
        let mut graph = Graph::new(&edges);

        // set the weights
        let cost = table.get("cost").unwrap();
        for (e, &c) in cost.iter().enumerate() {
            graph.set_weight(e, c);
        }

        // shortest paths
        graph.shortest_paths_fw();

        // check paths
        assert_eq!(graph.path(0, 20).unwrap(), &[0, 2, 11, 12, 23, 20]);
        assert_eq!(graph.path(2, 21).unwrap(), &[2, 11, 12, 23, 22, 21]);
        assert_eq!(graph.path(9, 15).unwrap(), &[9, 15]);
        assert_eq!(graph.path(10, 11).unwrap(), &[10, 11]);
        assert_eq!(graph.path(3, 20).unwrap(), &[3, 2, 11, 12, 23, 20]);
        assert_eq!(graph.path(8, 10).unwrap(), &[8, 9, 10]);
        assert_eq!(graph.path(11, 21).unwrap(), &[11, 12, 23, 22, 21]);
        assert_eq!(graph.path(5, 16).unwrap(), &[5, 7, 6, 17, 15, 16]);
        assert_eq!(graph.path(9, 11).unwrap(), &[9, 10, 11]);

        if SAVE_FIGURE {
            // define the graph layout data
            let columns = vec![
                vec![0, 2, 11, 12],
                vec![3, 10, 13, 22, 23],
                vec![4, 8, 9, 14, 21, 20],
                vec![1, 5, 7, 15, 16, 18, 19],
                vec![6, 17],
            ];
            let y_coords = vec![
                vec![7.0, 6.0, 4.0, 0.0],                // col0
                vec![6.0, 4.0, 2.0, 1.0, 0.0],           // col1
                vec![6.0, 5.0, 4.0, 2.0, 1.0, 0.0],      // col2
                vec![7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 0.0], // col3
                vec![5.0, 4.0],                          // col4
            ];
            let scale_x = 1.8;
            let scale_y = 1.3;

            // create coordinates
            let mut coords = vec![vec![0.0, 0.0]; graph.get_nnode()];
            for (j, col) in columns.iter().enumerate() {
                let x = j as f64 * scale_x;
                for (i, n) in col.iter().enumerate() {
                    coords[*n][0] = x;
                    coords[*n][1] = y_coords[j][i] * scale_y;
                }
            }

            // draw the graph
            graph
                .draw(
                    "/tmp/russell/test_shortest_paths_fw_works_4.svg",
                    &coords,
                    true,
                    true,
                    Some(1),
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(800.0),
                )
                .unwrap();
        }
    }
}
