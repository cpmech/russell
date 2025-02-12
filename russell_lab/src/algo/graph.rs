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

    /// (optional) Holds all node coordinates
    ///
    /// This array is only used if the real coordinates of points are
    /// needed to calculate the distances.
    ///
    /// Size: 0 or (nnode, ndim)
    coords: Vec<Vec<f64>>,

    /// Specifies that the distances are based on coordinates
    coords_based_dist: bool,

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
            coords: Vec::new(),
            coords_based_dist: false,
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

    /// Sets the coordinates of a node
    pub fn set_coords(&mut self, node: usize, coordinates: &[f64]) -> &mut Self {
        let ndim = coordinates.len();
        if self.coords.is_empty() {
            let nnode = self.shares.len();
            self.coords = vec![vec![0.0; ndim]; nnode];
        } else {
            assert_eq!(self.coords[0].len(), ndim, "coordinates must have the same dimension");
        }
        for i in 0..ndim {
            self.coords[node][i] = coordinates[i];
        }
        self.ready_path = false;
        self
    }

    /// Activates the use of node coordinates to calculate distances
    pub fn set_coords_based_dist(&mut self, value: bool) -> &mut Self {
        self.coords_based_dist = value;
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
    ///          *10
    ///    0 ––––––––––→ 3
    ///    │     (1)     ↑
    ///    │             │
    /// *5 │(0)       (3)│ *1
    ///    │             │
    ///    ↓     (2)     |
    ///    1 ––––––––––→ 2
    ///          *3
    ///
    /// numbers in parentheses indicate edge ids
    /// starred numbers indicate weights
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
        // constants
        let nedge = self.edges.nrow();
        let nnode = self.shares.len();

        // initialize 'dist' and 'next' matrices
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
        for e in 0..nedge {
            // get edge nodes
            let (i, j) = (self.edges.get(e, 0), self.edges.get(e, 1));

            // initialize distance
            let mut d = 1.0;

            // compute Euclidean distance if vertex coordinates are provided
            // and the coords_based_dist flag is true
            if self.coords_based_dist && self.coords.len() == nnode {
                d = 0.0;
                let xa = &self.coords[i];
                let xb = &self.coords[j];
                for dim in 0..xa.len() {
                    d += (xa[dim] - xb[dim]).powi(2);
                }
                d = d.sqrt();
            }

            // apply edge weight
            d *= self.weights[e];

            // update distance and next-hop matrices
            self.dist.set(i, j, d);
            self.next.set(i, j, j);
            assert!(self.dist.get(i, j) >= 0.0, "distance must be ≥ 0");
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
    /// * `labels_n`: (optional) A dictionary with node labels
    /// * `labels_e`: (optional) A dictionary with edge labels
    /// * `radius`: (optional) The radius of the vertex circle
    /// * `gap_arrows`: (optional) The gap between arrows
    /// * `gap_labels`: (optional) The gap between the edge arrows and the respective label
    /// * `fig_width_pt`: (optional) The figure width in points
    pub fn draw(
        &self,
        full_path: &str,
        labels_n: Option<HashMap<usize, String>>,
        labels_e: Option<HashMap<usize, String>>,
        radius: Option<f64>,
        gap_arrows: Option<f64>,
        gap_labels: Option<f64>,
        fig_width_pt: Option<f64>,
    ) -> Result<(), StrError> {
        // check
        if self.coords.is_empty() {
            return Err("vertices coordinates are required to draw graph");
        }

        // drawing objects
        let mut txt1 = Text::new();
        txt1.set_color("#d70d0d")
            .set_fontsize(8.0)
            .set_align_vertical("center")
            .set_align_horizontal("center");
        let mut txt2 = Text::new();
        txt2.set_color("#2732c6")
            .set_fontsize(7.0)
            .set_align_vertical("center")
            .set_align_horizontal("center");
        let mut canvas1 = Canvas::new();
        canvas1
            .set_face_color("none")
            .set_edge_color("black")
            .set_line_width(0.8);
        let mut canvas2 = Canvas::new();
        canvas2.set_arrow_style("->").set_arrow_scale(12.0);

        // constants
        let nnode = self.coords.len();
        let mut gap_arrows = gap_arrows.unwrap_or(0.15);
        let gap_labels = gap_labels.unwrap_or(0.12);
        let radius = radius.unwrap_or(0.2);

        // draw nodes and find limits
        let mut xmin = self.coords[0][0];
        let mut ymin = self.coords[0][1];
        let mut xmax = xmin;
        let mut ymax = ymin;
        for i in 0..nnode {
            let (x, y) = (self.coords[i][0], self.coords[i][1]);

            // plot vertex label
            let mut lbl = &format!("{}", i);
            if let Some(labels_n) = labels_n.as_ref() {
                lbl = labels_n.get(&i).unwrap_or(&lbl);
            }
            txt1.draw(x, y, lbl);

            // plot vertex circle with partition color if available
            canvas1.draw_circle(x, y, radius);

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
                xa[i] = self.coords[self.edges.get(k, 0)][i];
                xb[i] = self.coords[self.edges.get(k, 1)][i];
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

            canvas2.draw_arrow(xi[0], xi[1], xj[0], xj[1]);

            let mut lbl = &format!("{}", k);
            if let Some(labels_e) = labels_e.as_ref() {
                lbl = labels_e.get(&k).unwrap_or(lbl);
            }
            txt2.draw(xc[0], xc[1], lbl);
        }

        // update range
        xmin -= 1.2 * radius;
        xmax += 1.2 * radius;
        ymin -= 1.2 * radius;
        ymax += 1.2 * radius;

        // add objects to plot
        let width = fig_width_pt.unwrap_or(600.0);
        let mut plot = Plot::new();
        plot.set_range(xmin, xmax, ymin, ymax)
            .set_equal_axes(true)
            .add(&canvas1)
            .add(&canvas2)
            .add(&txt1)
            .add(&txt2)
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

    #[test]
    fn new_works_1() {
        //  0 ––––––––––→ 3
        //  │     (1)     ↑
        //  │             │
        //  │(0)       (3)│
        //  │             │
        //  ↓     (2)     |
        //  1 ––––––––––→ 2
        //
        // numbers in parentheses indicate edge ids

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

        // check coordinates
        assert_eq!(graph.coords.len(), 0);

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
        //  │     (1)     ↑
        //  │             │
        //  │(0)       (3)│
        //  │             │
        //  ↓     (2)     |
        //  1 ––––––––––→ 2     equal unitary weights
        //
        // numbers in parentheses indicate edge ids

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
    }

    #[test]
    fn shortest_paths_fw_works_2() {
        //          *10
        //    0 ––––––––––→ 3
        //    │     (1)     ↑
        //    │             │
        // *5 │(0)       (3)│ *1
        //    │             │
        //    ↓     (2)     |
        //    1 ––––––––––→ 2
        //          *3
        //
        // numbers in parentheses indicate edge ids
        // starred numbers indicate weights

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
    }

    #[test]
    fn shortest_paths_fw_works_3() {
        //              *3
        //      4 ––––––––––––––→ 5 . (6)
        //      ↑       (0)       │  `. *4
        //      │                 │    `.
        //      │                 │      `v
        //  *11 │(1)           *7 │(4)     3
        //      |                 │     ,^
        //      │                 │   ,' *9
        //      │   (2)     (3)   ↓ ,' (5)
        //      1 ←––––– 0 –––––→ 2
        //           *6      *8
        //
        // numbers in parentheses indicate edge ids
        // starred numbers indicate weights

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
    }

    #[test]
    fn draw_graph_works_1() {
        //              *3
        //      4 ––––––––––––––→ 5 . (6)
        //      ↑       (0)       │  `. *4
        //      │                 │    `.
        //      │                 │      `v
        //  *11 │(1)           *7 │(4)     3
        //      |                 │     ,^
        //      │                 │   ,' *9
        //      │   (2)     (3)   ↓ ,' (5)
        //      1 ←––––– 0 –––––→ 2
        //           *6      *8
        //
        // numbers in parentheses indicate edge ids
        // starred numbers indicate weights

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
        graph
            .set_coords(0, &[1.0, 0.0])
            .set_coords(1, &[0.0, 0.0])
            .set_coords(2, &[2.0, 0.0])
            .set_coords(3, &[3.0, 1.0])
            .set_coords(4, &[0.0, 2.0])
            .set_coords(5, &[2.0, 2.0]);

        let labels_n = HashMap::from([
            (0, "0".to_string()),
            (1, "1".to_string()),
            (2, "2".to_string()),
            (3, "3".to_string()),
            (4, "4".to_string()),
            (5, "5".to_string()),
        ]);
        let labels_e = HashMap::from([
            (0, "(0)".to_string()),
            (1, "(1)".to_string()),
            (2, "(2)".to_string()),
            (3, "(3)".to_string()),
            (4, "(4)".to_string()),
            (5, "(5)".to_string()),
            (6, "(6)".to_string()),
        ]);
        graph
            .draw(
                "/tmp/russell/test_draw_graph_works_1.svg",
                Some(labels_n),
                Some(labels_e),
                Some(0.1),
                Some(0.0),
                None,
                Some(400.0),
            )
            .unwrap();
    }

    #[test]
    fn draw_graph_works_2() {
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
        for (j, col) in columns.iter().enumerate() {
            let x = j as f64 * scale_x;
            for (i, &vid) in col.iter().enumerate() {
                graph.set_coords(vid, &[x, y_coords[j][i] * scale_y]);
            }
        }

        // draw the graph
        graph
            .draw(
                "/tmp/russell/test_draw_graph_works_2.svg",
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
