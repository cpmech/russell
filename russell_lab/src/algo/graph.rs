use crate::{AsArray2D, StrError};
use crate::{Matrix, NumMatrix};
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
    coordinates: Vec<Vec<f64>>,

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
            weights: vec![1.0; nnode],
            shares,
            coordinates: Vec::new(),
            dist: Matrix::new(nnode, nnode),
            next: NumMatrix::new(nnode, nnode),
            ready_path: false,
        }
    }

    /// Sets the weight of an edge
    ///
    /// **Important**: The weight must be non-negative.
    pub fn set_weight(&mut self, edge: usize, non_neg_value: f64) -> &mut Self {
        assert!(non_neg_value >= 0.0, "edge weight must be ≥ 0");
        self.weights[edge] = non_neg_value;
        self
    }

    /// Sets the coordinates of a node
    pub fn set_node_coordinates(&mut self, node: usize, coordinates: &[f64]) -> &mut Self {
        let ndim = coordinates.len();
        if self.coordinates.is_empty() {
            let nnode = self.shares.len();
            self.coordinates = vec![vec![0.0; ndim]; nnode];
        } else {
            assert_eq!(
                self.coordinates[0].len(),
                ndim,
                "coordinates must have the same dimension"
            );
        }
        for i in 0..ndim {
            self.coordinates[node][i] = coordinates[i];
        }
        self
    }

    /// Computes the shortest paths using the Floyd-Warshall algorithm
    ///
    /// An example of a graph with weights:
    ///
    /// ```text
    ///        [10]
    ///     0 ––––––→ 3       []: numbers in brackets
    ///     │         ↑           indicate weights
    /// [5] │         │ [1]
    ///     ↓         │
    ///     1 ––––––→ 2
    ///         [3]
    /// ```
    ///
    /// The initial distance matrix is:
    ///
    /// ```text
    /// j=  0  1  2  3
    ///   ┌            ┐ i=
    ///   │ 0  5  ∞ 10 │  0  ⇒  w(0→1)=5, w(0→3)=10
    ///   │ ∞  0  3  ∞ │  1  ⇒  w(1→2)=3
    ///   │ ∞  ∞  0  1 │  2  ⇒  w(2→3)=1
    ///   │ ∞  ∞  ∞  0 │  3
    ///   └            ┘
    /// ∞ means that there are no connections from i to j
    /// ```
    ///
    /// See <https://algorithms.discrete.ma.tum.de/graph-algorithms/spp-floyd-warshall/index_en.html>
    pub fn shortest_paths_fw(&mut self) {
        self.calc_dist_and_next();
        println!("{}", self.str_mat(false));
        let nnode = self.dist.nrow();
        for k in 0..nnode {
            for i in 0..nnode {
                for j in 0..nnode {
                    let sum = self.dist.get(i, k) + self.dist.get(k, j);
                    if self.dist.get(i, j) > sum {
                        self.dist.set(i, j, sum);
                        self.next.set(i, j, self.next.get(i, k));
                        println!("{}", self.str_mat(false));
                    }
                }
            }
        }
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
    fn calc_dist_and_next(&mut self) {
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
            if self.coordinates.len() == nnode {
                d = 0.0;
                let xa = &self.coordinates[i];
                let xb = &self.coordinates[j];
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
}

#[cfg(test)]
mod tests {
    use super::Graph;

    #[test]
    fn new_works_1() {
        //  0 ––––––––→ 3    (): numbers in parentheses
        //  │    (1)    ↑        indicate edge ids
        //  │(0)        │
        //  │        (3)│
        //  ↓    (2)    |
        //  1 ––––––––→ 2

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
        assert_eq!(graph.coordinates.len(), 0);

        // check 'dist'
        assert_eq!(
            format!("{}", graph.str_mat(false)),
            "┌         ┐\n\
             │ 0 ∞ ∞ ∞ │\n\
             │ ∞ 0 ∞ ∞ │\n\
             │ ∞ ∞ 0 ∞ │\n\
             │ ∞ ∞ ∞ 0 │\n\
             └         ┘"
        );

        // check 'next'
        assert_eq!(
            format!("{}", graph.str_mat(true)),
            "┌         ┐\n\
             │ ∞ ∞ ∞ ∞ │\n\
             │ ∞ ∞ ∞ ∞ │\n\
             │ ∞ ∞ ∞ ∞ │\n\
             │ ∞ ∞ ∞ ∞ │\n\
             └         ┘"
        );

        // check 'ready_path'
        assert_eq!(graph.ready_path, false);
    }

    #[test]
    fn shortest_paths_fw_works_1() {
        //  0 ––––––––→ 3    (): numbers in parentheses
        //  │    (1)    ↑        indicate edge ids
        //  │(0)        │
        //  │        (3)│    Note: equal weights
        //  ↓    (2)    |
        //  1 ––––––––→ 2

        // edge:       0       1       2       3
        let edges = [[0, 1], [0, 3], [1, 2], [2, 3]];
        let mut graph = Graph::new(&edges);

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
        //           [10]
        //      0 ––––––––→ 3      (): numbers in parentheses
        //      |    (1)    ↑          indicate edge ids
        //   [5]|(0)        |
        //      |        (3)|[1]
        //      ↓    (2)    |      []: numbers in brackets
        //      1 ––––––––→ 2          indicate weights
        //           [3]

        // edge:       0       1       2       3
        let edges = [[0, 1], [0, 3], [1, 2], [2, 3]];
        let mut graph = Graph::new(&edges);
        graph
            .set_weight(0, 5.0)
            .set_weight(1, 10.0)
            .set_weight(2, 3.0)
            .set_weight(3, 1.0);

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
}
