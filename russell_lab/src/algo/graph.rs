use crate::AsArray2D;
use crate::{Matrix, NumMatrix};
use std::cmp;
use std::fmt::Write;
use std::{collections::HashMap, vec};

/// Graph defines a (directed) graph structure
///
/// Warning: The algorithms here are not as optimized as they could be.
/// A specialized graph library certainly would provide better performance.
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

    /// Indicates whether the distances are already calculated and next has been initialized
    ready: bool,
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
        let mut distances = Matrix::filled(nnode, nnode, f64::MAX);
        for i in 0..nnode {
            distances.set(i, i, 0.0);
        }
        Graph {
            edges: NumMatrix::from(edges),
            weights: vec![1.0; nnode],
            shares,
            coordinates: Vec::new(),
            dist: distances,
            next: NumMatrix::filled(nnode, nnode, usize::MAX),
            ready: false,
        }
    }

    /// Sets the weight of an edge
    pub fn set_weight(&mut self, edge: usize, value: f64) -> &mut Self {
        assert!(value >= 0.0, "edge weight must be ≥ 0");
        self.weights[edge] = value;
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

    /// Calculates the distance matrix and initializes the next-hop matrix
    fn calc_dist_and_next(&mut self) {
        let nedge = self.edges.nrow();
        let nnode = self.shares.len();
        for e in 0..nedge {
            // Get edge nodes
            let (i, j) = (self.edges.get(e, 0), self.edges.get(e, 1));

            // Initialize distance
            let mut d = 1.0;

            // Compute Euclidean distance if vertex coordinates are provided
            if self.coordinates.len() == nnode {
                d = 0.0;
                let xa = &self.coordinates[i];
                let xb = &self.coordinates[j];
                for dim in 0..xa.len() {
                    d += (xa[dim] - xb[dim]).powi(2);
                }
                d = d.sqrt();
            }

            // Apply edge weight
            d *= self.weights[e];

            // Update distance and next-hop matrices
            self.dist.set(i, j, d);
            self.next.set(i, j, j);
            assert!(self.dist.get(i, j) >= 0.0, "distance must be ≥ 0");
        }
    }

    /// Returns a string representation of the distance matrix
    pub fn str_dist_matrix(&self) -> String {
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
                let val = self.dist.get(i, j);
                if val == f64::MAX {
                    write!(&mut buf, "∞").unwrap();
                } else {
                    write!(&mut buf, "{}", val).unwrap();
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
                write!(&mut buf, " │\n");
            }
            for j in 0..ncol {
                if j == 0 {
                    write!(&mut buf, "│").unwrap();
                }
                let val = self.dist.get(i, j);
                if val == f64::MAX {
                    write!(&mut buf, "{:>1$}", "∞", width).unwrap();
                } else {
                    write!(&mut buf, "{:>1$}", val, width).unwrap();
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
    fn new_works_simple_1() {
        //  0 ––––––––→ 3    (): numbers in parentheses
        //  |    (1)    ↑        indicate edge ids
        //  |(0)        |
        //  |        (3)|
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

        // check distances
        assert_eq!(
            format!("{}", graph.str_dist_matrix()),
            "┌         ┐\n\
             │ 0 ∞ ∞ ∞ │\n\
             │ ∞ 0 ∞ ∞ │\n\
             │ ∞ ∞ 0 ∞ │\n\
             │ ∞ ∞ ∞ 0 │\n\
             └         ┘"
        );

        // check 'next'
        assert_eq!(graph.next.dims(), (4, 4));
    }
}
