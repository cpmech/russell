#![allow(unused)]

//! Algorithms for graph computations

use std::collections::HashMap;

/// Graph defines a graph structure
pub struct Graph {
    /// Edges (connectivity)
    ///
    /// Size: `[nedge][2]`
    edges: Vec<Vec<i32>>,

    /// Weights of edges
    ///
    /// Size: `[nedge]`
    weights_e: Option<Vec<f64>>,

    /// Nodes (Vertices)
    ///
    /// Size: `[nnode][ndim]`
    nodes: Option<Vec<Vec<f64>>>,

    /// Weights of vertices
    ///
    /// Size: `[nnode]`
    weights_v: Option<Vec<f64>>,

    /// Edges sharing a node/vertex
    ///
    /// Size: `[nnode]`
    shares: HashMap<i32, Vec<i32>>,

    /// Maps (i,j) node to edge index
    key2edge: HashMap<i32, i32>,

    /// Distances
    ///
    /// Size: `[nnode][nnode]`
    dist: Vec<Vec<f64>>,

    /// Next tree connection. -1 means no connection
    ///
    /// Size: `[nnode][nnode]`
    next: Vec<Vec<i32>>,
}

impl Graph {
    /// Creates a new graph
    pub fn new() -> Self {
        Graph {
            edges: Vec::new(),
            weights_e: None,
            nodes: None,
            weights_v: None,
            shares: HashMap::new(),
            key2edge: HashMap::new(),
            dist: Vec::new(),
            next: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Graph;

    #[test]
    fn test_new() {
        let graph = Graph::new();
        assert!(graph.edges.is_empty());
        assert!(graph.weights_e.is_none());
        assert!(graph.nodes.is_none());
        assert!(graph.weights_v.is_none());
        assert!(graph.shares.is_empty());
        assert!(graph.key2edge.is_empty());
        assert!(graph.dist.is_empty());
        assert!(graph.next.is_empty());
    }
}
