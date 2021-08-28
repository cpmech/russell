#[allow(dead_code)]

/// Holds triples representing a sparse matrix
pub struct SparseTriplet {
    pub(crate) m: usize,        // number of rows
    pub(crate) n: usize,        // number of columns
    pub(crate) pos: usize,      // current index => nnz in the end
    pub(crate) max: usize,      // max allowed number of entries
    pub(crate) one_based: bool, // indices (i; j) start with 1 instead of 0 (e.g. for MUMPS)
    pub(crate) symmetric: bool, // symmetric matrix?, but WITHOUT both sides of the diagonal
    pub(crate) i: Vec<i32>,     // zero- or one-based indices stored here
    pub(crate) j: Vec<i32>,     // zero- or one-based indices stored here
    pub(crate) x: Vec<f64>,     // the non-zero entries in the matrix
}

impl SparseTriplet {
    pub fn new() -> Self {
        SparseTriplet {
            m: 0,
            n: 0,
            pos: 0,
            max: 0,
            one_based: false,
            symmetric: false,
            i: vec![0; 0],
            j: vec![0; 0],
            x: vec![0.0; 0],
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    // use russell_chk::*;

    #[test]
    fn new_works() {
        let t = SparseTriplet::new();
        assert_eq!(t.m, 0);
        assert_eq!(t.n, 0);
        assert_eq!(t.pos, 0);
        assert_eq!(t.max, 0);
        assert_eq!(t.one_based, false);
        assert_eq!(t.symmetric, false);
        assert_eq!(t.i, &[]);
        assert_eq!(t.j, &[]);
        assert_eq!(t.x, &[]);
    }
}
