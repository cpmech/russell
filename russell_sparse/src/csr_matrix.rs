use super::Layout;
use crate::CooMatrix;
use russell_openblas::to_i32;

/// Holds the arrays needed for a CSR (compressed sparse row) matrix
pub struct CsrMatrix {
    /// Defines the stored layout: lower-triangular, upper-triangular, full-matrix
    pub layout: Layout,

    /// Holds the number of rows (must fit i32)
    pub nrow: usize,

    /// Holds the number of columns (must fit i32)
    pub ncol: usize,

    /// Defines the row pointers array with size = n_row + 1
    pub row_pointers: Vec<i32>,

    /// Defines the column indices array with size = nnz (number of non-zeros)
    pub col_indices: Vec<i32>,

    /// Defines the values array with size = nnz (number of non-zeros)
    pub values: Vec<f64>,
}

impl CsrMatrix {
    pub fn from(coo: &CooMatrix) -> Self {
        // Based on the SciPy code from here:
        //
        // https://github.com/scipy/scipy/blob/main/scipy/sparse/sparsetools/coo.h
        //
        // Notes:
        //
        // * The row and column indices may be unordered (NOT TRUE)
        // * Linear complexity: O(nnz(A) + max(n_row,n_col))

        // access triplet data
        let ai = &coo.indices_i;
        let aj = &coo.indices_j;
        let ax = &coo.values_aij;

        // allocate vectors
        let nrow = coo.nrow;
        let nnz = coo.pos;
        let mut csr = CsrMatrix {
            layout: coo.layout,
            nrow: coo.nrow,
            ncol: coo.ncol,
            row_pointers: vec![0; nrow + 1],
            col_indices: vec![0; nnz],
            values: vec![0.0; nnz],
        };
        let bp = &mut csr.row_pointers;
        let bj = &mut csr.col_indices;
        let bx = &mut csr.values;

        // compute number of non-zero entries per row of A
        for k in 0..nnz {
            bp[ai[k] as usize] += 1;
        }

        // perform the cumulative sum of the nnz per row to get bp[]
        let mut cum_sum = 0;
        for i in 0..nrow {
            let temp = bp[i];
            bp[i] = cum_sum;
            cum_sum += temp;
        }
        bp[nrow] = to_i32(nnz);

        // write aj and ax into bj and bx (will use bp as workspace)
        for k in 0..nnz {
            let row = ai[k];
            let dest = bp[row as usize];
            bj[dest as usize] = aj[k];
            bx[dest as usize] = ax[k];
            bp[row as usize] += 1;
        }

        // fix bp
        let mut last = 0;
        for i in 0..nrow {
            let temp = bp[i];
            bp[i] = last;
            last = temp;
        }

        // sort rows
        let mut temp: Vec<(i32, f64)> = Vec::new();
        for i in 0..nrow {
            let row_start = bp[i];
            let row_end = bp[i + 1];
            temp.resize((row_end - row_start) as usize, (0, 0.0));
            let mut n = 0;
            for jj in row_start..row_end {
                temp[n].0 = bj[jj as usize];
                temp[n].1 = bx[jj as usize];
                n += 1;
            }
            temp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            n = 0;
            for jj in row_start..row_end {
                bj[jj as usize] = temp[n].0;
                bx[jj as usize] = temp[n].1;
                n += 1;
            }
        }

        // sum duplicates
        let final_nnz = csr_sum_duplicates(nrow, bp, bj, bx);
        bj.resize(final_nnz, 0);
        bx.resize(final_nnz, 0.0);

        // results
        csr
    }
}

/// brief Sums duplicate column entries in each row of a CSR matrix
///
/// Returns The final number of non-zeros (nnz) after duplicates have been handled
fn csr_sum_duplicates(nrow: usize, ap: &mut [i32], aj: &mut [i32], ax: &mut [f64]) -> usize {
    // Based on the SciPy code from here:
    //
    // https://github.com/scipy/scipy/blob/main/scipy/sparse/sparsetools/csr.h
    //
    // * ap[n_row+1] -- row pointer
    // * aj[nnz(A)]  -- column indices
    // * ax[nnz(A)]  -- non-zeros
    // * The column indices within each row must be sorted
    // * Explicit zeros are retained
    // * ap, aj, and ax will be modified in place

    let mut nnz: i32 = 0;
    let mut row_end = 0;
    for i in 0..nrow {
        let mut jj = row_end;
        row_end = ap[i + 1];
        while jj < row_end {
            let j = aj[jj as usize];
            let mut x = ax[jj as usize];
            jj += 1;
            while jj < row_end && aj[jj as usize] == j {
                x += ax[jj as usize];
                jj += 1;
            }
            aj[nnz as usize] = j;
            ax[nnz as usize] = x;
            nnz += 1;
        }
        ap[i + 1] = nnz;
    }
    nnz as usize
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::CsrMatrix;
    use crate::{CooMatrix, Layout};
    use russell_chk::vec_approx_eq;

    #[test]
    fn csr_matrix_first_triplet_with_shuffled_entries() {
        //  1  -1   .  -3   .
        // -2   5   .   .   .
        //  .   .   4   6   4
        // -4   .   2   7   .
        //  .   8   .   .  -5
        // first triplet with shuffled entries
        let mut coo = CooMatrix::new(Layout::Full, 5, 5, 13).unwrap();
        coo.put(2, 4, 4.0).unwrap();
        coo.put(4, 1, 8.0).unwrap();
        coo.put(0, 1, -1.0).unwrap();
        coo.put(2, 2, 4.0).unwrap();
        coo.put(4, 4, -5.0).unwrap();
        coo.put(3, 0, -4.0).unwrap();
        coo.put(0, 3, -3.0).unwrap();
        coo.put(2, 3, 6.0).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 5.0).unwrap();
        coo.put(3, 2, 2.0).unwrap();
        coo.put(1, 0, -2.0).unwrap();
        coo.put(3, 3, 7.0).unwrap();
        let csr = CsrMatrix::from(&coo);
        // solution
        let correct_p = vec![0, 3, 5, 8, 11, 13];
        let correct_j = vec![
            0, 1, 3, /**/ 0, 1, /**/ 2, 3, 4, /**/ 0, 2, 3, /**/ 1, 4,
        ];
        let correct_x = vec![1.0, -1.0, -3.0, -2.0, 5.0, 4.0, 6.0, 4.0, -4.0, 2.0, 7.0, 8.0, -5.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_small_triplet_with_shuffled_entries() {
        // 1  2  .  .  .
        // 3  4  .  .  .
        // .  .  5  6  .
        // .  .  7  8  .
        // .  .  .  .  9
        // small triplet with shuffled entries
        let mut coo = CooMatrix::new(Layout::Full, 5, 5, 9).unwrap();
        coo.put(4, 4, 9.0).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 0, 3.0).unwrap();
        coo.put(2, 2, 5.0).unwrap();
        coo.put(2, 3, 6.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(3, 2, 7.0).unwrap();
        coo.put(1, 1, 4.0).unwrap();
        coo.put(3, 3, 8.0).unwrap();
        let csr = CsrMatrix::from(&coo);
        // solution
        let correct_p = vec![0, 2, 4, 6, 8, 9];
        let correct_j = vec![0, 1, 0, 1, 2, 3, 2, 3, 4];
        let correct_x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_small_triplet_with_duplicates() {
        // 1  2  .  .  .
        // 3  4  .  .  .
        // .  .  5  6  .
        // .  .  7  8  .
        // .  .  .  .  9
        // with duplicates
        let mut coo = CooMatrix::new(Layout::Full, 5, 5, 11).unwrap();
        coo.put(4, 4, 9.0).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 0, 3.0).unwrap();
        coo.put(2, 2, 5.0).unwrap();
        coo.put(2, 3, 3.0).unwrap(); // <<
        coo.put(0, 1, 2.0).unwrap();
        coo.put(3, 2, 7.0).unwrap();
        coo.put(1, 1, 2.0).unwrap(); // <<
        coo.put(3, 3, 8.0).unwrap();
        coo.put(2, 3, 3.0).unwrap(); // << duplicate
        coo.put(1, 1, 2.0).unwrap(); // << duplicate
        let csr = CsrMatrix::from(&coo);
        // solution
        let correct_p = vec![0, 2, 4, 6, 8, 9];
        let correct_j = vec![0, 1, 0, 1, 2, 3, 2, 3, 4];
        let correct_x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_upper_triangular_with_ordered_entries() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // upper triangular with ordered entries
        let mut coo = CooMatrix::new(Layout::Upper, 5, 5, 9).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(0, 1, 1.5).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(0, 2, 6.0).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(0, 3, 0.75).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(0, 4, 3.0).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        let csr = CsrMatrix::from(&coo);
        // solution
        let correct_p = vec![0, 5, 6, 7, 8, 9];
        let correct_j = vec![0, 1, 2, 3, 4, 1, 2, 3, 4];
        let correct_x = vec![9.0, 1.5, 6.0, 0.75, 3.0, 0.5, 12.0, 0.625, 16.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_upper_triangular_with_shuffled_entries() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // upper triangular with shuffled entries
        let mut coo = CooMatrix::new(Layout::Upper, 5, 5, 9).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(0, 1, 1.5).unwrap();
        coo.put(0, 2, 6.0).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        coo.put(0, 3, 0.75).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(0, 4, 3.0).unwrap();
        let csr = CsrMatrix::from(&coo);
        // solution
        let correct_p = vec![0, 5, 6, 7, 8, 9];
        let correct_j = vec![0, 1, 2, 3, 4, 1, 2, 3, 4];
        let correct_x = vec![9.0, 1.5, 6.0, 0.75, 3.0, 0.5, 12.0, 0.625, 16.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_upper_triangular_with_diagonal_entries_first() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // upper triangular with diagonal entries being set first
        let mut coo = CooMatrix::new(Layout::Upper, 5, 5, 9).unwrap();
        // diagonal
        coo.put(0, 0, 9.0).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        // upper diagonal
        coo.put(0, 1, 1.5).unwrap();
        coo.put(0, 2, 6.0).unwrap();
        coo.put(0, 3, 0.75).unwrap();
        coo.put(0, 4, 3.0).unwrap();
        let csr = CsrMatrix::from(&coo);
        // solution
        let correct_p = vec![0, 5, 6, 7, 8, 9];
        let correct_j = vec![0, 1, 2, 3, 4, 1, 2, 3, 4];
        let correct_x = vec![9.0, 1.5, 6.0, 0.75, 3.0, 0.5, 12.0, 0.625, 16.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_lower_triangular_with_ordered_entries() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // lower diagonal with ordered entries
        let mut coo = CooMatrix::new(Layout::Lower, 5, 5, 9).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(1, 0, 1.5).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(2, 0, 6.0).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(3, 0, 0.75).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(4, 0, 3.0).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        let csr = CsrMatrix::from(&coo);
        // solution
        let correct_p = vec![0, 1, 3, 5, 7, 9];
        let correct_j = vec![0, 0, 1, 0, 2, 0, 3, 0, 4];
        let correct_x = vec![9.0, 1.5, 0.5, 6.0, 12.0, 0.75, 0.625, 3.0, 16.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_lower_triangular_with_diagonal_first() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // lower triangular with diagonal entries being set first
        let mut coo = CooMatrix::new(Layout::Lower, 5, 5, 9).unwrap();
        // diagonal
        coo.put(0, 0, 9.0).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        // lower diagonal
        coo.put(1, 0, 1.5).unwrap();
        coo.put(2, 0, 6.0).unwrap();
        coo.put(3, 0, 0.75).unwrap();
        coo.put(4, 0, 3.0).unwrap();
        let csr = CsrMatrix::from(&coo);
        // solution
        let correct_p = vec![0, 1, 3, 5, 7, 9];
        let correct_j = vec![0, 0, 1, 0, 2, 0, 3, 0, 4];
        let correct_x = vec![9.0, 1.5, 0.5, 6.0, 12.0, 0.75, 0.625, 3.0, 16.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }
}
