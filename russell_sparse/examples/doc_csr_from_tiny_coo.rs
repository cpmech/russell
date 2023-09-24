use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // allocate a square matrix and store as COO matrix
    // ┌          ┐
    // │  1  0  2 │
    // │  0  0  3 │ << the diagonal 0 entry is optional,
    // │  4  5  6 │    but should be saved for Intel DSS
    // └          ┘
    let (nrow, ncol, nnz) = (3, 3, 6);
    let mut coo = CooMatrix::new(nrow, ncol, nnz, None, false)?;
    coo.put(0, 0, 1.0)?;
    coo.put(0, 2, 2.0)?;
    coo.put(1, 2, 3.0)?;
    coo.put(2, 0, 4.0)?;
    coo.put(2, 1, 5.0)?;
    coo.put(2, 2, 6.0)?;

    // convert to CSR matrix
    let csr = CsrMatrix::from(&coo)?;
    let correct_v = &[
        //                               p
        1.0, 2.0, //      i = 0, count = 0, 1
        3.0, //           i = 1, count = 2
        4.0, 5.0, 6.0, // i = 2, count = 3, 4, 5
             //                  count = 6
    ];
    let correct_j = &[
        //                         p
        0, 2, //    i = 0, count = 0, 1
        2, //       i = 1, count = 2
        0, 1, 2, // i = 2, count = 3, 4, 5
           //              count = 6
    ];
    let correct_p = &[0, 2, 3, 6];

    // check
    assert_eq!(&csr.row_pointers, correct_p);
    assert_eq!(&csr.col_indices, correct_j);
    assert_eq!(&csr.values, correct_v);
    Ok(())
}
