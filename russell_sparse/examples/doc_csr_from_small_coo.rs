use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // allocate a square matrix and store as COO matrix
    // ┌                ┐
    // │  2  3  0  0  0 │
    // │  3  0  4  0  6 │
    // │  0 -1 -3  2  0 │
    // │  0  0  1  0  0 │
    // │  0  4  2  0  1 │
    // └                ┘
    let (nrow, ncol, nnz) = (5, 5, 13);
    let mut coo = CooMatrix::new(nrow, ncol, nnz, None, false)?;
    coo.put(0, 0, 1.0)?; // << (0, 0, a00/2) duplicate
    coo.put(0, 0, 1.0)?; // << (0, 0, a00/2) duplicate
    coo.put(1, 0, 3.0)?;
    coo.put(0, 1, 3.0)?;
    coo.put(2, 1, -1.0)?;
    coo.put(4, 1, 4.0)?;
    coo.put(1, 2, 4.0)?;
    coo.put(2, 2, -3.0)?;
    coo.put(3, 2, 1.0)?;
    coo.put(4, 2, 2.0)?;
    coo.put(2, 3, 2.0)?;
    coo.put(1, 4, 6.0)?;
    coo.put(4, 4, 1.0)?;

    // convert to CSR matrix
    let csr = CsrMatrix::from_coo(&coo)?;
    let correct_j = &[
        //                         p
        0, 1, //    i = 0, count = 0, 1
        0, 2, 4, // i = 1, count = 2, 3, 4
        1, 2, 3, // i = 2, count = 5, 6, 7
        2, //       i = 3, count = 8
        1, 2, 4, // i = 4, count = 9, 10, 11
           //              count = 12
    ];
    let correct_v = &[
        //                                 p
        2.0, 3.0, //        i = 0, count = 0, 1
        3.0, 4.0, 6.0, //   i = 1, count = 2, 3, 4
        -1.0, -3.0, 2.0, // i = 2, count = 5, 6, 7
        1.0, //             i = 3, count = 8
        4.0, 2.0, 1.0, //   i = 4, count = 9, 10, 11
             //                    count = 12
    ];
    let correct_p = &[0, 2, 5, 8, 9, 12];

    // check
    let pp = csr.get_row_pointers();
    let jj = csr.get_col_indices();
    let vv = csr.get_values();
    let final_nnz = pp[nrow] as usize;
    assert_eq!(final_nnz, 12);
    assert_eq!(pp, correct_p);
    assert_eq!(&jj[0..final_nnz], correct_j);
    assert_eq!(&vv[0..final_nnz], correct_v);
    Ok(())
}
