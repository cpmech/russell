use russell_lab::Matrix;
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // allocate a square matrix and store as CSR matrix
    // ┌                ┐
    // │  2  3  0  0  0 │
    // │  3  0  4  0  6 │
    // │  0 -1 -3  2  0 │
    // │  0  0  1  0  0 │
    // │  0  4  2  0  1 │
    // └                ┘
    let csr = CsrMatrix {
        symmetry: None,
        nrow: 5,
        ncol: 5,
        row_pointers: vec![0, 2, 5, 8, 9, 12],
        col_indices: vec![
            //                         p
            0, 1, //    i = 0, count = 0, 1
            0, 2, 4, // i = 1, count = 2, 3, 4
            1, 2, 3, // i = 2, count = 5, 6, 7
            2, //       i = 3, count = 8
            1, 2, 4, // i = 4, count = 9, 10, 11
               //              count = 12
        ],
        values: vec![
            //                                 p
            2.0, 3.0, //        i = 0, count = 0, 1
            3.0, 4.0, 6.0, //   i = 1, count = 2, 3, 4
            -1.0, -3.0, 2.0, // i = 2, count = 5, 6, 7
            1.0, //             i = 3, count = 8
            4.0, 2.0, 1.0, //   i = 4, count = 9, 10, 11
                 //                    count = 12
        ],
    };

    // covert to dense
    let mut a = Matrix::new(5, 5);
    csr.to_matrix(&mut a)?;
    let correct = "┌                ┐\n\
                   │  2  3  0  0  0 │\n\
                   │  3  0  4  0  6 │\n\
                   │  0 -1 -3  2  0 │\n\
                   │  0  0  1  0  0 │\n\
                   │  0  4  2  0  1 │\n\
                   └                ┘";
    assert_eq!(format!("{}", a), correct);
    Ok(())
}
