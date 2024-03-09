use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // allocate a square matrix and store as CSR matrix
    //  2  3  .  .  .
    //  3  .  4  .  6
    //  . -1 -3  2  .
    //  .  .  1  .  .
    //  .  4  2  .  1
    let nrow = 5;
    let ncol = 5;
    let row_pointers = vec![0, 2, 5, 8, 9, 12];
    let col_indices = vec![
        //                         p
        0, 1, //    i = 0, count = 0, 1
        0, 2, 4, // i = 1, count = 2, 3, 4
        1, 2, 3, // i = 2, count = 5, 6, 7
        2, //       i = 3, count = 8
        1, 2, 4, // i = 4, count = 9, 10, 11
           //              count = 12
    ];
    let values = vec![
        //                                 p
        2.0, 3.0, //        i = 0, count = 0, 1
        3.0, 4.0, 6.0, //   i = 1, count = 2, 3, 4
        -1.0, -3.0, 2.0, // i = 2, count = 5, 6, 7
        1.0, //             i = 3, count = 8
        4.0, 2.0, 1.0, //   i = 4, count = 9, 10, 11
             //                    count = 12
    ];
    let sym = None;
    let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym)?;

    // covert to dense
    let a = csr.as_dense();
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
