use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // allocate a square matrix and store as CSC matrix
    // ┌                ┐
    // │  2  3  0  0  0 │
    // │  3  0  4  0  6 │
    // │  0 -1 -3  2  0 │
    // │  0  0  1  0  0 │
    // │  0  4  2  0  1 │
    // └                ┘
    let csc = CscMatrix {
        symmetry: None,
        nrow: 5,
        ncol: 5,
        col_pointers: vec![0, 2, 5, 9, 10, 12],
        row_indices: vec![
            //                             p
            0, 1, //       j = 0, count =  0, 1,
            0, 2, 4, //    j = 1, count =  2, 3, 4,
            1, 2, 3, 4, // j = 2, count =  5, 6, 7, 8,
            2, //          j = 3, count =  9,
            1, 4, //       j = 4, count = 10, 11,
               //                         12
        ],
        values: vec![
            //                                      p
            2.0, 3.0, //            j = 0, count =  0, 1,
            3.0, -1.0, 4.0, //      j = 1, count =  2, 3, 4,
            4.0, -3.0, 1.0, 2.0, // j = 2, count =  5, 6, 7, 8,
            2.0, //                 j = 3, count =  9,
            6.0, 1.0, //            j = 4, count = 10, 11,
                 //                                12
        ],
    };

    // covert to dense
    let a = csc.as_matrix()?;
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
