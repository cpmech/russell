use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // allocate a square matrix and store as COO matrix
    //  2  3  .  .  .
    //  3  .  4  .  6
    //  . -1 -3  2  .
    //  .  .  1  .  .
    //  .  4  2  .  1
    let (nrow, ncol, nnz) = (5, 5, 13);
    let mut coo = CooMatrix::new(nrow, ncol, nnz, None)?;
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

    // convert to CSC matrix
    let csc = CscMatrix::from_coo(&coo)?;
    let correct_pp = vec![0, 2, 5, 9, 10, 12];
    let correct_ii = vec![
        //                             p
        0, 1, //       j = 0, count =  0, 1,
        0, 2, 4, //    j = 1, count =  2, 3, 4,
        1, 2, 3, 4, // j = 2, count =  5, 6, 7, 8,
        2, //          j = 3, count =  9,
        1, 4, //       j = 4, count = 10, 11,
           //                         12
    ];
    let correct_vv = vec![
        //                                      p
        2.0, 3.0, //            j = 0, count =  0, 1,
        3.0, -1.0, 4.0, //      j = 1, count =  2, 3, 4,
        4.0, -3.0, 1.0, 2.0, // j = 2, count =  5, 6, 7, 8,
        2.0, //                 j = 3, count =  9,
        6.0, 1.0, //            j = 4, count = 10, 11,
             //                                12
    ];

    // check
    let pp = csc.get_col_pointers();
    let ii = csc.get_row_indices();
    let vv = csc.get_values();
    let final_nnz = pp[nrow] as usize;
    assert_eq!(final_nnz, 12);
    assert_eq!(pp, correct_pp);
    assert_eq!(&ii[0..final_nnz], correct_ii);
    assert_eq!(&vv[0..final_nnz], correct_vv);
    Ok(())
}
