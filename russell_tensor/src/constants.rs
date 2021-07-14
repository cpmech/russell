// sqrt(2) https://oeis.org/A002193
pub const SQRT_2: f64 = 1.41421356237309504880168872420969807856967187537694807317667973799073247846210703885038753432764157f64;

// sqrt(3) https://oeis.org/A002194
pub const SQRT_3: f64 = 1.7320508075688772935274463415058723669428052538103806280558069794519330169088000370811461867572485756756261414154f64;

// sqrt(6) https://oeis.org/A010464
pub const SQRT_6: f64 = 2.44948974278317809819728407470589139196594748065667012843269256725096037745731502653985943310464023f64;

// sqrt(2/3) https://oeis.org/A157697
pub const SQRT_2_BY_3: f64 = 0.816496580927726032732428024901963797321982493552223376144230855750320125819105008846619811034880078272864f64;

// sqt(3/2) https://oeis.org/A115754
pub const SQRT_3_BY_2: f64 = 1.22474487139158904909864203735294569598297374032833506421634628362548018872865751326992971655232011f64;

// 1/3
pub const ONE_BY_3: f64 = 0.33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333f64;

// 2/3
pub const TWO_BY_3: f64 = 0.66666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666f64;

// maps the component (i,j) of a second order tensor to the i-position in the Mandel vector
pub const IJ_TO_MANDEL_VECTOR_I: [[usize; 3]; 3] = [
    [0, 3, 5], // comment to prevent auto format
    [6, 1, 4], // comment to prevent auto format
    [8, 7, 2], // comment to prevent auto format
];

// maps the component (i,j) of a symmetric second order tensor to the i-position in the Mandel vector
pub const IJ_SYM_TO_MANDEL_VECTOR_I: [[usize; 3]; 3] = [
    [0, 3, 5], // comment to prevent auto format
    [3, 1, 4], // comment to prevent auto format
    [5, 4, 2], // comment to prevent auto format
];

// maps the component (i,j,k,l) of a fourth order tensor to the i-position in the Mandel matrix
pub const IJKL_TO_MANDEL_MATRIX_I: [[[[usize; 3]; 3]; 3]; 3] = [
    [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]], // [0][0][.][.]
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]], // [0][1][.][.]
        [[5, 5, 5], [5, 5, 5], [5, 5, 5]], // [0][2][.][.]
    ],
    [
        [[6, 6, 6], [6, 6, 6], [6, 6, 6]], // [1][0][.][.]
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]], // [1][1][.][.]
        [[4, 4, 4], [4, 4, 4], [4, 4, 4]], // [1][2][.][.]
    ],
    [
        [[8, 8, 8], [8, 8, 8], [8, 8, 8]], // [2][0][.][.]
        [[7, 7, 7], [7, 7, 7], [7, 7, 7]], // [2][1][.][.]
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]], // [2][2][.][.]
    ],
];

// maps the component (i,j,k,l) of a fourth order tensor to the j-position in the Mandel matrix
pub const IJKL_TO_MANDEL_MATRIX_J: [[[[usize; 3]; 3]; 3]; 3] = [
    [
        [[0, 3, 5], [6, 1, 4], [8, 7, 2]], // [0][0][.][.]
        [[0, 3, 5], [6, 1, 4], [8, 7, 2]], // [0][1][.][.]
        [[0, 3, 5], [6, 1, 4], [8, 7, 2]], // [0][2][.][.]
    ],
    [
        [[0, 3, 5], [6, 1, 4], [8, 7, 2]], // [1][0][.][.]
        [[0, 3, 5], [6, 1, 4], [8, 7, 2]], // [1][1][.][.]
        [[0, 3, 5], [6, 1, 4], [8, 7, 2]], // [1][2][.][.]
    ],
    [
        [[0, 3, 5], [6, 1, 4], [8, 7, 2]], // [2][0][.][.]
        [[0, 3, 5], [6, 1, 4], [8, 7, 2]], // [2][1][.][.]
        [[0, 3, 5], [6, 1, 4], [8, 7, 2]], // [2][2][.][.]
    ],
];

// maps the component (i,j,k,l) of a symmetric fourth order tensor to the i-position in the Mandel matrix
pub const IJKL_SYM_TO_MANDEL_MATRIX_I: [[[[usize; 3]; 3]; 3]; 3] = [
    [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]], // [0][0][.][.]
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]], // [0][1][.][.]
        [[5, 5, 5], [5, 5, 5], [5, 5, 5]], // [0][2][.][.]
    ],
    [
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]], // [1][0][.][.]
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]], // [1][1][.][.]
        [[4, 4, 4], [4, 4, 4], [4, 4, 4]], // [1][2][.][.]
    ],
    [
        [[5, 5, 5], [5, 5, 5], [5, 5, 5]], // [2][0][.][.]
        [[4, 4, 4], [4, 4, 4], [4, 4, 4]], // [2][1][.][.]
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]], // [2][2][.][.]
    ],
];

// maps the component (i,j,k,l) of a fourth order tensor to the j-position in the Mandel matrix
pub const IJKL_SYM_TO_MANDEL_MATRIX_J: [[[[usize; 3]; 3]; 3]; 3] = [
    [
        [[0, 3, 5], [3, 1, 4], [5, 4, 2]], // [0][0][.][.]
        [[0, 3, 5], [3, 1, 4], [5, 4, 2]], // [0][1][.][.]
        [[0, 3, 5], [3, 1, 4], [5, 4, 2]], // [0][2][.][.]
    ],
    [
        [[0, 3, 5], [3, 1, 4], [5, 4, 2]], // [1][0][.][.]
        [[0, 3, 5], [3, 1, 4], [5, 4, 2]], // [1][1][.][.]
        [[0, 3, 5], [3, 1, 4], [5, 4, 2]], // [1][2][.][.]
    ],
    [
        [[0, 3, 5], [3, 1, 4], [5, 4, 2]], // [2][0][.][.]
        [[0, 3, 5], [3, 1, 4], [5, 4, 2]], // [2][1][.][.]
        [[0, 3, 5], [3, 1, 4], [5, 4, 2]], // [2][2][.][.]
    ],
];

///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_correct() {
        assert_eq!(SQRT_2, 2_f64.sqrt());
        assert_eq!(SQRT_3, 3_f64.sqrt());
        assert_eq!(SQRT_6, 6_f64.sqrt());
        assert_eq!(SQRT_2_BY_3, (2_f64 / 3_f64).sqrt());
        assert_eq!(SQRT_3_BY_2, (3_f64 / 2_f64).sqrt());
        assert_eq!(ONE_BY_3, 1_f64 / 3_f64);
        assert_eq!(TWO_BY_3, 2_f64 / 3_f64);
    }

    #[test]
    fn ij_to_mandel_map_is_correct() {
        #[rustfmt::skip]
        let vec = [
            (0, 0), (1, 1), (2, 2), // 0,1,2 => diagonal
            (0, 1), (1, 2), (0, 2), // 3,4,5 => upper-diagonal
            (1, 0), (2, 1), (2, 0), // 6,7,8 => lower-diagonal
        ];
        for a in 0..9 {
            let (i, j) = vec[a];
            assert_eq!(IJ_TO_MANDEL_VECTOR_I[i][j], a);
        }
    }

    #[test]
    fn ij_sym_to_mandel_map_is_correct() {
        #[rustfmt::skip]
        let vec = [
            (0, 0), (1, 1), (2, 2), // 0,1,2 => diagonal
            (0, 1), (1, 2), (0, 2), // 3,4,5 => upper-diagonal
        ];
        for a in 0..6 {
            let (i, j) = vec[a];
            assert_eq!(IJ_SYM_TO_MANDEL_VECTOR_I[i][j], a);
        }
    }

    #[test]
    fn ijkl_to_mandel_matrix_maps_are_correct() {
        #[rustfmt::skip]
        let mat = [
            [(0,0,0,0), (0,0,1,1), (0,0,2,2), (0,0,0,1), (0,0,1,2), (0,0,0,2), (0,0,1,0), (0,0,2,1), (0,0,2,0)], // 0
            [(1,1,0,0), (1,1,1,1), (1,1,2,2), (1,1,0,1), (1,1,1,2), (1,1,0,2), (1,1,1,0), (1,1,2,1), (1,1,2,0)], // 1
            [(2,2,0,0), (2,2,1,1), (2,2,2,2), (2,2,0,1), (2,2,1,2), (2,2,0,2), (2,2,1,0), (2,2,2,1), (2,2,2,0)], // 2
            [(0,1,0,0), (0,1,1,1), (0,1,2,2), (0,1,0,1), (0,1,1,2), (0,1,0,2), (0,1,1,0), (0,1,2,1), (0,1,2,0)], // 3
            [(1,2,0,0), (1,2,1,1), (1,2,2,2), (1,2,0,1), (1,2,1,2), (1,2,0,2), (1,2,1,0), (1,2,2,1), (1,2,2,0)], // 4
            [(0,2,0,0), (0,2,1,1), (0,2,2,2), (0,2,0,1), (0,2,1,2), (0,2,0,2), (0,2,1,0), (0,2,2,1), (0,2,2,0)], // 5
            [(1,0,0,0), (1,0,1,1), (1,0,2,2), (1,0,0,1), (1,0,1,2), (1,0,0,2), (1,0,1,0), (1,0,2,1), (1,0,2,0)], // 6
            [(2,1,0,0), (2,1,1,1), (2,1,2,2), (2,1,0,1), (2,1,1,2), (2,1,0,2), (2,1,1,0), (2,1,2,1), (2,1,2,0)], // 7
            [(2,0,0,0), (2,0,1,1), (2,0,2,2), (2,0,0,1), (2,0,1,2), (2,0,0,2), (2,0,1,0), (2,0,2,1), (2,0,2,0)], // 8
        ];
        for a in 0..9 {
            for b in 0..9 {
                let (i, j, k, l) = mat[a][b];
                assert_eq!(IJKL_TO_MANDEL_MATRIX_I[i][j][k][l], a);
                assert_eq!(IJKL_TO_MANDEL_MATRIX_J[i][j][k][l], b);
            }
        }
    }

    #[test]
    fn ijkl_sym_to_mandel_matrix_maps_are_correct() {
        #[rustfmt::skip]
        let mat = [
            [(0,0,0,0), (0,0,1,1), (0,0,2,2), (0,0,0,1), (0,0,1,2), (0,0,0,2)], // 0
            [(1,1,0,0), (1,1,1,1), (1,1,2,2), (1,1,0,1), (1,1,1,2), (1,1,0,2)], // 1
            [(2,2,0,0), (2,2,1,1), (2,2,2,2), (2,2,0,1), (2,2,1,2), (2,2,0,2)], // 2
            [(0,1,0,0), (0,1,1,1), (0,1,2,2), (0,1,0,1), (0,1,1,2), (0,1,0,2)], // 3
            [(1,2,0,0), (1,2,1,1), (1,2,2,2), (1,2,0,1), (1,2,1,2), (1,2,0,2)], // 4
            [(0,2,0,0), (0,2,1,1), (0,2,2,2), (0,2,0,1), (0,2,1,2), (0,2,0,2)], // 5
        ];
        for a in 0..6 {
            for b in 0..6 {
                let (i, j, k, l) = mat[a][b];
                assert_eq!(IJKL_SYM_TO_MANDEL_MATRIX_I[i][j][k][l], a);
                assert_eq!(IJKL_SYM_TO_MANDEL_MATRIX_J[i][j][k][l], b);
            }
        }
    }
}
