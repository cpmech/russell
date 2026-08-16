use super::SQRT_2;

/// Holds fourth-order tensor samples
pub struct SamplesTensor3;

impl SamplesTensor3 {
    #[rustfmt::skip]
    pub const SAMPLE1: [[[f64; 3]; 3]; 3] = [
        // [0]
        [
            [ 1.0,  2.0,  3.0], // [0][0][0], [0][0][1], [0][0][2]
            [10.0, 11.0, 12.0], // [0][1][0], [0][1][1], [0][1][2]
            [16.0, 17.0, 18.0], // [0][2][0], [0][2][1], [0][2][2]
        ],
        // [1]
        [
            [19.0, 20.0, 21.0], // [1][0][0], [1][0][1], [1][0][2]
            [ 4.0,  5.0,  6.0], // [1][1][0], [1][1][1], [1][1][2]
            [13.0, 14.0, 15.0], // [1][2][0], [1][2][1], [1][2][2]
        ],
        // [2]
        [
            [25.0, 26.0, 27.0], // [2][0][0], [2][0][1], [2][0][2]
            [22.0, 23.0, 24.0], // [2][1][0], [2][1][1], [2][1][2]
            [ 7.0,  8.0,  9.0], // [2][2][0], [2][2][1], [2][2][2]
        ],
    ];

    /// Sample matrix representation with standard components
    #[rustfmt::skip]
    pub const SAMPLE1_STD_MATRIX: [[f64; 3]; 9] = [
        [ 1.0,  2.0,  3.0], // [0][0]...
        [ 4.0,  5.0,  6.0], // [1][1]...
        [ 7.0,  8.0,  9.0], // [2][2]...
        [10.0, 11.0, 12.0], // [0][1]...
        [13.0, 14.0, 15.0], // [1][2]...
        [16.0, 17.0, 18.0], // [0][2]...
        [19.0, 20.0, 21.0], // [1][0]...
        [22.0, 23.0, 24.0], // [2][1]...
        [25.0, 26.0, 27.0], // [2][0]...
    ];

    /// Sample matrix representation with Kelvin components
    #[rustfmt::skip]
    pub const SAMPLE1_KELVIN_MATRIX:[[f64; 3]; 9] = [
        [         1.0 ,          2.0 ,          3.0 ],
        [         4.0 ,          5.0 ,          6.0 ],
        [         7.0 ,          8.0 ,          9.0 ],
        [ 29.0/SQRT_2 ,  31.0/SQRT_2 ,  33.0/SQRT_2 ],
        [ 35.0/SQRT_2 ,  37.0/SQRT_2 ,  39.0/SQRT_2 ],
        [ 41.0/SQRT_2 ,  43.0/SQRT_2 ,  45.0/SQRT_2 ],
        [ -9.0/SQRT_2 ,  -9.0/SQRT_2 ,  -9.0/SQRT_2 ],
        [ -9.0/SQRT_2 ,  -9.0/SQRT_2 ,  -9.0/SQRT_2 ],
        [ -9.0/SQRT_2 ,  -9.0/SQRT_2 ,  -9.0/SQRT_2 ],
    ];

    /*
    let mut sample2 = [[[0.0_f64; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                sample2[i][j][k] = ((i + 1) * 100 + (j + 1) * 10 + (k + 1)) as f64;
            }
        }
    }
    */
    pub const SAMPLE2: [[[f64; 3]; 3]; 3] = [
        // [0]
        [
            [111_f64, 112_f64, 113_f64], // [0][0][0], [0][0][1], [0][0][2]
            [121_f64, 122_f64, 123_f64], // [0][1][0], [0][1][1], [0][1][2]
            [131_f64, 132_f64, 133_f64], // [0][2][0], [0][2][1], [0][2][2]
        ],
        // [1]
        [
            [211_f64, 212_f64, 213_f64], // [1][0][0], [1][0][1], [1][0][2]
            [221_f64, 222_f64, 223_f64], // [1][1][0], [1][1][1], [1][1][2]
            [231_f64, 232_f64, 233_f64], // [1][2][0], [1][2][1], [1][2][2]
        ],
        // [2]
        [
            [311_f64, 312_f64, 313_f64], // [2][0][0], [2][0][1], [2][0][2]
            [321_f64, 322_f64, 323_f64], // [2][1][0], [2][1][1], [2][1][2]
            [331_f64, 332_f64, 333_f64], // [2][2][0], [2][2][1], [2][2][2]
        ],
    ];

    /*
    fn map_tensor_to_voigt_9x3(tensor: &[[[f64; 3]; 3]; 3]) -> [[f64; 3]; 9] {
        let mut matrix = [[0.0; 3]; 9];

        // Explicit 0-indexed mapping for (i, j) based on your rule
        const ROW_TO_IJ: [(usize, usize); 9] = [
            (0, 0), // 1 -> 11
            (1, 1), // 2 -> 22
            (2, 2), // 3 -> 33
            (0, 1), // 4 -> 12
            (1, 2), // 5 -> 23
            (0, 2), // 6 -> 13
            (1, 0), // 7 -> 21
            (2, 1), // 8 -> 32
            (2, 0), // 9 -> 31
        ];

        for row in 0..9 {
            let (i, j) = ROW_TO_IJ[row];
            for k in 0..3 {
                matrix[row][k] = tensor[i][j][k];
            }
        }

        matrix
    }
    */

    /*
    fn map_voigt_9x3_to_tensor(matrix: &[[f64; 3]; 9]) -> [[[f64; 3]; 3]; 3] {
        let mut tensor = [[[0.0; 3]; 3]; 3];

        // Reverse lookup table mapping row index to (i, j)
        const ROW_TO_IJ: [(usize, usize); 9] = [
            (0, 0), (1, 1), (2, 2), // Rows 0, 1, 2 -> Diagonals
            (0, 1), (1, 2), (0, 2), // Rows 3, 4, 5 -> Upper Triangular
            (1, 0), (2, 1), (2, 0), // Rows 6, 7, 8 -> Lower Triangular
        ];

        for row in 0..9 {
            let (i, j) = ROW_TO_IJ[row];
            for k in 0..3 {
                tensor[i][j][k] = matrix[row][k];
            }
        }

        tensor
    }
    */

    /*
    fn map_mandel_9x3_to_tensor(matrix: &[[f64; 3]; 9]) -> [[[f64; 3]; 3]; 3] {
        let mut tensor = [[[0.0; 3]; 3]; 3];
        let sqrt2 = 2.0_f64.sqrt();

        const ROW_TO_IJ: [(usize, usize); 9] = [
            (0, 0), (1, 1), (2, 2),
            (0, 1), (1, 2), (0, 2),
            (1, 0), (2, 1), (2, 0),
        ];

        for row in 0..9 {
            let (i, j) = ROW_TO_IJ[row];
            // Rows 3 to 8 are off-diagonal shear components
            let weight = if row >= 3 { sqrt2 } else { 1.0 };

            for k in 0..3 {
                // Divide by sqrt(2) to revert Mandel normalization back to pure tensor values
                tensor[i][j][k] = matrix[row][k] / weight;
            }
        }

        tensor
    }
    */

    #[rustfmt::skip]
    pub const SYM_SAMPLE1: [[[f64; 3]; 3]; 3] = [
        // [0]
        [
            [ 1.0,  2.0,  3.0], // [0][0][0], [0][0][1], [0][0][2]
            [10.0, 11.0, 12.0], // [0][1][0], [0][1][1], [0][1][2]
            [16.0, 17.0, 18.0], // [0][2][0], [0][2][1], [0][2][2]
        ],
        // [1]
        [
            [10.0, 11.0, 12.0], // [1][0][0], [1][0][1], [1][0][2]
            [ 4.0,  5.0,  6.0], // [1][1][0], [1][1][1], [1][1][2]
            [13.0, 14.0, 15.0], // [1][2][0], [1][2][1], [1][2][2]
        ],
        // [2]
        [
            [16.0, 17.0, 18.0], // [2][0][0], [2][0][1], [2][0][2]
            [13.0, 14.0, 15.0], // [2][1][0], [2][1][1], [2][1][2]
            [ 7.0,  8.0,  9.0], // [2][2][0], [2][2][1], [2][2][2]
        ],
    ];

    /// Sample matrix representation of symmetric tensor with standard components
    #[rustfmt::skip]
    pub const SYM_SAMPLE1_STD_MATRIX: [[f64; 3]; 9] = [
        [ 1.0,  2.0,  3.0], // [0][0]...
        [ 4.0,  5.0,  6.0], // [1][1]...
        [ 7.0,  8.0,  9.0], // [2][2]...
        [10.0, 11.0, 12.0], // [0][1]...
        [13.0, 14.0, 15.0], // [1][2]...
        [16.0, 17.0, 18.0], // [0][2]...
        [10.0, 11.0, 12.0], // [1][0]...
        [13.0, 14.0, 15.0], // [2][1]...
        [16.0, 17.0, 18.0], // [2][0]...
    ];

    /// Sample matrix representation of symmetric tensor with Kelvin components
    #[rustfmt::skip]
    pub const SYM_SAMPLE1_KELVIN_MATRIX:[[f64; 3]; 6] = [
        [ 1.0       ,  2.0       ,  3.0       ],
        [ 4.0       ,  5.0       ,  6.0       ],
        [ 7.0       ,  8.0       ,  9.0       ],
        [10.0*SQRT_2, 11.0*SQRT_2, 12.0*SQRT_2],
        [13.0*SQRT_2, 14.0*SQRT_2, 15.0*SQRT_2],
        [16.0*SQRT_2, 17.0*SQRT_2, 18.0*SQRT_2],
    ];

    #[rustfmt::skip]
    pub const SYM_2D_SAMPLE1: [[[f64; 3]; 3]; 3] = [
        // [0]
        [
            [ 1.0,  2.0,  3.0], // [0][0][0], [0][0][1], [0][0][2]
            [10.0, 11.0, 12.0], // [0][1][0], [0][1][1], [0][1][2]
            [ 0.0,  0.0,  0.0], // [0][2][0], [0][2][1], [0][2][2]
        ],
        // [1]
        [
            [10.0, 11.0, 12.0], // [1][0][0], [1][0][1], [1][0][2]
            [ 4.0,  5.0,  6.0], // [1][1][0], [1][1][1], [1][1][2]
            [ 0.0,  0.0,  0.0], // [1][2][0], [1][2][1], [1][2][2]
        ],
        // [2]
        [
            [ 0.0,  0.0,  0.0], // [2][0][0], [2][0][1], [2][0][2]
            [ 0.0,  0.0,  0.0], // [2][1][0], [2][1][1], [2][1][2]
            [ 7.0,  8.0,  9.0], // [2][2][0], [2][2][1], [2][2][2]
        ],
    ];

    /// Sample matrix representation for 2D spaces with standard components
    #[rustfmt::skip]
    pub const SYM_2D_SAMPLE1_STD_MATRIX: [[f64; 3]; 9] = [
        [ 1.0,  2.0,  3.0], // [0][0]...
        [ 4.0,  5.0,  6.0], // [1][1]...
        [ 7.0,  8.0,  9.0], // [2][2]...
        [10.0, 11.0, 12.0], // [0][1]...
        [ 0.0,  0.0,  0.0], // [1][2]...
        [ 0.0,  0.0,  0.0], // [0][2]...
        [10.0, 11.0, 12.0], // [1][0]...
        [ 0.0,  0.0,  0.0], // [2][1]...
        [ 0.0,  0.0,  0.0], // [2][0]...
    ];

    /// Sample matrix representation for 2D spaces with Kelvin components
    #[rustfmt::skip]
    pub const SYM_2D_SAMPLE1_KELVIN_MATRIX:[[f64; 3]; 4] = [
        [ 1.0       ,  2.0       ,  3.0       ],
        [ 4.0       ,  5.0       ,  6.0       ],
        [ 7.0       ,  8.0       ,  9.0       ],
        [10.0*SQRT_2, 11.0*SQRT_2, 12.0*SQRT_2],
    ];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::SamplesTensor3;
    use crate::constants::IJK_TO_MN;

    #[test]
    fn sample1_is_ok() {
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let (m, n) = IJK_TO_MN[i][j][k];
                    let val = SamplesTensor3::SAMPLE1_STD_MATRIX[m][n];
                    assert_eq!(SamplesTensor3::SAMPLE1[i][j][k], val);
                }
            }
        }
    }

    #[test]
    fn sample2_is_ok() {
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let val = (i + 1) * 100 + (j + 1) * 10 + (k + 1);
                    assert_eq!(SamplesTensor3::SAMPLE2[i][j][k], val as f64);
                }
            }
        }
    }

    #[test]
    fn sample1_sym_is_ok() {
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let (a, b) = IJK_TO_MN[i][j][k];
                    let val = SamplesTensor3::SYM_SAMPLE1_STD_MATRIX[a][b];
                    assert_eq!(SamplesTensor3::SYM_SAMPLE1[i][j][k], val);
                }
            }
        }
    }

    #[test]
    fn sample1_sym_2d_is_ok() {
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let (m, n) = IJK_TO_MN[i][j][k];
                    let val = SamplesTensor3::SYM_2D_SAMPLE1_STD_MATRIX[m][n];
                    assert_eq!(SamplesTensor3::SYM_2D_SAMPLE1[i][j][k], val);
                }
            }
        }
    }
}
