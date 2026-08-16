use super::SQRT_2;

/// Holds third-order tensor samples
pub struct SamplesTensor3;

impl SamplesTensor3 {
    /// Tensor3 specified by standard 3x3x3 components
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

    /// Matrix representation of SAMPLE1
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

    /// Kelvin matrix representation of SAMPLE1
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

    /// Tensor3 specified by standard 3x3x3 components
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

    /// Minor-symmetric Tensor3 specified by 3x3x3 components
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

    /// Matrix representation of SYM_SAMPLE1
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

    /// Kelvin matrix representation of SYM_SAMPLE1
    #[rustfmt::skip]
    pub const SYM_SAMPLE1_KELVIN_MATRIX:[[f64; 3]; 6] = [
        [ 1.0       ,  2.0       ,  3.0       ],
        [ 4.0       ,  5.0       ,  6.0       ],
        [ 7.0       ,  8.0       ,  9.0       ],
        [10.0*SQRT_2, 11.0*SQRT_2, 12.0*SQRT_2],
        [13.0*SQRT_2, 14.0*SQRT_2, 15.0*SQRT_2],
        [16.0*SQRT_2, 17.0*SQRT_2, 18.0*SQRT_2],
    ];

    /// Minor-symmetric Tensor3 specified by 3x3x3 components (2D problems)
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

    /// Matrix representation of SYM_2D_SAMPLE1
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

    /// Kelvin matrix representation of SYM_2D_SAMPLE1
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
    use super::{SQRT_2, SamplesTensor3};
    use crate::constants::IJK_TO_MN;
    use russell_lab::approx_eq;

    #[test]
    fn sample1_is_ok() {
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let (m, n) = IJK_TO_MN[i][j][k];
                    let val = SamplesTensor3::SAMPLE1_STD_MATRIX[m][n];
                    assert_eq!(SamplesTensor3::SAMPLE1[i][j][k], val);
                    if i == j {
                        assert_eq!(
                            SamplesTensor3::SAMPLE1_KELVIN_MATRIX[m][n],
                            SamplesTensor3::SAMPLE1[i][j][k]
                        );
                    } else if i < j {
                        assert_eq!(
                            SamplesTensor3::SAMPLE1_KELVIN_MATRIX[m][n],
                            (SamplesTensor3::SAMPLE1[i][j][k] + SamplesTensor3::SAMPLE1[j][i][k]) / SQRT_2
                        );
                    } else {
                        assert_eq!(
                            SamplesTensor3::SAMPLE1_KELVIN_MATRIX[m][n],
                            (SamplesTensor3::SAMPLE1[j][i][k] - SamplesTensor3::SAMPLE1[i][j][k]) / SQRT_2
                        );
                    }
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
                    let (m, n) = IJK_TO_MN[i][j][k];
                    let val = SamplesTensor3::SYM_SAMPLE1_STD_MATRIX[m][n];
                    assert_eq!(SamplesTensor3::SYM_SAMPLE1[i][j][k], val);
                    if i == j {
                        assert_eq!(
                            SamplesTensor3::SYM_SAMPLE1_KELVIN_MATRIX[m][n],
                            SamplesTensor3::SYM_SAMPLE1[i][j][k]
                        );
                    } else if i < j {
                        approx_eq(
                            SamplesTensor3::SYM_SAMPLE1_KELVIN_MATRIX[m][n],
                            (SamplesTensor3::SYM_SAMPLE1[i][j][k] + SamplesTensor3::SYM_SAMPLE1[j][i][k]) / SQRT_2,
                            1e-14,
                        );
                    } else {
                        if m < 6 {
                            assert_eq!(
                                SamplesTensor3::SYM_SAMPLE1_KELVIN_MATRIX[m][n],
                                (SamplesTensor3::SYM_SAMPLE1[j][i][k] - SamplesTensor3::SYM_SAMPLE1[i][j][k]) / SQRT_2
                            );
                        }
                    }
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
                    if i == j {
                        assert_eq!(
                            SamplesTensor3::SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n],
                            SamplesTensor3::SYM_2D_SAMPLE1[i][j][k]
                        );
                    } else if i < j {
                        if m < 4 {
                            approx_eq(
                                SamplesTensor3::SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n],
                                (SamplesTensor3::SYM_2D_SAMPLE1[i][j][k] + SamplesTensor3::SYM_2D_SAMPLE1[j][i][k])
                                    / SQRT_2,
                                1e-14,
                            );
                        }
                    } else {
                        if m < 4 {
                            assert_eq!(
                                SamplesTensor3::SYM_2D_SAMPLE1_KELVIN_MATRIX[m][n],
                                (SamplesTensor3::SYM_2D_SAMPLE1[j][i][k] - SamplesTensor3::SYM_2D_SAMPLE1[i][j][k])
                                    / SQRT_2
                            );
                        }
                    }
                }
            }
        }
    }
}
