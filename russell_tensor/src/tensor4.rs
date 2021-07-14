use super::*;

/// Implements a fourth order tensor
pub struct Tensor4 {
    comps_mandel: Vec<f64>, // components in Mandel basis. len = 81 or 36 (minor-symmetric). col-major => Fortran
    minor_symmetric: bool,  // this tensor has minor-symmetry
}

impl Tensor4 {
    /// Returns a new Tensor4, minor-symmetric or not, with 0-valued components
    pub fn new(minor_symmetric: bool) -> Self {
        let size = if minor_symmetric { 36 } else { 81 };
        Tensor4 {
            comps_mandel: vec![0.0; size],
            minor_symmetric,
        }
    }

    pub fn from_tensor(dd: &[[[[f64; 3]; 3]; 3]; 3], minor_symmetric: bool) -> Self {
        let size = if minor_symmetric { 6 } else { 9 };
        let mut dd_bar = vec![0.0; size * size];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        let a = IJKL_TO_I[i][j][k][l];
                        let b = IJKL_TO_J[i][j][k][l];
                        let p = a + b * size; // col-major
                        if i == j && k == l {
                            dd_bar[p] = dd[i][j][k][l];
                        }
                        if i == j && k < l {
                            dd_bar[p] = (dd[i][j][k][l] + dd[i][j][l][k]) / SQRT_2;
                        }
                        if i == j && k > l {
                            dd_bar[p] = (dd[i][j][l][k] - dd[i][j][k][l]) / SQRT_2;
                        }

                        if i < j && k == l {
                            dd_bar[p] = (dd[i][j][k][l] + dd[j][i][k][l]) / SQRT_2;
                        }
                        if i < j && k < l {
                            dd_bar[p] = (dd[i][j][k][l] + dd[i][j][l][k] + dd[j][i][k][l] + dd[j][i][l][k]) / 2.0;
                        }
                        if i < j && k > l {
                            dd_bar[p] = (dd[i][j][l][k] - dd[i][j][k][l] + dd[j][i][l][k] - dd[j][i][k][l]) / 2.0;
                        }

                        if i > j && k == l {
                            dd_bar[p] = (dd[j][i][k][l] - dd[i][j][k][l]) / SQRT_2;
                        }
                        if i > j && k < l {
                            dd_bar[p] = (dd[j][i][k][l] + dd[j][i][l][k] - dd[i][j][k][l] - dd[i][j][l][k]) / 2.0;
                        }
                        if i > j && k > l {
                            dd_bar[p] = (dd[j][i][l][k] - dd[j][i][k][l] - dd[i][j][l][k] + dd[i][j][k][l]) / 2.0;
                        }
                    }
                }
            }
        }
        Tensor4 {
            comps_mandel: dd_bar,
            minor_symmetric,
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn new_tensor4_works() {
        let t4 = Tensor4::new(false);
        let correct = &[0.0; 81];
        assert_vec_approx_eq!(t4.comps_mandel, correct, 1e-15);
    }

    #[test]
    fn from_tensor_works() {
        #[rustfmt::skip]
        let comps_std: [[[[f64; 3]; 3]; 3]; 3] = [
            // [0]
            [
                // [0][0]
                [
                    [1111_f64, 1112_f64, 1113_f64], // [0][0][0][...]
                    [1121_f64, 1122_f64, 1123_f64], // [0][0][1][...]
                    [1131_f64, 1132_f64, 1133_f64], // [0][0][2][...]
                ],
                // [0][1]
                [
                    [1211_f64, 1212_f64, 1213_f64], // [0][1][0][...]
                    [1221_f64, 1222_f64, 1223_f64], // [0][1][1][...]
                    [1231_f64, 1232_f64, 1233_f64], // [0][1][2][...]
                ],
                // [0][2]
                [
                    [1311_f64, 1312_f64, 1313_f64], // [0][2][0][...]
                    [1321_f64, 1322_f64, 1323_f64], // [0][2][1][...]
                    [1331_f64, 1332_f64, 1333_f64], // [0][2][2][...]
                ],
            ],
            // [1]
            [
                // [1][0]
                [
                    [2111_f64, 2112_f64, 2113_f64], // [1][0][0][...]
                    [2121_f64, 2122_f64, 2123_f64], // [1][0][1][...]
                    [2131_f64, 2132_f64, 2133_f64], // [1][0][2][...]
                ],
                // [1][1]
                [
                    [2211_f64, 2212_f64, 2213_f64], // [1][1][0][...]
                    [2221_f64, 2222_f64, 2223_f64], // [1][1][1][...]
                    [2231_f64, 2232_f64, 2233_f64], // [1][1][2][...]
                ],
                // [1][2]
                [
                    [2311_f64, 2312_f64, 2313_f64], // [1][2][0][...]
                    [2321_f64, 2322_f64, 2323_f64], // [1][2][1][...]
                    [2331_f64, 2332_f64, 2333_f64], // [1][2][2][...]
                ],
            ],
            // [3]
            [
                // [3][0]
                [
                    [3111_f64, 3112_f64, 3113_f64], // [2][0][0][...]
                    [3121_f64, 3122_f64, 3123_f64], // [2][0][1][...]
                    [3131_f64, 3132_f64, 3133_f64], // [2][0][2][...]
                ],
                // [3][1]
                [
                    [3211_f64, 3212_f64, 3213_f64], // [2][1][0][...]
                    [3221_f64, 3222_f64, 3223_f64], // [2][1][1][...]
                    [3231_f64, 3232_f64, 3233_f64], // [2][1][2][...]
                ],
                // [3][2]
                [
                    [3311_f64, 3312_f64, 3313_f64], // [2][2][0][...]
                    [3321_f64, 3322_f64, 3323_f64], // [2][2][1][...]
                    [3331_f64, 3332_f64, 3333_f64], // [2][2][2][...]
                ],
            ],
        ];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        let val = (i + 1) * 1000 + (j + 1) * 100 + (k + 1) * 10 + (l + 1);
                        assert_eq!(comps_std[i][j][k][l], val as f64);
                    }
                }
            }
        }
        let t4 = Tensor4::from_tensor(&comps_std, false);
        println!("{:?}", t4.comps_mandel);
    }
}
