use crate::{SQRT_2_BY_3, SQRT_3};

/// Collects some values related to a sample Tensor2
pub struct SampleTensor2 {
    /// Sets the description
    pub desc: &'static str,

    /// Defines the matrix representation (standard components, not Mandel)
    pub matrix: [[f64; 3]; 3],

    /// Defines the matrix representation of the deviator tensor
    pub deviator: [[f64; 3]; 3],

    /// Holds the Frobenius norm of the corresponding matrix representation
    pub norm: f64,

    /// Holds the trace (equals the first invariant Iᴛ)
    pub trace: f64,

    /// Holds the second principal invariant IIᴛ
    pub second_invariant: f64,

    /// Holds the determinant of the corresponding matrix representation (equals the third invariant IIIᴛ)
    pub determinant: f64,

    /// Holds the Frobenius norm of deviator tensor
    pub deviator_norm: f64,

    /// Holds the second principal invariant IIᴛ of the deviator tensor
    pub deviator_second_invariant: f64,

    /// Holds the determinant of the deviator tensor
    pub deviator_determinant: f64,

    /// Collects the eigenvalues if the tensor is symmetric
    pub eigenvalues: Option<[f64; 3]>,

    /// Collects the eigenprojectors if the tensor is symmetric
    pub eigenprojectors: Option<[[[f64; 3]; 3]; 3]>,
}

/// Holds some second-order tensor samples
pub struct SamplesTensor2 {}

impl SamplesTensor2 {
    /// Collects data for a symmetric tensor with all zero components (Tensor O)
    pub const TENSOR_O: SampleTensor2 = SampleTensor2 {
        desc: "Tensor O: symmetric tensor with all zero components",
        matrix: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        deviator: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        norm: 0.0,
        trace: 0.0,
        second_invariant: 0.0,
        determinant: 0.0,
        deviator_norm: 0.0,
        deviator_second_invariant: 0.0,
        deviator_determinant: 0.0,
        eigenvalues: Some([0.0, 0.0, 0.0]),
        eigenprojectors: Some([
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]),
    };

    /// Collects data for a symmetric diagonal tensor, the identity tensor (Tensor I)
    pub const TENSOR_I: SampleTensor2 = SampleTensor2 {
        desc: "Tensor I: symmetric diagonal tensor (identity tensor)",
        matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        deviator: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        norm: SQRT_3,
        trace: 3.0,
        second_invariant: 3.0,
        determinant: 1.0,
        deviator_norm: 0.0,
        deviator_second_invariant: 0.0,
        deviator_determinant: 0.0,
        eigenvalues: Some([1.0, 1.0, 1.0]),
        eigenprojectors: Some([
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]),
    };

    /// Collects data for a symmetric tensor in 2D (as in plane-stress analyses) (Tensor X)
    pub const TENSOR_X: SampleTensor2 = SampleTensor2 {
        desc: "Tensor X: symmetric 2D tensor with zero out-of-plane component (T22)",
        matrix: [[7.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
        deviator: [[10.0 / 3.0, 2.0, 0.0], [2.0, 1.0 / 3.0, 0.0], [0.0, 0.0, -11.0 / 3.0]],
        norm: 8.54400374531753, // f64::sqrt(73.0)
        trace: 11.0,
        second_invariant: 24.0,
        determinant: 0.0,
        deviator_norm: 7.0 * SQRT_2_BY_3,
        deviator_second_invariant: -49.0 / 3.0,
        deviator_determinant: 286.0 / 27.0,
        eigenvalues: Some([8.0, 3.0, 0.0]),
        eigenprojectors: Some([
            [
                [4.0 / 5.0, 2.0 / 5.0, 0.0],
                [2.0 / 5.0, 1.0 / 5.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [1.0 / 5.0, -2.0 / 5.0, 0.0],
                [-2.0 / 5.0, 4.0 / 5.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]),
    };

    /// Collects data for a symmetric tensor in 2D (as in plane-stress analyses) (Tensor Y)
    pub const TENSOR_Y: SampleTensor2 = SampleTensor2 {
        desc: "Tensor Y: symmetric 2D tensor with zero out-of-plane component (T22)",
        matrix: [[11.0, 3.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 9.0]],
        deviator: [[3.0, 3.0, 0.0], [3.0, -4.0, 0.0], [0.0, 0.0, 1.0]],
        norm: 15.3622914957372, // 2.0 * f64::sqrt(59.0)
        trace: 24.0,
        second_invariant: 170.0,
        determinant: 315.0,
        deviator_norm: 6.6332495807108, // 2.0 * f64::sqrt(11.0)
        deviator_second_invariant: -22.0,
        deviator_determinant: -21.0,
        eigenvalues: Some([12.1097722286464, 2.89022777135355, 9.0]),
        eigenprojectors: Some([
            [
                [0.8796283011826486, 0.32539568672798447, 0.0],
                [0.32539568672798447, 0.12037169881735181, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.12037169881735181, -0.3253956867279844, 0.0],
                [-0.3253956867279844, 0.8796283011826483, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]),
    };

    /// Collects data for a symmetric tensor in 2D (Tensor Z)
    pub const TENSOR_Z: SampleTensor2 = SampleTensor2 {
        desc: "Tensor Z: symmetric tensor in 2D",
        matrix: [[1.0, 2.0, 0.0], [2.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        deviator: [[-5.0 / 3.0, 2.0, 0.0], [2.0, 1.0 / 3.0, 0.0], [0.0, 0.0, 4.0 / 3.0]],
        norm: 5.8309518948453, // f64::sqrt(34.0)
        trace: 8.0,
        second_invariant: 15.0,
        determinant: -4.0,
        deviator_norm: 3.55902608401044, // f64::sqrt(38.0 / 3.0)
        deviator_second_invariant: -19.0 / 3.0,
        deviator_determinant: -164.0 / 27.0,
        eigenvalues: Some([-0.23606797749978803, 4.23606797749979, 4.0]),
        eigenprojectors: Some([
            [
                [0.723606797749979, -0.44721359549995776, 0.0],
                [-0.44721359549995776, 0.2763932022500208, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.2763932022500209, 0.4472135954999578, 0.0],
                [0.4472135954999578, 0.7236067977499788, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]),
    };

    /// Collects data for a symmetric tensor in 3D (Tensor U)
    pub const TENSOR_U: SampleTensor2 = SampleTensor2 {
        desc: "Tensor U: symmetric tensor in 3D",
        matrix: [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
        deviator: [[-8.0 / 3.0, 2.0, 3.0], [2.0, 1.0 / 3.0, 5.0], [3.0, 5.0, 7.0 / 3.0]],
        norm: 11.3578166916005, // f64::sqrt(129.0)
        trace: 11.0,
        second_invariant: -4.0,
        determinant: -1.0,
        deviator_norm: 9.41629792788369, // f64::sqrt(266.0 / 3.0)
        deviator_second_invariant: -133.0 / 3.0,
        deviator_determinant: 3031.0 / 27.0,
        eigenvalues: Some([0.170915188827179, -0.515729471589257, 11.3448142827621]),
        eigenprojectors: Some([
            [
                [0.34929169541608923, -0.4355596199317577, 0.19384226684174433],
                [-0.4355596199317577, 0.5431339622578344, -0.24171735309001413],
                [0.19384226684174433, -0.24171735309001413, 0.10757434232607645],
            ],
            [
                [0.5431339622578346, 0.24171735309001352, -0.435559619931758],
                [0.24171735309001352, 0.10757434232607586, -0.1938422668417439],
                [-0.435559619931758, -0.1938422668417439, 0.3492916954160896],
            ],
            [
                [0.10757434232607616, 0.19384226684174424, 0.24171735309001374],
                [0.19384226684174424, 0.3492916954160899, 0.43555961993175796],
                [0.24171735309001374, 0.43555961993175796, 0.5431339622578341],
            ],
        ]),
    };

    /// Collects data for a symmetric tensor in 3D (Tensor S)
    pub const TENSOR_S: SampleTensor2 = SampleTensor2 {
        desc: "Tensor S: symmetric tensor in 3D",
        matrix: [[5.0, 4.0, 3.0], [4.0, 6.0, 1.0], [3.0, 1.0, 1.0]],
        deviator: [[1.0, 4.0, 3.0], [4.0, 2.0, 1.0], [3.0, 1.0, -3.0]],
        norm: 10.6770782520313, // f64::sqrt(114.0)
        trace: 12.0,
        second_invariant: 15.0,
        determinant: -21.0,
        deviator_norm: 8.12403840463596, // f64::sqrt(66.0)
        deviator_second_invariant: -33.0,
        deviator_determinant: 47.0,
        eigenvalues: Some([2.46647252957463, 10.3557010334017, -0.822173562976294]),
        eigenprojectors: Some([
            [
                [0.238267467437297, -0.34172021416371, 0.254407894197923],
                [-0.34172021416371, 0.490090846325138, -0.364868611839051],
                [0.254407894197923, -0.364868611839051, 0.271641686237565],
            ],
            [
                [0.45076513819893, 0.458387397610942, 0.193537908676564],
                [0.458387397610942, 0.466138546401525, 0.196810557825709],
                [0.193537908676564, 0.196810557825709, 0.0830963153995457],
            ],
            [
                [0.310967394363773, -0.116667183447231, -0.447945802874487],
                [-0.116667183447231, 0.0437706072733379, 0.168058054013342],
                [-0.447945802874487, 0.168058054013342, 0.645261998362889],
            ],
        ]),
    };

    /// Collects data for a non-symmetric tensor in 3D (Tensor R)
    pub const TENSOR_R: SampleTensor2 = SampleTensor2 {
        desc: "Tensor R: non-symmetric tensor",
        matrix: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        deviator: [[-4.0, 2.0, 3.0], [4.0, 0.0, 6.0], [7.0, 8.0, 4.0]],
        norm: 16.8819430161341, // f64::sqrt(285.0)
        trace: 15.0,
        second_invariant: -18.0,
        determinant: 0.0,
        deviator_norm: 14.4913767461894, // f64::sqrt(210.0)
        deviator_second_invariant: -93.0,
        deviator_determinant: 340.0,
        eigenvalues: None,
        eigenprojectors: None,
    };

    /// Collects data for a non-symmetric tensor in 3D (Tensor T)
    pub const TENSOR_T: SampleTensor2 = SampleTensor2 {
        desc: "Tensor T: non-symmetric tensor",
        matrix: [[6.0, 1.0, 2.0], [3.0, 12.0, 4.0], [5.0, 6.0, 15.0]],
        deviator: [[-5.0, 1.0, 2.0], [3.0, 1.0, 4.0], [5.0, 6.0, 4.0]],
        norm: 22.2710574513201, // 4.0 * f64::sqrt(31.0)
        trace: 33.0,
        second_invariant: 305.0,
        determinant: 827.0,
        deviator_norm: 11.5325625946708, // f64::sqrt(133.0)
        deviator_second_invariant: -58.0,
        deviator_determinant: 134.0,
        eigenvalues: None,
        eigenprojectors: None,
    };
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{SampleTensor2, SamplesTensor2};
    use russell_lab::{mat_approx_eq, Matrix};

    fn check_spectral(sample: &SampleTensor2, tolerance: f64) {
        match sample.eigenvalues {
            Some(l) => match sample.eigenprojectors {
                Some(pps) => {
                    let mut m = Matrix::new(3, 3);
                    for i in 0..3 {
                        for j in 0..3 {
                            m.set(i, j, l[0] * pps[0][i][j] + l[1] * pps[1][i][j] + l[2] * pps[2][i][j]);
                        }
                    }
                    mat_approx_eq(&m, &sample.matrix, tolerance);
                }
                None => panic!("eigenprojectors are not available for this tensor"),
            },
            None => panic!("eigenvalues are not available for this tensor"),
        }
    }

    #[test]
    fn samples_are_ok() {
        check_spectral(&SamplesTensor2::TENSOR_O, 1e-15);
        check_spectral(&SamplesTensor2::TENSOR_I, 1e-15);
        check_spectral(&SamplesTensor2::TENSOR_U, 1e-13);
        check_spectral(&SamplesTensor2::TENSOR_S, 1e-13);
        check_spectral(&SamplesTensor2::TENSOR_X, 1e-15);
        check_spectral(&SamplesTensor2::TENSOR_Y, 1e-13);
        check_spectral(&SamplesTensor2::TENSOR_Z, 1e-14);
    }
}
