use crate::Mandel;

pub struct SampleTensor2 {
    pub desc: &'static str,
    pub case: Mandel,
    pub matrix: [[f64; 3]; 3],
    pub eigenvalues: [f64; 3],
    pub eigenprojectors: [[[f64; 3]; 3]; 3],
}

pub struct SamplesTensor2 {}

impl SamplesTensor2 {
    pub const SAMPLE1: SampleTensor2 = SampleTensor2 {
        desc: "Symmetric tensor in 3D",
        case: Mandel::Symmetric,
        matrix: [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
        eigenvalues: [0.170915188827179, -0.515729471589257, 11.3448142827621],
        eigenprojectors: [
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
        ],
    };

    pub const SAMPLE2: SampleTensor2 = SampleTensor2 {
        desc: "Symmetric tensor in 2D",
        case: Mandel::Symmetric2D,
        matrix: [[1.0, 2.0, 0.0], [2.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        eigenvalues: [-0.23606797749978803, 4.23606797749979, 4.0],
        eigenprojectors: [
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
        ],
    };

    pub const SAMPLE3: SampleTensor2 = SampleTensor2 {
        desc: "Tensor with all zero components in 3D",
        case: Mandel::Symmetric,
        matrix: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        eigenvalues: [0.0, 0.0, 0.0],
        eigenprojectors: [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
    };

    pub const SAMPLE4: SampleTensor2 = SampleTensor2 {
        desc: "Diagonal tensor in 3D",
        case: Mandel::Symmetric,
        matrix: [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        eigenvalues: [2.0, 3.0, 4.0],
        eigenprojectors: [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
    };
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{SampleTensor2, SamplesTensor2};
    use russell_lab::{mat_approx_eq, Matrix};

    #[rustfmt::skip]
    fn check_spectral(sample: &SampleTensor2, tolerance: f64) {
        let mut m = Matrix::new(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                m.set(
                    i,
                    j,
                    sample.eigenvalues[0] * sample.eigenprojectors[0][i][j]
                  + sample.eigenvalues[1] * sample.eigenprojectors[1][i][j]
                  + sample.eigenvalues[2] * sample.eigenprojectors[2][i][j],
                );
            }
        }
        mat_approx_eq(&m, &sample.matrix, tolerance);
    }

    #[test]
    fn samples_are_ok() {
        check_spectral(&SamplesTensor2::SAMPLE1, 1e-13);
        check_spectral(&SamplesTensor2::SAMPLE2, 1e-14);
        check_spectral(&SamplesTensor2::SAMPLE3, 1e-15);
        check_spectral(&SamplesTensor2::SAMPLE4, 1e-15);
    }
}
