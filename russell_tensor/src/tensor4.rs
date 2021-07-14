use super::*;

/// Implements a fourth order tensor
pub struct Tensor4 {
    comps_mandel: Vec<f64>, // components in Mandel basis. len = 81 or 36 (minor-symmetric)
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
}
