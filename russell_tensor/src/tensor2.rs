#[derive(Debug)]
pub struct Tensor2 {
    components_mandel: Vec<f64>, // components in Mandel basis
    size: usize,                 // length of components_mandel: 9 or 6 (symmetric)
    symmetric: bool,             // represents a symmetric tensor
}

impl Tensor2 {
    pub fn new(symmetric: bool) -> Self {
        let size = if symmetric { 6 } else { 9 };
        Tensor2 {
            components_mandel: vec![0.0; size],
            size,
            symmetric,
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn new_tensor2_works() {
        let t2 = Tensor2::new(false);
        let correct = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(t2.size, 9);
        assert_vec_approx_eq!(t2.components_mandel, correct, 1e-15);
    }

    #[test]
    fn new_symmetric_tensor2_works() {
        let t2 = Tensor2::new(true);
        let correct = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(t2.size, 6);
        assert_vec_approx_eq!(t2.components_mandel, correct, 1e-15);
    }
}
