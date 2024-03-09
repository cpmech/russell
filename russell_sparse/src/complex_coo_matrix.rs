use crate::StrError;
use crate::{ComplexCooMatrix, CooMatrix};
use num_complex::Complex64;
use russell_lab::cpx;

impl ComplexCooMatrix {
    /// Assigns this matrix to the values of another real matrix (scaled)
    ///
    /// Performs:
    ///
    /// ```text
    /// this = (α + βi) · other
    /// ```
    ///
    /// Thus:
    ///
    /// ```text
    /// this[p].real = α · other[p]
    /// this[p].imag = β · other[p]
    ///
    /// other[p] ∈ Reals
    /// p = [0, nnz(other)]
    /// ```
    ///
    /// **Warning:** make sure to allocate `max_nnz ≥ nnz(other)`.
    pub fn assign_real(&mut self, alpha: f64, beta: f64, other: &CooMatrix) -> Result<(), StrError> {
        if other.nrow != self.nrow {
            return Err("matrices must have the same nrow");
        }
        if other.ncol != self.ncol {
            return Err("matrices must have the same ncol");
        }
        if other.symmetry != self.symmetry {
            return Err("matrices must have the same symmetry");
        }
        self.reset();
        for p in 0..other.nnz {
            let i = other.indices_i[p] as usize;
            let j = other.indices_j[p] as usize;
            self.put(i, j, cpx!(alpha * other.values[p], beta * other.values[p]))?;
        }
        Ok(())
    }

    /// Augments this matrix with the entries of another real matrix (scaled)
    ///
    /// Effectively, performs:
    ///
    /// ```text
    /// this += (α + βi) · other
    /// ```
    ///
    /// Thus:
    ///
    /// ```text
    /// this[p].real += α · other[p]
    /// this[p].imag += β · other[p]
    ///
    /// other[p] ∈ Reals
    /// p = [0, nnz(other)]
    /// ```
    ///
    /// **Warning:** make sure to allocate `max_nnz ≥ nnz(this) + nnz(other)`.
    pub fn augment_real(&mut self, alpha: f64, beta: f64, other: &CooMatrix) -> Result<(), StrError> {
        if other.nrow != self.nrow {
            return Err("matrices must have the same nrow");
        }
        if other.ncol != self.ncol {
            return Err("matrices must have the same ncol");
        }
        if other.symmetry != self.symmetry {
            return Err("matrices must have the same symmetry");
        }
        for p in 0..other.nnz {
            let i = other.indices_i[p] as usize;
            let j = other.indices_j[p] as usize;
            self.put(i, j, cpx!(alpha * other.values[p], beta * other.values[p]))?;
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use crate::{ComplexCooMatrix, CooMatrix, Storage, Sym};
    use num_complex::Complex64;
    use russell_lab::cpx;

    #[test]
    fn assign_capture_errors() {
        let sym = Some(Sym::General(Storage::Full));
        let nnz_a = 1;
        let nnz_b = 2; // wrong: must be ≤ nnz_a
        let mut a_1x2 = ComplexCooMatrix::new(1, 2, nnz_a, None).unwrap();
        let b_2x1 = CooMatrix::new(2, 1, nnz_b, None).unwrap();
        let b_1x3 = CooMatrix::new(1, 3, nnz_b, None).unwrap();
        let b_1x2_sym = CooMatrix::new(1, 2, nnz_b, sym).unwrap();
        let mut b_1x2 = CooMatrix::new(1, 2, nnz_b, None).unwrap();
        a_1x2.put(0, 0, cpx!(123.0, 321.0)).unwrap();
        b_1x2.put(0, 0, 456.0).unwrap();
        b_1x2.put(0, 1, 654.0).unwrap();
        assert_eq!(
            a_1x2.assign_real(2.0, 3.0, &b_2x1).err(),
            Some("matrices must have the same nrow")
        );
        assert_eq!(
            a_1x2.assign_real(2.0, 3.0, &b_1x3).err(),
            Some("matrices must have the same ncol")
        );
        assert_eq!(
            a_1x2.assign_real(2.0, 3.0, &b_1x2_sym).err(),
            Some("matrices must have the same symmetry")
        );
        assert_eq!(
            a_1x2.assign_real(2.0, 3.0, &b_1x2).err(),
            Some("COO matrix: max number of items has been reached")
        );
    }

    #[test]
    fn assign_works() {
        let nnz = 2;
        let mut a = ComplexCooMatrix::new(3, 2, nnz, None).unwrap();
        let mut b = CooMatrix::new(3, 2, nnz, None).unwrap();
        a.put(2, 1, cpx!(1000.0, 2000.0)).unwrap();
        b.put(0, 0, 10.0).unwrap();
        b.put(2, 1, 20.0).unwrap();
        assert_eq!(
            format!("{}", a.as_dense()),
            "┌                       ┐\n\
             │       0+0i       0+0i │\n\
             │       0+0i       0+0i │\n\
             │       0+0i 1000+2000i │\n\
             └                       ┘"
        );
        a.assign_real(3.0, 2.0, &b).unwrap();
        assert_eq!(
            format!("{}", a.as_dense()),
            "┌               ┐\n\
             │ 30+20i   0+0i │\n\
             │   0+0i   0+0i │\n\
             │   0+0i 60+40i │\n\
             └               ┘"
        );
    }

    #[test]
    fn augment_capture_errors() {
        let sym = Some(Sym::General(Storage::Full));
        let nnz_a = 1;
        let nnz_b = 1;
        let mut a_1x2 = ComplexCooMatrix::new(1, 2, nnz_a /* + nnz_b */, None).unwrap();
        let b_2x1 = CooMatrix::new(2, 1, nnz_b, None).unwrap();
        let b_1x3 = CooMatrix::new(1, 3, nnz_b, None).unwrap();
        let b_1x2_sym = CooMatrix::new(1, 2, nnz_b, sym).unwrap();
        let mut b_1x2 = CooMatrix::new(1, 2, nnz_b, None).unwrap();
        a_1x2.put(0, 0, cpx!(123.0, 321.0)).unwrap();
        b_1x2.put(0, 0, 456.0).unwrap();
        assert_eq!(
            a_1x2.augment_real(2.0, 3.0, &b_2x1).err(),
            Some("matrices must have the same nrow")
        );
        assert_eq!(
            a_1x2.augment_real(2.0, 3.0, &b_1x3).err(),
            Some("matrices must have the same ncol")
        );
        assert_eq!(
            a_1x2.augment_real(2.0, 3.0, &b_1x2_sym).err(),
            Some("matrices must have the same symmetry")
        );
        assert_eq!(
            a_1x2.augment_real(2.0, 3.0, &b_1x2).err(),
            Some("COO matrix: max number of items has been reached")
        );
    }

    #[test]
    fn augment_works() {
        let nnz_a = 1;
        let nnz_b = 2;
        let mut a = ComplexCooMatrix::new(3, 2, nnz_a + nnz_b, None).unwrap();
        let mut b = CooMatrix::new(3, 2, nnz_b, None).unwrap();
        a.put(2, 1, cpx!(1000.0, 2000.0)).unwrap();
        b.put(0, 0, 10.0).unwrap();
        b.put(2, 1, 20.0).unwrap();
        assert_eq!(
            format!("{}", a.as_dense()),
            "┌                       ┐\n\
             │       0+0i       0+0i │\n\
             │       0+0i       0+0i │\n\
             │       0+0i 1000+2000i │\n\
             └                       ┘"
        );
        a.augment_real(3.0, 2.0, &b).unwrap();
        assert_eq!(
            format!("{}", a.as_dense()),
            "┌                       ┐\n\
             │     30+20i       0+0i │\n\
             │       0+0i       0+0i │\n\
             │       0+0i 1060+2040i │\n\
             └                       ┘"
        );
    }
}
