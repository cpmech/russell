use serde::{Deserialize, Serialize};

#[allow(unused)]
use crate::{Tensor2, Tensor4}; // for documentation

/// Specifies the type of matrix representation (Kelvin notation) of Tensor2 and Tensor4
///
/// In the Kelvin basis, a second-order tensor is mapped to a column matrix
/// (vector) and a fourth-order tensor is mapped to a square matrix.
///
/// This enum specifies the following representations:
///
/// 1. General -- keeps all components of the tensor.
/// 2. Symmetric -- drops the last three rows of the vector and the last
///    three rows and columns of the matrix. Used for symmetric tensors in 3D,
///    or in 2D problems that still require the 6×1 or 6×6 representation
///    because nonzeros remain at the last rows/columns.
/// 3. Symmetric2D -- drops the last five rows of the vector and the last
///    five rows and columns of the matrix. Used for symmetric tensors in 2D.
///
/// **NOTE:** For Tensor4, "symmetric" means **minor-symmetric**
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum Rep {
    /// Representation for general operations (3D or 2D)
    ///
    /// * [Tensor2] becomes a 9×1 column matrix (vector), keeping all components.
    /// * [Tensor4] becomes a 9×9 square matrix, keeping all components.
    ///
    /// In Kelvin notation, a [Tensor2] is mapped as follows:
    ///
    /// ```text
    ///                       ┌                ┐
    ///                    00 │      T00       │ 0
    ///                    11 │      T11       │ 1
    /// ┌             ┐    22 │      T22       │ 2
    /// │ T00 T01 T02 │    01 │ (T01+T10) / √2 │ 3
    /// │ T10 T11 T12 │ => 12 │ (T12+T21) / √2 │ 4
    /// │ T20 T21 T22 │    02 │ (T02+T20) / √2 │ 5
    /// └             ┘    10 │ (T01-T10) / √2 │ 6
    ///                    21 │ (T12-T21) / √2 │ 7
    ///                    20 │ (T02-T20) / √2 │ 8
    ///                       └                ┘
    /// ```
    ///
    /// And a [Tensor4] is mapped as a 9×9 matrix; the rows and columns follow the
    /// same index order as the [Tensor2] vector above:
    ///
    /// ```text
    ///      0 0    0 1    0 2     0 3    0 4    0 5     0 6    0 7    0 8
    ///    ----------------------------------------------------------------
    /// 0 │ M0000  M0011  M0022   M0001  M0012  M0002   M0010  M0021  M0020
    /// 1 │ M1100  M1111  M1122   M1101  M1112  M1102   M1110  M1121  M1120
    /// 2 │ M2200  M2211  M2222   M2201  M2212  M2202   M2210  M2221  M2220
    ///   │
    /// 3 │ M0100  M0111  M0122   M0101  M0112  M0102   M0110  M0121  M0120
    /// 4 │ M1200  M1211  M1222   M1201  M1212  M1202   M1210  M1221  M1220
    /// 5 │ M0200  M0211  M0222   M0201  M0212  M0202   M0210  M0221  M0220
    ///   │
    /// 6 │ M1000  M1011  M1022   M1001  M1012  M1002   M1010  M1021  M1020
    /// 7 │ M2100  M2111  M2122   M2101  M2112  M2102   M2110  M2121  M2120
    /// 8 │ M2000  M2011  M2022   M2001  M2012  M2002   M2010  M2021  M2020
    ///    ----------------------------------------------------------------
    ///      8 0    8 1    8 2     8 3    8 4    8 5     8 6    8 7    8 8
    /// ```
    ///
    /// where `M` denotes the Kelvin-mapped component (see the [Tensor4] documentation).
    General,

    /// Representation for symmetric tensors (3D; or 2D when 6 components are required)
    ///
    /// * [Tensor2] becomes a 6×1 column matrix (vector), dropping the last three rows.
    /// * [Tensor4] becomes a 6×6 square matrix, dropping the last three rows and columns.
    ///
    /// In Kelvin notation, a [Tensor2] is mapped as follows:
    ///
    /// ```text
    ///                       ┌          ┐
    /// ┌             ┐    00 │   T00    │ 0
    /// │ T00 T01 T02 │    11 │   T11    │ 1
    /// │ T01 T11 T12 │ => 22 │   T22    │ 2
    /// │ T02 T12 T22 │    01 │ T01 * √2 │ 3
    /// └             ┘    12 │ T12 * √2 │ 4
    ///                    02 │ T02 * √2 │ 5
    ///                       └          ┘
    /// ```
    ///
    /// And a [Tensor4] is mapped as a 6×6 matrix:
    ///
    /// ```text
    ///      0 0       0 1       0 2        0 3       0 4       0 5
    ///    ------------------------------------------------------------
    /// 0 │ D0000     D0011     D0022      D0001*√2  D0012*√2  D0002*√2
    /// 1 │ D1100     D1111     D1122      D1101*√2  D1112*√2  D1102*√2
    /// 2 │ D2200     D2211     D2222      D2201*√2  D2212*√2  D2202*√2
    ///   │
    /// 3 │ D0100*√2  D0111*√2  D0122*√2   D0101*2   D0112*2   D0102*2
    /// 4 │ D1200*√2  D1211*√2  D1222*√2   D1201*2   D1212*2   D1202*2
    /// 5 │ D0200*√2  D0211*√2  D0222*√2   D0201*2   D0212*2   D0202*2
    ///    ------------------------------------------------------------
    ///      5 0       5 1       5 2        5 3       5 4       5 5
    /// ```
    ///
    /// **NOTE:** For [Tensor4], "symmetric" means **minor-symmetric**
    Symmetric,

    /// Representation for symmetric tensors (2D)
    ///
    /// * [Tensor2] becomes a 4×1 column matrix (vector), dropping the last five rows.
    /// * [Tensor4] becomes a 4×4 square matrix, dropping the last five rows and columns.
    ///
    /// In Kelvin notation, a [Tensor2] is mapped as follows:
    ///
    /// ```text
    /// ┌             ┐       ┌          ┐
    /// │ T00 T01     │    00 │   T00    │ 0
    /// │ T01 T11     │ => 11 │   T11    │ 1
    /// │         T22 │    22 │   T22    │ 2
    /// └             ┘    01 │ T01 * √2 │ 3
    ///                       └          ┘
    /// ```
    ///
    /// And a [Tensor4] is mapped as a 4×4 matrix:
    ///
    /// ```text
    ///      0 0       0 1       0 2        0 3
    ///    ----------------------------------------
    /// 0 │ D0000     D0011     D0022      D0001*√2
    /// 1 │ D1100     D1111     D1122      D1101*√2
    /// 2 │ D2200     D2211     D2222      D2201*√2
    ///   │
    /// 3 │ D0100*√2  D0111*√2  D0122*√2   D0101*2
    ///    ----------------------------------------
    ///      3 0       3 1       3 2        3 3
    /// ```
    ///
    /// **NOTE:** For [Tensor4], "symmetric" means **minor-symmetric**
    Symmetric2D,
}

impl Rep {
    /// Returns a new representation given the vector size (4, 6, 9)
    pub fn new(vector_dim: usize) -> Self {
        match vector_dim {
            4 => Rep::Symmetric2D,
            6 => Rep::Symmetric,
            _ => Rep::General,
        }
    }

    /// Returns the dimension of the Kelvin vector
    pub fn dim(&self) -> usize {
        match self {
            Rep::General => 9,
            Rep::Symmetric => 6,
            Rep::Symmetric2D => 4,
        }
    }

    /// Returns whether the space dimension is 2D or not
    ///
    /// Note: only Symmetric2D yields "true".
    pub fn two_dim(&self) -> bool {
        match self {
            Rep::General => false,
            Rep::Symmetric => false,
            Rep::Symmetric2D => true,
        }
    }

    /// Returns whether the Kelvin vector or matrix corresponds a symmetric tensor or not
    pub fn symmetric(&self) -> bool {
        if *self == Rep::General { false } else { true }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Rep;

    #[test]
    fn derive_works() {
        let rep = Rep::General.clone();
        assert_eq!(rep, Rep::General);
        assert_eq!(format!("{:?}", rep), "General");
        assert_eq!(rep, Rep::General);
    }

    #[test]
    fn new_works() {
        assert_eq!(Rep::new(4), Rep::Symmetric2D);
        assert_eq!(Rep::new(6), Rep::Symmetric);
        assert_eq!(Rep::new(9), Rep::General);
        assert_eq!(Rep::new(123), Rep::General);
    }

    #[test]
    fn member_functions_work() {
        // dim
        assert_eq!(Rep::General.dim(), 9);
        assert_eq!(Rep::Symmetric.dim(), 6);
        assert_eq!(Rep::Symmetric2D.dim(), 4);
        // two_dim
        assert_eq!(Rep::General.two_dim(), false);
        assert_eq!(Rep::Symmetric.two_dim(), false);
        assert_eq!(Rep::Symmetric2D.two_dim(), true);
        // symmetric
        assert_eq!(Rep::General.symmetric(), false);
        assert_eq!(Rep::Symmetric.symmetric(), true);
        assert_eq!(Rep::Symmetric2D.symmetric(), true);
    }
}
