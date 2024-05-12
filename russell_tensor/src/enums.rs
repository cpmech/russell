use serde::{Deserialize, Serialize};

/// Specifies the Mandel representation
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum Mandel {
    /// General representation of a 3×3 Tensor2 as a 9D vector
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
    General,

    /// Mandel representation of a symmetric 3×3 Tensor2 as a 6D vector
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
    /// **NOTE:** For Tensor4, "symmetric" means **minor-symmetric**
    Symmetric,

    /// Mandel representation of a symmetric 3×3 Tensor2 as a 4D vector (useful in 2D simulations)
    ///
    /// ```text
    /// ┌             ┐       ┌          ┐
    /// │ T00 T01     │    00 │   T00    │ 0
    /// │ T01 T11     │ => 11 │   T11    │ 1
    /// │         T22 │    22 │   T22    │ 2
    /// └             ┘    01 │ T01 * √2 │ 3
    ///                       └          ┘
    ///
    /// **NOTE:** For Tensor4, "symmetric" means **minor-symmetric**
    Symmetric2D,
}

impl Mandel {
    /// Returns a new Mandel enum given the vector size (4, 6, 9)
    pub fn new(vector_dim: usize) -> Self {
        match vector_dim {
            4 => Mandel::Symmetric2D,
            6 => Mandel::Symmetric,
            _ => Mandel::General,
        }
    }

    /// Returns the dimension of the Mandel vector
    pub fn dim(&self) -> usize {
        match self {
            Mandel::General => 9,
            Mandel::Symmetric => 6,
            Mandel::Symmetric2D => 4,
        }
    }

    /// Returns whether the space dimension is 2D or not
    ///
    /// Note: only Symmetric2D yields "true".
    pub fn two_dim(&self) -> bool {
        match self {
            Mandel::General => false,
            Mandel::Symmetric => false,
            Mandel::Symmetric2D => true,
        }
    }

    /// Returns whether the Mandel vector or matrix corresponds a symmetric tensor or not
    pub fn symmetric(&self) -> bool {
        if *self == Mandel::General {
            false
        } else {
            true
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Mandel;

    #[test]
    fn derive_works() {
        let mandel = Mandel::General.clone();
        assert_eq!(mandel, Mandel::General);
        assert_eq!(format!("{:?}", mandel), "General");
        assert_eq!(mandel, Mandel::General);
    }

    #[test]
    fn new_works() {
        assert_eq!(Mandel::new(4), Mandel::Symmetric2D);
        assert_eq!(Mandel::new(6), Mandel::Symmetric);
        assert_eq!(Mandel::new(9), Mandel::General);
        assert_eq!(Mandel::new(123), Mandel::General);
    }

    #[test]
    fn member_functions_work() {
        // dim
        assert_eq!(Mandel::General.dim(), 9);
        assert_eq!(Mandel::Symmetric.dim(), 6);
        assert_eq!(Mandel::Symmetric2D.dim(), 4);
        // two_dim
        assert_eq!(Mandel::General.two_dim(), false);
        assert_eq!(Mandel::Symmetric.two_dim(), false);
        assert_eq!(Mandel::Symmetric2D.two_dim(), true);
        // symmetric
        assert_eq!(Mandel::General.symmetric(), false);
        assert_eq!(Mandel::Symmetric.symmetric(), true);
        assert_eq!(Mandel::Symmetric2D.symmetric(), true);
    }
}
