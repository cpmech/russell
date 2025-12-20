/// Specifies the (boundary) side of a rectangle
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Xmin = 0,
    Xmax = 1,
    Ymin = 2,
    Ymax = 3,
}

impl Side {
    /// Creates a `Side` from its index
    ///
    /// # Panics
    ///
    /// A panic occurs if the index is not in {0, 1, 2, 3}.
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Side::Xmin,
            1 => Side::Xmax,
            2 => Side::Ymin,
            3 => Side::Ymax,
            _ => panic!("Side::from_index(): invalid index {}", index),
        }
    }
}
