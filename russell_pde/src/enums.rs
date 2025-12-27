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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Side;

    #[test]
    fn test_side_from_index() {
        assert_eq!(Side::from_index(0), Side::Xmin);
        assert_eq!(Side::from_index(1), Side::Xmax);
        assert_eq!(Side::from_index(2), Side::Ymin);
        assert_eq!(Side::from_index(3), Side::Ymax);
    }

    #[test]
    #[should_panic(expected = "Side::from_index(): invalid index 123")]
    fn test_side_from_index_error() {
        Side::from_index(123);
    }

    #[test]
    fn test_side_debug() {
        assert_eq!(format!("{:?}", Side::Xmin), "Xmin");
        assert_eq!(format!("{:?}", Side::Xmax), "Xmax");
        assert_eq!(format!("{:?}", Side::Ymin), "Ymin");
        assert_eq!(format!("{:?}", Side::Ymax), "Ymax");
    }

    #[test]
    fn test_side_clone_copy() {
        let side = Side::Xmin;
        let copy = side;
        assert_eq!(side, copy);
    }
}
