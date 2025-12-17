/// Specifies the (boundary) side of a rectangle
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Xmin = 0,
    Xmax = 1,
    Ymin = 2,
    Ymax = 3,
}
