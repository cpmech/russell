/// Defines a stack-allocated square matrix with numeric components
///
/// This is a type alias to a fixed-size nested array `[[T; N]; N]`, which is
/// allocated on the stack and stored in **row-major** order. In contrast with
/// the heap-allocated [`crate::NumMatrix`], the components are accessed directly with
/// the `a[i][j]` notation and no memory is allocated on the heap.
///
/// The type parameter `T` must implement a set of numeric traits (e.g. the
/// `Num` trait from `num_traits`) for the associated operations to be available.
///
/// # Examples
///
/// ```
/// use russell_lab::SmallMatrix;
///
/// let mut a: SmallMatrix<f64, 3> = [[0.0; 3]; 3];
/// a[0][0] = 1.0;
/// a[1][1] = 2.0;
/// a[2][2] = 3.0;
/// assert_eq!(a[1][1], 2.0);
/// ```
pub type SmallMatrix<T, const N: usize> = [[T; N]; N];
