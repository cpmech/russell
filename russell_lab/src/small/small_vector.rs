/// Defines a stack-allocated vector with numeric components
///
/// This is a type alias to a fixed-size array `[T; N]`, which is allocated on
/// the stack. In contrast with the heap-allocated [`crate::NumVector`], the components
/// are accessed directly with the `v[i]` notation and no memory is allocated on
/// the heap.
///
/// The type parameter `T` must implement a set of numeric traits (e.g. the
/// `Num` trait from `num_traits`) for the associated operations to be available.
///
/// # Examples
///
/// ```
/// use russell_lab::SmallVector;
///
/// let mut v: SmallVector<f64, 3> = [0.0; 3];
/// v[0] = 1.0;
/// v[1] = 2.0;
/// assert_eq!(v[1], 2.0);
/// ```
pub type SmallVector<T, const N: usize> = [T; N];
