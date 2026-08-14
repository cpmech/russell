# The `small` module <!-- omit in toc -->

This module implements operations on small matrices and vectors that are
allocated on the stack, e.g. the `[[f64; 9]; 9]` matrix used internally by
`Tensor4`.

```rust
pub type SmallMatrix<T, const N: usize> = [[T; N]; N];
pub type SmallVector<T, const N: usize> = [T; N];
```

The primary goal of these stack-allocated containers is **maximum
performance**, even though `Tensor4` sacrifices memory by always carrying a
`9x9` matrix when a smaller one would suffice. This document records the
analysis behind the key design decision: whether to hardwire `f64` or keep the
element type generic via [`num_traits`](https://crates.io/crates/num-traits).

## Table of contents <!-- omit in toc -->

- [Design decision: generic `T` vs. hardwired `f64`](#design-decision-generic-t-vs-hardwired-f64)
- [Why generics are free at runtime](#why-generics-are-free-at-runtime)
  - [Monomorphization](#monomorphization)
  - [Static dispatch, not `dyn`](#static-dispatch-not-dyn)
  - [How `num_traits` maps to `f64`](#how-num_traits-maps-to-f64)
- [The real caveats](#the-real-caveats)
- [Warning: misuse in a hot loop](#warning-misuse-in-a-hot-loop)
- [Recommendation](#recommendation)
- [When hardwiring `f64` would still make sense](#when-hardwiring-f64-would-still-make-sense)
- [References](#references)

## Design decision: generic `T` vs. hardwired `f64`

**Conclusion:** a generic `SmallMatrix<T, const N>` bounded by `num_traits`
traits does **not** lose any runtime performance compared to hardwiring `f64`.
This is one of Rust's core "zero-cost abstraction" guarantees, and it is the
same design already used by the heap-allocated `NumMatrix<T>`
(`src/matrix/num_matrix.rs`).

The Rust Book is explicit (ch. 10.1, "Performance of Code Using Generics"):

> *"using generic types won't make your program run any slower than it would
> with concrete types."* … *"we pay no runtime cost for using generics."*

## Why generics are free at runtime

### Monomorphization

Generic code is *monomorphized* at compile time: the compiler emits a
specialized copy of the code for every concrete type it is instantiated with.
`SmallMatrix<f64, 9>` therefore compiles to **identical** machine code as a
hardwired `[[f64; 9]; 9]`. There is no type erasure, no boxing, and no
indirection.

### Static dispatch, not `dyn`

A `T: Num` bound is resolved at compile time. This is the crucial distinction
from a `dyn Trait` object, which the `std::dyn` documentation describes as
carrying *two* pointers (data + vtable) and whose methods "generally cannot be
inlined". With generic trait bounds there is **no vtable** and calls are
statically resolved — exactly like calling a concrete function.

### How `num_traits` maps to `f64`

The `num_traits` traits are thin wrappers over `std::ops`, so the
monomorphized code is the exact same instruction sequence:

| `num_traits` call                  | `f64` instantiation     | machine code            |
| ---------------------------------- | ----------------------- | ----------------------- |
| `Num::add(a, b)` / `a + b`         | `std::ops::Add for f64` | `addsd`                 |
| `Num::mul(a, b)` / `a * b`         | `std::ops::Mul for f64` | `mulsd`                 |
| `Zero::zero()` / `One::one()`      | `0.0` / `1.0`           | immediate constant      |
| `Signed::abs(x)` / `Float::abs(x)` | `f64::abs`              | `andpd` (mask sign bit) |
| `Float::sqrt(x)`                   | `f64::sqrt`             | `sqrtsd`                |
| `MulAdd::mul_add(a, b, c)`         | `f64::mul_add`          | fused `vfmadd`/FMA      |

`Num` extends `NumOps`, which extends
`std::ops::Add + Sub + Mul + Div + Rem + Neg`, so for `f64` the implementation
*is* the standard one — there is no layer in between.

## The real caveats

None of these are runtime dispatch; they are the things that *could* cost
performance if misused, and how this module avoids them.

1. **Cross-crate inlining.** The only thing that can separate generic code from
   peak performance is that a generic function in a *library* crate is not
   inlined into downstream crates unless marked `#[inline]`. This is a
   compile-time visibility issue, not a dispatch issue. All hot functions in
   this module use `#[inline]`, and they are tiny loops that inline trivially.

2. **`to_f64()` in hot loops.** If code uses `NumCast::to_f64()` (as the
   `array_approx_eq` check does), for `f64` it is the identity and the
   optimizer removes it entirely; for `f32`/`i32` it becomes a real `cvt`
   conversion. In this module such conversions live only in the
   *check/assertion* helpers, never in the matmul/inverse hot path.

3. **Compile time / binary size.** Each concrete instantiation (`f64`, `f32`,
   `Complex64`, …) duplicates the code. This is a build artifact, not a runtime
   cost, and with an f64-dominant library it is negligible.

4. **Vectorization is unaffected.** Because the type is concrete after
   monomorphization, LLVM auto-vectorizes and fuses (FMA) exactly as it would
   for hand-written `f64`. Generics do not hide `f64` from the optimizer; the
   SIMD/bounds-check hints in `small_mat_inv` remain fully effective.

## Warning: misuse in a hot loop

Both of the following foot-guns only exist because the element type is a type
parameter — hardwiring `f64` makes them impossible. They are the "remote risk"
referred to above.

**`to_f64()` in a hot loop.** A contributor may "cheat" to avoid adding a
`Signed`/`Float` bound, e.g. inside the elimination loop of `small_mat_inv`:

```rust
// BAD: arithmetic silently promoted to f64
target_a[j] -= T::from(
    factor.to_f64().unwrap() * source_a[j].to_f64().unwrap(),
).unwrap();
```

For `T = f64` this is free (identity), but for `T = f32` every element does two
conversions (`cvtss2sd`/`cvtsd2ss`), the loop stops vectorizing at f32 width,
and the numerics change. The fix keeps the arithmetic on `T`:

```rust
target_a[j] = target_a[j] - factor * source_a[j];   // T: Num + Copy (+ Signed)
```

**`dyn` in a hot loop.** A contributor may box the scalar to avoid threading
`<T>` through every helper:

```rust
trait Scalar { fn mul(self, rhs: f64) -> f64; }   // invented, object-safe
fn update(b: &mut [[f64; 9]; 9], alpha: &dyn Scalar, a: &[[f64; 9]; 9], n: usize) {
    for i in 0..n {
        for j in 0..n {
            b[i][j] += alpha.mul(a[i][j]);   // BAD: indirect call per element
        }
    }
}
```

`alpha.mul(..)` goes through a **vtable** every iteration — an indirect call the
compiler cannot inline or vectorize, typically an order of magnitude slower than
the monomorphized `mulsd`/`addsd`. The fix keeps the scalar generic (`T: Num +
Copy`) and relies on static dispatch.

## Recommendation

Keep `T` generic and mirror the bound already used by `NumMatrix`
(`src/matrix/num_matrix.rs`):

```rust
T: AddAssign + MulAssign + Num + NumCast + Copy
```

- `Num` already implies `Zero + One + PartialEq + Add + Sub + Mul + Div + Rem
  + Neg`, so `zero()`, `one()`, `+`, `*`, and `/` all work out of the box.
- Add **`Signed`** when a function needs `.abs()` (e.g. `small_mat_inv` for its
  singularity check), or **`Float`** when it also needs `.sqrt()`/`.powf()`.
- Keep `#[inline]` on every hot function.
- Never use `dyn` in the arithmetic path.

## When hardwiring `f64` would still make sense

Not for speed — for **API ergonomics**. `Tensor4` is currently hardwired to
`[[f64; 9]; 9]`, and a generic `small` module means generic parameters (`T`,
`N`) leak into every public signature (`small_mat_add::<T, const N>`, …). If
the module is never instantiated with anything but `f64`, hardwiring simplifies
the public API and removes the (remote) risk of a future contributor placing a
`to_f64()` or a `dyn` in a hot loop. On raw performance, the two approaches are
indistinguishable.

## References

- [The Rust Programming Language, ch. 10.1 — Performance of Code Using Generics](https://doc.rust-lang.org/book/ch10-01-syntax.html#performance-of-code-using-generics)
- [std::keyword::dyn — dynamic dispatch and the vtable trade-off](https://doc.rust-lang.org/std/keyword.dyn.html)
- [num-traits crate — `Num`, `Float`, `Signed`, `MulAdd`, `NumCast`](https://docs.rs/num-traits/latest/num_traits/)
- [num-traits `Num` trait (extends `std::ops` for zero-cost arithmetic)](https://docs.rs/num-traits/latest/num_traits/trait.Num.html)
