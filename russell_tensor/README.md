# Russell Tensor - Tensor analysis structures and functions for continuum mechanics

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

## Contents

* [Introduction](#introduction)
* [Installation on Debian/Ubuntu/Linux](#installation)
* [Setting Cargo.toml](#cargo)
* [Examples](#examples)

## <a name="introduction"></a> Introduction

This crate implements structures and functions to perform tensor analysis in continuum mechanics. We give focus to second and fourth order tensors expressed by their components placed in a vector or matrix. We also consider the Mandel basis.

Documentation:

- [russell_tensor documentation](https://docs.rs/russell_tensor)

## <a name="installation"></a> Installation on Debian/Ubuntu/Linux

This crate depends on `russell_lab`, which, in turn, depends on an efficient BLAS library such as [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) and [Intel MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/overview.html). Thus, we have two options:

1. Use the standard Debian packages based on OpenBLAS (default)
2. **(XOR)** Install Intel MKL, which includes LAPACK

Option 2 requires the following environment variable:

```bash
export RUSSELL_LAB_USE_INTEL_MKL=1
```

For convenience, you may use the scripts in the [zscripts](https://github.com/cpmech/russell/tree/main/russell_stat/zscripts) directory.

**1.** Use the standard Debian packages based on OpenBLAS:

```bash
bash zscripts/01-ubuntu-openblas.bash
```

**2.** Install Intel MKL:

```bash
bash zscripts/02-ubuntu-intel-mkl.bash
```

## <a name="cargo"></a> Setting Cargo.toml

[![Crates.io](https://img.shields.io/crates/v/russell_tensor.svg)](https://crates.io/crates/russell_tensor)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
russell_tensor = "*"
```

## <a name="examples"></a> Examples

* [russell_tensor/examples](https://github.com/cpmech/russell/tree/main/russell_tensor/examples)

### Allocating Second Order Tensors

```rust
use russell_tensor::{Mandel, StrError, Tensor2, SQRT_2};

fn main() -> Result<(), StrError> {
    // general
    let a = Tensor2::from_matrix(
        &[
            [1.0, SQRT_2 * 2.0, SQRT_2 * 3.0],
            [SQRT_2 * 4.0, 5.0, SQRT_2 * 6.0],
            [SQRT_2 * 7.0, SQRT_2 * 8.0, 9.0],
        ],
        Mandel::General,
    )?;
    assert_eq!(
        format!("{:.1}", a.vec),
        "┌      ┐\n\
         │  1.0 │\n\
         │  5.0 │\n\
         │  9.0 │\n\
         │  6.0 │\n\
         │ 14.0 │\n\
         │ 10.0 │\n\
         │ -2.0 │\n\
         │ -2.0 │\n\
         │ -4.0 │\n\
         └      ┘"
    );

    // symmetric-3D
    let b = Tensor2::from_matrix(
        &[
            [1.0, 4.0 / SQRT_2, 6.0 / SQRT_2],
            [4.0 / SQRT_2, 2.0, 5.0 / SQRT_2],
            [6.0 / SQRT_2, 5.0 / SQRT_2, 3.0],
        ],
        Mandel::Symmetric,
    )?;
    assert_eq!(
        format!("{:.1}", b.vec),
        "┌     ┐\n\
         │ 1.0 │\n\
         │ 2.0 │\n\
         │ 3.0 │\n\
         │ 4.0 │\n\
         │ 5.0 │\n\
         │ 6.0 │\n\
         └     ┘"
    );

    // symmetric-2D
    let c = Tensor2::from_matrix(
        &[[1.0, 4.0 / SQRT_2, 0.0], [4.0 / SQRT_2, 2.0, 0.0], [0.0, 0.0, 3.0]],
        Mandel::Symmetric2D,
    )?;
    assert_eq!(
        format!("{:.1}", c.vec),
        "┌     ┐\n\
         │ 1.0 │\n\
         │ 2.0 │\n\
         │ 3.0 │\n\
         │ 4.0 │\n\
         └     ┘"
    );
    Ok(())
}
```
