# Russell Chk - Functions to check vectors and other data in tests

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

This repository implements macros to assist in tests (numerical checks).

Documentation:

- [API reference (docs.rs)](https://docs.rs/russell_chk)

## Installation

[![Crates.io](https://img.shields.io/crates/v/russell_chk.svg)](https://crates.io/crates/russell_chk)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
russell_chk = "*"
```

## Examples

### Check float point numbers

```rust
use russell_chk::approx_eq;

fn main() {
    approx_eq(0.123456789, 0.12345678, 1e-8);
    approx_eq(0.123456789, 0.1234567, 1e-7);
    approx_eq(0.123456789, 0.123456, 1e-6);
    approx_eq(0.123456789, 0.12345, 1e-5);
    approx_eq(0.123456789, 0.1234, 1e-4);
}
```

### Check a vector of float point numbers

```rust
use russell_chk::vec_approx_eq;

fn main() {
    let a = [0.123456789, 0.123456789, 0.123456789];
    let b = [0.12345678,  0.1234567,   0.123456];
    vec_approx_eq(&a, &b, 1e-6);
}
```

### Check derivatives

```rust
use russell_chk::deriv_approx_eq;

struct Arguments {}

fn main() {
    let f = |x: f64, _: &mut Arguments| -x;
    let args = &mut Arguments {};
    let at_x = 8.0;
    let dfdx = -1.01;
    deriv_approx_eq(dfdx, at_x, f, args, 1e-2);
}
```

### Check complex numbers

```rust
use russell_chk::complex_vec_approx_eq;
use num_complex::Complex64;

fn main() {
    let a = &[
        Complex64::new(0.123456789, 5.01),
        Complex64::new(0.123456789, 5.01),
        Complex64::new(0.123456789, 5.01)];
    let b = &[
        Complex64::new(0.12345678, 5.01),
        Complex64::new(0.1234567, 5.01),
        Complex64::new(0.123456, 5.01)];
    complex_vec_approx_eq(a, b, 1e-6);
}
```
