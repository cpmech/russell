# Russell - Rust Scientific Library

[![codecov](https://codecov.io/gh/cpmech/russell/branch/main/graph/badge.svg?token=PQWSKMZQXT)](https://codecov.io/gh/cpmech/russell)

Work in progress...

**Russell** assists in the development of scientific computations using the Rust language. We mainly consider the development of numerical methods and solvers for differential equations.

Available crates:

- [chk](https://github.com/cpmech/russell/tree/main/russell_chk) implements macros to assist in tests (numerical checks)
- [lab](https://github.com/cpmech/russell/tree/main/russell_lab) is a "laboratory" for vectors and matrices
- [stat](https://github.com/cpmech/russell/tree/main/russell_stat) contains structures and functions to work with statistics and probability distributions

## Installation

Add this to your Cargo.toml:

```toml
[dependencies]
russell_lab = "0.1"
russell_stat = "0.1"
```
