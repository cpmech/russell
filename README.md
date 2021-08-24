# Russell - Rust Scientific Library

[![codecov](https://codecov.io/gh/cpmech/russell/branch/main/graph/badge.svg?token=PQWSKMZQXT)](https://codecov.io/gh/cpmech/russell)

Work in progress...

![Bertrand Russell](zassets/Bertrand_Russell_1957.jpg)

(Figure: [Bertrand Russell](https://en.wikipedia.org/wiki/Bertrand_Russell) [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en))

**Russell** assists in the development of scientific computations using the Rust language. We mainly consider the development of numerical methods and solvers for differential equations.

Available crates:

- [chk](https://github.com/cpmech/russell/tree/main/russell_chk) Functions to check vectors and other data in tests
- [lab](https://github.com/cpmech/russell/tree/main/russell_lab) Matrix-vector laboratory including linear algebra tools
- [openblas](https://github.com/cpmech/russell/tree/main/russell_openblas) Thin wrapper to some OpenBLAS routines
- [stat](https://github.com/cpmech/russell/tree/main/russell_stat) Statistics calculations, probability distributions, and pseudo random numbers
- [tensor](https://github.com/cpmech/russell/tree/main/russell_tensor) Tensor analysis structures and functions for continuum mechanics

## Installation

Install OpenBLAS:

```bash
sudo apt-get install libopenblas-dev
```

Add this to your Cargo.toml (select only the crates you want):

```toml
[dependencies]
russell_chk = "*"
russell_lab = "*"
russell_openblas = "*"
russell_stat = "*"
russell_tensor = "*"
```
