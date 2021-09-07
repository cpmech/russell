# Russell - Rust Scientific Library

[![codecov](https://codecov.io/gh/cpmech/russell/branch/main/graph/badge.svg?token=PQWSKMZQXT)](https://codecov.io/gh/cpmech/russell)

Work in progress...

![Bertrand Russell](zassets/Bertrand_Russell_1957.jpg)

([CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en). Photo: [Bertrand Russell](https://en.wikipedia.org/wiki/Bertrand_Russell))

**Russell** assists in the development of scientific computations using the Rust language. We mainly consider the development of numerical methods and solvers for differential equations.

Available crates:

- [chk](https://github.com/cpmech/russell/tree/main/russell_chk) Functions to check vectors and other data in tests
- [lab](https://github.com/cpmech/russell/tree/main/russell_lab) Matrix-vector laboratory including linear algebra tools
- [openblas](https://github.com/cpmech/russell/tree/main/russell_openblas) Thin wrapper to some OpenBLAS routines
- [sparse](https://github.com/cpmech/russell/tree/main/russell_sparse) Sparse matrix tools and solvers
- [stat](https://github.com/cpmech/russell/tree/main/russell_stat) Statistics calculations, probability distributions, and pseudo random numbers
- [tensor](https://github.com/cpmech/russell/tree/main/russell_tensor) Tensor analysis structures and functions for continuum mechanics

## Installation

Install the following Debian packages:

```bash
sudo apt-get install \
    liblapacke-dev \
    libmumps-seq-dev \
    libopenblas-dev \
    libsuitesparse-dev
```

Add this to your Cargo.toml (select only the crates you want and replace the right version):

```toml
[dependencies]
russell_chk = "*"
russell_lab = "*"
russell_openblas = "*"
russell_sparse = "*"
russell_stat = "*"
russell_tensor = "*"
```

## Examples

### Solving a sparse linear system

```rust
use russell_lab::*;
use russell_sparse::*;

fn main() -> Result<(), &'static str> {

    // allocate a square matrix
    let mut trip = SparseTriplet::new(5, 5, 13, false)?;
    trip.put(0, 0, 1.0); // << (0, 0, a00/2)
    trip.put(0, 0, 1.0); // << (0, 0, a00/2)
    trip.put(1, 0, 3.0);
    trip.put(0, 1, 3.0);
    trip.put(2, 1, -1.0);
    trip.put(4, 1, 4.0);
    trip.put(1, 2, 4.0);
    trip.put(2, 2, -3.0);
    trip.put(3, 2, 1.0);
    trip.put(4, 2, 2.0);
    trip.put(2, 3, 2.0);
    trip.put(1, 4, 6.0);
    trip.put(4, 4, 1.0);

    // print matrix
    let (m, n) = trip.dims();
    let mut a = Matrix::new(m, n);
    trip.to_matrix(&mut a)?;
    let correct = "┌                ┐\n\
                   │  2  3  0  0  0 │\n\
                   │  3  0  4  0  6 │\n\
                   │  0 -1 -3  2  0 │\n\
                   │  0  0  1  0  0 │\n\
                   │  0  4  2  0  1 │\n\
                   └                ┘";
    assert_eq!(format!("{}", a), correct);

    // allocate x and rhs
    let mut x = Vector::new(5);
    let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);

    // initialize, factorize, and solve
    let config = ConfigSolver::new();
    let mut solver = Solver::new(config)?;
    solver.initialize(&trip, false)?;
    solver.factorize(false)?;
    solver.solve(&mut x, &rhs, false)?;
    let correct = "┌          ┐\n\
                   │ 1.000000 │\n\
                   │ 2.000000 │\n\
                   │ 3.000000 │\n\
                   │ 4.000000 │\n\
                   │ 5.000000 │\n\
                   └          ┘";
    assert_eq!(format!("{:.6}", x), correct);
    Ok(())
}
```
