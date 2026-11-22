# Russell Sparse — Sparse matrices and linear solvers <!-- omit from toc -->

<p align="left">
  <a href="https://docs.rs/russell_sparse/"><img alt="documentation" src="https://docs.rs/russell_sparse/badge.svg"></a>
  <a href="https://crates.io/crates/russell_sparse"><img alt="crates.io" src="https://img.shields.io/crates/v/russell_sparse.svg"></a>
  <a href="https://github.com/cpmech/matrix-market"><img alt="matrix assets" src="https://img.shields.io/badge/assets-cpmech%2Fmatrix--market-1f425f"></a>
  <a href="https://doi.org/10.1002/nme.7545"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1002%2Fnme.7545-blue"></a>
</p>

`russell_sparse` provides sparse matrix storage (COO, CSC, CSR), six solver backends,
Matrix Market I/O, and the benchmark tools used to compare them. It is part of the
[Russell - Rust Scientific Library](https://github.com/cpmech/russell).

> **Benchmark assets:** the 44 Matrix Market matrices used by the tools in
> [`src/bin`](https://github.com/cpmech/russell/tree/main/russell_sparse/src/bin)
> are hosted separately in [cpmech/matrix-market](https://github.com/cpmech/matrix-market).
> That repository contains seven compressed archives, SHA-256 checksums, and
> extraction instructions.

## Contents <!-- omit from toc -->

- [References](#references)
- [Introduction](#introduction)
  - [Solver backends](#solver-backends)
  - [Complex Numbers Support](#complex-numbers-support)
  - [Documentation](#documentation)
- [Installation](#installation)
  - [Setting Cargo.toml](#setting-cargotoml)
  - [Optional features](#optional-features)
- [Performance](#performance)
- [🌟 Examples](#-examples)
  - [Solve a tiny sparse linear system using the Faer solver](#solve-a-tiny-sparse-linear-system-using-the-faer-solver)
  - [Solve a complex sparse linear system using the Faer solver](#solve-a-complex-sparse-linear-system-using-the-faer-solver)
  - [Iterative solvers](#iterative-solvers)
- [Tools](#tools)
  - [Matrix Market assets](#matrix-market-assets)
  - [solve_matrix_market](#solve_matrix_market)
  - [xtract_matrix_market](#xtract_matrix_market)
  - [Benchmark runner](#benchmark-runner)
- [For developers](#for-developers)

## References

1. Pedroso DM (2024) [Caveats of three direct linear solvers for finite element analyses](https://onlinelibrary.wiley.com/doi/10.1002/nme.7545), _International Journal for Numerical Methods in Engineering_, doi.org/10.1002/nme.7545.

## Introduction

This library implements tools for handling sparse matrices and functions to solve sparse linear systems with six backends:

### Solver backends

| Backend | Kind | Availability | Notes |
|---|---|---|---|
| **Faer** | Sparse LU | Pure Rust, always available | The default backend; `FaerMethod` selects simplicial or supernodal |
| **BasicLU** | Sparse LU with rank-1 updates | Pure Rust, always available | Square general systems; written for LP bases |
| **QDLDL** | `L D Lᵀ` factorization | Pure Rust, always available | Symmetric quasi-definite systems; upper triangle |
| **Iterative** | Krylov + AMG | Pure Rust, always available | CG, BiCGStab, and restarted GMRES |
| [**UMFPACK**](https://github.com/DrTimothyAldenDavis/SuiteSparse) | Unsymmetric multifrontal LU | `umfpack` feature | Recommended for general sequential use |
| [**MUMPS**](https://mumps-solver.org) | Multifrontal LU | `mumps` feature | Very large systems; supports multithreaded execution |

> **The native iterative solvers are limited.** With the default algebraic-multigrid settings they
> reach a relative error of 1e-6 on a minority of the real SuiteSparse matrices measured in
> [benchmark-russell](https://github.com/cpmech/benchmark-russell): BiCGStab on 12 of 33, GMRES on 7,
> and CG on 1, since CG requires a symmetric positive definite matrix. The setup and the kernels are
> tested — what is limited is how often the preconditioner makes the iteration converge. For
> production work use **Faer**, **UMFPACK**, or **MUMPS**, and use the iterative path where it has
> been measured to converge on your own matrix.

Performance benchmarks and the comparison harness are available in
[benchmark-russell](https://github.com/cpmech/benchmark-russell).

### Complex Numbers Support

This crate fully supports both real (`f64`) and complex (`Complex64`) numbers. You can use the complex variants of the data structures and solvers, such as `ComplexCooMatrix` and `ComplexSolverUMFPACK`, seamlessly.

The crate also ships pure-Rust iterative solvers: preconditioned conjugate gradients (CG), BiCGStab, and restarted GMRES, with either algebraic multigrid (AMG) or identity preconditioning. They need no external library and are available in every build, but their convergence is matrix-dependent -- see the limitation above.

This library implements three storage formats for sparse matrices:

* COO: COOrdinates matrix, also known as a sparse triplet.
* CSC: Compressed Sparse Column matrix
* CSR: Compressed Sparse Row matrix

Additionally, to unify the handling of the above sparse matrix data structures, this library implements:

* SparseMatrix: Either a COO, CSC, or CSR matrix

The COO matrix is the best when we need to update the values of the matrix because it has easy access to the triples (i, j, aij). For instance, the repetitive access is the primary use case for codes based on the finite element method (FEM) for approximating partial differential equations. Moreover, the COO matrix allows storing duplicate entries; for example, the triple `(0, 0, 123.0)` can be stored as two triples `(0, 0, 100.0)` and `(0, 0, 23.0)`. Again, this is the primary need for FEM codes because of the so-called assembly process where elements add to the same positions in the "global stiffness" matrix. Nonetheless, the duplicate entries must be summed up at some stage for the linear solver (e.g., MUMPS, UMFPACK). These linear solvers also use the more memory-efficient storage formats CSC and CSR. See the [russell_sparse documentation](https://docs.rs/russell_sparse) for further information.

This library also provides functions to read and write Matrix Market files containing (huge) sparse matrices that can be used in performance benchmarking or other studies. The [read_matrix_market()] function reads a Matrix Market file and returns a [CooMatrix]. To write a Matrix Market file, we can use the function [write_matrix_market()], which takes a [SparseMatrix] and, thus, automatically converts COO to CSC or COO to CSR, also performing the sum of duplicates. The `write_matrix_market` also writes an SMAT file (almost like the Matrix Market format) without the header and with zero-based indices. The SMAT file can be given to the fantastic [Vismatrix](https://github.com/cpmech/vismatrix) tool to visualize the sparse matrix structure and values interactively; see the example below.

![readme-vismatrix](https://raw.githubusercontent.com/cpmech/russell/main/russell_sparse/data/figures/readme-vismatrix.png)

### Documentation

* [![documentation](https://docs.rs/russell_sparse/badge.svg)](https://docs.rs/russell_sparse/) — [russell_sparse documentation](https://docs.rs/russell_sparse/)
* See also a [paper concerning some caveats with the sparse solvers (freely available)](https://onlinelibrary.wiley.com/doi/10.1002/nme.7545).


## Installation

The default build needs no external library: the sparse solver is pure Rust (the Faer backend). The optional `umfpack` and `mumps` features link external solvers, and `open_blas`/`intel_mkl` accelerate the dense kernels; [the main README file](https://github.com/cpmech/russell) has the installation steps for each platform.

### Setting Cargo.toml

[![Crates.io](https://img.shields.io/crates/v/russell_sparse.svg)](https://crates.io/crates/russell_sparse)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
russell_sparse = "*"
```

### Optional features

The following optional (Rust) features are available:

* `open_blas`: Use OpenBLAS for the dense kernels (needs its development headers)
* `intel_mkl`: Use Intel oneAPI MKL for the dense kernels; takes precedence over `open_blas` when both are enabled
* `umfpack`: Add UMFPACK (SuiteSparse) as a sparse direct solver; a local install under `RUSSELL_LOCAL_PREFIX` takes precedence over the package-manager library
* `mumps`: Add MUMPS as a sparse direct solver; requires a locally compiled MUMPS under `RUSSELL_LOCAL_PREFIX` (Linux, macOS, and Windows with the GNU toolchain)

UMFPACK is optional and off by default. With `umfpack`, the build first checks for a local install at `$RUSSELL_LOCAL_PREFIX/include/umfpack` and `$RUSSELL_LOCAL_PREFIX/lib/umfpack`. If that installation is not there — or `RUSSELL_LOCAL_PREFIX` is unset — it falls back to the system library found with `pkg-config` (or the usual include/library directories), and fails as usual if neither is available. The local installation therefore takes precedence, and a build links only one of the two sources. The default build uses the Faer solver; enable `umfpack` to add UMFPACK. On Windows the sources differ by toolchain: the GNU toolchain uses the MSYS2 package (through `MSYS2_PREFIX`), and the MSVC toolchain uses a vcpkg install (`VCPKG_ROOT` or `VCPKG_INSTALLATION_ROOT`). MUMPS needs the GNU toolchain there: its Windows build is MinGW objects plus gfortran, which MSVC cannot link.

MUMPS is independent of UMFPACK and always comes from `RUSSELL_LOCAL_PREFIX`, which must then be set and exported. The layout is fixed:

```text
$RUSSELL_LOCAL_PREFIX/include/umfpack     $RUSSELL_LOCAL_PREFIX/lib/umfpack
$RUSSELL_LOCAL_PREFIX/include/mumps       $RUSSELL_LOCAL_PREFIX/lib/mumps
```

A prefix that is set but does not contain UMFPACK produces a build warning and the system fallback; a missing MUMPS install fails the build naming the path it expected. The directory of the selected UMFPACK must also be reachable by the dynamic loader at run time, since the two SuiteSparse builds share the `libsolver.so.6` soname.

The [main README file](https://github.com/cpmech/russell) has the installation steps for each platform.



### WebAssembly

With the default features (Faer solver, no external libraries) the crate builds for `wasm32-unknown-unknown` and `wasm32-wasip1`. On the browser target the Matrix Market and JSON file helpers compile but return errors, since there is no filesystem; WASI provides one when the runtime preopens a directory. A smoke test preprocesses and solves a small system on the browser target, where there is neither a clock nor a filesystem, so the Faer path is exercised there rather than only compiled.

## Performance

The `Faer` solver (`Genie::Faer`) is the pure-Rust sparse solver, using `FaerMethod::Simplicial` by default. Note that `algebra_linear`'s supernodal path (`FaerMethod::Supernodal`) omits contribution blocks between supernodes and instead updates each front directly from the stored `L` and `U` columns, so large fronts are slower than simplicial. The table below compares the default simplicial path with UMFPACK on the SuiteSparse benchmark matrices that completed; the data is from [benchmark-russell](https://github.com/cpmech/benchmark-russell).

| Matrix | NROW | NNZ | Faer | UMFPACK | UMFPACK / Faer |
| --- | ---: | ---: | ---: | ---: | ---: |
| bwm2000 | 2,000 | 7,996 | 1.08 ms | 0.750 ms | 1.44× faster |
| mhd1280b (c) | 1,280 | 22,778 | 0.740 ms | 0.383 ms | 1.93× faster |
| rdb5000 | 5,000 | 29,600 | 35.2 ms | 13.6 ms | 2.59× faster |
| mplate (c) | 5,962 | 142,190 | 3.12 s | 627 ms | 4.97× faster |
| Goodwin_040 | 17,922 | 561,677 | 508 ms | 107 ms | 4.75× faster |
| fp | 7,548 | 848,553 | 942 ms | 154 ms | 6.10× faster |
| xenon1 | 48,600 | 1,181,120 | 55.0 s | 598 ms | 92× faster |

`(c)` marks a complex-valued matrix; rows are sorted by NNZ. **NNZ is the matrix-file non-zero count.** Times are outlier-free averages on the benchmark machine (Arch Linux, Intel Core i9-13900KF): 5 runs for UMFPACK, and for Faer 3 runs on the fast/medium matrices (`<= 30 s`) and 1 run on `xenon1`.

The Faer solver is fast on small systems: up to about 30,000 non-zeros it takes at most a few tens of milliseconds. It becomes very slow for matrices with more than **1 million** non-zeros: the `xenon1` case (NROW=48,600, 1,181,120 non-zeros) takes **55.0 s**, and matrices above 1.2 million pattern entries (starting with `twotone`, NROW=120,750, 1,224,224 non-zeros) are skipped by default in the benchmark harness. For matrices with more than **1 million** non-zeros, expect **nearly a minute** or more.

The threshold is not a function of NNZ alone. The complex `mplate` (NROW=5,962, 142,190 non-zeros) takes **3.12 s**, and `RFdevice` (NROW=74,104, 365,580 non-zeros) did not finish within the **3-minute** probe; sparsity structure and numerical type matter as much as the non-zero count.

UMFPACK is faster than the Faer solver on every matrix in the table; the gap grows with the fill-in, reaching **92×** on `xenon1`. The figure below shows the same ratios on a log scale (the dashed line is 1×):

![UMFPACK speed-up over the Faer solver](https://raw.githubusercontent.com/cpmech/russell/main/russell_sparse/data/figures/plot_speedup_umfpack.png)

## 🌟 Examples

This section illustrates how to use `russell_sparse`. See also:

* [More examples in the documentation](https://docs.rs/russell_sparse/)
* [Examples directory](https://github.com/cpmech/russell/tree/main/russell_sparse/examples)



### Solve a tiny sparse linear system using the Faer solver

```rust
use russell_lab::{vec_approx_eq, Vector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // constants
    let ndim = 3; // number of rows = number of columns
    let nnz = 5; // number of non-zero values

    // allocate solver
    let mut solver = SolverFaer::new()?;

    // allocate the coefficient matrix
    let mut coo = CooMatrix::new(ndim, ndim, nnz, Sym::No)?;
    coo.put(0, 0, 0.2)?;
    coo.put(0, 1, 0.2)?;
    coo.put(1, 0, 0.5)?;
    coo.put(1, 1, -0.25)?;
    coo.put(2, 2, 0.25)?;

    // print matrix
    let a = coo.as_dense();
    let correct = "┌                   ┐\n\
                   │   0.2   0.2     0 │\n\
                   │   0.5 -0.25     0 │\n\
                   │     0     0  0.25 │\n\
                   └                   ┘";
    assert_eq!(format!("{}", a), correct);

    // call preprocess
    solver.preprocess(&coo, None)?;

    // allocate two right-hand side vectors
    let b = Vector::from(&[1.0, 1.0, 1.0]);

    // calculate the solution
    let mut x = Vector::new(ndim);
    solver.solve(&mut x, &b, false)?;
    let correct = vec![3.0, 2.0, 4.0];
    vec_approx_eq(&x, &correct, 1e-14);
    Ok(())
}
```

### Solve a complex sparse linear system using the Faer solver

```rust
use russell_lab::{complex_vec_approx_eq, cpx, ComplexVector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    let ndim = 3;
    let nnz = 7;

    // allocate solver
    let mut solver = ComplexSolverFaer::new()?;

    // allocate the coefficient matrix
    //   ┌                      ┐
    //   │  2+1i  -1-1i     0   │
    //   │ -1-1i   2+2i  -1+1i  │
    //   │   0    -1+1i   2-1i  │
    //   └                      ┘
    let mut coo = ComplexCooMatrix::new(ndim, ndim, nnz, Sym::No)?;
    coo.put(0, 0, cpx!(2.0, 1.0))?;
    coo.put(0, 1, cpx!(-1.0, -1.0))?;
    coo.put(1, 0, cpx!(-1.0, -1.0))?;
    coo.put(1, 1, cpx!(2.0, 2.0))?;
    coo.put(1, 2, cpx!(-1.0, 1.0))?;
    coo.put(2, 1, cpx!(-1.0, 1.0))?;
    coo.put(2, 2, cpx!(2.0, -1.0))?;

    // call preprocess
    solver.preprocess(&coo, None)?;

    // right-hand side vector
    let b = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);

    // calculate the solution
    let mut x = ComplexVector::new(ndim);
    solver.solve(&mut x, &b, false)?;

    // check the result
    let correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];
    complex_vec_approx_eq(&x, correct, 1e-14);
    Ok(())
}
```

See [russell_sparse documentation](https://docs.rs/russell_sparse) for more examples.

See also the folder `examples`.

### Iterative solvers

Select an iterative method with `Genie::Iterative(IterativeMethod::{Cg, BiCgStab, Gmres})`. All three use algebraic multigrid (AMG) preconditioning by default, or the identity preconditioner. Their parameters live in `LinSolParams::iterative`: tolerance, absolute tolerance, maximum number of iterations, GMRES restart, number of threads, AMG coarsening, smoother, coarse-enough size, and maximum number of levels.

```rust
use russell_lab::{vec_approx_eq, Vector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // ┌             ┐
    // │  4 -1    0  │
    // │ -1  4   -1  │
    // │  0 -1    4  │
    // └             ┘
    let ndim = 3;
    let mut coo = CooMatrix::new(ndim, ndim, 7, Sym::YesFull)?;
    coo.put(0, 0, 4.0)?;
    coo.put(0, 1, -1.0)?;
    coo.put(1, 0, -1.0)?;
    coo.put(1, 1, 4.0)?;
    coo.put(1, 2, -1.0)?;
    coo.put(2, 1, -1.0)?;
    coo.put(2, 2, 4.0)?;

    // solve with conjugate gradients
    let mut solver = SolverIterative::new(IterativeMethod::Cg)?;
    solver.preprocess(&coo, None)?;
    let rhs = Vector::from(&[3.0, 2.0, 3.0]);
    let mut x = Vector::new(ndim);
    solver.solve(&mut x, &rhs, false)?;
    vec_approx_eq(&x, &[1.0, 1.0, 1.0], 1e-10);

    // print the iterative statistics
    let mut stats = StatsLinSol::new();
    solver.update_stats(&mut stats);
    let iterative = stats.iterative_stats.as_ref().expect("iterative statistics");
    println!("method = {}", iterative.method);
    println!("preconditioner = {}", iterative.preconditioner);
    println!("iterations = {:?}", iterative.iterations_array);
    println!("relative residual = {:?}", iterative.relative_residual_array);
    println!("AMG levels = {}", iterative.amg_levels);
    Ok(())
}
```

If the method does not converge within the iteration limit, `solve` returns `"the iterative solver did not converge"` after storing the report in the statistics. The identity preconditioner (`params.iterative.preconditioner = Preconditioner::Identity`) is the escape hatch for nonsymmetric systems when the AMG setup is unsuitable.



## Tools

The [`src/bin`](https://github.com/cpmech/russell/tree/main/russell_sparse/src/bin)
directory contains command-line tools for testing and comparing the sparse solvers:

| Tool | Purpose |
|---|---|
| `solve_matrix_market` | Solves one Matrix Market system with a selected backend and writes JSON statistics |
| `xtract_matrix_market` | Reads a directory of result JSON files and prints a solver comparison table |
| `run_solve_matrix_market.bash` | Runs the benchmark suite over the standard matrix set, one result per matrix and solver |

### Matrix Market assets

The 44 matrices used by these tools (34 real and 10 complex, 13.565 GB uncompressed)
are not stored in this repository. They live in
[**cpmech/matrix-market**](https://github.com/cpmech/matrix-market), split into seven
`tar.xz` archives with SHA-256 checksums and extraction instructions.

Download the archives from the latest release, extract them, and flatten the files
into one directory:

```bash
# Follow the instructions in cpmech/matrix-market first.
export INPUT_DIR="$HOME/Downloads/matrix-market"
```

The benchmark runner expects that single flat directory. You can also use a single
matrix directly with `solve_matrix_market`; the crate includes a small example at
`russell_sparse/data/matrix_market/bfwb62.mtx`.

### solve_matrix_market

`solve_matrix_market` reads a [Matrix Market file](https://math.nist.gov/MatrixMarket/formats.html)
and solves:

```text
A ⋅ x = b
```

with a right-hand side `b` containing only ones.

```bash
cargo run --release --bin solve_matrix_market -- ~/Downloads/matrix-market/bfwb62.mtx
cargo run --release --bin solve_matrix_market -- --help
```

The default backend is Faer. Use `--genie` (`-g`) to select another one:
`faer`, `cg`, `bicgstab`, `gmres`, `basiclu`, `qdldl`, `mumps`, or `umfpack`.

```bash
cargo run --release --bin solve_matrix_market -- -g mumps ~/Downloads/matrix-market/bfwb62.mtx
```

Useful options:

| Option | Meaning |
|---|---|
| `-g, --genie <name>` | Solver backend |
| `--faer-method` | Faer LU strategy: `auto`, `simplicial`, or `supernodal` |
| `--faer-flop-ratio <f>` | Threshold for `--faer-method auto` |
| `-r, --nrun <n>` | Number of runs; timing statistics are averaged |
| `--time-guard <seconds>` | Abort each run after this many seconds; every repetition gets its own budget. Default `200`; `0` disables it |
| `--information` | Read the matrix, print its metadata, and exit |
| `--hide-json` | Suppress the JSON output |
| `--vismatrix` | Also write a Vismatrix `.smat` file |

The iterative solvers are configured with the following options, which `faer`,
`basiclu`, `qdldl`, `mumps`, and `umfpack` ignore (`--help` prints the defaults):

| Option | Meaning |
|---|---|
| `--iter-tol` | Relative residual tolerance |
| `--iter-abs-tol` | Absolute residual tolerance |
| `--iter-max` | Maximum number of Krylov iterations |
| `--iter-precond` | Preconditioner: `amg` or `identity` |
| `--iter-restart` | GMRES restart dimension |
| `--iter-threads` | Threads for the Krylov and AMG kernels |
| `--iter-min-parallel-len` | Minimum vector length before threads are used |
| `--amg-coarsening` | `ruge-stuben`, `aggregation`, or `smoothed-aggregation` |
| `--amg-smoother` | `gauss-seidel`, `damped-jacobi`, `ilu0`, `chebyshev`, `spai0`, or `spai1` |
| `--amg-coarse-enough` | AMG size at which the coarsest level is solved directly |
| `--amg-max-levels` | Maximum number of AMG levels |
| `--amg-max-memory-mb` | Memory budget in MiB for AMG and Krylov (`0` = unlimited) |

<details>
<summary><b>Example JSON output (abridged)</b></summary>

```json
{
  "main": {
    "platform": "Russell",
    "blas_lib": "OpenBLAS",
    "solver": "MUMPS",
    "has_mumps": true,
    "has_umfpack": true,
    "solved": true,
    "stopped_by_time_guard": false,
    "out_of_memory": false,
    "error_from_solver": null
  },
  "matrix": {
    "name": "bfwb62",
    "nrow": 62,
    "ncol": 62,
    "nnz": 202,
    "nnz_actual": 342,
    "complex": false,
    "symmetric": "YesLower"
  },
  "time_human": {
    "read_matrix": "43.1µs",
    "initialize": "266.6µs",
    "preprocess": "196.8µs",
    "solve": "166.9µs",
    "total_ifs": "630.3µs",
    "verify": "2.2µs"
  }
}
```

</details>

### xtract_matrix_market

`xtract_matrix_market` reads the JSON files produced by `solve_matrix_market`
(recursively) and prints two comparison tables: one for real-valued matrices and
one for complex-valued matrices:

```bash
cargo run --release --bin xtract_matrix_market -- [results_dir]
```

With no argument it uses `$OUTPUT_DIR` when set, otherwise
`/tmp/russell_sparse/results/$SYSTEM` (`SYSTEM` defaults to `arch`).

- Every solver column shows its absolute total time (initialize + preprocess +
  solve); no MUMPS baseline is required.
- Failed runs appear as keys such as `stop-time`, `no-mem`, `only-sym`,
  `only-spd`, `no-conv`, `singular`, and `not-avail`; the key descriptions are
  printed after the tables.
- Blank cells mean no result file was found for that matrix/solver.

### sys_info

`sys_info` exports the OS, GPU, CPU, memory, and solver-library versions:

```bash
cargo run --release --bin sys_info
cargo run --release --bin sys_info -- system-information.txt
```

With no file argument it prints to stdout. The report uses the same features as
the build, so `--features umfpack,mumps` reports those libraries as compiled in.

### Benchmark runner

`run_solve_matrix_market.bash` runs the standard matrix set and stores one JSON file
per matrix and solver under `$OUTPUT_DIR/<solver>/`. Before running, it writes the
benchmark configuration and the `sys_info` report to
`$OUTPUT_DIR/system-information.txt` (override with `INFO_FILE`):

```bash
cd russell_sparse/src/bin
export INPUT_DIR="$HOME/Downloads/matrix-market"

# One solver:
./run_solve_matrix_market.bash -s -g mumps -f 10

# Every available genie, solving the systems:
./run_solve_matrix_market.bash -s -G -f 10

# Compare the results:
cargo run --release --bin xtract_matrix_market
```

| Option | Meaning |
|---|---|
| `-i, --information` | Only print matrix metadata (default) |
| `-s, --solve` | Solve the systems |
| `-g, --genie <name>` | Run one solver |
| `-G, --all-genies` | Run every available solver |
| `-f, --first <n>` | First `n` real matrices |
| `-F, --first-complex <n>` | First `n` complex matrices |
| `-1` … `-5` | Number of runs per matrix |

The default `OUTPUT_DIR` is `/tmp/russell_sparse/results/$SYSTEM`. Results are split
by solver, so a MUMPS run, an UMFPACK run, and a Faer run can coexist and then be
combined by `xtract_matrix_market`.

The iterative solvers (`cg`, `bicgstab`, and `gmres`) run only on the
real-valued matrices; the runner skips the complex matrices for them. `basiclu`
and `qdldl` also have no complex variant, so complex matrices are skipped for
them as well. `qdldl` additionally runs only on the known symmetric matrices.

## For developers

* The `c_code` directory contains a thin wrapper to the sparse solvers (MUMPS, UMFPACK)
* The `build.rs` file uses the crate `cc` to build the C-wrappers
* The `zscripts` directory also contains the following:
    * `memcheck.bash`: Checks for memory leaks on the C-code using Valgrind
    * `run-examples`: Runs all examples in the `examples` directory
    * `run-solve-matrix-market.bash`: Runs the solve-matrix-market tool from the `bin` directory
