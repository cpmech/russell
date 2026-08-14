# Russell Lab — Benchmarks <!-- omit from toc -->

This directory contains the [Criterion](https://github.com/bheisler/criterion.rs) benchmarks for `russell_lab`.

## Contents <!-- omit from toc -->

- [Benchmarks](#benchmarks)
  - [Chebyshev polynomial evaluation](#chebyshev-polynomial-evaluation)
  - [Matrix and vector operations](#matrix-and-vector-operations)
  - [Small matrix inversion](#small-matrix-inversion)
- [How to run](#how-to-run)

## Benchmarks

### Chebyshev polynomial evaluation

`algo_chebyshev.rs` — compares `InterpChebyshev::eval` (Clenshaw algorithm) against `InterpChebyshev::eval_using_trig` (trigonometric functions).

### Matrix and vector operations

`matvec_benchmark.rs` — benchmarks `vec_add`, `mat_eigen_sym` (LAPACK DSYEV) and `mat_eigen_sym_jacobi` (Jacobi rotation).

### Small matrix inversion

`small_mat_inv_benchmark.rs` — compares three approaches to inverting an (n×n) matrix, for n = 3 … 9:

| group                   | function                                                                            | pivoting      | implementation                                         |
| ----------------------- | ----------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------ |
| `mat_inverse`           | [`mat_inverse`](https://docs.rs/russell_lab/latest/russell_lab/fn.mat_inverse.html) | LU            | LAPACK `dgetrf`/`dgetri` (analytic formulas for n ≤ 3) |
| `small_mat_inv_partial` | `small_mat_inv(..., false)`                                                         | partial (row) | pure Rust (Gauss-Jordan)                               |
| `small_mat_inv_full`    | `small_mat_inv(..., true)`                                                          | full          | compiled C (Numerical Recipes `gaussj`)                |

All three groups invert the same, well-conditioned, diagonally dominant matrix

```text
aᵢⱼ = n + 1   (diagonal)
aᵢⱼ = 1       (off-diagonal)
```

which is guaranteed to be non-singular.

Indicative results (single machine, spot-check):

| size | `mat_inverse` | `small_mat_inv_partial` | `small_mat_inv_full` |
| ---- | ------------- | ----------------------- | -------------------- |
| 3×3  | 6.6 ns        | 30 ns                   | 55 ns                |
| 9×9  | 309 ns        | 213 ns                  | 645 ns               |

The partial-pivoting Rust implementation is faster than LAPACK at 9×9, while the full-pivoting C code is the slowest (it pays for `malloc`/`free` and the column-unscrambling bookkeeping on every call).

## How to run

Install [cargo-criterion](https://github.com/bheisler/criterion.rs):

```bash
cargo install cargo-criterion
```

Run a single benchmark (from the `russell_lab` directory):

```bash
cargo bench --bench small_mat_inv_benchmark --all-features
```

Run only a given size (e.g., 3×3) using a Criterion filter:

```bash
cargo bench --bench small_mat_inv_benchmark --all-features -- '3'
```

> **Note:** `--all-features` selects Intel MKL (when available) instead of OpenBLAS.
