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

Results (median time, single machine):

| size | `mat_inverse` | `small_mat_inv_partial` | `small_mat_inv_full` |
| ---- | ------------- | ----------------------- | -------------------- |
| 3×3  | 6.58 ns       | 30.0 ns                 | 55.6 ns              |
| 4×4  | 43.3 ns       | 46.9 ns                 | 98.3 ns              |
| 5×5  | 69.1 ns       | 68.1 ns                 | 155 ns               |
| 6×6  | 103 ns        | 120 ns                  | 235 ns               |
| 7×7  | 192 ns        | 128 ns                  | 333 ns               |
| 8×8  | 243 ns        | 196 ns                  | 512 ns               |
| 9×9  | 308 ns        | 218 ns                  | 648 ns               |

At the smallest sizes, the LAPACK-based `mat_inverse` is fastest (especially for n ≤ 3, where it uses closed-form formulas). Around n = 5–6 the two pivoting strategies are roughly on par, and for n ≥ 7 the partial-pivoting Rust implementation overtakes LAPACK. The full-pivoting C code is always the slowest — it pays for `malloc`/`free` and the column-unscrambling bookkeeping on every call.

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
