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

| group                   | function                                                                            | pivoting      | implementation                                                          |
| ----------------------- | ----------------------------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------- |
| `mat_inverse`           | [`mat_inverse`](https://docs.rs/russell_lab/latest/russell_lab/fn.mat_inverse.html) | LU            | LAPACK `dgetrf`/`dgetri` (closed-form for n ≤ 3)                         |
| `small_mat_inv_partial` | `small_mat_inv(..., false)`                                                         | partial (row) | closed-form for n ≤ 3; Gauss-Jordan (pure Rust) for n ≥ 4                |
| `small_mat_inv_full`    | `small_mat_inv(..., true)`                                                          | full          | closed-form for n ≤ 3; Numerical Recipes `gaussj` (compiled C) for n ≥ 4 |

Note: `small_mat_inv` uses the closed-form formulas for n ≤ 3 regardless of `full_pivot`, so the two `small_mat_inv` rows coincide at n = 3.

All three groups invert the same, well-conditioned, diagonally dominant matrix

```text
aᵢⱼ = n + 1   (diagonal)
aᵢⱼ = 1       (off-diagonal)
```

which is guaranteed to be non-singular.

Results (median time, single machine):

| size | `mat_inverse` | `small_mat_inv_partial` | `small_mat_inv_full` |
| ---- | ------------- | ----------------------- | -------------------- |
| 3×3  | 6.57 ns       | 3.66 ns                 | 3.65 ns              |
| 4×4  | 42.9 ns       | 44.8 ns                 | 97.9 ns              |
| 5×5  | 71.6 ns       | 66.8 ns                 | 154 ns               |
| 6×6  | 102 ns        | 90.2 ns                 | 226 ns               |
| 7×7  | 190 ns        | 144 ns                  | 337 ns               |
| 8×8  | 243 ns        | 193 ns                  | 467 ns               |
| 9×9  | 310 ns        | 225 ns                  | 644 ns               |

For n ≤ 3, `small_mat_inv` uses the same closed-form formulas as `mat_inverse` but operates on stack arrays, so it is faster (~3.7 ns vs ~6.6 ns). For n ≥ 4, the partial-pivoting Rust implementation is on par with LAPACK at n = 4 and faster for n ≥ 5. The full-pivoting C code is the slowest for n ≥ 4 — it pays for `malloc`/`free` and the column-unscrambling bookkeeping on every call.

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
