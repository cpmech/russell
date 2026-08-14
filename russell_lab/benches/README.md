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

| group                    | function                                                                            | pivoting      | implementation                                            |
| ------------------------ | ----------------------------------------------------------------------------------- | ------------- | --------------------------------------------------------- |
| `mat_inverse`            | [`mat_inverse`](https://docs.rs/russell_lab/latest/russell_lab/fn.mat_inverse.html) | LU            | LAPACK `dgetrf`/`dgetri` (closed-form for n ≤ 3)          |
| `small_mat_inv`          | `small_mat_inv(..., n)`                                                             | partial (row) | closed-form for n ≤ 3; Gauss-Jordan (pure Rust) for n ≥ 4 |
| `num_recipes_gaussj_inv` | `num_recipes_gaussj_inv(...)`                                                       | full          | Numerical Recipes `gaussj` (compiled C)                   |

All three groups invert the same, well-conditioned, diagonally dominant matrix

```text
aᵢⱼ = n + 1   (diagonal)
aᵢⱼ = 1       (off-diagonal)
```

which is guaranteed to be non-singular.

Results (median time, single machine):

| size | `mat_inverse` | `small_mat_inv` | `num_recipes_gaussj_inv` |
| ---- | ------------- | --------------- | ------------------------ |
| 3×3  | 6.57 ns       | 1.10 ns         | 55.6 ns                  |
| 4×4  | 43.5 ns       | 46.0 ns         | 98.5 ns                  |
| 5×5  | 70.3 ns       | 71.0 ns         | 155 ns                   |
| 6×6  | 103 ns        | 96.1 ns         | 238 ns                   |
| 7×7  | 189 ns        | 127 ns          | 335 ns                   |
| 8×8  | 244 ns        | 164 ns          | 489 ns                   |
| 9×9  | 311 ns        | 222 ns          | 645 ns                   |

For n ≤ 3, `small_mat_inv` uses the same closed-form formulas as `mat_inverse` but operates on stack arrays, so it is much faster (~1.1 ns vs ~6.6 ns). For n ≥ 4, the partial-pivoting Rust implementation is on par with LAPACK at n = 4–5 and faster for n ≥ 6. The full-pivoting C code (`num_recipes_gaussj_inv`) is always the slowest — it pays for `malloc`/`free` and the column-unscrambling bookkeeping on every call.

small_mat_inv (partial) is now faster than before at n ≥ 6 — removing the full_pivot branch (which had an opaque extern "C" call) let the compiler optimize the pure-Rust path better.

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
