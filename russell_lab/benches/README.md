# Russell Lab — Benchmarks <!-- omit from toc -->

This directory contains the [Criterion](https://github.com/bheisler/criterion.rs) benchmarks for `russell_lab`.

## System information

The benchmark results in this directory were computed on the following machine
(output of the `get_system_info` example from `russell_sparse`, run with Intel MKL):

```text
--- OS ---
NAME="Arch Linux"
KERNEL=7.1.4-arch1-1

--- GPU ---
GPU[0]: NVIDIA GeForce RTX 4090, 595.71.05, 24564 MiB

--- CPU ---
Architecture       : x86_64
CPU(s)             : 32
On-line CPU(s) list: 0-31
Model name         : 13th Gen Intel(R) Core(TM) i9-13900KF
Thread(s) per core : 2
Core(s) per socket : 24
Socket(s)          : 1
CPU(s) scaling MHz : 43%
CPU max MHz        : 5800.0000
CPU min MHz        : 800.0000
BogoMIPS           : 5990.40
L1d cache          : 896 KiB (24 instances)
L1i cache          : 1.3 MiB (24 instances)
L2 cache           : 32 MiB (12 instances)
L3 cache           : 36 MiB (1 instance)
NUMA node0 CPU(s)  : 0-31
Vulnerability L1tf : Not affected

--- Memory ---
MemTotal:       32615544 kB
MemFree:          605312 kB
MemAvailable:   22415912 kB
SwapTotal:      36810532 kB
```

> **Note:** the `CPU(s) scaling MHz`, `MemFree`, and `MemAvailable` fields are
> instantaneous snapshots and vary between runs; the remaining fields are stable.

## Contents <!-- omit from toc -->

- [System information](#system-information)
- [Benchmarks](#benchmarks)
  - [Chebyshev polynomial evaluation](#chebyshev-polynomial-evaluation)
  - [Matrix and vector operations](#matrix-and-vector-operations)
  - [Small matrix inversion](#small-matrix-inversion)
  - [Small operations](#small-operations)
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
| `num_recipes_gaussj_inv` | `num_recipes_gaussj_inv(...)`                                                       | full          | Numerical Recipes `gaussj` (pure Rust)                    |

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

For n ≤ 3, `small_mat_inv` uses the same closed-form formulas as `mat_inverse` but operates on stack arrays, so it is much faster (~1.1 ns vs ~6.6 ns). For n ≥ 4, the partial-pivoting Rust implementation is on par with LAPACK at n = 4–5 and faster for n ≥ 6. The full-pivoting Rust code (`num_recipes_gaussj_inv`) is always the slowest — full pivoting pays for the pivot-search and column-unscrambling bookkeeping on every call.

small_mat_inv (partial) is now faster than before at n ≥ 6 — removing the full_pivot branch (which had an opaque extern "C" call) let the compiler optimize the pure-Rust path better.

### Small operations

`small_ops_benchmark.rs` — benchmarks the stack-allocated `small`-module functions against
their heap-allocated (BLAS/LAPACK) counterparts: `small_mat_add` vs `mat_add`,
`small_mat_update` vs `mat_update`, `small_mat_mat_mul` vs `mat_mat_mul`,
`small_vec_add` vs `vec_add`, `small_vec_update` vs `vec_update`, and
`small_solve_lin_sys` vs `solve_lin_sys`. Median times, single machine, Intel MKL backend.

#### Methodology

- Criterion settings: 20 samples per benchmark, 300 ms warm-up, 500 ms measurement.
- The small versions operate on `[[f64; N]; N]` / `[f64; N]` stack arrays (`N` known
  at compile time); the heap versions allocate `Matrix` / `Vector`.
- For **in-place** operations (`mat_update`, `vec_update`, `solve_lin_sys`), the
  mutable input is reset on every iteration: a heap `clone()` (allocation) for the
  heap version, vs a stack copy for the small version.
- All operations use a well-conditioned, diagonally dominant matrix
  `aᵢⱼ = 2` (diagonal) / `aᵢⱼ = 0.1` (off-diagonal).
- The `speedup` / `heap/small` column is `heap time ÷ small time`; values greater
  than 1 mean the small (stack) version is faster.

#### `mat_add` — `c := α⋅a + β⋅b`

| size | `mat_add` (heap) | `small_mat_add` | speedup |
| ---- | ---------------- | --------------- | ------- |
| 3×3  | 4.03 ns          | 1.28 ns         | 3.1×    |
| 4×4  | 5.06 ns          | 1.94 ns         | 2.6×    |
| 5×5  | 17.8 ns          | 5.73 ns         | 3.1×    |
| 6×6  | 16.7 ns          | 7.63 ns         | 2.2×    |
| 7×7  | 19.5 ns          | 7.92 ns         | 2.5×    |
| 8×8  | 21.0 ns          | 9.13 ns         | 2.3×    |
| 9×9  | 24.6 ns          | 9.98 ns         | 2.5×    |

#### `mat_update` — `b += α⋅a`

| size | `mat_update` (heap) | `small_mat_update` | speedup |
| ---- | ------------------- | ------------------ | ------- |
| 3×3  | 9.50 ns             | 0.538 ns           | 18×     |
| 4×4  | 9.39 ns             | 0.847 ns           | 11×     |
| 5×5  | 10.2 ns             | 1.28 ns            | 7.9×    |
| 6×6  | 11.5 ns             | 1.77 ns            | 6.5×    |
| 7×7  | 12.7 ns             | 2.38 ns            | 5.3×    |
| 8×8  | 13.8 ns             | 3.04 ns            | 4.5×    |
| 9×9  | 15.6 ns             | 3.84 ns            | 4.1×    |

#### `mat_mat_mul` — `c := α⋅a⋅b + β⋅c`

| size | `mat_mat_mul` (heap) | `small_mat_mat_mul` | speedup |
| ---- | -------------------- | ------------------- | ------- |
| 3×3  | 56.3 ns              | 3.49 ns             | 16×     |
| 4×4  | 57.8 ns              | 6.16 ns             | 9.4×    |
| 5×5  | 61.5 ns              | 12.9 ns             | 4.8×    |
| 6×6  | 46.0 ns              | 30.3 ns             | 1.5×    |
| 7×7  | 67.2 ns              | 33.5 ns             | 2.0×    |
| 8×8  | 69.2 ns              | 67.6 ns             | 1.0×    |
| 9×9  | 84.0 ns              | 68.0 ns             | 1.2×    |

#### `vec_add` — `w := α⋅u + β⋅v`

| size | `vec_add` (heap) | `small_vec_add` | speedup |
| ---- | ---------------- | --------------- | ------- |
| 4    | 1.43 ns          | 0.549 ns        | 2.6×    |
| 8    | 1.65 ns          | 0.955 ns        | 1.7×    |
| 16   | 4.75 ns          | 1.94 ns         | 2.4×    |
| 32   | 17.3 ns          | 7.05 ns         | 2.5×    |
| 64   | 22.2 ns          | 8.93 ns         | 2.5×    |
| 128  | 25.5 ns          | 14.3 ns         | 1.8×    |

#### `vec_update` — `v += α⋅u`

| size | `vec_update` (heap) | `small_vec_update` | speedup |
| ---- | ------------------- | ------------------ | ------- |
| 4    | 9.18 ns             | 0.251 ns           | 37×     |
| 8    | 9.02 ns             | 0.435 ns           | 21×     |
| 16   | 9.40 ns             | 0.800 ns           | 12×     |
| 32   | 10.4 ns             | 1.53 ns            | 6.8×    |
| 64   | 14.2 ns             | 3.00 ns            | 4.7×    |
| 128  | 20.8 ns             | 23.4 ns            | 0.9×    |

#### `solve_lin_sys` — `a⋅x = b`

| size | `solve_lin_sys` (heap) | `small_solve_lin_sys` | heap/small |
| ---- | ---------------------- | --------------------- | ---------- |
| 3×3  | 33.9 ns                | 29.2 ns               | 1.2×       |
| 4×4  | 49.7 ns                | 54.6 ns               | 0.91×      |
| 5×5  | 63.0 ns                | 77.7 ns               | 0.81×      |
| 6×6  | 93.8 ns                | 113 ns                | 0.83×      |
| 7×7  | 121 ns                 | 169 ns                | 0.72×      |
| 8×8  | 171 ns                 | 230 ns                | 0.74×      |
| 9×9  | 211 ns                 | 316 ns                | 0.67×      |

#### Observations

- **Arithmetic ops are much faster on the stack.** `small_mat_add`, `small_mat_mat_mul`,
  `small_vec_add` and `small_vec_update` avoid both the heap allocation and the BLAS
  dispatch overhead (`daxpy`/`dgemm` have a fixed function-call cost of ~5–60 ns), so
  they win decisively for small sizes. The gap closes only for larger matrices/vectors
  (e.g. `vec_update` at n = 128, `mat_mat_mul` at n ≥ 8).
- **In-place updates show the largest speedup.** `mat_update`/`vec_update` reset their
  mutable input each iteration; the heap version pays for a `clone()` allocation
  (~9 ns, nearly constant), which dominates the timing at small sizes.
- **The solver is the exception.** `solve_lin_sys` (LAPACK `dgesv`, LU) beats
  `small_solve_lin_sys` (Gauss-Jordan full pivoting) for n ≥ 4, because full-pivoting
  Gauss-Jordan does more work per pivot (bookkeeping, column unscrambling) than LU.
  The small version is only competitive at n = 3, where LAPACK's fixed overhead still
  dominates.

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
