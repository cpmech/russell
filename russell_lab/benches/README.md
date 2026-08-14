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
`small_solve_lin_sys` vs `solve_lin_sys`. See [RESULTS.md](RESULTS.md).

Headline results (median, MKL):

- **Arithmetic ops are much faster on the stack** — `small_mat_add` ~2–3×, `small_mat_mat_mul`
  ~16× at 3×3 (narrowing to ~1× at 8×8/9×9), `small_vec_add` ~2×. The heap versions pay
  BLAS dispatch overhead (`daxpy`/`dgemm`) that swamps tiny workloads.
- **In-place updates show the largest gap** — `small_mat_update` ~18× at 3×3, `small_vec_update`
  ~37× at n = 4, because the heap version re-`clone()`s (heap alloc) each iteration while the
  stack version just copies. They converge only near n = 128.
- **The solver is the exception** — `solve_lin_sys` (LAPACK `dgesv`/LU) beats
  `small_solve_lin_sys` (full-pivoting Gauss-Jordan) for n ≥ 4 (~1.5× at 9×9), since full
  pivoting does more per-pivot bookkeeping than LU. The small version only wins at n = 3,
  where LAPACK's fixed overhead dominates.

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
