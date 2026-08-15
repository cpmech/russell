# Russell Tensor — Benchmarks

This directory contains [Criterion](https://github.com/bheisler/criterion.rs) benchmarks that
compare the **stack-allocated** (`russell_tensor`) and **heap-allocated**
(`russell_tensor_heap`) implementations of selected tensor functions.

The two crates expose the same function names with the same signatures, but differ in their
internal storage:

| crate                 | `Tensor2` storage | `Tensor4` storage         |
| --------------------- | ----------------- | ------------------------- |
| `russell_tensor`      | `vec: [f64; 9]`   | `mat: [[f64; 9]; 9]`      |
| `russell_tensor_heap` | `vec: Vector`     | `mat: Matrix` (col-major) |

## System information

| component | value                                              |
| --------- | -------------------------------------------------- |
| OS        | Arch Linux (kernel 7.1.4)                          |
| CPU       | 13th Gen Intel(R) Core(TM) i9-13900KF (32 threads) |
| GPU       | NVIDIA GeForce RTX 4090                            |
| Memory    | 32 GB                                              |
| BLAS      | Intel MKL (`--all-features`)                       |

## Benchmarked functions

Each function is benchmarked in two modes, controlled by the `use_loops` flag:

- `unrolled` — `use_loops = false` (the default, production path; direct component access)
- `loops` — `use_loops = true` (loop-based; uses the `get`/`set` accessors)

| function               | description                                       |
| ---------------------- | ------------------------------------------------- |
| `t2_ssd`               | self-sum-dyadic operation `D = s (A ⊗ A + A ⊗ A̅)` |
| `t2_qsd_t2`            | quartic-sum-dyadic operation                      |
| `deriv2_invariant_jj3` | second derivative of the J3 invariant             |

All benchmarks use a fixed symmetric 3×3 input tensor.

## Results

Median times (single machine, Intel MKL):

| function               | stack/unrolled | heap/unrolled | stack/loops | heap/loops |
| ---------------------- | -------------- | ------------- | ----------- | ---------- |
| `t2_ssd`               | 5.53 ns        | 10.17 ns      | 211.02 ns   | 211.12 ns  |
| `t2_qsd_t2`            | 7.25 ns        | 14.63 ns      | 436.57 ns   | 437.80 ns  |
| `deriv2_invariant_jj3` | 74.64 ns       | 99.18 ns      | 495.21 ns   | 527.16 ns  |

## Observations

- **Unrolled path:** the stack version is ~2× faster across the board. The heap version's
  `Matrix::set_unchecked` carries the column-major access overhead, whereas the stack version
  writes directly to `[[f64; 9]; 9]`.
- **Loops path:** stack and heap are essentially identical (~0.05–6% difference) — the loop
  overhead (iteration, `M_TO_IJ` lookups, `get_std`/`set` calls, and the conditional `√2`
  factors) dominates and masks the storage-layout difference.
- **Unrolled vs loops:** the unrolled path is ~40–60× faster for `t2_ssd`/`t2_qsd_t2` and ~7×
  faster for `deriv2_invariant_jj3` (which spends proportionally more time in the
  `deviator`/matrix-multiplication steps).

## How to run

Run the benchmark (from the workspace root):

```bash
cargo bench -p russell_tensor --all-features --bench tensor_benchmark
```

Filter to a single function, e.g. `t2_ssd`:

```bash
cargo bench -p russell_tensor --all-features --bench tensor_benchmark -- t2_ssd
```

> **Note:** `--all-features` selects Intel MKL (when available) instead of OpenBLAS.
