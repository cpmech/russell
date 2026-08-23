# Russell Tensor — Benchmarks

This directory contains [Criterion](https://github.com/bheisler/criterion.rs) benchmarks for the
`russell_tensor` crate.

The crate has a `heap` cargo feature that selects between two internal storage layouts at compile
time:

| `Tensor2` storage | `Tensor4` storage         | selected by          |
| ----------------- | ------------------------- | -------------------- |
| `vec: [f64; 9]`   | `mat: [[f64; 9]; 9]`      | (no `heap` feature)  |
| `vec: Vector`     | `mat: Matrix` (col-major) | `--features heap`    |

To compare the **stack** and **heap** layouts, run the benchmark twice (once with and once without
`--features heap`) and compare the results.

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
| `deriv2_invariant_lode`| second derivative of the Lode invariant           |

All benchmarks use fixed 3×3 input tensors.

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
# stack (MKL, no heap feature)
cargo bench -p russell_tensor --features intel_mkl --bench tensor_benchmark

# heap
cargo bench -p russell_tensor --all-features --bench tensor_benchmark
```

Filter to a single function, e.g. `t2_ssd`:

```bash
cargo bench -p russell_tensor --all-features --bench tensor_benchmark -- t2_ssd
```

> **Note:** `--all-features` enables both Intel MKL (when available) and the `heap` feature. To
> benchmark the stack layout, use `--features intel_mkl` instead.

---

## Polar decomposition benchmark

`polar_decomp_benchmark` compares the speed of the two polar-decomposition
algorithms across the condition-number range:

| algorithm | description |
| --------- | ----------- |
| `brannon` | `polar_decomp` — iterative fixed-point (Bjorck–Bowie) |
| `higham`  | `polar_decomp_higham` — quaternion-based, direct (Higham & Noferini, 2016) |

Cases (the condition number κ is the ratio of the largest to the smallest
singular value of `F`):

| case | κ | input |
| ---- | - | ----- |
| `well_conditioned` | ≈ 4 | example 03 (McGinty) |
| `moderate_conditioned` | ≈ 6·10² | Higham test 5.2, `y = 1e-3` |
| `ill_conditioned` | ≈ 6·10⁷ | Higham test 5.2, `y = 1e-8` |

Median times (single machine, Intel MKL):

| case | `brannon` | `higham` | higham speedup |
| ---- | --------- | -------- | -------------- |
| `well_conditioned` | 208 ns | 125 ns | 1.7× |
| `moderate_conditioned` | 731 ns | 165 ns | 4.4× |
| `ill_conditioned` | 1.91 µs | 202 ns | 9.5× |

### Observations

- **Higham is faster in every case**, including well-conditioned ones. Its
  direct algorithm does a fixed amount of work (~125–200 ns), whereas Brannon's
  fixed-point iteration performs several iterations even for well-conditioned
  `F`.
- **The gap widens with conditioning** (1.7× → 4.4× → 9.5×): Brannon iterates
  more as κ grows, while Higham's cost stays nearly constant (the small rise is
  from the extra pivoting/QR fallback paths taken for ill-conditioned inputs).
- Combined with the accuracy cross-check (Higham stays at ~1e-16 for
  ill-conditioned `F` where Brannon degrades to ~1e-12), Higham's algorithm is
  both faster and more robust here; Brannon's remains the reference for its
  simplicity and for the 2×2 case.

### How to run

```bash
cargo bench -p russell_tensor --features intel_mkl --bench polar_decomp_benchmark
```
