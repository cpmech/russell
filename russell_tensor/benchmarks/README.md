# Russell Tensor — Benchmarks

This directory contains [Criterion](https://github.com/bheisler/criterion.rs) benchmarks for the
`russell_tensor` crate.

The crate has a `heap` cargo feature that selects between two internal storage layouts at compile
time:

| `Tensor2` storage | `Tensor4` storage         | selected by         |
| ----------------- | ------------------------- | ------------------- |
| `vec: [f64; 9]`   | `mat: [[f64; 9]; 9]`      | (no `heap` feature) |
| `vec: Vector`     | `mat: Matrix` (col-major) | `--features heap`   |

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

| function                | description                             |
| ----------------------- | --------------------------------------- |
| `ssd_fn`                | self-sum-dyadic operation               |
| `qsd_fn`                | quad-sum-dyadic operation               |
| `deriv2_invariant_jj3`  | second derivative of the J3 invariant   |
| `deriv2_invariant_lode` | second derivative of the Lode invariant |

All benchmarks use fixed 3×3 input tensors.

## Results

Median times (single machine, Intel MKL):

| function               | stack/unrolled | heap/unrolled | stack/loops | heap/loops |
| ---------------------- | -------------- | ------------- | ----------- | ---------- |
| `ssd_fn`               | 5.53 ns        | 10.17 ns      | 211.02 ns   | 211.12 ns  |
| `qsd_fn`               | 7.25 ns        | 14.63 ns      | 436.57 ns   | 437.80 ns  |
| `deriv2_invariant_jj3` | 74.64 ns       | 99.18 ns      | 495.21 ns   | 527.16 ns  |

## Observations

- **Unrolled path:** the stack version is ~2× faster across the board. The heap version's
  `Matrix::set_unchecked` carries the column-major access overhead, whereas the stack version
  writes directly to `[[f64; 9]; 9]`.
- **Loops path:** stack and heap are essentially identical (~0.05–6% difference) — the loop
  overhead (iteration, `M_TO_IJ` lookups, `get_std`/`set` calls, and the conditional `√2`
  factors) dominates and masks the storage-layout difference.
- **Unrolled vs loops:** the unrolled path is ~40–60× faster for `ssd_fn`/`qsd_fn` and ~7×
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

Filter to a single function, e.g. `ssd_fn`:

```bash
cargo bench -p russell_tensor --all-features --bench tensor_benchmark -- ssd_fn
```

> **Note:** `--all-features` enables both Intel MKL (when available) and the `heap` feature. To
> benchmark the stack layout, use `--features intel_mkl` instead.

---

## Polar decomposition benchmark

`polar_decomp_benchmark` compares the speed of the polar-rotation algorithms:

| algorithm   | description                                                |
| ----------- | ---------------------------------------------------------- |
| `brannon`   | `polar_rotation_brannon` — iterative fixed-point (3×3)     |
| `brannon2d` | `polar_rotation_brannon2d` — closed-form (in-plane only)   |
| `higham`    | `polar_quaternion_higham` — quaternion-based, direct (3×3) |

> **Note:** `polar_quaternion_higham` computes the stretch `H` together with the
> rotation `Q` (the quaternion algorithm does not separate them), whereas the
> Brannon routines compute only `R`.

### General (3×3): Brannon vs Higham

| case                   | κ       | `brannon` | `higham` | higham speedup |
| ---------------------- | ------- | --------- | -------- | -------------- |
| `well_conditioned`     | ≈ 4     | 195 ns    | 124 ns   | 1.6×           |
| `moderate_conditioned` | ≈ 6·10² | 718 ns    | 164 ns   | 4.4×           |
| `ill_conditioned`      | ≈ 6·10⁷ | 1.89 µs   | 202 ns   | 9.4×           |

### In-plane: all three algorithms

| algorithm   | time   |
| ----------- | ------ |
| `brannon`   | 255 ns |
| `brannon2d` | 6.4 ns |
| `higham`    | 129 ns |

### Observations

- **Higham is faster than the iterative Brannon in every case** — by ~1.6× for
  well-conditioned `F`, growing to ~9.4× for ill-conditioned `F` (Brannon
  iterates more as κ grows, while Higham's cost stays nearly constant).
- **For in-plane `F`, the closed-form `brannon2d` is by far the fastest**
  (~6 ns), about 40× faster than the iterative `brannon` and 20× faster than
  `higham`.
- Combined with the accuracy cross-check (Higham stays at ~1e-16 for
  ill-conditioned `F` where the iterative Brannon degrades to ~1e-12), Higham's
  algorithm is both faster and more robust; the iterative Brannon remains the
  reference for its simplicity, and `brannon2d` is the clear choice for planar
  deformations.

### How to run

```bash
cargo bench -p russell_tensor --features intel_mkl --bench polar_decomp_benchmark
```
