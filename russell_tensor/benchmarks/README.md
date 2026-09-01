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

Each function is benchmarked in two variants:

- `unrolled` — the production implementation (manually unrolled, direct component access)
- `loops` — the loop-based reference implementation from `z_reference_loop_fns`

| function                | description                                        |
| ----------------------- | -------------------------------------------------- |
| `ssd_fn`                | self-sum-dyadic operation                          |
| `qsd_fn`                | quad-sum-dyadic operation                          |
| `deriv2_invariant_jj3`  | second derivative of the J3 invariant              |
| `deriv2_invariant_lode` | second derivative of the Lode invariant            |
| `deriv_squared_tensor`  | derivative of the squared tensor (general Tensor2) |

All benchmarks use fixed 3×3 input tensors.

## Results

Median times (single machine, Intel MKL):

| function                | stack/unrolled | heap/unrolled | stack/loops | heap/loops |
| ----------------------- | -------------- | ------------- | ----------- | ---------- |
| `ssd_fn`                | 3.52 ns        | 5.99 ns       | 181.28 ns   | 252.29 ns  |
| `qsd_fn`                | 6.72 ns        | 10.41 ns      | 372.14 ns   | 512.96 ns  |
| `deriv2_invariant_jj3`  | 19.48 ns       | 25.58 ns      | 347.93 ns   | 434.22 ns  |
| `deriv2_invariant_lode` | 98.59 ns       | 147.86 ns     | 494.03 ns   | 663.02 ns  |
| `deriv_squared_tensor`  | 23.28 ns       | 47.37 ns      | 77.73 ns    | 110.54 ns  |

## Observations

- **Unrolled path:** the stack version is ~1.3–2× faster across the board. The heap version's
  `Matrix` carries the column-major access overhead, whereas the stack version writes directly
  to `[[f64; 9]; 9]`.
- **Loops path:** the stack version is ~1.3–1.4× faster; the loop overhead (iteration,
  `M_TO_IJ`/`MN_TO_IJKL` lookups, and `get_std`/`set` accessors) dominates but does not fully
  mask the storage-layout difference.
- **Unrolled vs loops:** the unrolled path is ~40–55× faster for `ssd_fn`/`qsd_fn`, ~17× for
  `deriv2_invariant_jj3`, ~5× for `deriv2_invariant_lode`, and ~2–3× for `deriv_squared_tensor`.

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

`polar_decomp_benchmark` compares the speed of the polar-decomposition algorithms:

| algorithm | description                                                    |
| --------- | -------------------------------------------------------------- |
| `brannon` | `polar_rotation_brannon` — iterative fixed-point (3×3)         |
| `higham`  | `polar_quaternion_higham` — quaternion-based, direct (3×3)     |
| `eigen`   | `PolarAlgo::Eigen` — classic: eigenvalues of `C = Fᵀ F` (3×3)  |
| `svd`     | `PolarAlgo::SVD` — classic: singular value decomposition (3×3) |

> **Note:** all algorithms are benchmarked through the unified `polar_decomp`
> dispatcher, which computes the rotation `R` and the right stretch `U` together
> for every algorithm.

### General (3×3): all algorithms

| case                   | κ       | `brannon` | `higham` | `eigen` | `svd`  |
| ---------------------- | ------- | --------- | -------- | ------- | ------ |
| `well_conditioned`     | ≈ 4     | 216 ns    | 126 ns   | 786 ns  | 740 ns |
| `moderate_conditioned` | ≈ 6·10² | 768 ns    | 164 ns   | 711 ns  | 636 ns |
| `ill_conditioned`      | ≈ 6·10⁷ | 2.02 µs   | 200 ns   | —       | 582 ns |

### In-plane: all algorithms

| algorithm | time   |
| --------- | ------ |
| `brannon` | 275 ns |
| `higham`  | 131 ns |
| `eigen`   | 455 ns |
| `svd`     | 320 ns |

### Observations

- **Higham is the fastest in every case**, and its cost is nearly constant
  (~126–200 ns). The iterative `brannon` is competitive only for well-conditioned
  `F` and degrades sharply as κ grows (216 ns → 2.02 µs).
- **The classic `eigen`/`svd` algorithms are the slowest** (~320–786 ns) because
  they call general LAPACK routines (`dsyev`/`dgesvd`) instead of a
  3×3-specialized method. `svd` is somewhat faster than `eigen`.
- **`eigen` squares the condition number** (via `C = Fᵀ F`), so it fails for very
  ill-conditioned `F` (`det(F) < 1e-15`); it is not benchmarked for the
  ill-conditioned case. This makes the SVD-based classic algorithm the more
  robust of the two, and the only classic choice for ill-conditioned `F`.
- Accuracy-wise, `higham`, `eigen`, and `svd` all match the published reference
  values for well-conditioned `F`; for ill-conditioned `F`, `higham` and `svd`
  stay accurate while the iterative `brannon` degrades.

### How to run

```bash
cargo bench -p russell_tensor --all-features --bench polar_decomp_benchmark
```

