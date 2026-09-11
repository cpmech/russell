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
| OS        | Arch Linux (kernel 7.1.9)                          |
| CPU       | 13th Gen Intel(R) Core(TM) i9-13900KF (32 threads) |
| GPU       | NVIDIA GeForce RTX 4090                            |
| Memory    | 32 GB                                              |
| BLAS      | Intel MKL (`--features intel_mkl`)                  |

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
| `ssd_fn`                | 3.36 ns        | 6.07 ns       | 20.90 ns    | 70.27 ns   |
| `qsd_fn`                | 5.44 ns        | 9.03 ns       | 125.58 ns   | 140.28 ns  |
| `deriv2_invariant_jj3`  | 6.71 ns        | 11.77 ns      | 126.35 ns   | 135.36 ns  |
| `deriv2_invariant_lode` | 48.72 ns       | 69.04 ns      | 162.93 ns   | 205.25 ns  |
| `deriv_squared_tensor`  | 6.47 ns        | 43.03 ns      | 77.62 ns    | 82.17 ns   |

## Observations

- **Unrolled path:** the stack version is faster than the heap version, with the gap
  ranging from ~1.4× (`deriv2_invariant_lode`) to ~6.7× (`deriv_squared_tensor`). The
  heap version's `Matrix` carries the column-major access overhead, whereas the stack
  version writes directly to `[[f64; N]; N]`.
- **Loops path:** the stack version is only ~1.1–1.3× faster (`ssd_fn` is the
  exception at ~3.4×); the loop overhead (iteration, `M_TO_IJ`/`MN_TO_IJKL` lookups,
  and `get_std`/`set` accessors) dominates and largely masks the storage-layout
  difference.
- **Unrolled vs loops:** the unrolled path is ~23× faster for `qsd_fn`, ~19× for
  `deriv2_invariant_jj3`, ~12× for `deriv_squared_tensor`, ~6× for `ssd_fn`, and
  ~3.3× for `deriv2_invariant_lode`.

## How to run

Run the benchmark (from the workspace root):

```bash
# stack (MKL, no heap feature)
cargo bench -p russell_tensor --features intel_mkl --bench tensor_benchmark

# heap
cargo bench -p russell_tensor --features intel_mkl,heap --bench tensor_benchmark
```

Filter to a single function, e.g. `ssd_fn`:

```bash
cargo bench -p russell_tensor --features intel_mkl,heap --bench tensor_benchmark -- ssd_fn
```

> **Note:** the `heap` feature selects the heap-allocated storage layout. To
> benchmark the stack layout, use `--features intel_mkl` instead.

Alternatively, run all benchmarks and regenerate `RESULTS.md` in one go with:

```bash
python3 run_all.py
```

---

## Polar decomposition benchmark

`polar_decomp_benchmark` compares the speed of the polar-decomposition algorithms:

| algorithm | description                                                    |
| --------- | -------------------------------------------------------------- |
| `brannon` | `polar_rotation_brannon` — iterative fixed-point (3×3)         |
| `higham`  | `polar_quaternion_higham` — quaternion-based, direct (3×3)     |
| `eigen`   | `PolarAlgo::Eigen` — eigen-decomposition of `C = Fᵀ F` via `Spectral2` (3×3) |
| `svd`     | `PolarAlgo::SVD` — classic: singular value decomposition (3×3) |

> **Note:** all algorithms are benchmarked through the unified `polar_decomp`
> dispatcher, which computes the rotation `R` and the right stretch `U` together
> for every algorithm.

### General (3×3): all algorithms

| case                   | κ       | `brannon` | `higham`  | `eigen`   | `svd`     |
| ---------------------- | ------- | --------- | --------- | --------- | --------- |
| `well_conditioned`     | ≈ 4     | 208.50 ns | 115.86 ns | 94.25 ns  | 716.10 ns |
| `moderate_conditioned` | ≈ 6·10² | 729.33 ns | 157.19 ns | 144.48 ns | 611.59 ns |
| `ill_conditioned`      | ≈ 6·10⁷ | 1.92 µs   | 195.06 ns | —         | 557.94 ns |

### In-plane: all algorithms

| algorithm | time      |
| --------- | --------- |
| `brannon` | 263.52 ns |
| `higham`  | 120.87 ns |
| `eigen`   | 93.70 ns  |
| `svd`     | 295.43 ns |

### Observations

- **`eigen` is the fastest in every case** (~94–144 ns), after switching from the
  LAPACK `dsyev` path to the 3×3-specialized stack operations (`small_mat_*`) and
  `Spectral2` for the eigen-decomposition of `C = Fᵀ F`.
- **`higham` is a close second** (~116–195 ns), with a nearly constant cost. The
  iterative `brannon` is competitive only for well-conditioned `F` and degrades
  sharply as κ grows (208 ns → 1.92 µs).
- **`svd` is the slowest** (~295–716 ns) because it calls the general LAPACK
  `dgesvd` routine instead of a 3×3-specialized method.
- **`eigen` squares the condition number** (via `C = Fᵀ F`), so it fails for very
  ill-conditioned `F` (`det(F) < 1e-15`); it is not benchmarked for the
  ill-conditioned case, where `svd` is the robust classic choice.
- Accuracy-wise, `higham`, `eigen`, and `svd` all match the published reference
  values for well-conditioned `F`; for ill-conditioned `F`, `higham` and `svd`
  stay accurate while the iterative `brannon` degrades.

### How to run

```bash
cargo bench -p russell_tensor --features intel_mkl --bench polar_decomp_benchmark
```

---

## Eigenvalues benchmark

`spectral2_benchmark` compares the speed of the four eigenvalue methods available in
`Spectral2::calc_eigenvalues_mx` (eigenvalues only, without the eigenprojectors):

| method              | description                                          |
| ------------------- | ---------------------------------------------------- |
| `habera_zilian`     | stable closed-form, Habera & Zilian (2025)           |
| `harari_albocher22` | Box-1 discriminant, Harari & Albocher (2022)         |
| `harari_albocher23` | seven-square discriminant, Harari & Albocher (2023)  |
| `jacobi`            | iterative Jacobi rotations                           |

Two symmetric input tensors are used: `distinct` (well-separated eigenvalues) and
`coalescent` (two nearly equal eigenvalues).

### Results

Median times (single machine, Intel MKL):

| case         | `habera_zilian` | `harari_albocher22` | `harari_albocher23` | `jacobi`  |
| ------------ | --------------- | ------------------- | ------------------- | --------- |
| `distinct`   | 52.63 ns        | 42.84 ns            | 44.23 ns            | 205.83 ns |
| `coalescent` | 53.96 ns        | 41.50 ns            | 42.92 ns            | 205.42 ns |

### Observations

- The three closed-form methods take ~42–54 ns, roughly **4–5× faster** than the
  iterative `jacobi` (~206 ns).
- Among the closed-form methods, `harari_albocher22` is the fastest (~42 ns),
  followed by `harari_albocher23` (~44 ns) and `habera_zilian` (~53 ns).
- The coalescent case costs about the same as the distinct case for every method:
  the analytic formulas branch on the discriminant but never iterate.

### How to run

```bash
cargo bench -p russell_tensor --features intel_mkl --bench spectral2_benchmark
```

