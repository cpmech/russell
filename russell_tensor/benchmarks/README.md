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

---

## Polar decomposition benchmark

`polar_decomp_benchmark` compares the speed of the polar-decomposition algorithms:

| algorithm    | description                                                                  |
| ------------ | ---------------------------------------------------------------------------- |
| `iterative`  | `PolarAlgo::Iterative` — Brannon's iterative fixed-point (3×3)               |
| `quaternion` | `PolarAlgo::Quaternion` — Higham & Noferini quaternion-based, direct (3×3)   |
| `eigen`      | `PolarAlgo::Eigen` — eigen-decomposition of `C = Fᵀ F` via `Spectral2` (3×3) |
| `svd`        | `PolarAlgo::SVD` — classic: singular value decomposition (3×3)               |

> **Note:** all algorithms are benchmarked through the unified `polar_decomp_mx`
> dispatcher, which computes the rotation `R` and the right stretch `U` together
> for every algorithm.

Mildly-, well-, moderately-, and ill-conditioned `F` are benchmarked, plus an in-plane `F`.
The `eigen` algorithm squares the condition number (via `C = Fᵀ F`), so it is not
benchmarked for the ill-conditioned case.

### How to run

```bash
cargo bench -p russell_tensor --features intel_mkl --bench polar_decomp_benchmark
```

---

## Eigenvalues benchmark

`spectral2_benchmark` compares the speed of the four eigenvalue methods available in
`Spectral2::calc_eigenvalues_mx` (eigenvalues only, without the eigenprojectors):

| method            | description                                                                      |
| ----------------- | -------------------------------------------------------------------------------- |
| `analytical_hz`   | `EigMethod::AnalyticalHZ` — stable closed-form (Habera & Zilian 2025)            |
| `analytical_ha22` | `EigMethod::AnalyticalHA22` — Box-1 discriminant (Harari & Albocher 2022)        |
| `analytical_ha23` | `EigMethod::AnalyticalHA23` — seven-square discriminant (Harari & Albocher 2023) |
| `iterative`       | `EigMethod::Iterative` — iterative Jacobi rotations                              |

Two symmetric input tensors are used: `distinct` (well-separated eigenvalues) and
`coalescent` (two nearly equal eigenvalues).

### How to run

```bash
cargo bench -p russell_tensor --features intel_mkl --bench spectral2_benchmark
```

---

## Results

The full results (system information and the median times for every benchmark above)
are auto-generated into [`RESULTS.md`](RESULTS.md). To regenerate it, run:

```bash
python3 run_all.py
```
