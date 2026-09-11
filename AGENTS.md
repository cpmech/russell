# AGENTS.md

Notes for automated agents and contributors working in this repository.

## Build backend (read this first)

`russell_lab` links BLAS/LAPACK through the C shim in
`russell_lab/c_code/interface_blas.c`. Two backends are available:

- **OpenBLAS** (default) — resolved via `pkg-config`, with a Homebrew fallback.
  This requires the OpenBLAS/LAPACK development headers (e.g. `lapack.h`).
- **Intel MKL** — enabled with the `intel_mkl` feature. MKL is expected under
  `/opt/intel/oneapi/mkl/<version>`; set `MKL_VERSION` to override the default
  (`latest`).

On machines without a complete OpenBLAS/LAPACK install the default backend fails
to compile. **Use `--features intel_mkl` for local builds and tests.**

The `intel_mkl` feature is forwarded from every dependent crate
(`russell_tensor`, `russell_sparse`, `russell_stat`, `russell_ode`,
`russell_pde`, `russell_nonlin`).

Extra features:

- `russell_tensor/heap` — enables heap-backed tensor test cases. Run the tests
  both with and without it.
- `local_sparse` (and `cudss`) on `russell_sparse`, `russell_ode`,
  `russell_pde`, `russell_nonlin` — require locally compiled MUMPS/SuiteSparse
  (see `zscripts/*-compile-mumps.bash` and `zscripts/*-compile-suitesparse.bash`).

## Test / lint commands

```bash
# Tests (per crate; add intel_mkl when the OpenBLAS headers are unavailable)
cargo test -p russell_lab --features intel_mkl
cargo test -p russell_tensor --features intel_mkl          # stack
cargo test -p russell_tensor --features intel_mkl,heap     # stack + heap

# Whole workspace (sparse crates need the locally compiled libraries)
cargo test --workspace --features local_sparse

# Formatting (CI runs the check)
cargo fmt --all
cargo fmt --all -- --check

# Lint
cargo clippy --all-targets --features intel_mkl

# Coverage (nightly + llvm-tools-preview); project requires >95% lines
cargo llvm-cov --workspace --features local_sparse \
  --ignore-filename-regex 'build.rs|mem_check.rs|mem_check_lab.rs|solve_matrix_market.rs|amplifier1t.rs|brusselator_pde.rs' \
  --fail-under-lines 95
```

`cargo test` also runs the doc examples. Always keep the tree rustfmt-clean and
warning-free.

## CI

Workflows live in `.github/workflows/` (`ubuntu.yml`, `macos.yml`, `rocky.yml`,
`arch.yml`, `windows.yml`). The Ubuntu job compiles MUMPS/SuiteSparse via
`zscripts/debian-compile-*.bash`, runs `cargo test --features local_sparse`, and
on nightly enforces the coverage threshold shown above.
