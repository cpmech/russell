# AGENTS.md

Notes for automated agents and contributors working in this repository.

> **Work in progress:** also read [`AGENTS-WIP.md`](AGENTS-WIP.md) (a local,
> gitignored file) for the current in-flight work, design decisions, and next
> steps. Keep it up to date as you work.

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

## Coding conventions

- **Double lower-case names denote a single capital letter**, matching the symbols
  used in doc comments and the mathematical formulas: `aa` ⇔ `A`, `bb` ⇔ `B`,
  `pp` ⇔ `P`, ... For example, the scalar coefficients `A`, `B`, `C` in the
  derivative formulas of the invariants are coded as `aa`, `bb`, `cc`.
- **Index ranges:** `m`, `n` (and `p`, `q`) are reserved for the Kelvin-Mandel
  components, ranging over `1..N`. Do **not** use them as dummy Cartesian
  indices (which range over `1,2,3`); use `r`, `s`, `t`, ... for those instead
  (e.g. `∂I3/∂a_ij = ½ ϵ_ikl ϵ_jrs a_kr a_ls`).
- **Permutation (Levi-Civita) tensor:** use the *lunate* epsilon `ϵ` (U+03F5) in
  formulas, not the curly `ε` (U+03B5), which is reserved for strain.
- **Prefer the `small_*` helpers** from `russell_lab` over the generic `mat_*` / `Matrix`
  versions whenever the size is known at compile time — e.g. for 3×3 (second-order tensor)
  and 2×2 operations. Use `small_mat_mat_mul`, `small_mat_t_mat_mul`, `small_mat_inv`,
  `small_mat_eigen_sym_jacobi`, etc., which operate on stack `[[f64; N]; N]` / `[f64; N]`
  arrays and avoid heap allocation; get the stack matrix from a tensor with
  `Tensor2::to_std_matrix_slice` (or `to_std_matrix_slice`-style helpers).

## Code intelligence (CodeGraph)

This repository is indexed by CodeGraph (a `.codegraph/` directory exists at the
root; see the global AGENTS.md for the full guidance). Reach for
`codegraph_explore` **before** grep/read when you need to understand or locate
code — one call returns the relevant symbols' verbatim source plus the call
paths, which is cheaper and more accurate than a search/read loop.

The index keeps itself fresh: a file watcher with a debounced auto-sync
(~2 s, `CODEGRAPH_WATCH_DEBOUNCE_MS`), a per-file staleness banner on tool
responses, and a connect-time catch-up. A path can therefore look stale for a
few seconds right after a rename/create — this once surfaced `eigen2_values.rs`
moments after it had been renamed to `eigen_values.rs` (the next check already
showed the correct name). When that happens, or when a tool response carries the
staleness banner, verify the filename on disk (`ls` / glob) and `Read` the
specific file for line-level edits.

If a path is still wrong after the debounce window — or the watcher is disabled
(e.g. a sandbox, or `CODEGRAPH_NO_DAEMON=1`) — force a refresh:
`codegraph status` (reports a `### Pending sync:` list), `codegraph sync`
(incremental) or `codegraph index` (full rebuild); `codegraph unlock` clears a
stale lock file.

## CI

Workflows live in `.github/workflows/` (`ubuntu.yml`, `macos.yml`, `rocky.yml`,
`arch.yml`, `windows.yml`). The Ubuntu job compiles MUMPS/SuiteSparse via
`zscripts/debian-compile-*.bash`, runs `cargo test --features local_sparse`, and
on nightly enforces the coverage threshold shown above.
