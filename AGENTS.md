# AGENTS.md

Notes for automated agents and contributors working in this repository.

## Build backend (read this first)

`russell_lab` has **two** BLAS/LAPACK backends, and the default needs nothing
installed:

- **`algebra_linear` (default)** — a separate crate of pure Rust kernels, nothing to
  find and nothing to link. It builds for any target, including
  `wasm32-unknown-unknown`, which has no libc for the C shim and no BLAS to link.
  This is what `cargo build` and `cargo test` use with no extra flags.
- **External library** — the C shim in `russell_lab/c_code/interface_blas.c`,
  selected by exactly one of:
  - **`open_blas`** — OpenBLAS, resolved via `pkg-config` with a Homebrew fallback.
    Needs the development headers (e.g. `lapack.h`). Also the reference netlib
    BLAS/LAPACK.
  - **`intel_mkl`** — Intel oneAPI MKL, expected under
    `/opt/intel/oneapi/mkl/<version>`; set `MKL_VERSION` to override `latest`.

`intel_mkl` takes precedence: enabling both links MKL (OpenBLAS is compiled in only under
`open_blas` without `intel_mkl`), and `build.rs` says so in the build log. There is no
`compile_error!` for the combination, so `--all-features` is a valid build and exercises the
MKL path. Enabling neither prints which path was taken, so a build log always records it.

The `open_blas` and `intel_mkl` features are forwarded from every dependent crate
(`russell_tensor`, `russell_sparse`, `russell_stat`, `russell_ode`,
`russell_pde`, `russell_nonlin`). Their own defaults are empty: they do **not**
force an external library, and Cargo cannot disable a dependency's default
features from the outside, so a crate that wants one has to say so in the
dependency declaration.

### Reference parity lives in `algebra_linear`

This crate has no reference-LAPACK harness any more: the `dev_lapack` feature moved
with the kernels. To compare the kernels against a locally built LAPACK 3.12.1, run
it there (`ALGEBRA_LINEAR_DEV_LAPACK_ROOT=... cargo test --features dev_lapack` in
the `algebra_linear` checkout). Nothing here needs a reference build, and there is no
`RUSSELL_DEV_LAPACK_ROOT` any more.

Extra features:

- `local_umfpack` and `mumps` on `russell_sparse`, `russell_ode`, `russell_pde`,
  `russell_nonlin` — use the locally compiled SuiteSparse and MUMPS installed by
  `zscripts/*-compile-suitesparse.bash` and `zscripts/*-compile-mumps.bash`. Both
  read `RUSSELL_LOCAL_PREFIX` (`.../include/umfpack`, `.../lib/umfpack`,
  `.../include/mumps`, `.../lib/mumps`) and fail the build if it is unset or wrong.
  Without `local_umfpack` the UMFPACK comes from the package manager, found with
  pkg-config; UMFPACK is required in every configuration, `mumps` is optional.

## The BLAS/LAPACK kernels live in `algebra_linear`

The port is no longer in this repository: it was extracted to
[`algebra_linear`](https://crates.io/crates/algebra_linear) (`BSD-3-Clause`), which
this crate depends on. Everything about the port — its conventions, its measured
state, the bug history, the reference-parity harness (`dev_lapack` lives there, not
here), and the traps it has already paid for — is in that crate's `AGENTS.md` and
`README`. Read those before touching a kernel; nothing in this repository implements
one.

What stays here: the dispatchers (`base::blas_dispatch`, `base::lapack_dispatch`),
the C shim and `build.rs` that link OpenBLAS or MKL, `Matrix`/`Vector`, the `small_*`
stack kernels, and every public API.

### Performance: where the kernels are

`mat_mat_mul` routes through `base::blas_dispatch::dgemm`, which is
`algebra_linear::blas::dgemm` -- a Goto-style blocked kernel with a 4x4 AVX2+FMA
micro-kernel (MC=128, KC=256, NC=256) -- unless an accelerated feature is on.
There is no `portable` module: `wasm32` takes the same path and gets the kernel's
scalar fallback, and the suite runs there under `wasmtime` in CI.

The kernel-side notes -- the BLAS-3-free pipeline of the LAPACK layer, what
`dsytrd` did and did not buy, and the blocking rules of thumb -- are in
`algebra_linear`'s `AGENTS.md`, where the code is. What is measured here is the
routing: `benchmarks/matmul_benchmark.rs` and `benchmarks/decomp_benchmark.rs`.

### Loose ends

- The crate is warning-free and clippy-clean (`cargo clippy --workspace
  --all-targets` exits 0). Keeping it that way needs the pattern described under
  "Dispatch between the native and external backends": the shim-only imports are
  `cfg`-gated and the functions with a shim branch carry a `cfg_attr`'d allow.
  Note that a warning count is not a pass/fail signal by itself -- clippy's
  deny-by-default lints are *errors*, so read the exit code, not the count (a
  `| grep -c warning` once reported "0 warnings" on a build that had failed).
- `--features open_blas` builds and passes on this machine: the headers are under
  `/usr/include/openblas` and `pkg-config` finds them. All configurations (plain,
  `open_blas`, `intel_mkl`, `--all-features`, and the wasm target) are runnable
  here, so an accelerated branch never has to be reasoned about without being
  checked. The unported kernel paths (`dlaqr0`, the QR/LQ pre-reduction of
  `dgesvd`) are loose ends of `algebra_linear`, recorded there.
- The web/WASM consumer of this crate has not been re-checked since the kernels
  became a dependency.

## Test / lint commands

```bash
# Tests, native backend: no external library needed.
cargo test -p russell_lab
cargo test -p russell_tensor

# Tests, external backend (needs OpenBLAS headers, or MKL installed).
cargo test -p russell_lab --features open_blas
cargo test -p russell_lab --features intel_mkl

# Numerical parity of the kernels against the reference LAPACK. The harness
# moved with the code, so it runs in the algebra_linear checkout, not here:
(cd ../algebra_linear && ALGEBRA_LINEAR_DEV_LAPACK_ROOT=/path/to/lapack/build \
  cargo test --features dev_lapack)

# Whole workspace (sparse crates need the locally compiled libraries)
RUSSELL_LOCAL_PREFIX=/usr/local cargo test --workspace --features local_umfpack,mumps

# Formatting (CI runs the check)
cargo fmt --all
cargo fmt --all -- --check

# Lint
cargo clippy --all-targets

# Coverage (nightly + llvm-tools-preview); project requires >95% lines, counted once
# per source line (llvm-cov's `--fail-under-lines` counts a line once per
# monomorphization, so const-generic dimension dispatch reads as uncovered; the
# `DA:` records of the lcov export are merged per line and are what the gate uses)
cargo llvm-cov --workspace --features local_umfpack,mumps \
  --ignore-filename-regex 'build.rs|mem_check.rs|mem_check_lab.rs|solve_matrix_market.rs|amplifier1t.rs|brusselator_pde.rs' \
  --lcov --output-path lcov.info
bash zscripts/check-coverage.bash lcov.info 95
```


## Deliberately uncovered code

The 95% gate counts source lines, so what is left uncovered is a decision, not an
accident. This is where those decisions are recorded. **A PR that adds code no test
can reach adds it here in the same PR** — otherwise the next person spends a day
rediscovering why a file sits at 84%.

Measured 2026-10-21, after the kernels moved to `algebra_linear`, with the command
above: workspace **59,445 / 60,166 lines = 98.80%**, 721 missed. The kernel rows left
with the code: they are in that crate's `AGENTS.md`, together with the older per-file
measurements taken while the two were one crate.

| where | missed | why |
|---|---|---|
| `russell_pde/src/transfinite_samples.rs`, `russell_lab/src/algo/bspline.rs` | 79, 43 | `if SAVE_FIGURE { ... }` blocks: plotting code that writes SVG to a developer's disk |
| `russell_tensor/src/polar_higham.rs` | 67 | the `dd2 == 0` branch, which needs an isotropic spectrum, and the ill-conditioned fallbacks |
| `russell_nonlin/src/solver.rs`, `solver_arclength.rs` | 31, 27 | scenario code: the `dlambda/ds` guard needs a turning point reached while the listed step is still legal, the precise-stop redo needs `Stop::UComp`, and the failure limits need a deliberately failing system |
| `russell_sparse/src/util.rs` | 31 | the `nvidia-smi` probe without a GPU, the `/proc/cpuinfo` fallback that `lscpu` pre-empts, and the false arms of `cfg!(feature = ...)` in a build that has the feature |
| `russell_tensor/src/eigen_projectors.rs` | 26 | `if VERBOSE { ... }` diagnostics |

The remaining ~420 lines are spread over many files, none above 25; the largest are
`russell_sparse/src/solver_mumps.rs` at 22 and `complex_solver_mumps.rs`,
`transfinite_2d.rs` and `radau5.rs` at 21.

Three rules that follow:

- **Do not refactor production code to satisfy the tool.** The five dyadic kernels in
  `russell_tensor` read as 50% covered because `--fail-under-lines` counts a line once
  per monomorphization; the `DA:` gate exists so that this is not a reason to rewrite
  working code.
- **When a gap is called unreachable, say what was tried.** The LAPACK entry above is
  trustworthy because five generators were ported and verified against LAPACK's own
  build (`families.rs`) before the sweeps were run.
- **The percentage moves when function layout changes, and that is the tool, not the
  tests.** llvm-cov's `DA:` records include lines that carry no code — comments, a
  `const` declared inside a function, a literal argument on its own line — whenever
  inlining puts a region boundary there. Consolidating two copies of `dlaln2` into
  `auxiliary.rs` moved the gate from 96.60% to 95.37% while every executed line kept
  its hit count (the `dgebal` entry went from 139 to 139): of `eig.rs`'s 586 missed
  lines, 416 have no code in them. Judge a refactor by the hit counts of the lines
  that do work, and say so in the PR instead of hunting the total.

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
  `small_mat_svd` is the one exception to "no heap": it wraps `dgesvd` (native
  Rust by default, the C shim with `open_blas`/`intel_mkl`) and `dgesvd` allocates its own
  workspace. Do not reintroduce an `AᵀA`-based SVD for it — forming `AᵀA` squares
  the condition number, and at κ = 1e8 that left `russell_tensor`'s polar
  rotation non-orthogonal by 0.28.
- **No developer paths in source.** Not in code, not in defaults, not in doc
  comments. An absolute path under someone's home directory ends up published the
  moment the crate is. Use an environment variable, and prefer *requiring* it over
  supplying a default.

## BLAS/LAPACK invariants

These are the rules this repository's code depends on. Each one has been violated at
least once. The kernels' own invariants -- scaling before squaring, the convergence
contracts, the triangle and contraction rules as they apply inside a driver -- are in
`algebra_linear`'s `AGENTS.md`.

- **Matrix layout.** Everything here and in the kernels is **column-major**:
  `algebra_linear::blas` and `algebra_linear::lapack` index `data[i + j * n]`, to
  match `Matrix`. `lapack.h` binds directly to the Fortran routines and is therefore
  column-major too. **`LAPACKE_*` is the row-major API and will silently transpose
  every matrix** — never use it here.
- **LAPACK's string arguments are a hidden-argument ABI.** `lapack.h` in a build
  with `LAPACK_FORTRAN_STRLEN_END` passes hidden trailing string lengths
  (`FORTRAN_STRLEN` = `size_t`). Prefer the shim's `c_*` wrappers, which take
  `C_BOOL` and supply those lengths themselves; hand-declaring the ABI is an easy
  way to get garbage rather than an error.
- **A routine that writes one triangle leaves the other alone.** `dpotrf` with
  `'L'` writes only the lower triangle and `'U'` only the upper; the rest keeps
  whatever the caller had. The dispatchers mirror the triangle they read into the one
  they need and write back only the requested one; `dsyrk` behaves the same way.
- **Check the contraction length, not just the output shape.** `C := A Aᵀ` and
  `C := Aᵀ A` are both `(n,n)` but contract over different lengths — `k` from `A`'s
  columns in one case and from its rows in the other. A dispatch that passes the
  same shape to both is silently wrong for non-square `A`. This bit
  `mat_sym_rank_op`.
- **`idamax` is 1-based in BLAS and 0-based in Rust.** `find_index_abs_max` and
  `algebra_linear::blas::idamax` return 0-based indices; the first maximum wins.
- **Non-convergence is an error, not a debug assertion.** At the boundary,
  `base::lapack_dispatch` maps an `INFO > 0` to `LapackError::Convergence`, and the
  Rust driver returns the same variant itself, so a caller never receives a partly
  converged result presented as a success. A `debug_assert` would only fire in a
  debug build.

## Dispatch between the native and external backends

BLAS and the main LAPACK drivers go through one module, not a `#[cfg]` pair per
call site:

- `base::blas_dispatch` -- dgemm (+tn/nt/tt), dgemv (+trans), ddot, dnrm2, dasum,
  idamax, dlange, zgemm (+tn/nt/tt), dger, zger, dsyrk, zherk, daxpy, zaxpy,
  zscal, dznrm2, zgemv (+trans), zsyrk.
- `base::lapack_dispatch` -- dgesv, dgetrf, dgetri, dgeev (+ left vectors, and
  eigenvalues only), dpotrf, dsyev, dgesvd, dggev, zggev.

Both mirror `algebra_linear`'s signatures (`usize` lengths, no workspace argument,
`Result<(), LapackError>`), so a call site reads the same in every configuration.
Add an entry there, not a branch in the caller. `intel_mkl` wins when both
features are on; there is no `compile_error!` for the pair, so `--all-features`
is a valid build and exercises the MKL path.

**A routine stays on the old per-file path when it has no twin to mirror or no
measured gain.** That is a decision per routine, documented where it is made, not
a TODO: the complex routines and `small_mat_svd` (a stack array, which a LAPACK
workspace defeats); and on the BLAS side `dcopy`/`zcopy` (one `memmove`), `dscal` and `zlange` (measured slower
through the library at the sizes that matter). Those keep this shape:

```rust
#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
{
    // the C call, with a comment naming the shim's symbol
}
#[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
// the native call, which uses usize lengths and no workspace argument
```

Before adding to either path:

1. **A new shim entry goes in both `c_code/interface.h` and the `.c`.** The
   header is what `build.rs` tracks now, so a wrapper added to the source file
   alone can be silently not rebuilt.
2. **The `extern "C"` declaration must be gated with the same condition** -- in
   the files that still hold one. If it is not, a native-only build fails to
   link. If it *is* but the call site is not dispatched, the build fails to
   compile, so migrate the dispatch **first** and gate the declaration second,
   one file at a time. The native arm calls `algebra_linear`.
3. **Routines without a native implementation return
   `LapackError::NotImplemented`** rather than being left to fail at link time. A
   runtime error can name the feature to enable; a link error about a mangled
   symbol cannot.
4. **A native driver returns `Result<(), LapackError>`, never LAPACK's `INFO`.**
   `INFO < 0` and `INFO > 0` both become an error at the driver boundary, so a
   caller never has to remember which integer means what. Internal helpers
   (`dlahqr`, `dhgeqz`, `dsteqr`, `dbdsqr`, `dtrevc3`, ...) keep returning `INFO`
   because that is what the reference source says and what a diff against it
   should show.
5. **The function-scope `lda`/`ldu`/`lwork`/`work`/`info` declarations and their
   `#[cfg_attr(... allow(unused_variables, ...))]` belong only to the files still
   on the per-file path.** The nine dispatched drivers keep their scratch arrays
   inside `lapack_dispatch`, which sizes the workspace from LAPACK's
   `lwork = -1` query instead of the hard-coded minimum two call sites used.


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
`zscripts/debian-compile-*.bash` (into `/usr/local`, with `RUSSELL_LOCAL_PREFIX`
set accordingly), runs `cargo test --features local_umfpack,mumps`, and on nightly
enforces the coverage threshold shown above.
