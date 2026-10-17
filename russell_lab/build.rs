//! This build script compiles and links `c_code/interface_blas.c`, the C shim
//! that exposes BLAS/LAPACK to `russell_lab`.
//!
//! Two accelerators are supported, selected at compile time and mutually
//! exclusive:
//!
//! * `intel_mkl` -- link against Intel oneAPI MKL (statically).
//! * `open_blas` -- link against OpenBLAS, resolved via pkg-config when
//!   possible, with a fallback to hardcoded paths (mainly for Homebrew, where
//!   the `.pc` files are keg-only).
//!
//! With neither feature on, nothing is compiled or linked here: the crate is
//! in its default configuration and uses the native Rust BLAS/LAPACK in
//! `src/native`. That is also the only configuration available on targets with
//! no C toolchain or BLAS, chiefly `wasm32-unknown-unknown` (there is no wasm
//! libc for the shim to compile against -- `inttypes.h` is missing).
//!
//! An optional third mode, `dev_lapack`, builds a prefixed copy of the same
//! shim against a locally built reference LAPACK so the native routines can be
//! checked for parity; see `compile_dev_lapack` below.
//!
//! Note that the guard has to be here in `main` rather than inside
//! `compile_blas`: the `#[cfg(feature = ...)]` blocks below then simply never
//! apply, so no `cc`/`pkg-config` work is attempted at all.

/// Default MKL version used when the `MKL_VERSION` environment variable is unset.
#[cfg(feature = "intel_mkl")]
const MKL_VERSION: &str = "latest";

/// Compiles and links against Intel oneAPI MKL.
#[cfg(feature = "intel_mkl")]
fn compile_blas() {
    // Allow overriding the MKL/compiler version via the environment (e.g., "2024.2.0").
    let version = std::env::var("MKL_VERSION").unwrap_or_else(|_| MKL_VERSION.to_string());
    let mkl_root = format!("/opt/intel/oneapi/mkl/{}", version);
    let compiler_root = format!("/opt/intel/oneapi/compiler/{}", version);

    cc::Build::new()
        .file("c_code/interface_blas.c")
        .include(format!("{}/include", mkl_root))
        .define("USE_INTEL_MKL", None)
        .compile("c_code_interface_blas");

    // MKL's own library dir, the compiler's library dir (for the OpenMP runtime
    // `libiomp5`), and a few system libraries.
    println!("cargo:rustc-link-search=native={}/lib/intel64", mkl_root);
    println!("cargo:rustc-link-search=native={}/lib", compiler_root);
    println!("cargo:rustc-link-lib=static=mkl_intel_lp64");
    println!("cargo:rustc-link-lib=static=mkl_intel_thread");
    println!("cargo:rustc-link-lib=static=mkl_core");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=iomp5");
}

/// Compiles and links against OpenBLAS (and LAPACK).
#[cfg(all(feature = "open_blas", not(feature = "intel_mkl")))]
fn compile_blas() {
    #[cfg(target_os = "windows")]
    {
        // On Windows (MSYS2), everything lives under `MSYS2_PREFIX`.
        let prefix = std::env::var("MSYS2_PREFIX").expect("MSYS2_PREFIX environment variable not set");
        cc::Build::new()
            .file("c_code/interface_blas.c")
            .include(format!("{}/include/openblas", prefix))
            .compile("c_code_interface_blas");
        println!("cargo:rustc-link-search=native={}/lib", prefix);
        println!("cargo:rustc-link-lib=dylib=openblas");
    }

    #[cfg(not(target_os = "windows"))]
    {
        let mut build = cc::Build::new();
        build.file("c_code/interface_blas.c");

        match probe_blas_lapack() {
            // pkg-config knows the answer: use its include/link information.
            Some(found) => {
                for inc in &found.include_paths {
                    build.include(inc);
                }
                build.compile("c_code_interface_blas");
                for dir in &found.link_paths {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                }
                for lib in &found.libs {
                    println!("cargo:rustc-link-lib=dylib={}", lib);
                }
            }
            // Fallback for setups pkg-config knows nothing about (mainly
            // Homebrew, where openblas/lapack are keg-only and their `.pc`
            // files are off the default PKG_CONFIG_PATH).
            None => {
                let inc_dirs = existing_dirs(&[
                    "/usr/include/openblas",
                    "/opt/homebrew/opt/lapack/include",
                    "/opt/homebrew/opt/openblas/include",
                    "/usr/local/opt/lapack/include",
                    "/usr/local/opt/openblas/include",
                ]);
                let lib_dirs = existing_dirs(&[
                    "/opt/homebrew/opt/lapack/lib",
                    "/opt/homebrew/opt/openblas/lib",
                    "/usr/local/opt/lapack/lib",
                    "/usr/local/opt/openblas/lib",
                ]);
                build.includes(&inc_dirs);
                build.compile("c_code_interface_blas");
                for d in &lib_dirs {
                    println!("cargo:rustc-link-search=native={}", d);
                }
                println!("cargo:rustc-link-lib=dylib=openblas");
                println!("cargo:rustc-link-lib=dylib=lapack");
            }
        }
    }
}

/// Keeps only the directories that actually exist, so the fallback does not
/// emit `-I`/`-L` flags pointing at non-existent directories (e.g., on a Mac
/// where Homebrew lives under `/opt/homebrew` rather than `/usr/local`).
#[cfg(all(feature = "open_blas", not(target_os = "windows"), not(feature = "intel_mkl")))]
fn existing_dirs(dirs: &[&str]) -> Vec<String> {
    dirs.iter()
        .filter(|d| std::path::Path::new(d).is_dir())
        .map(|d| d.to_string())
        .collect()
}

/// The include/link information needed to build and link `interface_blas.c`.
#[cfg(all(feature = "open_blas", not(target_os = "windows"), not(feature = "intel_mkl")))]
struct BlasLapack {
    include_paths: Vec<std::path::PathBuf>,
    link_paths: Vec<std::path::PathBuf>,
    libs: Vec<String>,
}

/// Finds BLAS/LAPACK via `pkg-config`. Returns `None` if there is no usable
/// answer, so the caller can fall back to hardcoded paths.
///
/// Tries `openblas` first (one module with both headers), then the neutral
/// `cblas` + `lapack` pair that Nix uses.
///
/// A successful `pkg-config` run is not enough: we also check that the headers
/// are really there. The `lapack` module means different things on different
/// distros — on RHEL/Rocky it is netlib LAPACK, which registers `lapack.pc`
/// but ships no `lapack.h`, so we would end up with no usable `-I` and fail
/// to compile instead of falling back.
#[cfg(all(feature = "open_blas", not(target_os = "windows"), not(feature = "intel_mkl")))]
fn probe_blas_lapack() -> Option<BlasLapack> {
    // Only emit the cargo directives once we know the answer is usable.
    let probe = |name: &str| pkg_config::Config::new().cargo_metadata(false).probe(name).ok();

    let candidates: Vec<Vec<pkg_config::Library>> = vec![
        probe("openblas").into_iter().collect(),
        probe("cblas").into_iter().chain(probe("lapack")).collect::<Vec<_>>(),
    ];

    for libraries in candidates {
        if libraries.is_empty() {
            continue;
        }
        let found = BlasLapack {
            include_paths: libraries.iter().flat_map(|l| l.include_paths.clone()).collect(),
            link_paths: libraries.iter().flat_map(|l| l.link_paths.clone()).collect(),
            libs: libraries.iter().flat_map(|l| l.libs.clone()).collect(),
        };
        if headers_available(&found.include_paths) {
            return Some(found);
        }
    }
    None
}

/// Checks that both headers used by `interface_blas.c` can actually be found.
///
/// The default include dirs count too: Debian/Ubuntu put `lapack.h` directly
/// in `/usr/include`, where no `-I` is needed.
#[cfg(all(feature = "open_blas", not(target_os = "windows"), not(feature = "intel_mkl")))]
fn headers_available(include_paths: &[std::path::PathBuf]) -> bool {
    let implicit = [
        std::path::PathBuf::from("/usr/include"),
        std::path::PathBuf::from("/usr/local/include"),
    ];
    ["cblas.h", "lapack.h"].iter().all(|header| {
        include_paths
            .iter()
            .chain(&implicit)
            .any(|dir| dir.join(header).is_file())
    })
}

fn main() {
    // The reference build is independent of the accelerators: it exists so the *native*
    // routines can be tested against the reference implementation, so it is
    // handled before the accelerator branches below.
    #[cfg(feature = "dev_lapack")]
    compile_dev_lapack();

    // Native Rust BLAS/LAPACK is the default: nothing to build, nothing to link,
    // and the crate compiles for any target. The two `cfg` blocks below then
    // compile out, so a native build has no further work to do here.
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    {
        println!("cargo:warning=russell_lab: using the native Rust BLAS/LAPACK (no external library)");
    }

    // Enabling both would mean two different libraries providing the same
    // symbols. Fail here rather than at link time, where the error is far less
    // clear.
    #[cfg(all(feature = "open_blas", feature = "intel_mkl"))]
    {
        panic!(
            "russell_lab: features `open_blas` and `intel_mkl` are mutually exclusive; \
             enable exactly one accelerator, or neither for the native Rust routines"
        );
    }

    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    compile_blas();
}

/// Compiles the C shim against a locally built LAPACK and links it, so the tests
/// have a reference to compare the native routines against.
///
/// The shim itself is `c_code/interface_blas.c` -- already known to drive the
/// **column-major** `LAPACK_*` interface correctly because `lapack.h` binds
/// straight to the Fortran routines, unlike `LAPACKE_*`, which is row-major and
/// would silently transpose every matrix. That one source is what keeps this
/// reference trustworthy; the copy compiled here is derived from it mechanically
/// (see [`prefixed_shim`]) so it cannot drift from the shim the crate actually
/// ships with.
///
/// # Layout expected
///
/// A CMake build directory of LAPACK 3.12.1 that has `include/` and `lib/`
/// containing `lapack.h`, `cblas.h`, `liblapack.so` and `libcblas.so`. Point
/// `RUSSELL_DEV_LAPACK_ROOT` at it. There is deliberately no default: recording one
/// would put a developer's directory layout into released source.
#[cfg(feature = "dev_lapack")]
fn compile_dev_lapack() {
    let root = std::env::var("RUSSELL_DEV_LAPACK_ROOT").unwrap_or_else(|_| {
        panic!(
            "russell_lab: the `dev_lapack` feature needs RUSSELL_DEV_LAPACK_ROOT set to the CMake \
             build directory of a LAPACK 3.12.1 build (the one containing include/ and lib/)"
        )
    });

    let include = std::path::Path::new(&root).join("include");
    let lib = std::path::Path::new(&root).join("lib");

    if !include.join("lapack.h").is_file() || !include.join("cblas.h").is_file() {
        panic!(
            "russell_lab: the `dev_lapack` feature needs a LAPACK build for its headers, \
             looked for {}/lapack.h and {}/cblas.h. \
             Set RUSSELL_DEV_LAPACK_ROOT to the CMake build directory of a LAPACK 3.12.1 build.",
            include.display(),
            include.display()
        );
    }
    if !lib.join("libcblas.so").is_file() {
        panic!(
            "russell_lab: the `dev_lapack` feature needs {}/libcblas.so for the `cblas_*` symbols; \
             set RUSSELL_DEV_LAPACK_ROOT to the CMake build directory of a LAPACK 3.12.1 build",
            lib.display()
        );
    }
    if !lib.join("liblapack.so").is_file() {
        panic!(
            "russell_lab: the `dev_lapack` feature needs {}/liblapack.so; \
             set RUSSELL_DEV_LAPACK_ROOT to the CMake build directory of a LAPACK 3.12.1 build",
            lib.display()
        );
    }

    // A second copy of the shim can be linked alongside the normal one only if
    // its entry points have different names, so this build derives one on the
    // fly with each `c_*` renamed to `dev_lapack_c_*`. Deriving it mechanically
    // (rather than keeping a copy in the tree) means the two can never drift:
    // there is one `interface_blas.c`, and the prefixing is the only difference.
    let out_dir =
        std::path::PathBuf::from(std::env::var("OUT_DIR").expect("russell_lab: OUT_DIR must be set by cargo"));
    let shim =
        std::fs::read_to_string("c_code/interface_blas.c").expect("russell_lab: cannot read c_code/interface_blas.c");
    let generated = prefixed_shim(&shim);
    let prefixed_path = out_dir.join("interface_blas_dev_lapack.c");
    std::fs::write(&prefixed_path, &generated).expect("russell_lab: cannot write the prefixed shim into OUT_DIR");
    cc::Build::new()
        .file(&prefixed_path)
        // The generated file sits in OUT_DIR, so the shim's own `constants.h`
        // has to be found through an explicit include directory.
        .include("c_code")
        .include(&include)
        .compile("dev_lapack_interface_blas");

    println!("cargo:rustc-link-search=native={}", lib.display());
    // Order matters for a static link and does no harm for a shared one: the
    // shim needs LAPACK, which needs BLAS. CBLAS is separate: `interface_blas.c`
    // wraps only LAPACK, so the `cblas_*` symbols it calls come from here.
    println!("cargo:rustc-link-lib=dylib=lapack");
    println!("cargo:rustc-link-lib=dylib=cblas");
    println!("cargo:rustc-link-lib=dylib=blas");
    // The reference is Fortran, so its runtime is needed.
    println!("cargo:rustc-link-lib=dylib=gfortran");
    println!("cargo:rerun-if-env-changed=RUSSELL_DEV_LAPACK_ROOT");
    // The prefixed copy is derived from this file, so watching it is enough.
    println!("cargo:rerun-if-changed=c_code/interface_blas.c");
    println!("cargo:warning=russell_lab: `dev_lapack` feature ON -- tests compare against {root}");
}

/// Returns `interface_blas.c` with every `c_*` entry point renamed to
/// `dev_lapack_c_*`, so the two shims can be linked into the same binary.
///
/// The rename has to cover the identifiers only; it is applied to the whole
/// source because the shim calls none of its own entry points (each wrapper
/// goes straight to a `FN_*` macro), so every `c_` token in the file is a
/// definition. Anything that is not an identifier character before the token
/// guards against matching inside a longer name such as `cblas_*`.
#[cfg(feature = "dev_lapack")]
fn prefixed_shim(source: &str) -> String {
    // The entry points are already `c_*`, so this yields `dev_lapack_c_*`.
    const PREFIX: &str = "dev_lapack_";
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let chars: Vec<char> = source.chars().collect();
    let mut out = String::with_capacity(source.len() + 4096);
    out.push_str("// Generated from interface_blas.c by build.rs -- do not edit.\n");
    out.push_str("// Entry points carry the `dev_lapack_c_` prefix so this copy can be\n");
    out.push_str("// linked alongside the normal shim without duplicate symbols.\n\n");
    let mut i = 0;
    while i < chars.len() {
        let starts_entry_point =
            chars[i] == 'c' && i + 1 < chars.len() && chars[i + 1] == '_' && (i == 0 || !is_ident(chars[i - 1]));
        if starts_entry_point {
            let start = i;
            while i < chars.len() && is_ident(chars[i]) {
                i += 1;
            }
            out.push_str(PREFIX);
            out.extend(&chars[start..i]);
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}
