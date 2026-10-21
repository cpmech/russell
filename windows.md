# Using Russell on Windows

Two toolchains work, and the crate supports both:

- **MSVC**, the default Rust toolchain on Windows. The Native solver is pure
  Rust, so nothing else is needed; UMFPACK comes from [vcpkg](https://vcpkg.io).
  MUMPS is not available here.
- **MSYS2 with the GNU toolchain**, which also provides OpenBLAS, SuiteSparse and
  the MinGW/gfortran build of MUMPS. This is the only way to use MUMPS on
  Windows, and it needs the GNU target (`x86_64-pc-windows-gnu`).

## Option 1 — MSVC (default toolchain)

1. Install Rust from [rust-lang.org](https://rust-lang.org/learn/get-started/),
   keeping the preselected `x86_64-pc-windows-msvc` host triple.
2. Run the tests. The Native solver needs no external library:

```bash
cargo test
```

3. To add UMFPACK, install it with vcpkg (the dynamic triplet is the supported
   one; a static triplet renames the libraries and would need the BLAS and its
   runtime linked by hand):

```bat
vcpkg install suitesparse-umfpack[core]:x64-windows
```

4. Point the build at the vcpkg tree by setting `VCPKG_ROOT` to the vcpkg clone
   (the GitHub runners set `VCPKG_INSTALLATION_ROOT`, which is also understood;
   `RUSSELL_VCPKG_TRIPLET` overrides the triplet). The build script looks for
   `%VCPKG_ROOT%\installed\x64-windows\include\suitesparse` and
   `...\lib\umfpack.lib`.
5. The DLLs must be found at run time, so put them on `PATH` for the tests:

```bat
set PATH=%VCPKG_ROOT%\installed\x64-windows\bin;%PATH%
cargo test --features umfpack
```

## Option 2 — MSYS2 with the GNU toolchain (required for MUMPS)

### Install MSYS2

1. Download the [MSYS2 installer](https://www.msys2.org) (the 64-bit version is recommended).
2. Run the installer and select the default installation path (typically `C:\msys64`).
3. After the installation, launch the `MSYS2 UCRT64` terminal from the Start Menu.
4. Run the following commands in the `MSYS2 UCRT64` terminal:

```bash
pacman -Syu
pacman -S mingw-w64-ucrt-x86_64-openblas
pacman -S mingw-w64-ucrt-x86_64-suitesparse
```

The above commands will update the package database and install `openblas` and `suitesparse`, as required by this project.

Finally, set the environment variable in the `MSYS2 UCRT64` terminal. Open a text editor (e.g., Notepad) and add the following line to `~/.bashrc`:

```bash
export MSYS2_PREFIX='/ucrt64'
```

### Install and configure Rust

1. Go to [the official Rust website](https://rust-lang.org/learn/get-started/) and download **rustup-init.exe** (e.g., for x64).
2. Run `rustup-init.exe` and select (in the terminal that opens):

```text
* Type **3** for `3) Don't install the prerequisites (if you're targeting the GNU ABI)`.
* Type **2** for `2) Customize installation`.
* Answer the `Default host triple` question with `x86_64-pc-windows-gnu`, **not** the preselected msvc option.
* You may choose the default answers for the next questions, then, *Proceed with the selected options*.
```

Then, open the `MSYS2 UCRT64` terminal and run the following command to add the GNU target:

```bash
rustup target add x86_64-pc-windows-gnu
```

If you have already installed Rust with the MSVC toolchain, you can switch to the GNU toolchain by running:

```bash
rustup default stable-x86_64-pc-windows-gnu
```

### Compile and test the code

Open the `MSYS2 UCRT64` terminal, navigate to the Russell project directory and run `cargo test`. For example, if the project is located at `C:\russell`, you can run:

```bash
cd /c/russell
cargo test
```

## (Optional) Compile and install the MUMPS solver — GNU toolchain only

The Windows build of MUMPS is MinGW objects plus gfortran, so MSVC cannot link
it; use the GNU toolchain from the section above.

Open the `MSYS2 UCRT64` terminal, change to the `russell` project directory
(e.g., `cd /c/russell`), and run the following script to download and compile
the MUMPS solver (*use this script at your own risk; carefully check the scripts
before running them*):

```bash
bash ./zscripts/windows-compile-mumps.bash
```

The script installs the libraries under `$RUSSELL_LOCAL_PREFIX` (the MSYS2 root,
`/ucrt64`), which is where the build script reads them from. Set it and run the
tests:

```bash
export RUSSELL_LOCAL_PREFIX='/ucrt64'
cargo test --features umfpack,mumps
```

With `RUSSELL_LOCAL_PREFIX` set, UMFPACK itself is still taken from
`MSYS2_PREFIX`, because the MSYS2 package does not use the crate's
`include/umfpack` layout; the build prints a warning saying so and falls back to
the system library.
