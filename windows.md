# Using Russell on Windows

## Installation

The installation steps are as follows:

**1. Install MSYS2**

Download the installer from the MSYS2 website https://www.msys2.org/ (the 64-bit version is recommended).  

Run the installer and select the default installation path (typically `C:\msys64`).  

**2. Launch the MSYS2 Terminal**

After the installation, launch the `MSYS2 UCRT64` terminal from the Start Menu or the installation directory (recommended, as it best matches the Rust GNU toolchain).  

**3. Install the required packages in MSYS2**

Run the following commands in the `MSYS2 UCRT64` terminal:  

```bash
# Update the package database
pacman -Syu

# Install the Rust GNU toolchain (if not already installed)
rustup target add x86_64-pc-windows-gnu

# Install libraries required for compilation
pacman -S mingw-w64-ucrt-x86_64-openblas
pacman -S mingw-w64-ucrt-x86_64-suitesparse
```

**4. Set Environment Variable**

Set the environment variable in the `MSYS2 UCRT64` terminal (add to `~/.bashrc` to make it permanent):

```bash
export MSYS2_PREFIX='/ucrt64'
```

**5. Compile and Test**

In the `MSYS2 UCRT64` terminal, navigate to the Russell project directory and run:

```bash
# Navigate to the project directory
cd /c/path/to/russell

# Clean previous build cache
cargo clean

# Compile and run tests
cargo test
```

**6. Notes**

* Ensure that all cargo commands are executed within the MSYS2 UCRT64 terminal.
* Add export `MSYS2_PREFIX='/ucrt64'` to `~/.bashrc` to make the environment variable persistent.
* The Rust toolchain in MSYS2 defaults to GNU, which best matches the MSYS2 environment.

## Optional feature "with_mumps"

On Windows (MSYS2), MUMPS can be compiled using the following steps:

*Install compilation dependencies (in MSYS2 UCRT64 terminal):*

```bash
pacman -S mingw-w64-ucrt-x86_64-gcc-fortran
pacman -S mingw-w64-ucrt-x86_64-make
pacman -S mingw-w64-ucrt-x86_64-cmake
pacman -S mingw-w64-ucrt-x86_64-clang
pacman -S mingw-w64-ucrt-x86_64-metis
```

*Download and compile MUMPS:*

```bash
cd /tmp
curl http://deb.debian.org/debian/pool/main/m/mumps/mumps_5.8.2.orig.tar.gz -o mumps_5.8.2.orig.tar.gz
tar xzf mumps_5.8.2.orig.tar.gz
cd MUMPS_5.8.2

# Copy the MSYS2 configuration file
cp /c/path/to/russell/zscripts/makefiles-mumps/Makefile.inc.msys2 Makefile.inc

# Compile double precision (real) version
make d

# Compile complex version
make clean
make z
```

*Install MUMPS libraries and headers:*

```bash
# Copy libraries to /ucrt64/lib/mumps
mkdir -p /ucrt64/lib/mumps
cp lib/*.a /ucrt64/lib/mumps/

# Copy headers to /ucrt64/include/mumps
mkdir -p /ucrt64/include/mumps
cp include/*.h /ucrt64/include/mumps/
```