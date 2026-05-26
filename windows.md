# Using Russell on Windows

## Install MSYS2

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

## Install and Configure Rust on Windows

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

## Compile and Test the Code

Open the `MSYS2 UCRT64` terminal, navigate to the Russell project directory and run `cargo test`. For example, if the project is located at `C:\russell`, you can run:

```bash
cd /c/russell
cargo test
```

## (Optional) Compile and Install the MUMPS Solver

Open the `MSYS2 UCRT64` terminal, make sure to change to the `russell` project directory (e.g., `cd /c/russell`), and run the following script to download and compile the MUMPS solver (*use this script at your own risk; carefully check the scripts before running them*):

```bash
bash ./zscripts/windows-compile-mumps.bash
```

You may now test the code with MUMPS by running:

```bash
cargo test --features local_sparse
```
