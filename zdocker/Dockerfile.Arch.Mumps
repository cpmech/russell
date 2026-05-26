FROM archlinux:base-devel

# Update keyring first to avoid GPG errors on a fresh Arch image,
# then upgrade the system and install Rust and the required libraries
RUN pacman -Sy --noconfirm archlinux-keyring && \
    pacman -Syu --noconfirm rust blas-openblas suitesparse

# Create a non-root user for building and testing
RUN useradd -m user

# Copy project files; --chown ensures the destination directory and all
# files are owned by user, so cargo can write target/ during the build
COPY --chown=user:user . /usr/local/src/russell/
WORKDIR /usr/local/src/russell

# Compile and install MUMPS with OpenBLAS backend; must run as root
# (before USER user) because pacman and ldconfig need it
RUN bash zscripts/arch-compile-mumps.bash

USER user

# Run tests
RUN cargo test --features with_mumps
