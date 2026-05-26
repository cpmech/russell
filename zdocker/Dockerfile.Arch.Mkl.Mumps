FROM archlinux:base-devel

# Update keyring first to avoid GPG errors on a fresh Arch image,
# then upgrade the system and install Rust and core dependencies;
# suitesparse is installed without blas-openblas because MKL will be used instead
RUN pacman -Sy --noconfirm archlinux-keyring && \
    pacman -Syu --noconfirm rust suitesparse

# Create a non-root user for building and testing
RUN useradd -m user

# Give user passwordless sudo so yay/makepkg can install AUR packages
RUN echo 'user ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

# Copy project files; --chown ensures the destination directory and all
# files are owned by user, so cargo can write target/ during the build
COPY --chown=user:user . /usr/local/src/russell/
WORKDIR /usr/local/src/russell

# Install the Intel oneAPI toolkit (MKL + compilers);
# must run as root because pacman needs it
RUN bash zscripts/arch-install-intel-toolkit.bash

# Install yay (AUR helper) needed for the metis package from AUR;
# yay-bin is a pre-compiled binary so only git + base-devel are needed;
# must build as non-root (user) because makepkg refuses to run as root
RUN pacman -S --noconfirm git && \
    su - user -c "git clone https://aur.archlinux.org/yay-bin.git /tmp/yay-bin && \
        cd /tmp/yay-bin && \
        makepkg -si --noconfirm"

# Compile and install MUMPS with Intel MKL backend;
# must run as root because pacman and ldconfig need it
RUN bash zscripts/arch-compile-mumps.bash 1

# Set the user
USER user

# MKL requires sourcing setvars.sh in the same shell as cargo test
ENV RUST_BACKTRACE=1
RUN bash -c "source /opt/intel/oneapi/setvars.sh && cargo test --features intel_mkl,local_sparse"
