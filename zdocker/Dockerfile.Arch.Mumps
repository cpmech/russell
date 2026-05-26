FROM archlinux:base-devel

# Update keyring first to avoid GPG errors on a fresh Arch image,
# then upgrade the system and install Rust and core dependencies;
# blas-openblas must be in the same transaction as suitesparse so that
# pacman resolves suitesparse's blas dependency via blas-openblas instead
# of first installing the default blas package (which would then conflict)
RUN pacman -Sy --noconfirm archlinux-keyring && \
    pacman -Syu --noconfirm rust blas-openblas suitesparse

# Create a non-root user for building and testing
RUN useradd -m user

# Give user passwordless sudo so yay/makepkg can install AUR packages
RUN echo 'user ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

# Copy project files; --chown ensures the destination directory and all
# files are owned by user, so cargo can write target/ during the build
COPY --chown=user:user . /usr/local/src/russell/
WORKDIR /usr/local/src/russell

# Install yay (AUR helper) needed for the metis package from AUR;
# yay-bin is a pre-compiled binary so only git + base-devel are needed;
# must build as non-root (user) because makepkg refuses to run as root
RUN pacman -S --noconfirm git && \
    su - user -c "git clone https://aur.archlinux.org/yay-bin.git /tmp/yay-bin && \
        cd /tmp/yay-bin && \
        makepkg -si --noconfirm"

# Compile and install MUMPS with OpenBLAS backend;
# must run as root because pacman and ldconfig need it
RUN bash zscripts/arch-compile-mumps.bash 0

# Set the user
USER user

# Test the code
ENV RUST_BACKTRACE=1
RUN cargo test --features local_sparse
