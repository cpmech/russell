#!/bin/bash
#
# Publishes every crate, in dependency order. `cargo publish` is run with the
# default features: feature selection only affects the verification build, never
# the published artifact, and enabling `intel_mkl`/`umfpack`/`mumps` would make a
# release depend on those libraries being installed on this machine.
set -euo pipefail

cd russell_lab
cargo publish
cd ..
sleep 5

cd russell_sparse
cargo publish
cd ..
sleep 5

cd russell_stat
cargo publish
cd ..
sleep 5

cd russell_tensor
cargo publish
cd ..
sleep 5

cd russell_pde
cargo publish
cd ..
sleep 5

cd russell_ode
cargo publish
cd ..
sleep 5

cd russell_nonlin
cargo publish
cd ..
sleep 5

echo
echo "All done!"