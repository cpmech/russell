#!/bin/bash

cd russell_lab
cargo publish --features intel_mkl
cd ..
sleep 5

cd russell_sparse
cargo publish --features local_suitesparse,with_mumps,intel_mkl
cd ..
sleep 5

cd russell_ode
cargo publish --features local_suitesparse,with_mumps,intel_mkl
cd ..
sleep 5

cd russell_stat
cargo publish --features intel_mkl
cd ..
sleep 5

cd russell_tensor
cargo publish --features intel_mkl
cd ..

echo
echo "All done!"