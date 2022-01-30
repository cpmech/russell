#!/bin/bash

set -e

cargo run --example distribution_frechet
cargo run --example distribution_gumbel
cargo run --example distribution_lognormal
cargo run --example distribution_normal
cargo run --example distribution_uniform
