#!/bin/bash

sudo apt-get install pkg-config libssl-dev
cargo install cargo-workspaces

cargo ws --help
