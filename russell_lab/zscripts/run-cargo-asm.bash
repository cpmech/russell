#!/bin/bash

#cargo asm -C opt-level=3 -p russell_lab --example asm01 asm01::set_approach > /tmp/set_approach.asm
#cargo asm -C opt-level=3 -p russell_lab --example asm01 asm01::index_approach > /tmp/index_approach.asm
#meld /tmp/set_approach.asm /tmp/index_approach.asm

cargo asm -p russell_lab --example asm01 asm01::set_approach > /tmp/set_approach.asm
cargo asm -p russell_lab --example asm01 asm01::index_approach > /tmp/index_approach.asm
meld /tmp/set_approach.asm /tmp/index_approach.asm

