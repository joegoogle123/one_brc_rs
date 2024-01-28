#!/bin/sh
cargo build --release
hyperfine  --warmup 1 --runs 5 './target/release/one_brc_rs measurements.txt'
