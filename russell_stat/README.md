# Russell Stat - Statistics calculations, probability distributions, and pseudo random numbers

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

This repository contains structures and functions to work with statistics and probability distributions. Internally, we wrap and use the [rand_distr](https://docs.rs/rand_distr/latest/rand_distr/) crate for generating samples.

Documentation:

- [API reference (docs.rs)](https://docs.rs/russell_stat)

## Installation

Add this to your Cargo.toml (choose the right version):

```toml
[dependencies]
russell_stat = "*"
```

## Examples

```rust
use russell_stat::{statistics, DistributionFrechet, Histogram, ProbabilityDistribution, StrError};


fn main() -> Result<(), StrError> {
    // generate samples
    let mut rng = rand::thread_rng();
    let dist = DistributionFrechet::new(0.0, 1.0, 1.0)?;
    let nsamples = 10_000;
    let mut data = vec![0.0; nsamples];
    for i in 0..nsamples {
        data[i] = dist.sample(&mut rng);
    }
    println!("{}", statistics(&data));

    // text-plot
    let stations = (0..20).map(|i| (i as f64) * 0.5).collect::<Vec<f64>>();
    let mut hist = Histogram::new(&stations)?;
    hist.count(&data);
    println!("{}", hist);
    Ok(())
}
```

Example output

```text
min = 0.09073675834496424
max = 7694.599272007603
mean = 10.392955760859788
std_dev = 137.11729225249485

[  0,0.5) | 1407 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦
[0.5,  1) | 2335 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦
[  1,1.5) | 1468 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦
[1.5,  2) |  913 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦
[  2,2.5) |  640 🟦🟦🟦🟦🟦🟦🟦🟦
[2.5,  3) |  420 🟦🟦🟦🟦🟦
[  3,3.5) |  358 🟦🟦🟦🟦
[3.5,  4) |  269 🟦🟦🟦
[  4,4.5) |  207 🟦🟦
[4.5,  5) |  185 🟦🟦
[  5,5.5) |  143 🟦
[5.5,  6) |  137 🟦
[  6,6.5) |   99 🟦
[6.5,  7) |   76
[  7,7.5) |   88 🟦
[7.5,  8) |   73
[  8,8.5) |   51
[8.5,  9) |   63
[  9,9.5) |   53
      sum = 8985
```
