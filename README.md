
[![Latest version](https://img.shields.io/crates/v/stochy.svg)](https://crates.io/crates/stochy)
[![Documentation](https://docs.rs/stochy/badge.svg)](https://docs.rs/stochy)
[![License](https://img.shields.io/crates/l/stochy.svg)](https://choosealicense.com/licenses/)
[![msrv](https://img.shields.io/crates/msrv/stochy)](https://www.rust-lang.org)

# Overview

The `stochy` crate is a collection of stochastic approximation algorithms:

- [`RSPSA`](https://docs.rs/stochy/latest/stochy/struct.RspsaAlgo.html) (Resilient Simultaneous Perturbation Stochastic Approximation) 
- [`SPSA`](https://docs.rs/stochy/latest/stochy/struct.SpsaAlgo.html) (Simultaneous Perturbation Stochastic Approximation) 

You can use `stochy` to:

- Minimize functions with multiple parameters, without requiring a gradient.
- Optimize parameters in game-playing programs using [relative difference functions](https://docs.rs/stochy/latest/stochy/#relative_difference).

`stochy` is compatible with both the [stepwise](https://crates.io/crates/stepwise) algorithm API and 
the [argmin](https://crates.io/crates/argmin) solver API (enable via the `argmin` feature flag). Difference functions are only supported under `stepwise`.

- [Documentation](https://docs.rs/stochy)
- [Changelog](https://github.com/akanalytics/stochy/blob/main/CHANGELOG.md)
- [Releases](https://github.com/akanalytics/stochy/releases)


# Usage

Example `Cargo.toml`:

```toml
[dependencies]
stochy = "0.0.3" 

# if using argmin, replace the above with:
# stochy = { version = "0.0.3", features = ["argmin"] } 
```

## Example 1

```rust
use stepwise::{Driver as _, fixed_iters, assert_approx_eq};
use stochy::{SpsaAlgo, SpsaParams};

let f = |x: &[f64]| (x[0] - 1.5).powi(2) + x[1].powi(2);

let hyperparams = SpsaParams::default();
let initial_guess = vec![1.0, 1.0];
let spsa = SpsaAlgo::from_fn(hyperparams, initial_guess, f).expect("bad hyperparams!");

let (solved, final_step) = fixed_iters(spsa, 20_000)
    .on_step(|algo, step| println!("{} {:?}", step.iteration(), algo.x()))
    .solve()
    .expect("solving failed!");

assert_approx_eq!(solved.x(), &[1.5, 0.0]);
println!("Solved in {} iterations.", final_step.iteration());
```


## Example 2 (argmin)
This example is equivalent to Example 1, but uses the `argmin` crate to manage the SPSA algorithm. 

```rust
use stepwise::assert_approx_eq;
struct MySimpleCost;

# #[cfg(feature = "argmin")]
impl argmin::core::CostFunction for MySimpleCost {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok((x[0] - 1.5).powi(2) + x[1].powi(2))
    }
}

let hyperparams = stochy::SpsaParams::default();
let algo = stochy::SpsaSolverArgmin::new(hyperparams);

let exec = argmin::core::Executor::new(MySimpleCost, algo);

let initial_param = vec![1.0, 1.0];
let result = exec
    .configure(|step| step.param(initial_param).max_iters(20_000))
    .run()
    .unwrap();

let best_param = result.state.best_param.unwrap();
assert_approx_eq!(best_param.as_slice(), &[1.5, 0.0]);
println!("Solved in {} iterations.", result.state.iter);
```


# Comparison
Table 1: Feature comparison of the algorithms contrasted with the more familiar Gradient Descent algorithm.

| Gradient Descent (reference) | RSPSA | SPSA |
| :--- | :--- | :--- |
| <a href="https://postimg.cc/nsHPN1tg">
   <img src="https://i.postimg.cc/nsHPN1tg/plot3d-Sphere-GD.png" width="300" /></a> | <a href="https://postimg.cc/5Yyk2Nym"><img src="https://i.postimg.cc/5Yyk2Nym/plot3d-Sphere-RSPSA.png" width="300" /></a> | <a href="https://postimg.cc/7CmW4TH6"><img src="https://i.postimg.cc/7CmW4TH6/plot3d-Sphere-SPSA.png" width="300" /></a> |
| Requires gradient function | No gradient function required | No gradient function required |
| Difference function insufficient | Accepts relative difference function | Accepts relative difference function |
| One gradient evaluation per iteration | Two function evaluations per iteration | Two function evaluations per iteration |
| Single learning-rate hyperparameter | Less sensitive to hyperparameters than SPSA | Very sensitive to hyperparameters |
| Continuous convergence progression  | Convergence saturation | Continuous convergence progression | 






