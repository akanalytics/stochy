# Overview

Stochy is a collection of stochastic approximation algorithms - methods for optimizing systems with multiple unknown parameters.
The algos can be used with [stepwise](<https://crates.io/crates/stochy>) (built-in, the default) or with [argmin](<https://crates.io/crates/argmin>) (using feature-flag `argmin`).

Neither algo requires a gradient function, only the objective function to be minimised. Alternatively a proxy to relative differences of objective function evaluations can be specified. See [Relative difference functions](#relative_difference)

## RSPSA algorithm
Resiliant Simultaneous Perturbation Stochastic Approximation.
[`RspsaSolver`]

## SPSA algorithm
Simultaneous Perturbation Stochastic Approximation.
[`SpsaSolver`]



# Crate features
By default no features are enabled.

Features:
- `argmin`: enable integration with [argmin](<https://crates.io/crates/argmin>).

Example `Cargo.toml`
```toml
[dependencies]
stochy = { version = "0.0.1", features = ["argmin"] }
```


# Example
The [`stepwise::Driver`] has functional style iteration control methods,
along with a `solve` call which returns the algo (in a solved state) 
along with final iteration step.
The `on_step()` logging and progress bar are optional, and can be omitted.


```rust
# use std::time::Duration;
use stochy::{SpsaParams, SpsaSolver};
use stepwise::{fixed_iters, Driver, assert_approx_eq};

let f = |x: &[f64]| (x[0] - 1.5).powi(2) + x[1] * x[1];

let hyperparams = SpsaParams::default();
let algo = SpsaSolver::from_fn(hyperparams, &[1.0, 1.0], f).expect("hyperparams!");

let driver = fixed_iters(algo, 1000)
    .on_step(|algo,step| println!("{:?} {:?}", step, algo.x()))
    .show_progress_bar_after(Duration::from_millis(200));

let (solved, step) = driver.solve().expect("solving failed!");

assert_approx_eq!(solved.x(), &[1.5, 0.0], 1e-2);
```


# Example (Argmin)
In the below, `use` statements have been replaced by full qualification of paths, to make clear what
structs or functions come from which crates.

```rust
struct MySimpleCost;

# #[cfg(feature = "argmin")]
impl argmin::core::CostFunction for MySimpleCost {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok((x[0] - 1.5).powi(2) + x[1] * x[1])
    }
}

let hyperparams = stochy::SpsaParams::default();
let algo = stochy::SpsaSolverArgmin::new(hyperparams);

let exec = argmin::core::Executor::new(MySimpleCost, algo);

let initial_param = vec![1.0, 1.0];
let result = exec
    .configure(|step| step.param(initial_param).max_iters(1000))
    .run()
    .unwrap();

let best_param = result.state.best_param.unwrap();
stepwise::assert_approx_eq!(best_param.as_slice(), &[1.5, 0.0], 1e-2);
```


# Comparison


| Gradient Descent (for comparison) | RSPSA | SPSA |
| :--- | :--- | :--- |
| <a href="https://postimg.cc/nsHPN1tg">
   <img src="https://i.postimg.cc/nsHPN1tg/plot3d-Sphere-GD.png" width="300" /></a> | <a href="https://postimg.cc/5Yyk2Nym"><img src="https://i.postimg.cc/5Yyk2Nym/plot3d-Sphere-RSPSA.png" width="300" /></a> | <a href="https://postimg.cc/7CmW4TH6"><img src="https://i.postimg.cc/7CmW4TH6/plot3d-Sphere-SPSA.png" width="300" /></a> |
| Gradient function required | No gradient function required | No gradient function required |
| Gradient function required | Accepts relative difference function | Accepts relative difference function |
| One gradient eval per iteration | Two function evals per iteration | Two function evals per iteration |
| Single learning-rate hyperparameter | Very sensitive to hyperparameters | Less sensitive to hyperparameters than SPSA |
| Continuous convergence progression  | Convergence saturation | Continuous convergence progression | 



# <a name="relative_difference">Relative Differences</a>

Rather than specifying an objective function to be minimised (cost function), a relative difference function can be specified.

```text
df(x1, x2) ~ f(x2) - f(x1)
```
which permits use in cases where an abolute value of objective function is unavailable. Typically a game playing program would seek to minimise `-df` (and hence maximize `df`) where `x₁` and `x₂` represent game playing parameters, and the difference function df represents the outcome of a single game or a series of games.

```text
           / +1   x₂ win vs x₁ loss
df(x₁, x₂) =  0   drawn game
           \ -1   x₂ loss vs x₁ win
```








