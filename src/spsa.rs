use rand::rngs::StdRng;
use rand::SeedableRng as _;
use std::ops::ControlFlow;
use stepwise::{Algo, BoxedError, VectorExt as _};

use crate::{
    common::{rademacher, FuncKind},
    StochyError,
};

/// Hyperparameters for the [`SpsaAlgo`] algorithm.
///
/// | Hyperparameter | Default             | Explanation |
/// |----------------|---------------------|-------------|
/// | `random_seed`  | `Some(0)`           | Use `Some(seed)` for reproducible results. <br> Use `None` for entropy-based seeding. |
/// | `alpha`        | `0.602`             | Learning rate decay exponent. |
/// | `gamma`        | `0.101`             | Step size decay exponent. |
/// | `big_a`        | `0.01 × 1_000`      | Set to ~1–10% of the estimated number of iterations. |
/// | `a`            | `0.1 × (A + 1)^α / \|g₀\|` | Initial learning rate. <br> `\|g₀\|` is the magnitude of the initial gradient estimate. |
/// | `c`            | `0.01`              | Initial step size. <br> Use a larger value for noisy objective functions. |
///
/// ### Notes on Hyperparameter Selection
///
/// **Alpha and Gamma**  
/// Practically effective (and theoretically valid) values are `alpha = 0.602` and `gamma = 0.101`.  
/// Asymptotically optimal values are `alpha = 1.0`, `gamma = 1/6`, though these are rarely used in practice.
///
/// **Parameter `c`**  
/// In high-noise settings (i.e., poor quality measurements of gradient), use a **larger `c`** and a **smaller `a`**.  
/// A good rule of thumb is to set `c` approximately equal to the standard deviation of the noise in `f(x)`
/// (estimated by evaluating `f(x)` multiple times at the same `x`).  
/// If `f(x)` is noise-free, a small positive value for `c` suffices.
///
/// **Parameters `big_a` and `a`**  
/// Choose `big_a` such that it's **much smaller** than the maximum number of iterations (commonly 10% or less).  
/// Choose `a` such that:
///
/// ```text
/// a / (A + 1)^alpha × |g₀| ≈ smallest_desired_change_in_x
/// ```
///
/// where `|g₀|` is the magnitude of the initial gradient estimate,  
/// and `smallest_desired_change_in_x` is the minimal meaningful step size in the parameter vector `x`.
#[derive(Clone, Debug)]
#[allow(missing_docs)] // documentation of pub fields at struct level instead
pub struct SpsaParams {
    pub random_seed: Option<u64>,
    pub alpha: f64,
    pub gamma: f64,
    pub c: f64,
    pub big_a: f64,
    pub a: Option<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct SpsaState {
    rng: StdRng,
    x: Vec<f64>,
    iter: usize,
    ck: f64,
    ak: f64,
    a0: f64,
}

/// The traditional SPSA algorithm.
///
/// ### Overview
/// This algorithm minimizes a given function using stochastic approximation.  
/// It requires only the objective (cost) function — no gradient is needed.
///
/// ### Reference
///
/// *James C. Spall (1998)*  
/// “An Overview of the Simultaneous Perturbation Method for Efficient Optimization”  
/// <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>
///
/// ### Algorithm
/// See [`SpsaParams`] for hyperparameter details.
///
/// ```matlab
/// N  <--- estimated number of iterations  
/// g0 <--- estimated gradient  
/// α  <--- 0.602  
/// γ  <--- 0.101  
/// A  <--- 0.01 * N  
/// a0 <--- 0.1 * (A + 1)^α / ∥g0∥  
/// c  <--- 0.01  
/// x  <--- initial guess
///
/// for k = 1, 2, ..., N do  
///     ak ← a0 / (k + A)^α  
///     ck ← c / k^γ  
///     δk ← vector from the Rademacher distribution  
///      g ← [f(x + ck·δk) − f(x − ck·δk)] / (2·ck·δk)  
///      x ← x − ak · g  
/// end for
/// ```
///
/// - Requires only the objective function (not its gradient)  
/// - Also accepts a relative difference function, ideal for game-playing tasks without an absolute objective  
/// - Requires two function evaluations per iteration  
/// - Offers strong convergence properties  
/// - Is *very* sensitive to hyperparameters
///
/// ### Constrained theta (optional)
/// If minimum and maximum values for the parameter vector `theta` can be specified (say `thetamin` and `thetamax`),  
/// you may apply clipping after the update step:
///
/// ```matlab
/// theta = min(theta, thetamax);
/// theta = max(theta, thetamin);
/// ```
///
/// ### Example
/// The [`stepwise::Driver`] provides a functional-style iteration controller,  
/// and a `.solve()` method that returns a `(solved_algo, final_step)` tuple.  
/// The `on_step()` callback enables optional iteration-time logging.
///
/// ```rust
/// use stochy::{SpsaParams, SpsaAlgo};
/// use stepwise::{fixed_iters, Driver, assert_approx_eq};
///
/// let f = |x: &[f64]| (x[0] - 1.5).powi(2) + x.iter().skip(1).map(|x| x * x).sum::<f64>();
///
/// let hyperparams = SpsaParams::default();
/// let algo = SpsaAlgo::from_fn(hyperparams, vec![1.0, 1.0], f).expect("bad hyperparams!");
///
/// let (solved, step) = fixed_iters(algo, 1000)
///     .on_step(|algo, step| println!("{:?} {:?}", step, algo.x()))
///     .solve()
///     .expect("failed to solve");
///
/// assert_approx_eq!(solved.x(), &[1.5, 0.0], 1e-2);
/// assert_eq!(step.iteration(), 1000);
/// ```
///
#[derive()]
pub struct SpsaAlgo<'a> {
    pub(crate) params: SpsaParams,
    pub(crate) state: SpsaState,
    func: FuncKind<'a>,
}

impl Default for SpsaParams {
    /// see [`SpsaParams`] for default values.
    fn default() -> Self {
        Self {
            random_seed: Some(0),
            alpha: 0.602,
            gamma: 0.101,
            c: 0.01,                // 0.00001
            big_a: 0.01 * 1000_f64, // 10% of 1000 iterations
            a: None,                //
        }
    }
}

impl std::fmt::Debug for SpsaAlgo<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RspsaAlgo")
            .field("params", &self.params)
            .field("step", &self.state)
            .finish()
    }
}

/// Required by stepwise crate, for the [stepwise::Driver] to function
impl Algo for SpsaAlgo<'_> {
    type Error = StochyError;

    ///
    fn step(&mut self) -> (ControlFlow<()>, Result<(), Self::Error>) {
        if let Err(e) = Self::step_df(&self.params, &mut self.state, &mut self.func) {
            (ControlFlow::Break(()), Err(e))
        } else {
            (ControlFlow::Continue(()), Ok(()))
        }
    }
}

impl<'a> SpsaAlgo<'a> {
    // Returns a driver that runs the algorithm for a fixed number of iterations.
    // pub fn fixed_iters(self, fixed_iters: usize) -> stepwise::BasicDriver<Self> {
    //     stepwise::fixed_iters(self, fixed_iters)
    // }

    /// The current best solution vector.
    pub fn x(&self) -> &[f64] {
        &self.state.x
    }

    /// The current step size.
    /// Evolved from [`SpsaParams::c`]
    ///
    pub fn ck(&self) -> f64 {
        self.state.ck
    }

    /// The current learning rate.
    /// Evolved from [`SpsaParams::a`]
    pub fn ak(&self) -> f64 {
        self.state.ak
    }

    /// Creates a new instance from a given objective (cost) function.
    ///
    /// # Errors
    /// [`StochyError::InvalidHyperparameter`] if the configuration is invalid.
    ///
    /// # Example
    /// ```
    /// # use stochy::{SpsaAlgo, SpsaParams};
    /// let hyperparameters = SpsaParams::default();
    /// let initial_vec = vec![0.23, 0.45, 1.34];
    /// let f = | x: &[f64] | 3.0 * x[0]*x[0] - 2.0 * x[1] + x[2].powi(3);
    /// let algo = SpsaAlgo::from_fn(hyperparameters, initial_vec, f).expect("error!");
    /// ```
    pub fn from_fn<F>(
        params: SpsaParams,
        initial_guess: Vec<f64>,
        mut f: F,
    ) -> Result<Self, StochyError>
    where
        F: FnMut(&[f64]) -> f64 + 'a,
    {
        let df = move |x: &[f64], y: &[f64]| -> Result<f64, BoxedError> { Ok((f)(y) - (f)(x)) };
        Self::from_difference_fn(params, initial_guess, df)
    }

    /// Creates a new instance from a given falliable function (which may return an error).
    ///
    ///
    /// # Errors
    /// [`StochyError::InvalidHyperparameter`] if the configuration is invalid.
    ///
    /// # Example
    /// ```
    /// # use stochy::{SpsaAlgo, SpsaParams};
    /// # use std::error::Error;
    /// let hyperparameters = SpsaParams::default();
    /// let initial_vec = vec![0.23, 0.45, 1.34];
    /// let f = |x: &[f64]| -> Result<f64, Box<dyn Error + Send+ Sync + 'static>> {
    ///     Ok(3.0 * x[0]*x[0] - 2.0 * x[1] + x[2].powi(3))
    /// };
    /// let algo = SpsaAlgo::from_falliable_fn(hyperparameters, initial_vec, f).expect("error!");
    /// ```
    pub fn from_falliable_fn<F>(
        params: SpsaParams,
        initial_guess: Vec<f64>,
        mut f: F,
    ) -> Result<Self, StochyError>
    where
        F: FnMut(&[f64]) -> Result<f64, BoxedError> + 'a,
    {
        let df = move |x: &[f64], y: &[f64]| -> Result<f64, BoxedError> { Ok((f)(y)? - (f)(x)?) };
        Self::from_difference_fn(params, initial_guess, df)
    }

    /// Creates a new instance from a given difference function (useful for tuning game play).
    ///
    /// See [relative difference functions](https://docs.rs/stochy/latest/stochy/index.html#relative_difference)
    ///
    /// # Errors
    /// [`StochyError::InvalidHyperparameter`] if the configuration is invalid.
    pub fn from_difference_fn<F>(
        params: SpsaParams,
        initial_guess: Vec<f64>,
        df: F,
    ) -> Result<Self, StochyError>
    where
        F: FnMut(&[f64], &[f64]) -> Result<f64, BoxedError> + 'a,
    {
        Self::from_function_kind(params, initial_guess, FuncKind::Difference(Box::new(df)))
    }

    fn from_function_kind(
        params: SpsaParams,
        x0: Vec<f64>,
        func: FuncKind<'a>,
    ) -> Result<Self, StochyError> {
        if params.c < 0.0 {
            return Err(StochyError::InvalidHyperparameter("c < 0.0".to_string()))?;
        }
        let x = x0;
        let rng = match params.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };
        let iter = 0;
        let step = SpsaState {
            rng,
            x,
            iter,
            ck: 0.,
            ak: 0.,
            a0: 0.,
        };
        Ok(Self {
            params,
            state: step,
            func,
        })
    }

    pub(crate) fn step_df(
        cfg: &SpsaParams,
        step: &mut SpsaState,
        func: &mut FuncKind<'a>,
    ) -> Result<ControlFlow<()>, <Self as Algo>::Error> {
        let SpsaState {
            rng,
            x,
            iter,
            ck,
            ak,
            a0,
        } = step;

        *iter += 1;
        // self.f_value = f(self.x.as_slice())?;

        let i = *iter as f64;

        *ck = cfg.c / i.powf(cfg.gamma); // step size (i starts at 1, so ck starts at c)

        let rx = rademacher(x.len(), rng);
        let dx: Vec<_> = rx.into_iter().map(|x| x * *ck).collect();
        let delta_f = func.difference(&x.sub_vec(&dx), &x.add_vec(&dx))?;
        // self.f_value = (f1 + f2) / 2.0;

        let g: Vec<_> = dx.iter().map(|dx| delta_f / (2.0 * dx)).collect();
        let _norm_g = g.norm_l2();
        let magnitude_g = g
            .iter()
            .copied()
            .map(f64::abs)
            .reduce(f64::max)
            .unwrap_or_default()
            / x.len() as f64;
        // let initial_step_size = self.a0;
        if *iter == 1 {
            *a0 = cfg.a.unwrap_or_else(|| {
                let initial_lr = 0.01;
                initial_lr * (cfg.big_a + 1.).powf(cfg.alpha) / magnitude_g
            });
        }
        *ak = *a0 / (i + cfg.big_a).powf(cfg.alpha); // learning rate

        #[allow(clippy::needless_range_loop)]
        for i in 0..x.len() {
            let g = delta_f / (2.0 * dx[i]);
            x[i] -= g * *ak;
            // bound X
        }

        Ok(ControlFlow::Continue(()))
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;
    use log::trace;
    use stepwise::{
        assert_approx_eq, fixed_iters,
        problems::{sigmoid, sphere},
    };
    use test_log::test;

    #[test]
    fn spsa_trad() -> Result<(), BoxedError> {
        let w = [1.0, 2.0];
        let paraboloid = |x: &[f64]| (x[0] - w[0]).powi(2) + 6. * (x[1] - w[1]).powi(2);
        let w0 = vec![5.2, 6.4];
        let cfg = SpsaParams {
            a: Some(0.3),
            c: 0.2,
            ..SpsaParams::default()
        };

        let algo = SpsaAlgo::from_fn(cfg, w0, paraboloid).unwrap();
        trace!("{algo:#?}");
        let (solved, step) = fixed_iters(algo, 300)
            .on_step(|s, _v| trace!("{s:?}"))
            .solve()?;
        let x = solved.x();
        assert_eq!(step.iteration(), 300);
        assert_approx_eq!(x, &w, 1e-3);
        Ok(())
    }

    #[test]
    fn spsa_sphere() -> Result<(), Box<dyn Error>> {
        let w0 = [0.0, 0.0];
        let mut total_iters = 0;
        for x0 in -10..10 {
            for y0 in -10..10 {
                if x0 == y0 || x0 == -y0 {
                    print!("{:>4}", '-');
                    continue;
                }
                let z0 = vec![w0[0] + 0.01 * f64::from(x0), w0[0] + 0.01 * f64::from(y0)];
                let cfg = SpsaParams::default();

                // need to specify types so that lifetimes can be elided
                let obj_func = |x: &[f64], y: &[f64]| Ok(sigmoid(sphere(y)) - sigmoid(sphere(x)));
                let algo = SpsaAlgo::from_difference_fn(cfg, z0, obj_func)?;
                let (solved, step) = fixed_iters(algo, 2_000)
                    .converge_when(|v, _s| v.x().dist_max(&w0) < 1e-6)
                    .fail_if(|_v, s| s.iteration() > 1000)
                    .solve()?;
                print!("{:>4}", step.iteration());
                assert!(
                    step.iteration() < 600,
                    "x0 = {x0} y0 = {y0} iters = {}",
                    step.iteration()
                );
                total_iters += step.iteration();
                assert!(solved.x().dist_max(&w0) < 1e-6);
            }
            println!();
        }
        println!("total iters {total_iters}");
        Ok(())
    }
}
