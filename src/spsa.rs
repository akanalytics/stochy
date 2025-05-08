use rand::rngs::StdRng;
use rand::SeedableRng as _;
use std::ops::ControlFlow;
use stepwise::{Algo, VectorExt as _};

use crate::{
    common::{rademacher, FuncKind},
    BoxedError, StochyError,
};

/// Hyperparameters for the [SPSA](SpsaSolver) algorithm
///
/// | Hyperparameter | Default | Explanation |
/// | :--- | :--- | :--- |
/// | random seed  |   Some(0)                         | `Some(seed)` for reproducible results.<br> Use `None` for an (arbitrary) entropy based seed |
/// | alpha        |   0.602                           | learning rate decay |
/// | gamma        |   0.101                           | step size decay |
/// | A (big_a)    |   0.01 x 1_000                    | recommended set to 0.01 x estimate_iterations (up to 0.1 x estimate_iterations) |
/// | a            |   0.1 (A + 1)^α / \|g0\|          | is the initial lr.<br>\|g0\| is the magnitutde of initial gradient estimate |
/// | c            |   0.01                            | initial step size (use larger for noisy objective functions) |
///
/// Spall's comments on hyperparameters:
///
/// * **Alpha and Gamma:**
///   Practically effective (and theoretically valid) values for alpha = 0.602 and gamma = 0.101,
///   respectively (the asymptotically optimal values of 1.0 and 1/6 may also be used);
///
/// * **Parameter c:**
///   One typically finds that in a high-noise setting (i.e., poor quality measurements of gradient)
///   it is necessary to pick a smaller a and larger c than in a low-noise setting.
///
///   As a rule-of-thumb (with the Bernoulli distribution for the elements of aₖ as suggested in Step 1),
///   it is effective to set c at a level approximately equal to the standard deviation of the measurement
///   noise in y(θ) in order to keep the p elements of ĝₖ(θₖ) from getting excessively large in magnitude
///   (the standard deviation can be estimated by collecting several y(θ) values at the initial guess θ₀;
///   a precise estimate is not required in practice).
///
///   In the case where one had perfect measurements of L(θ), then c should be chosen as some small positive number.
///
/// * **Parameters `big_a` and a:**
///   As guideline we have found useful is to choose A such that it is much less than the maximum number
///   of iterations allowed or expected, e.g., we frequently take it to be 10% (or less) of the maximum number
///   of expected/allowed iterations
///
///   and choose a such that ```a/(A + 1)^alpha``` times the magnitude of elements in ghat(θ₀) is approximately equal
///   to the smallest of the desired change magnitudes among the elements of θ in the early iterations.
///     
///   a = `smallest_desired_change_in_x` * (A+1)^alpha / `magnitude_of_initial_gradient`
///     
#[derive(Clone, Debug)]
#[allow(missing_docs)] // TODO
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

/// The traditional SPSA algorithm
///
/// ### Overview
/// The algorithm minimizes a given function using stochastic approximation. It requires only
/// the objective (cost) function, not a gradient.
///
/// ### Paper
///
/// "An Overview of the Simultaneous Perturbation Method for Efficient Optimization"
///
/// *James CSpall (1998)*
///
/// <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>
///
/// ### Algorithm
///
/// (see [`SpsaParams`] for details on the hyperparameters)
///
/// ```Matlab
/// N  <--- estimated number of iterations
/// g0 <--- estimated gradient
/// α  <--- 0.602
/// γ  <--- 0.101
/// A  <--- 0.01N
/// a0 <--- 0.1(A + 1)^α/∥g0∥
/// c  <--- 0.01
/// x  <--- initial guess
///
/// for k = 1,2,...,N do
///     ak <--- a0/(k + A)^α
///     ck <--- c/k^γ
///     δk <--- vector of rademacher distribution
///      g <--- [f(x + ck.δk) - f(x - ck.δk)] / 2
///      x <--- x - ak.g
/// end for
/// ```
///
/// - requires only the objective function, not the gradient function
/// - also accept a relative difference function, ideal for game-play, without an absolute objective function
/// - requires two function evaluations per iteration
/// - offers good convergence
/// - is *very* sensitive to hyper-parameters
///
///
///  theta
///    If maximum and minimum values on the values of theta
///    can be specified, say thetamax and thetamin, then the
///    following two lines can be added below the theta update
///    line to impose the constraints
///
/// ```matlab
///    theta = min(theta, thetamax);
///    theta = max(theta, thetamin);
/// ```
///
/// /// ### Example
/// The [`stepwise::Driver`] has functional style iteration control methods,
/// along with a `solve` which returns a pair-tuple of the algo (in a solved state)
/// and with final iteration step.
/// The on_step() logging is optional and can be omitted.
///
///
/// ```rust
/// use stochy::{SpsaParams, SpsaSolver};
/// use stepwise::{fixed_iters, Driver, assert_approx_eq};
///
/// let f = |x: &[f64]| (x[0] - 1.5).powi(2) + x.iter().skip(1).map(|x| x * x).sum::<f64>();
///
/// let hyperparams = SpsaParams::default();
/// let algo = SpsaSolver::from_fn(hyperparams, &[1.0, 1.0], f).expect("bad hyperparams!");
///
/// let (solved, step) = fixed_iters(algo, 1000)
///     .on_step(|algo,step| println!("{:?} {:?}", step, algo.x()))
///     .solve()
///     .expect("failed to solve");
///
/// assert_approx_eq!(solved.x(), &[1.5, 0.0], 1e-2);
/// assert_eq!(step.iteration(), 1000);
/// ```
///
#[derive()]
pub struct SpsaSolver<'a> {
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

impl std::fmt::Debug for SpsaSolver<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RspsaSolver")
            .field("params", &self.params)
            .field("step", &self.state)
            .finish()
    }
}

/// Required by stepwise crate, for the [stepwise::Driver] to function
impl Algo for SpsaSolver<'_> {
    ///
    type Error = StochyError;

    ///
    fn step(&mut self) -> ControlFlow<Result<(), Self::Error>, Result<(), Self::Error>> {
        if let Err(e) = Self::step_df(&self.params, &mut self.state, &mut self.func) {
            ControlFlow::Break(Err(e))
        } else {
            ControlFlow::Continue(Ok(()))
        }
    }
}

impl<'a> SpsaSolver<'a> {
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
    /// # use stochy::{SpsaSolver, SpsaParams};
    /// let hyperparameters = SpsaParams::default();
    /// let initial_vec = [0.23, 0.45, 1.34];
    /// let f = | x: &[f64] | 3.0 * x[0]*x[0] - 2.0 * x[1] + x[2].powi(3);
    /// let algo = SpsaSolver::from_fn(hyperparameters, &initial_vec, f).expect("error!");
    /// ```
    pub fn from_fn<F>(
        params: SpsaParams,
        initial_guess: &[f64],
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
    /// # use stochy::{SpsaSolver, SpsaParams};
    /// # use std::error::Error;
    /// let hyperparameters = SpsaParams::default();
    /// let initial_vec = [0.23, 0.45, 1.34];
    /// let f = |x: &[f64]| -> Result<f64, Box<dyn Error + Send+ Sync + 'static>> {
    ///     Ok(3.0 * x[0]*x[0] - 2.0 * x[1] + x[2].powi(3))
    /// };
    /// let algo = SpsaSolver::from_falliable_fn(hyperparameters, &initial_vec, f).expect("error!");
    /// ```
    pub fn from_falliable_fn<F>(
        params: SpsaParams,
        initial_guess: &[f64],
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
    /// The function is a relative difference function `df: (&[f64], &[f64]) -> Result<f64>`
    ///
    /// where df(x1, x2) ~ f(x2) - f(x1)
    ///
    /// this permits use in cases where an abolute value of objective function is unavailable.
    /// Typically a game playing program would seek to minimise `-df` (and hence maximize `df`)
    /// where `x₁` and `x₂` represent game playing parameters.
    ///
    /// ```math
    ///    / +1 x₂ win vs x₁ loss
    /// df =  0 drawn game
    ///    \ -1 x₂ loss vs x₁ win
    /// ```
    ///
    /// # Errors
    /// [`StochyError::InvalidHyperparameter`] if the configuration is invalid.
    pub fn from_difference_fn<F>(
        params: SpsaParams,
        initial_guess: &[f64],
        df: F,
    ) -> Result<Self, StochyError>
    where
        F: FnMut(&[f64], &[f64]) -> Result<f64, BoxedError> + 'a,
    {
        Self::from_function_kind(params, initial_guess, FuncKind::Difference(Box::new(df)))
    }

    fn from_function_kind(
        params: SpsaParams,
        x0: &[f64],
        func: FuncKind<'a>,
    ) -> Result<Self, StochyError> {
        if params.c < 0.0 {
            return Err(StochyError::InvalidHyperparameter("c < 0.0".to_string()))?;
        }
        let x = Vec::<f64>::from(x0);
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
        Driver,
    };
    use test_log::test;

    #[test]
    fn spsa_trad() -> Result<(), BoxedError> {
        let w = [1.0, 2.0];
        let paraboloid = |x: &[f64]| (x[0] - w[0]).powi(2) + 6. * (x[1] - w[1]).powi(2);
        let w0 = [5.2, 6.4];
        let cfg = SpsaParams {
            a: Some(0.3),
            c: 0.2,
            ..SpsaParams::default()
        };

        let algo = SpsaSolver::from_fn(cfg, &w0, paraboloid).unwrap();
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
                let z0 = [w0[0] + 0.01 * f64::from(x0), w0[0] + 0.01 * f64::from(y0)];
                let cfg = SpsaParams::default();

                // need to specify types so that lifetimes can be elided
                let obj_func = |x: &[f64], y: &[f64]| Ok(sigmoid(sphere(y)) - sigmoid(sphere(x)));
                let algo = SpsaSolver::from_difference_fn(cfg, &z0, obj_func)?;
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
