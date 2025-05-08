//! RSPSA algorithm
//!
//!

use rand::rngs::StdRng;
use rand::SeedableRng as _;
use std::error::Error;
use std::iter::zip;
use std::ops::ControlFlow;
use stepwise::Algo;
use stepwise::VectorExt as _;

use crate::common::rademacher;
use crate::common::FuncKind;
use crate::BoxedError;
use crate::StochyError;

/// Hyperparameters for the [RSPSA](RspsaSolver) algorithm
///
/// | Hyperparameter | Default | Explanation |
/// |:---|:---|:---|
/// | random seed | Some(0) | Some(seed) for reproducible results.<br>Use None for an (arbitrary) entropy based seed |
/// | epa_p | 1.1 | Step size growth for same sign df |
/// | epa_m | 0.85 | Step size growth for sign reversals |
/// | delta0 | 0.01 | Initial step size |
/// | delta_min | 1e-6 | Min delta |
///
#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub struct RspsaParams {
    pub random_seed: Option<u64>,
    pub eta_p: f64,
    pub eta_m: f64,
    pub delta0: f64,
    pub delta_min: f64,
    pub delta_max: Option<f64>,
    pub rho: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct RspsaState {
    rng: StdRng,
    x: Vec<f64>,
    delta: Vec<f64>,
    g: Vec<f64>,
    iter: usize,
}

/// RSPSA - the Resiliant SPSA algorithm
///
/// ### Overview
/// The algorithm minimizes a given function using stochastic approximation. It requires only
/// the objective (cost) function, not a gradient.
///
/// ### Paper
///
/// "RSPSA: Enhanced Parameter Optimisation in Games"
///
/// *Levente Kocsis, Csaba Szepesvari, Mark H.M. Winands*
///
/// <https://sites.ualberta.ca/~szepesva/papers/rspsa_acg.pdf>
///
/// ### Algorithm
///
/// (sign differs from paper as we are *minimising* the objective function not maximising)
///
/// θ\[t+1,i\] = θ\[t,i\] + sign(g\[ti\]) * δ\[ti\]  for  t = 1, 2,... and  i = 1, 2,...,d
///
/// where
///
/// δ\[ti\] ≥ 0 is the step size for the i-th component and
///
/// g\[t·\] is a gradient-like quantity:
///
/// g\[ti\] = I(g\[t−1,i\] * f'\[i\](θ\[t\]) ≥ 0) * f\[i\](θ\[t\])
/// gti equals the ith partial derivative of f at θ except when a sign reversal is
/// observed between the current and the previous partial derivative, in which case
/// gti is set to zero
///
///
/// p\[ti\] = g\[t−1,i\] * f'\[i\](θ\[t\])
///
/// η\[ti\] = I(p\[ti\] > 0) * η+
///       + I(p\[ti\] < 0) * η−
///       + I(p\[ti\] = 0) * 1.
///
/// δ\[ti\] = P\[δ−,δ+\] * η\[ti\] *  δ\[t−1,i\]
///
/// ### Example
/// The [`stepwise::Driver`] has functional style iteration control methods,
/// along with a `solve` which returns a pair-tuple of the algo (in a solved state)
/// and with final iteration step.
/// The on_step() logging is optional and can be omitted.
///
///
/// ```rust
/// use stochy::{RspsaParams, RspsaSolver};
/// use stepwise::{fixed_iters, Driver, assert_approx_eq};
///
/// let f = |x: &[f64]| (x[0] - 1.5).powi(2) + x.iter().skip(1).map(|x| x * x).sum::<f64>();
///
/// let hyperparams = RspsaParams::default();
/// let algo = RspsaSolver::from_fn(hyperparams, &[1.0, 1.0], f).expect("bad hyperparams!");
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
pub struct RspsaSolver<'a> {
    pub(crate) params: RspsaParams,
    pub(crate) state: RspsaState,
    func: FuncKind<'a>,
}

impl RspsaParams {
    /// either the delta_max specified or 100 * delta0 if delta_max is None
    pub fn delta_max(&self) -> f64 {
        self.delta_max.unwrap_or(100.0 * self.delta0)
    }
}

impl Default for RspsaParams {
    /// see [`RspsaParams`] for default values.
    fn default() -> Self {
        Self {
            random_seed: Some(0),
            eta_p: 1.1,
            eta_m: 0.85,
            delta0: 0.01,
            delta_min: 1e-6,
            delta_max: None,
            rho: 0.5,
        }
    }
}

impl std::fmt::Debug for RspsaSolver<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RspsaSolver")
            .field("params", &self.params)
            .field("step", &self.state)
            .finish()
    }
}

/// Required by stepwise crate, for the [stepwise::Driver] to function
impl Algo for RspsaSolver<'_> {
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

impl<'a> RspsaSolver<'a> {
    /// Returns the current best solution vector.
    pub fn x(&self) -> &[f64] {
        &self.state.x
    }

    /// Creates a new instance from a given function.
    ///
    ///
    /// # Errors
    /// [`StochyError::InvalidHyperparameter`] if the configuration is invalid.
    ///
    /// # Example
    /// ```
    /// # use stochy::{RspsaSolver, RspsaParams};
    ///
    /// let hyperparameters = RspsaParams::default();
    /// let initial_vec = [0.23, 0.45, 1.34];
    /// let f = |x: &[f64]| 3.0 * x[0]*x[0] - 2.0 * x[1] + x[2].powi(3);
    ///
    /// let algo = RspsaSolver::from_fn(hyperparameters, &initial_vec, f).unwrap();
    /// ```    
    ///
    pub fn from_fn<F>(params: RspsaParams, initial_guess: &[f64], f: F) -> Result<Self, StochyError>
    where
        F: FnMut(&[f64]) -> f64 + 'a,
    {
        Self::from_function_kind(params, initial_guess, FuncKind::Fn(Box::new(f)))
    }

    /// Creates a new instance from a given falliable function (which may return an error).
    ///
    ///
    /// # Errors
    /// [`StochyError::InvalidHyperparameter`] if the configuration is invalid.
    ///
    /// # Example
    /// ```
    /// # use stochy::{RspsaSolver, RspsaParams};
    /// # use std::error::Error;
    ///
    ///    let hyperparameters = RspsaParams::default();
    ///    let initial_vec = [0.23, 0.45, 1.34];
    ///    let f = |x: &[f64]| -> Result<f64, Box<dyn Error + Send + Sync + 'static>> {
    ///       Ok(3.0 * x[0]*x[0] - 2.0 * x[1] + x[2].powi(3))
    ///    };
    ///    let algo = RspsaSolver::from_falliable_fn(hyperparameters, &initial_vec, f).unwrap();
    /// ```    
    ///
    /// [`StochyError::InvalidHyperparameter`] if the configuration is invalid.
    pub fn from_falliable_fn<F>(
        params: RspsaParams,
        initial_guess: &[f64],
        mut f: F,
    ) -> Result<Self, StochyError>
    where
        F: FnMut(&[f64]) -> Result<f64, BoxedError> + 'a,
    {
        Self::from_function_kind(
            params,
            initial_guess,
            FuncKind::Falliable(Box::new(move |x| f(x))),
        )
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
    pub fn from_difference_fn<F, E>(
        params: RspsaParams,
        initial_guess: &[f64],
        mut df: F,
    ) -> Result<Self, StochyError>
    where
        F: FnMut(&[f64], &[f64]) -> Result<f64, E> + 'a,
        E: Error + Send + Sync + 'static,
    {
        Self::from_function_kind(
            params,
            initial_guess,
            FuncKind::Difference(Box::new(move |x, y| {
                df(x, y).map_err(|e| Box::new(e) as BoxedError)
            })),
        )
    }

    fn from_function_kind(
        cfg: RspsaParams,
        initial_guess: &[f64],
        func: FuncKind<'a>,
    ) -> Result<Self, StochyError> {
        if cfg.delta0 > cfg.delta_max() {
            let e = StochyError::InvalidHyperparameter("delta0 > delta_max".to_string());
            return Err(e);
        }
        let x = Vec::<f64>::from(initial_guess);
        let g = vec![0.0; x.len()];
        let delta = vec![cfg.delta0; x.len()];

        // trace!(
        //     "[   0] cfg = {cfg:?} x = {x:+.6?} g = {g:+11.6?} |g| = {norm_g:10.6} delta = {delta:+11.6?}",
        //     g = g.as_slice(),
        //     x = x.as_slice(),
        //     delta = delta.as_slice(),
        //     norm_g = g.norm_l2(),
        // );

        let rng = match cfg.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };
        let iter = 0;
        let step = RspsaState {
            rng,
            x,
            // prior_x: <[f64]>::nan(initial_guess.len()),
            delta,
            g,
            iter,
        };
        Ok(Self {
            params: cfg,
            state: step,
            func,
        })
    }

    // // for armin init
    // pub(crate) fn init<F>(&mut self, mut f: F, initial_guess: &[f64])
    // where
    //     F: FnMut(&[f64]) -> f64 + 'a,
    // {
    //     let df = move |x: &[f64], y: &[f64]| -> Result<f64, BoxError> {
    //         Ok((f)(y) - (f)(x))
    //     };

    //     self.df = Box::new(df);
    //     self.x = Vec::<f64>::from(initial_guess);
    //     self.g = vec![0.0; self.x.len()];
    //     self.delta = vec![self.cfg.delta0; self.x.len()];
    // }

    // fn est_grad<R>(x: &Vec<f64>, f: Func, rng: &mut R, delta: &Vec<f64>) -> Vec<f64>
    // where
    //     R: rand::Rng + ?Sized,
    // {
    //     let dx = Vec::rademacher(x.len(), rng).component_mul(delta);
    //     let df = f(x.component_add(&dx).as_slice()) - f((x.component_sub(&dx)).as_slice());
    //     Vec::vec_div(df, &dx).div(2.0)
    // }
    pub(crate) fn step_df(
        cfg: &RspsaParams,
        step: &mut RspsaState,
        func: &mut FuncKind<'a>,
    ) -> Result<ControlFlow<()>, <Self as Algo>::Error> {
        let RspsaState {
            x,
            delta,
            g,
            iter,
            rng,
        } = step;
        let rx = rademacher(x.len(), rng);
        let dx: Vec<_> = zip(&rx, &mut *delta).map(|(a, b)| *a * *b).collect();
        let delta_f = func.difference(&x.sub_vec(&dx), &x.add_vec(&dx))?;

        // f_value = (f1 + f2) / 2.0;

        #[allow(clippy::needless_range_loop)]
        for i in 0..x.len() {
            let df = delta_f / (2.0 * dx[i]);
            let p_i = g[i] * df;

            let gi = match p_i {
                _ if p_i >= 0.0 => df,
                _ => 0.0,
            };

            let eta_i = match p_i {
                _ if p_i > 0.0 => cfg.eta_p,
                _ if p_i < 0.0 => cfg.eta_m,
                _ => 1.0,
            };
            delta[i] = (delta[i] * eta_i).clamp(cfg.delta_min, cfg.delta_max());

            g[i] = gi;

            // let dir = match cfg.always_step {
            //     true => df,
            //     false => gi,
            // };
            // differs from paper as we are *minimising*
            match gi {
                _ if gi > 0.0 => x[i] -= cfg.rho * delta[i],
                _ if gi < 0.0 => x[i] += cfg.rho * delta[i],
                _ => {}
            }
            // constrain x (within xmax less delta?)
        }
        // trace!(
        //     "[{i:4}] rx = {rx:+.0?} x = {x:+.6?} dx = {dx:+11.6?} delta = {delta:+11.6?} gi = {g:+11.8?} |g| = {norm_g:10.8}  signum(g) = {signum_g:10.6?}",
        //     i = iter,
        //     rx = rx.as_slice(),
        //     x = x.as_slice(),
        //     dx = dx.as_slice(),
        //     delta = delta.as_slice(),
        //     // ghat = ghat.as_slice(),
        //     g = g.as_slice(),
        //     norm_g = g.norm_l2(),
        //     signum_g = g.iter().copied().map(f64::signum),
        // );
        *iter += 1;
        Ok(ControlFlow::Continue(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use stepwise::{
        assert_approx_eq, fixed_iters,
        problems::{sigmoid, sphere},
        BoxedError, Driver, VectorExt,
    };
    use test_log::test;

    #[test]
    fn rspsa_1d() -> Result<(), BoxedError> {
        let w0 = [1.5];
        let cfg = RspsaParams {
            ..RspsaParams::default()
        };

        let algo = RspsaSolver::from_fn(cfg, &w0, sphere)?;
        let driver = fixed_iters(algo, 100_000);
        let (solved, step) = driver
            // .on_step(|s, v| println!("{s:?} {} {:?}  {}", s.iteration(), v.x(), (v.x()[0] - 0.0).abs()< 1e-8))
            .fail_if(|_v, s| s.iteration() >= 100_000)
            .converge_when(|v, _s| (v.x()[0] - 0.0).abs() < 1e-8)
            .solve()?;

        // algo.minimize2((sphere, |s: &RspsaSolver| (s.x()[0] - 3.0).abs() < 1e-8))?;

        let x = solved.x();
        assert!(step.iteration() < 300, "{iters}", iters = step.iteration());
        assert_approx_eq!(x, &[0.0], 1e-8);
        Ok(())
    }

    #[test]
    fn rspsa_100d() -> Result<(), BoxedError> {
        let dim: usize = env::var("RUST_DIM").unwrap_or("100".to_string()).parse()?;
        let w0 = vec![1.5; dim];
        let cfg = RspsaParams {
            ..RspsaParams::default()
        };

        let algo = RspsaSolver::from_fn(cfg, &w0, sphere)?;
        let driver = fixed_iters(algo, 1_000_000);
        let (solved, step) = driver
            .converge_when(|v, _s| sphere(v.x()).abs() < 1e-6)
            .fail_if(|_v, s| s.iteration() >= 1_000_000)
            .solve()?;

        let x = solved.x();
        let f_val = sphere(solved.x()).abs();
        let iters = step.iteration();
        assert!(
            sphere(solved.x()).abs() < 1e-5,
            "f_val={f_val}\niters={iters}\n{x:9.6?}",
        );
        Ok(())
    }

    #[test]
    fn rspsa_sphere() -> Result<(), BoxedError> {
        let w0 = [0.0, 0.0];
        let mut total_iters = 0;
        for initial_guess in -10..10 {
            for y0 in -10..10 {
                if initial_guess == y0 || initial_guess == -y0 {
                    print!("{:>4}", '-');
                    continue;
                }
                let z0 = [
                    w0[0] + 0.01 * f64::from(initial_guess),
                    w0[0] + 0.01 * f64::from(y0),
                ];
                let cfg = RspsaParams {
                    ..RspsaParams::default()
                };
                let algo = RspsaSolver::from_fn(cfg, &z0, |x| sigmoid(sphere(x)))?;
                let driver = fixed_iters(algo, 1_000_000);
                let (solved, step) = driver
                    .converge_when(|v, _s| v.x().dist_max(&w0) < 1e-6)
                    .fail_if(|_v, s| s.iteration() > 10_000)
                    .solve()?;
                print!("{:>4}", step.iteration());
                assert!(step.iteration() < 600, "{initial_guess} {y0} {solved:?}");
                total_iters += step.iteration();
                assert!(solved.x().dist_max(&w0) < 1e-6);
            }
            println!();
        }
        println!("total iters {total_iters}");
        Ok(())
    }
}
