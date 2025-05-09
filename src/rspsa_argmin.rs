use std::ops::ControlFlow;

use argmin::argmin_error_closure;
use argmin::core::{
    CostFunction, Error, IterState, Problem, Solver, State, TerminationReason, TerminationStatus,
    KV,
};

use crate::common::FuncKind;
use crate::{RspsaAlgo, RspsaParams};

type Inp = Vec<f64>;
type Out = f64;
type F = f64; // float type
type IState = IterState<Inp, Out, (), (), (), F>;

/// A wrapper for the RSPSA algo implementing the Argmin `Algo` trait
pub struct RspsaSolverArgmin<'a> {
    params: Option<RspsaParams>,
    algo: Option<RspsaAlgo<'a>>,
    iters: u64,
    control_flow: ControlFlow<()>,
}

impl RspsaSolverArgmin<'_> {
    /// Creates a new instance with the given hyper-parameters.
    pub fn new(params: RspsaParams) -> Self {
        Self {
            params: Some(params),
            algo: None,
            iters: 0,
            control_flow: ControlFlow::Continue(()),
        }
    }
}

impl<O> Solver<O, IterState<Inp, Out, (), (), (), F>> for RspsaSolverArgmin<'_>
where
    O: CostFunction<Param = Inp, Output = Out>,
    // <Self as Algo>::Error : Send + Sync
{
    const NAME: &'static str = "RSPSA";

    fn next_iter(
        &mut self,
        p: &mut Problem<O>,
        step: IState,
    ) -> Result<(IState, Option<KV>), Error> {
        let Some(rspsa) = self.algo.as_mut() else {
            return Err(Error::msg("RspsaAlgo is not initialized"));
        };

        let mut f = |x: &[f64]| p.cost(&x.to_vec()).unwrap();
        let mut func = FuncKind::FnDyn(&mut f);
        self.control_flow = RspsaAlgo::step_df(&rspsa.params, &mut rspsa.state, &mut func)
            .map_err(|e| Error::msg(e.to_string()))?;
        self.iters += 1;
        let x = rspsa.x().to_vec();
        Ok((step.param(x), None))
    }

    fn init(
        &mut self,
        _: &mut Problem<O>,
        mut step: IState,
    ) -> Result<(IState, Option<KV>), Error> {
        let aec = argmin_error_closure!(NotInitialized, "Initial parameter vector required!");
        let x0 = step.take_param().ok_or_else(aec)?;
        let f = |_x: &[f64]| 0.0;
        let Some(params) = self.params.take() else {
            return Err(Error::msg("RspsaAlgo params is missing"));
        };
        let rspsa = RspsaAlgo::from_fn(params, x0, f).map_err(|e| Error::msg(e.to_string()))?;
        self.algo = Some(rspsa);
        Ok((step, None))
    }

    fn terminate_internal(&mut self, step: &IState) -> TerminationStatus {
        let ts = <Self as Solver<O, IState>>::terminate(self, step);
        if ts.terminated() {
            ts
        } else if self.iters >= step.get_max_iters() {
            TerminationStatus::Terminated(TerminationReason::MaxItersReached)
        } else if self.control_flow.is_break() {
            let r = TerminationReason::SolverExit("algo -> ControlFlow::Break".to_string());
            TerminationStatus::Terminated(r)
        } else {
            TerminationStatus::NotTerminated
        }
    }
}
