//! Simple example using the stochy crate with argmin
//! as both argmin and stochy used, with an overlap in yupe names such as Error,
//! full qualifying paths have been used for clarity
//!
//! requires the argmin feature:
//! ```shell
//! cargo test --test argmin  --features=argmin
//! ```

#[cfg(feature = "argmin")]
struct MySimpleCost;

#[cfg(feature = "argmin")]
impl argmin::core::CostFunction for MySimpleCost {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        // (x[0]-1.5)^ 2 + sum of remaining squares
        Ok((param[0] - 1.5).powi(2) + param.iter().skip(1).map(|x| x * x).sum::<f64>())
    }
}

#[cfg(feature = "argmin")]
#[test]
fn argmin_rspsa_example() -> Result<(), Box<dyn std::error::Error>> {
    let hyper_params = stochy::RspsaParams::default();
    let algo = stochy::RspsaSolverArgmin::new(hyper_params);

    // Create the executer
    let exec = argmin::core::Executor::new(MySimpleCost, algo);

    let initial_param = vec![1.0, 1.0];
    let res = exec
        .configure(|step| step.param(initial_param).max_iters(1000))
        .run();

    // Check the result
    assert!(res.is_ok());
    let result = res.unwrap();
    assert!(result.state.best_param.is_some());
    let best_param = result.state.best_param.unwrap();
    stepwise::assert_approx_eq!(best_param.as_slice(), [1.5, 0.0].as_slice(), 1e-2);
    Ok(())
}

#[cfg(feature = "argmin")]
#[test]
fn argmin_spsa_example() -> Result<(), Box<dyn std::error::Error>> {
    let hyper_params = stochy::SpsaParams::default();
    let algo = stochy::SpsaSolverArgmin::new(hyper_params);

    // Create the executer
    let exec = argmin::core::Executor::new(MySimpleCost, algo);

    let initial_param = vec![1.0, 1.0];
    let res = exec
        .configure(|step| step.param(initial_param).max_iters(1000))
        .run();

    // Check the result
    assert!(res.is_ok());
    let result = res.unwrap();
    assert!(result.state.best_param.is_some());
    let best_param = result.state.best_param.unwrap();
    stepwise::assert_approx_eq!(best_param.as_slice(), [1.5, 0.0].as_slice(), 1e-2);
    Ok(())
}
