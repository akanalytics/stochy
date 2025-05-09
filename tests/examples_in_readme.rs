use stepwise::{assert_approx_eq, fixed_iters, Driver as _};
use stochy::{SpsaAlgo, SpsaParams};

#[test]
fn example1_in_readme() {
    let f = |x: &[f64]| (x[0] - 1.5).powi(2) + x[1] * x[1];

    let hyperparams = SpsaParams::default();
    let spsa = SpsaAlgo::from_fn(hyperparams, vec![1.0, 1.0], f).expect("bad hyperparams!");

    let (solved, final_step) = fixed_iters(spsa, 20_000)
        .on_step(|algo, step| println!("{:>4} {:.8?}", step.iteration(), algo.x()))
        .solve()
        .expect("solving failed!");

    assert_approx_eq!(solved.x(), &[1.5, 0.0]);
    println!("solved in {} iters", final_step.iteration());
}

#[test]
#[cfg(feature = "argmin")]
fn example2_in_readme() {
    struct MySimpleCost;

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
        .configure(|step| step.param(initial_param).max_iters(20_000))
        .run()
        .unwrap();

    let best_param = result.state.best_param.unwrap();
    stepwise::assert_approx_eq!(best_param.as_slice(), &[1.5, 0.0]);
}
