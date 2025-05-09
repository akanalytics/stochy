use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::error::Error;
use stepwise::{assert_approx_eq, fixed_iters, problems::sphere, Driver};
use stochy::{RspsaAlgo, RspsaParams, SpsaAlgo, SpsaParams};

#[test]
fn noisy_sphere_rspsa() -> Result<(), Box<dyn Error>> {
    // Fixed seed for reproducibility
    let mut rng = Pcg64::seed_from_u64(42);

    let noisy_sphere = |x: &[f64]| sphere(x) + rng.random_range(-0.0005..0.0005);

    let params = RspsaParams::default();
    let algo = RspsaAlgo::from_fn(params, vec![1.5, 1.5], noisy_sphere)?;
    let (solved, step) = fixed_iters(algo, 10_000).solve()?;

    let x = solved.x();
    assert_eq!(step.iteration(), 10_000);
    assert_approx_eq!(x, &[0.0, 0.0], 0.02);
    Ok(())
}

#[test]
fn noisy_sphere_spsa() -> Result<(), Box<dyn Error>> {
    // Fixed seed for reproducibility
    let mut rng = Pcg64::seed_from_u64(42);

    let noisy_sphere = |x: &[f64]| sphere(x) + rng.random_range(-0.0005..0.0005);

    // default hyperparameters for SPSA require more iterations than RSPSA for same accuracy
    let params = SpsaParams::default();
    let algo = SpsaAlgo::from_fn(params, vec![1.5, 1.5], noisy_sphere)?;
    let driver = fixed_iters(algo, 100_000);
    let (solved, step) = driver.solve()?;

    let x = solved.x();
    assert_eq!(step.iteration(), 100_000);
    assert_approx_eq!(x, &[0.0, 0.0], 0.02);
    Ok(())
}
