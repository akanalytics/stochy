use std::error::Error;
use stepwise::{fixed_iters, Driver};
use stochy::{RspsaAlgo, RspsaParams, SpsaAlgo, SpsaParams};

// {\displaystyle f({\boldsymbol {x}})=\sum _{i=1}^{n-1}\left[100\left(x_{i+1}-x_{i}^{2}\right)^{2}+\left(1-x_{i}\right)^{2}\right]}

#[test]
fn convergence() -> Result<(), Box<dyn Error>> {
    let mut fixtures = Vec::<(Box<dyn Fn(&[f64]) -> f64>, _, _)>::new();

    let w0 = [5.2, 6.4];
    let w = [1.0, 2.0];
    let paraboloid1 = move |x: &[f64]| (x[0] - w[0]).powi(2) + 6. * (x[1] - w[1]).powi(2);
    fixtures.push((Box::new(paraboloid1), w0, &w));

    let w0 = [2.3, 5.5];
    let w = [1.0, 2.0];
    let paraboloid2 = move |x: &[f64]| (x[0] - w[0]).powi(2) + 5. * (x[1] - w[1]).powi(2);
    fixtures.push((Box::new(paraboloid2), w0, &w));

    let w0 = [230., 550.];
    let w = [100., 200.];
    let paraboloid3 = move |x: &[f64]| (x[0] - w[0]).powi(2) + 500. * (x[1] - w[1]).powi(2);
    fixtures.push((Box::new(paraboloid3), w0, &w));

    // loop through minimizing
    //      each function
    //      for each algo
    //
    for (mut f, x0, _w) in fixtures.into_iter() {
        let cfg = SpsaParams {
            ..SpsaParams::default()
        };

        let algo = SpsaAlgo::from_fn(cfg, x0.to_vec(), &mut f)?;
        let driver = fixed_iters(algo, 500);
        let (solved, _step) = driver.solve().unwrap();
        let _x = solved.x();
        // assert!((Vector2::from_row_slice(x) - *w).norm() < 0.01, "{x:?} {w:?}");

        let cfg = RspsaParams {
            delta0: x0.iter().copied().map(f64::abs).fold(f64::NAN, f64::max) * 0.1, // max norm
            ..Default::default()
        };
        drop(solved);

        let algo = RspsaAlgo::from_fn(cfg.clone(), x0.to_vec(), &mut f)?;
        let (solved, _step) = fixed_iters(algo, 300).solve()?;

        let _x = solved.x();
        drop(solved);

        let algo = RspsaAlgo::from_fn(cfg, x0.to_vec(), move |x| f(x))?;
        let mut driver = fixed_iters(algo, 300);
        while let Some((v, _s)) = driver.iter_step().unwrap() {
            if let Some(rspsa) = (v as &mut dyn std::any::Any).downcast_mut::<RspsaAlgo>() {
                // `rspsa` is now a &mut RspsaAlgo
                println!("This is an RspsaAlgo! {rspsa:?}");
                // You can now use `rspsa` as an `RspsaAlgo`
            }
        }
        let (solved, _step) = driver.solve()?;

        let _x = solved.x();
        // assert!((Vector2::from_row_slice(x) - *w).norm() < 0.01, "{x:?} {w:?}");
    }
    Ok(())
}
