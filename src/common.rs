use crate::BoxedError;
use rand::Rng;

#[allow(clippy::type_complexity, dead_code)]
pub enum FuncKind<'a> {
    FnDyn(&'a mut dyn FnMut(&[f64]) -> f64),
    Fn(Box<dyn 'a + FnMut(&[f64]) -> f64>),
    Falliable(Box<dyn 'a + FnMut(&[f64]) -> Result<f64, BoxedError>>),
    Difference(Box<dyn 'a + FnMut(&[f64], &[f64]) -> Result<f64, BoxedError>>),
}

impl FuncKind<'_> {
    pub fn difference(&mut self, xm: &[f64], xp: &[f64]) -> Result<f64, BoxedError> {
        match self {
            FuncKind::FnDyn(f) => Ok(f(xp) - f(xm)),
            FuncKind::Fn(f) => Ok(f(xp) - f(xm)),
            FuncKind::Falliable(f) => Ok(f(xp)? - f(xm)?),
            FuncKind::Difference(df) => df(xm, xp),
        }
    }
}

pub fn rademacher<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Vec<f64> {
    (0..n)
        .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;
    use rand_core::impls;
    use stepwise::assert_approx_eq;

    /// Marsaglia multiply with carry random number generator
    pub struct MarsagliaRng(pub u32, pub u32);

    impl RngCore for MarsagliaRng {
        fn next_u32(&mut self) -> u32 {
            self.0 = 36969 * (self.0 & 0xffff) + (self.0 >> 16);
            self.1 = 18000 * (self.1 & 0xffff) + (self.1 >> 16);
            (self.0 << 16) + (self.1 & 0xffff)
        }

        fn next_u64(&mut self) -> u64 {
            impls::next_u64_via_u32(self)
        }

        fn fill_bytes(&mut self, dst: &mut [u8]) {
            impls::fill_bytes_via_next(self, dst);
        }
    }

    #[test]
    fn random() {
        let mut my_rand = MarsagliaRng(42, 43);
        let avg = (0..100_000)
            .map(|_| my_rand.random_range(0.0..1.0))
            // .on_step(|a| println!("{a}"))
            .sum::<f64>()
            / 100_000.0;
        assert!((avg - 0.5).abs() < 0.001, "average {avg}");

        let var = (0..100_000)
            .map(|_| my_rand.random_range(0.0..1_f64)) // .on_step(|x| println!("{x}"))
            .map(|r| (r - 0.5).powi(2))
            .sum::<f64>()
            / 100_000.0;
        assert!(
            (var - 1. / 12.0).abs() < 0.001,
            "variance {var} {}",
            1. / 12.
        );
    }

    #[test]
    fn rademacher_core() {
        let mut my_rand = MarsagliaRng(42, 43);
        let vec = rademacher(5, &mut my_rand);
        println!("{vec:?}");
        assert_approx_eq!(vec[0], 1.0);
        assert_approx_eq!(vec[1], -1.0);
        assert_approx_eq!(vec[2], -1.0);
        assert_approx_eq!(vec[3], 1.0);
        assert_approx_eq!(vec[4], -1.0);
    }
}
