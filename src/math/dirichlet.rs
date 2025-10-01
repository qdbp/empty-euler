use rug::Integer as Int;
use std::ops::{AddAssign, Mul, SubAssign};

use num_traits::Zero;

pub fn dirichlet_hyperbola<T, F, FSum, G, Gsum>(n: u64, f: F, fsum: FSum, g: G, gsum: Gsum) -> T
where
    T: Zero + Mul<T, Output = T> + AddAssign<T> + SubAssign<T>,
    F: Fn(u64) -> T,
    FSum: Fn(u64) -> T,
    G: Fn(u64) -> T,
    Gsum: Fn(u64) -> T,
{
    let rn = Int::from(n).root(2).to_u64().unwrap();
    let mut out = T::zero();
    for k in 1..=rn {
        let ndk = n.div_euclid(k);
        out += f(k) * gsum(ndk);
        out += g(k) * fsum(ndk);
    }
    out -= fsum(rn) * gsum(rn);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirichlet_hyperbola() {
        use crate::int::divisors;
        let n = 100;
        let res = dirichlet_hyperbola(n, |_| 1u64, |m| m, |_| 1u64, |m| m);
        let mut expected = 0u64;
        for k in 1..=n {
            expected += divisors(k).len() as u64;
        }
        assert_eq!(res, expected);
    }
}
