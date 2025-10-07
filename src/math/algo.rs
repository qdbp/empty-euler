use std::{
    borrow::Borrow,
    ops::{Add, AddAssign, BitAnd, Mul, ShrAssign},
};

use num_traits::{Num, One, Zero};

use crate::int::introot_u;

// TODO proper abstract algebra tower

/// Exponentiation by squaring defined against maximally generic trait bounds. Does not provide a
/// modulus argument -- if this is required a type that internalized the modular logic should be
/// used, such as modular::Z.
pub fn expsq<X, A>(x: impl Borrow<X>, exp: impl Borrow<A>) -> X
where
    X: Clone + Zero + One + Mul<X> + Add<X>,
    A: Clone + Zero + One + ShrAssign<u32> + for<'a> BitAnd<&'a A, Output = A> + Eq,
    for<'a> &'a A: Borrow<A>,
{
    let mut base: X = x.borrow().clone();
    let mut acc = X::one();
    let mut exp: A = exp.borrow().clone();
    while !exp.is_zero() {
        if (A::one() & &exp) == A::one() {
            acc = acc * base.clone();
        }
        base = base.clone() * base;
        exp.shr_assign(1);
    }
    acc
}

/// Performs a general summation of the form
/// Σ{i=1}^=N f(i)g(N//i) in O(√N) where an O(1) partial summation function for f is available
/// and where (//) is floor division
pub fn bucketed_sum<T, F, FSum, G>(n: u64, f: F, f_sum: FSum, g: G) -> T
where
    F: Fn(u64) -> T,
    // inclusive!
    FSum: Fn(u64, u64) -> T,
    G: Fn(u64) -> T,
    T: Zero + std::fmt::Debug + Clone,
    for<'a> T: AddAssign<&'a T>,
    for<'a> T: Mul<T, Output = T>,
{
    let p = introot_u(n, 2);
    let mut out = T::zero();

    // will double-count iff p*p == n, otherwise we need =p on both loops or we miss some terms
    for k in 1..=(p - (n / p == p) as u64) {
        out += &(f(k) * g(n / k));
    }
    for j in 1..=p {
        let k_ub = n / j;
        let k_lb = n / (j + 1) + 1;
        out += &(f_sum(k_lb, k_ub).clone() * g(j).clone());
    }

    out
}

#[cfg(test)]
mod tests {
    use crate::modular::Z;

    use super::*;
    #[test]
    fn test_expsq() {
        assert_eq!(expsq(2u32, 10u32), 1024u32);
        assert_eq!(expsq(3u32, 5u32), 243u32);
        assert_eq!(expsq(5u64, 0u64), 1u64);
        assert_eq!(expsq(7u64, 1u64), 7u64);
        assert_eq!(expsq(7u64, 2u64), 49u64);
        assert_eq!(expsq(7u64, 3u64), 343u64);
        assert_eq!(expsq(Z::<13>(7), 3u64), Z::<13>(5));
    }

    #[test]
    fn test_bucketed_sum_nonsq() {
        // test summing just i * n / i
        let n = 12;
        let brute = (1..=n).map(|i| i * (n / i)).sum::<u64>();
        let f_sum = |l: u64, u: u64| (u * (u + 1)) / 2 - (l * (l - 1)) / 2;
        let f = |x: u64| x;
        let g = |x: u64| x;

        assert_eq!(brute, bucketed_sum(n, f, f_sum, g));
    }
    #[test]
    fn test_bucketed_sum_sq() {
        // test summing just i * n / i
        let n = 25;
        let brute = (1..=n).map(|i| i * (n / i)).sum::<u64>();
        let f_sum = |l: u64, u: u64| (u * (u + 1)) / 2 - (l * (l - 1)) / 2;
        let f = |x: u64| x;
        let g = |x: u64| x;

        assert_eq!(brute, bucketed_sum(n, f, f_sum, g));
    }
}
