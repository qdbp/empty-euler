use std::{
    borrow::Borrow,
    ops::{Add, BitAnd, Mul, ShrAssign},
};

use num_traits::{One, Zero};

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
}
