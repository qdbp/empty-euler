use std::borrow::Borrow;

use fixedbitset::FixedBitSet;
use num_traits::One;

// TODO move all the monoids and tags
// TODO this should be commutative monoid, split before moving/extending use beyond this module
// it's happening
pub trait CommutativeMonoid<Op>: Sized {
    fn mempty() -> Self;
    fn mappend(self, rhs: impl Borrow<Self>) -> Self;
    // this is a hard requirement to stop painfully slow reallocating
    // sieves from accidentally being written
    fn mappend_mut(&mut self, rhs: impl Borrow<Self>);
}

/// Generalized linear sieve to compute multiplicative functions.
/// The functions may range into any monoid -- specifically this lets us
/// express prime factor sieves, divisor sieves, etc., using the same logic.
/// All that needs to be provided is an (efficient) mappend implementation for
/// the chosen container type.
pub fn linear_sieve<T, Op, Fp, Fip>(n: u64, fp: Fp, fip: Fip) -> Vec<T>
where
    T: CommutativeMonoid<Op>,
    Fp: Fn(u64) -> T,
    Fip: Fn(u64, u64, &T) -> T,
{
    let mut is_composite = FixedBitSet::with_capacity(n as usize + 1);
    let mut primes = Vec::<u64>::with_capacity(n as usize / (n as f64).ln() as usize);
    let mut f_arr: Vec<T> = std::iter::repeat_with(T::mempty)
        .take(n as usize + 1)
        .collect();

    for i in 2..=n {
        if !is_composite[i as usize] {
            primes.push(i);
            f_arr[i as usize] = fp(i);
        }
        for p in primes.iter().take_while(|p| *p * i <= n) {
            let pi = p * i;
            is_composite.set(pi as usize, true);
            // not coprime, need fip
            if i.is_multiple_of(*p) {
                f_arr[pi as usize] = fip(i, *p, &f_arr[i as usize]);
                break;
            }
            unsafe {
                // SAFETY: we know i, pi, p are all in bounds and furtermore
                // aliasing f_i and f_p is fine since they are read only
                // f_ip != f_i, f_p
                let ptr = f_arr.as_mut_ptr();
                let f_ip = &mut *ptr.add(pi as usize);
                // f[pi] = mzero() so by monoid laws we can do this in any order
                // this lets us avoid making a temporary. assumption is two mut calls
                // will be faster than one alloc
                f_ip.mappend_mut(&*ptr.add(*p as usize));
                f_ip.mappend_mut(&*ptr.add(i as usize));
            }
        }
    }
    f_arr
}

pub fn linear_sieve_completely_multiplicative<T, Op, Fp>(n: u64, fp: Fp) -> Vec<T>
where
    T: CommutativeMonoid<Op>,
    Fp: Fn(u64) -> T,
{
    let mut is_composite = FixedBitSet::with_capacity(n as usize + 1);
    let mut primes = Vec::<u64>::with_capacity(n as usize / (n as f64).ln() as usize);
    let mut f_arr: Vec<T> = std::iter::repeat_with(T::mempty)
        .take(n as usize + 1)
        .collect();

    for i in 2..=n {
        if !is_composite[i as usize] {
            primes.push(i);
            f_arr[i as usize] = fp(i);
        }
        for p in primes.iter().take_while(|p| *p * i <= n) {
            let pi = p * i;
            is_composite.set(pi as usize, true);
            // completely multiplicative, just f(p)*f(i) always
            unsafe {
                // SAFETY: we know i, pi, p are all in bounds and furtermore
                // aliasing f_i and f_p since they are read only
                let ptr = f_arr.as_mut_ptr();
                let f_ip = &mut *ptr.add(pi as usize);
                f_ip.mappend_mut(&*ptr.add(*p as usize));
                f_ip.mappend_mut(&*ptr.add(i as usize));
            }
            if i.is_multiple_of(*p) {
                break;
            }
        }
    }
    f_arr
}

// number-theoretic monoid tags. these are meaningless by themselves
// -- the relevant place to look is the impl for the base type

// a * b
pub struct Multiplicative {}

impl<T> CommutativeMonoid<Multiplicative> for T
where
    T: One,
    for<'a> T: std::ops::Mul<&'a T, Output = T>,
    for<'a> T: std::ops::MulAssign<&'a T>,
{
    fn mempty() -> Self {
        T::one()
    }
    fn mappend(self, rhs: impl Borrow<Self>) -> Self {
        self * rhs.borrow()
    }
    fn mappend_mut(&mut self, rhs: impl Borrow<Self>) {
        *self *= rhs.borrow();
    }
}

// A ⊗ B := { ab | a ∈ A, b ∈ B }
pub struct MulSet {}

impl<T> CommutativeMonoid<MulSet> for Vec<T>
where
    T: One + Clone + Ord,
    for<'a> T: std::ops::Mul<&'a T, Output = T>,
    for<'a> T: std::ops::MulAssign<&'a T>,
{
    fn mempty() -> Self {
        vec![T::one()]
    }
    fn mappend(self, rhs: impl Borrow<Self>) -> Self {
        let rhs = rhs.borrow();
        let mut out = Vec::with_capacity(self.len() * rhs.len());
        for a in self.iter() {
            for b in rhs.iter() {
                let mut ab = a.clone();
                ab *= b;
                out.push(ab);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
    fn mappend_mut(&mut self, rhs: impl Borrow<Self>) {
        let out = std::mem::take(self).mappend(rhs.borrow());
        *self = out;
    }
}

// A ⊎ B
pub struct DiscriminatedUnion {}

// A ∪ B
impl<T> CommutativeMonoid<DiscriminatedUnion> for Vec<T>
where
    T: Clone + std::hash::Hash + std::cmp::Eq,
{
    fn mempty() -> Self {
        Vec::new()
    }
    fn mappend(self, rhs: impl Borrow<Self>) -> Self {
        let mut out = self.clone();
        out.mappend_mut(rhs);
        out
    }
    fn mappend_mut(&mut self, rhs: impl Borrow<Self>) {
        self.extend_from_slice(rhs.borrow());
    }
}

#[cfg(test)]
mod tests {
    use crate::int::{divisors, factorint, φ};

    use super::*;

    #[test]
    fn test_φ_sieve() {
        let n = 20;
        let out = linear_sieve(n, |p| p - 1, |_, p, f_i| p * f_i);
        let expected: Vec<u64> = (0..=n).map(φ).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_prime_fac_sieve() {
        let n = 20;
        let out: Vec<Vec<u64>> =
            linear_sieve_completely_multiplicative::<_, DiscriminatedUnion, _>(n, |p| vec![p]);
        let expected: Vec<Vec<u64>> = (0..=n)
            .map(|n| {
                let fac_map = factorint(n);
                let mut out = Vec::new();
                for (p, e) in fac_map {
                    out.extend(std::iter::repeat_n(p, e as usize));
                }
                out.sort_unstable();
                out
            })
            .collect();
        assert_eq!(
            out.into_iter()
                .map(|mut v| {
                    v.sort_unstable();
                    v
                })
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn test_divisor_sieve() {
        let n = 20;
        let out: Vec<Vec<u64>> =
            linear_sieve_completely_multiplicative::<_, MulSet, _>(n, |p| vec![1, p]);

        let expected: Vec<Vec<u64>> = (0..=n)
            .map(|n| {
                let mut divisors = divisors(n);
                divisors.sort_unstable();
                divisors
            })
            .collect();

        assert_eq!(
            out.into_iter()
                .map(|mut v| {
                    v.sort_unstable();
                    v
                })
                .collect::<Vec<_>>(),
            expected
        );
    }
}
