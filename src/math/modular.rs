use num_traits::{One, Zero};
use std::fmt;
use std::ops::AddAssign;
/// Modular integer type ℤ/Nℤ
/// We try to be generic and follow num_traits
/// For now we hardcode u64 backing store
// TODO be generic/auto-switch to rug::Integer?
use std::{
    borrow::Borrow,
    cmp::Ordering,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

use crate::int::{UnsignedInt, modinv_u};

/// Modular integer type ℤ/Nℤ. There is no requirement that N be prime,
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Z<const N: u64>(pub u64);

impl<const N: u64> Z<N> {
    // TODO allow bigint moduli, can get nasty slow?
    const WIDE: bool = N > u32::MAX as u64;
}

impl<const N: u64> Default for Z<N> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const N: u64, Rhs> AddAssign<Rhs> for Z<N>
where
    Rhs: Borrow<Z<N>>,
{
    fn add_assign(&mut self, rhs: Rhs) {
        let (s, carry) = self.0.overflowing_add(rhs.borrow().0);
        let ge = (s >= N) as u64 | (carry as u64);
        // subtract m iff ge == 1
        self.0 = s.wrapping_sub(N & 0u64.wrapping_sub(ge))
    }
}

impl<const N: u64, Rhs> Add<Rhs> for Z<N>
where
    Rhs: Borrow<Z<N>>,
{
    type Output = Self;
    fn add(self, rhs: Rhs) -> Self {
        let mut out = self;
        out += rhs;
        out
    }
}

impl<const N: u64> Sub for Z<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Z::<N>((self.0 + N - rhs.0 % N) % N)
    }
}

impl<const N: u64, Rhs> Mul<Rhs> for Z<N>
where
    Rhs: Borrow<Z<N>>,
{
    type Output = Self;
    fn mul(self, rhs: Rhs) -> Self {
        // if we fit into u32 we can never overflow u64
        if !Self::WIDE {
            Z::<N>((self.0 * rhs.borrow().0) % N)
        } else {
            Z::<N>(((self.0 as u128 * rhs.borrow().0 as u128) % (N as u128)) as u64)
        }
    }
}

impl<const N: u64> Rem for Z<N> {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        Z::<N>(self.0 % rhs.0)
    }
}

impl<const N: u64> Neg for Z<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Z::<N>((N - self.0 % N) % N)
    }
}

impl<const N: u64> PartialOrd for Z<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<const N: u64> Ord for Z<N> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<const N: u64, Rhs: UnsignedInt + From<u64> + TryInto<u64>> From<Rhs> for Z<N> {
    #[inline]
    fn from(x: Rhs) -> Self {
        // SAFETY: rem_euclid with N should always give an in-range value
        let r = x.rem_euclid(&N.into()).try_into().ok().unwrap();
        Self(r)
    }
}

impl<const N: u64> Div for Z<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Z::<N>((self.0 * modinv_u(rhs.0, N).expect("No modular inverse")) % N)
    }
}

impl<const N: u64> Zero for Z<N> {
    fn zero() -> Self {
        Z::<N>(0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl<const N: u64> One for Z<N> {
    fn one() -> Self {
        Z::<N>(1 % N)
    }
}

impl<const N: u64> fmt::Display for Z<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn z7(x: u64) -> Z<7> {
        Z::<7>(x % 7)
    }

    #[test]
    fn test_add() {
        assert_eq!(z7(3) + z7(5), z7(1));
    }

    #[test]
    fn test_sub() {
        assert_eq!(z7(2) - z7(5), z7(4));
    }

    #[test]
    fn test_mul() {
        assert_eq!(z7(3) * z7(4), z7(5));
    }

    #[test]
    fn test_neg() {
        assert_eq!(-z7(3), z7(4));
    }

    #[test]
    fn test_div() {
        // 4^-1 = 2 mod 7
        assert_eq!(z7(2) / z7(4), z7(4));
    }

    #[test]
    fn test_zero_one() {
        assert_eq!(Z::<7>::zero(), z7(0));
        assert_eq!(Z::<7>::one(), z7(1));
        assert!(Z::<7>::zero().is_zero());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", z7(5)), "5");
    }

    fn z_girthy(x: u64) -> Z<{ u64::MAX - 1 }> {
        Z::<{ u64::MAX - 1 }>(x % (u64::MAX - 1))
    }

    #[test]
    fn test_extreme_add() {
        assert_eq!(z_girthy(u64::MAX - 2) + z_girthy(3), z_girthy(2));
    }

    #[test]
    fn test_extreme_mul() {
        // -1 * 2 = -2
        assert_eq!(z_girthy(u64::MAX - 2) * z_girthy(2), z_girthy(u64::MAX - 3));
    }
}
