/// Modular integer type ℤ/Nℤ
/// We try to be generic and follow num_traits
/// For now we hardcode u64 backing store
// TODO be generic/auto-switch to rug::Integer?
use std::{
    borrow::Borrow,
    cmp::Ordering,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Z<const N: u64>(pub u64);

impl<const N: u64> Z<N> {
    const WIDE: bool = N > u32::MAX as u64;
}

impl<const N: u64> Add for Z<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let (s, carry) = self.0.overflowing_add(rhs.0);
        let ge = (s >= N) as u64 | (carry as u64);
        // subtract m iff ge == 1
        Self(s.wrapping_sub(N & 0u64.wrapping_sub(ge)))
    }
}

impl<const N: u64> Sub for Z<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Z::<N>((self.0 + N - rhs.0 % N) % N)
    }
}

impl<const N: u64> Mul for Z<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // if we fit into u32 we can never overflow u64
        if !Self::WIDE {
            Z::<N>((self.0 * rhs.0) % N)
        } else {
            Z::<N>(((self.0 as u128 * rhs.0 as u128) % (N as u128)) as u64)
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

impl<const N: u64> Borrow<u64> for Z<N> {
    #[inline]
    fn borrow(&self) -> &u64 {
        &self.0
    }
}

impl<const N: u64> From<u32> for Z<N> {
    #[inline]
    fn from(x: u32) -> Self {
        Self((x as u64) % N)
    }
}

impl<const N: u64> Div for Z<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Z::<N>((self.0 * modinv(rhs.0, N).expect("No modular inverse")) % N)
    }
}
use num_traits::{One, Zero};
use std::fmt;

use crate::primes::modinv;

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
