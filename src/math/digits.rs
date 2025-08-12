use num_traits::Zero;
use std::ops;

#[derive(Debug, Clone)]
pub struct Digits<const B: u32> {
    pub digits: Vec<u32>,
}

impl<const B: u32, T> From<T> for Digits<B>
where
    T: Zero + From<u32>,
    for<'a> &'a T: std::ops::Rem<&'a T> + std::ops::Div<&'a T, Output = T>,
    for<'a> <&'a T as std::ops::Rem<&'a T>>::Output: TryInto<u32>,
{
    fn from(mut n: T) -> Self {
        debug_assert!(B >= 2);
        let mut digits = Vec::new();
        let base = T::from(B);
        while !n.is_zero() {
            let r = &n % &base;
            digits.push(r.try_into().ok().unwrap());
            n = &n / &base;
        }
        digits.reverse();
        Self { digits }
    }
}

impl<const B: u32> Digits<B> {
    pub fn new<V: Into<Vec<u32>>>(n: V) -> Self {
        debug_assert!(B >= 2);
        let digits = n.into();
        Self { digits }
    }

    pub fn into_num<T>(&self) -> T
    where
        T: Zero + From<u32>,
        for<'a> T: ops::AddAssign<&'a T>,
        for<'a> T: ops::MulAssign<&'a T>,
    {
        let mut n = T::zero();
        let base = T::from(B);
        for &d in &self.digits {
            n *= &base;
            let tmp: T = d.into();
            n += &tmp;
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_digits() {
        let d: Digits<10> = Digits::from(12345u32);
        assert_eq!(d.digits, vec![1, 2, 3, 4, 5]);
        let d: Digits<2> = Digits::from(13u32);
        assert_eq!(d.digits, vec![1, 1, 0, 1]);
    }

    #[test]
    fn test_into_num() {
        let d: Digits<10> = Digits::from(12345u32);
        assert_eq!(d.into_num::<u32>(), 12345);
        let d: Digits<2> = Digits::from(13u32);
        assert_eq!(d.into_num::<u64>(), 13);
        let d: Digits<16> = Digits::from(0u32);
        assert_eq!(d.into_num::<rug::Integer>(), 0);
    }
}
