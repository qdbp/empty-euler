use num_traits::{One, Zero};
use std::{
    borrow::Borrow,
    collections::{HashMap, hash_map::Entry},
    ops::{Add, AddAssign, Deref, Mul, MulAssign},
};

pub trait PolyCoef:
    Clone
    + Zero
    + One
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
{
}

impl<T> PolyCoef for T where
    T: std::clone::Clone
        + Zero
        + One
        + for<'a> Mul<&'a T, Output = T>
        + for<'a> Add<&'a T, Output = T>
        + for<'a> AddAssign<&'a T>
{
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly<T: PolyCoef> {
    pub coefs: HashMap<u64, T>,
}

#[macro_export]
macro_rules! poly {

    // stop
    (@munch $m:ident, @end) => {};

    //  + C v ^ E   /  - C v ^ E
    (@munch $m:ident, + $c:literal * x ^ $e:literal $($rest:tt)* ) => {{
        poly!(@push $m, $e, $c);
        poly!(@munch $m, $($rest)*);
    }};
    (@munch $m:ident, - $c:literal * x ^ $e:literal $($rest:tt)* ) => {{
        poly!(@push $m, $e, -$c);
        poly!(@munch $m, $($rest)*);
    }};

    //  + v ^ E      /  - v ^ E
    (@munch $m:ident,  + x ^ $e:literal $($rest:tt)* ) => {{
        poly!(@push $m, $e, 1);
        poly!(@munch $m, $($rest)*);
    }};
    (@munch $m:ident,  - x ^ $e:literal $($rest:tt)* ) => {{
        poly!(@push $m, $e, -1);
        poly!(@munch $m, $($rest)*);
    }};

    //  + C * v        /  - C * v
    (@munch $m:ident,  + $c:literal * x $($rest:tt)* ) => {{
        poly!(@push $m, 1u64, $c);
        poly!(@munch $m, $($rest)*);
    }};
    (@munch $m:ident,  - $c:literal * x $($rest:tt)* ) => {{
        poly!(@push $m, 1u64, -$c);
        poly!(@munch $m, $($rest)*);
    }};

    //  + v          /  - v
    (@munch $m:ident,  + x $($rest:tt)* ) => {{
        poly!(@push $m, 1u64, 1);
        poly!(@munch $m, $($rest)*);
    }};
    (@munch $m:ident,  - x $($rest:tt)* ) => {{
        poly!(@push $m, 1u64, -1);
        poly!(@munch $m, $($rest)*);
    }};

    //  + C          /  - C
    (@munch $m:ident, + $c:literal $($rest:tt)* ) => {{
        poly!(@push $m, 0u64, $c);
        poly!(@munch $m, $($rest)*);
    }};
    (@munch $m:ident, - $c:literal $($rest:tt)* ) => {{
        poly!(@push $m, 0u64, -$c);
        poly!(@munch $m, $($rest)*);
    }};

    (@push $m:ident, $e:expr, $c:expr) => {{
        use ::std::collections::hash_map::Entry;
        match $m.entry($e as u64) {
            Entry::Occupied(mut o) => { *o.get_mut() += $c; }
            Entry::Vacant(v) => { if !$c.is_zero() { v.insert($c); } }
        }
    }};

    (@munch $($got:tt)+) => {
        compile_error!(concat!("unexpected tokens: ", stringify!($($got)*)));
    };

    // typed, default var x:  poly!(i64; 1 + 3x^2);
    ( $t:ty ; $($rest:tt)+ ) => {{
        let mut __m: ::std::collections::HashMap<u64, $t> =
            ::std::collections::HashMap::new();
        poly!(@munch __m, + $($rest)+);
        Poly { coefs: __m }
    }};

    // untyped, default var x:  let p = poly!(1 + 3x^2 + 5x^3);
    ( $($rest:tt)+ ) => {{
        let mut __m = ::std::collections::HashMap::new();
        poly!(@munch __m, + $($rest)+ @end);
        Poly { coefs: __m }
    }};
}

// invariant: we do not keep zero coefficients!
impl<T: PolyCoef> Poly<T> {
    pub fn new() -> Self {
        Poly {
            coefs: HashMap::new(),
        }
    }

    pub fn new_with_capacity(cap: usize) -> Self {
        Poly {
            coefs: HashMap::with_capacity(cap),
        }
    }

    /// Creates a monomial
    pub fn monomial(degree: impl Into<u64>, coef: impl Into<T>) -> Self {
        let mut coefs = HashMap::new();
        let coef = coef.into();
        if !coef.is_zero() {
            coefs.insert(degree.into(), coef);
        }
        Poly { coefs }
    }
}

impl<T: PolyCoef> Deref for Poly<T> {
    type Target = HashMap<u64, T>;
    fn deref(&self) -> &Self::Target {
        &self.coefs
    }
}

impl<T: PolyCoef> Zero for Poly<T> {
    fn zero() -> Self {
        Self::new()
    }

    fn is_zero(&self) -> bool {
        self.coefs.is_empty()
    }
}

impl<T: PolyCoef> Default for Poly<T> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<T: PolyCoef> One for Poly<T> {
    fn one() -> Self {
        Self::monomial(0u64, T::one())
    }
}

impl<T: PolyCoef> Add for Poly<T> {
    type Output = Poly<T>;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self.clone();
        out += rhs;
        out
    }
}

impl<T: PolyCoef> AddAssign for Poly<T> {
    fn add_assign(&mut self, rhs: Self) {
        for (pow, coef) in rhs.coefs.iter() {
            let entry = self.coefs.entry(*pow);
            match entry {
                Entry::Occupied(mut e) => {
                    let remove = {
                        let v = e.get_mut();
                        *v += coef;
                        v.is_zero()
                    };
                    if remove {
                        e.remove();
                    }
                }
                Entry::Vacant(e) => {
                    // no check here, we assume invariant holds and coef is not zero
                    e.insert(coef.clone());
                }
            }
        }
    }
}

impl<T: PolyCoef, Rhs> Mul<Rhs> for Poly<T>
where
    Rhs: Borrow<Poly<T>>,
{
    type Output = Self;

    fn mul(self, rhs: Rhs) -> Self::Output {
        // this is the minimum capacity estimate, for large poly's we'd rather
        // reallocate a few times than egregiously waste memory -- the upper bound
        // is quadratic in the sizes of the polys!
        let mut out = Self::new_with_capacity(self.len());
        let rhs = rhs.borrow();
        for (pow0, coef0) in self.coefs.iter() {
            for (pow1, coef1) in rhs.coefs.iter() {
                let new_coef = coef0.clone() * coef1;
                *out.coefs.entry(pow0 + pow1).or_insert_with(T::zero) += &new_coef;
            }
        }
        out
    }
}

impl<T: PolyCoef, Rhs> MulAssign<Rhs> for Poly<T>
where
    Rhs: Borrow<Poly<T>>,
{
    fn mul_assign(&mut self, rhs: Rhs) {
        let lhs = std::mem::take(self);
        *self = lhs * rhs;
    }
}

impl<T: PolyCoef + std::fmt::Display> std::fmt::Display for Poly<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut terms: Vec<String> = Vec::new();
        let mut sorted_keys: Vec<&u64> = self.coefs.keys().collect();
        sorted_keys.sort();
        for &pow in sorted_keys.iter() {
            let coef = &self.coefs[pow];
            let term = match pow {
                0 => format!("{}", coef),
                1 => format!("{}x", coef),
                _ => format!("{}x^{}", coef, pow),
            };
            terms.push(term);
        }
        write!(f, "{}", terms.join(" + "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_polynomial() {
        let poly: Poly<i32> = Poly::new();
        assert!(poly.is_zero());
    }

    #[test]
    fn test_from_coefficient() {
        let poly = poly!(5);
        assert_eq!(poly.coefs.get(&0), Some(&5));
        assert_eq!(poly.coefs.len(), 1);
    }

    #[test]
    fn test_addition() {
        let poly1 = poly!(3 + 2 * x);
        let poly2 = poly!(1 + 4 * x);

        let result = poly1 + poly2; // Should be 4 + 6x
        assert_eq!(result.coefs.get(&0), Some(&4));
        assert_eq!(result.coefs.get(&1), Some(&6));
    }

    #[test]
    fn test_add_assign() {
        let mut poly1 = poly!(3 + 2 * x);
        let poly2 = poly!(1 + 4 * x);
        poly1 += poly2; // Should be 4 + 6x
        assert_eq!(poly1, poly!(4 + 6 * x));
    }

    #[test]
    fn test_multiplication_with_terms() {
        let poly1 = poly!(2 + 3 * x);
        let poly2 = poly!(1 + x);

        let result = poly1 * poly2; // (2 + 3x)(1 + x) = 2 + 2x + 3x + 3x^2 = 2 + 5x + 3x^2
        assert_eq!(result, poly!(2 + 5 * x + 3 * x ^ 2));
    }

    #[test]
    fn test_mul_assign() {
        let mut poly1 = poly!(2 + 3 * x);
        let poly2 = poly!(1 + x);
        poly1 *= poly2; // (2 + 3x)(1 + x) = 2 + 5x + 3x^2
        assert_eq!(poly1, poly!(2 + 5 * x + 3 * x ^ 2));
    }

    #[test]
    fn test_zero_polynomial() {
        let zero_poly = poly!(0);
        let other_poly = poly!(5);

        let result1 = zero_poly.clone() + other_poly.clone();
        assert_eq!(result1.coefs, other_poly.coefs);

        let result2 = zero_poly * other_poly;
        assert!(result2.is_zero());
    }

    #[test]
    fn test_display() {
        let poly = poly!(3 + 2 * x + 5 * x ^ 2);
        assert_eq!(format!("{}", poly), "3 + 2x + 5x^2");
    }
}
