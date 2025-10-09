use num_traits::{One, Zero};
use std::{
    borrow::Borrow,
    mem,
    ops::{Add, AddAssign, Deref, Mul, MulAssign},
};

pub trait PolyCoef: Clone + Zero + One + for<'a> AddAssign<&'a Self> {}

impl<T> PolyCoef for T where T: std::clone::Clone + Zero + One + for<'a> AddAssign<&'a T> {}

/// An abstract monomial term in N (anonymous) variables, with no defined coefficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Mono<const N: usize = 1> {
    pub exps: [u32; N],
}

impl<const N: usize> Mono<N> {
    /// Creates a monomial from an array of exponents.
    pub fn new(coefs: [u32; N]) -> Self {
        Mono { exps: coefs }
    }

    /// Creates the multiplicative identity monomial (all exponents zero).
    pub fn one() -> Self {
        Mono { exps: [0u32; N] }
    }

    /// The degree of the monomial.
    #[inline(always)]
    pub fn deg(&self) -> u32 {
        self.exps.iter().sum()
    }

    /// Checks if deg(self.x) <= deg(other.x) for all vars x
    /// Can be used for pruning iterated products.
    #[inline(always)]
    pub fn all_leq(&self, other: &Self) -> bool {
        (0..N).all(|i| self.exps[i] <= other.exps[i])
    }

    #[inline(always)]
    pub fn all_le(&self, other: &Self) -> bool {
        (0..N).all(|i| self.exps[i] < other.exps[i])
    }
}

impl<const N: usize> PartialOrd for Mono<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> Ord for Mono<N> {
    /// Compares two monomials in degree-reverse-lexicographic order (degrevlex).
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let deg_self: u32 = self.deg();
        let deg_other: u32 = other.deg();
        if deg_self != deg_other {
            return deg_self.cmp(&deg_other);
        }
        // degrevlex
        // walk backwards from xn to x1, and reverse the outcome if unequal
        for i in (0..N).rev() {
            if self.exps[i] != other.exps[i] {
                // other.cmp(self) is correct here
                return other.exps[i].cmp(&self.exps[i]);
            }
        }
        std::cmp::Ordering::Equal
    }
}

impl<const N: usize> Deref for Mono<N> {
    type Target = [u32; N];
    fn deref(&self) -> &Self::Target {
        &self.exps
    }
}

// some sugar
// impl From<u32> for Mono<1> {
//     fn from(exp: u32) -> Self {
//         Mono { exps: [exp] }
//     }
// }

impl From<Mono<1>> for u32 {
    fn from(m: Mono<1>) -> Self {
        m.exps[0]
    }
}

impl<const N: usize, T> From<T> for Mono<N>
where
    T: Borrow<[u32; N]>,
{
    fn from(exps: T) -> Self {
        Mono {
            exps: *exps.borrow(),
        }
    }
}

impl<const N: usize> Default for Mono<N> {
    fn default() -> Self {
        Self::one()
    }
}

impl<const N: usize> Mul for Mono<N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Mono {
            exps: std::array::from_fn(|i| self.exps[i] + rhs.exps[i]),
        }
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<const N: usize> MulAssign for Mono<N> {
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.exps[i] += rhs.exps[i];
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term<T: PolyCoef, const N: usize> {
    pub mono: Mono<N>,
    pub coef: T,
}

impl<T: PolyCoef, const N: usize> From<(Mono<N>, T)> for Term<T, N> {
    fn from((m, c): (Mono<N>, T)) -> Self {
        Term { mono: m, coef: c }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly<T: PolyCoef = i32, const N: usize = 1> {
    pub terms: Vec<Term<T, N>>,
}

// invariant: we do not keep zero coefficients!
impl<T: PolyCoef, const N: usize> Poly<T, N> {
    pub fn new() -> Self {
        Poly { terms: vec![] }
    }

    /// Accepts any vector of terms, including duplicates, zero coefficients, and out of order.
    pub fn from_raw_terms(mut terms: Vec<Term<T, N>>) -> Self {
        canonicalize_terms_inplace(&mut terms);
        Poly { terms }
    }

    pub fn new_with_capacity(cap: usize) -> Self {
        Poly {
            terms: Vec::with_capacity(cap),
        }
    }

    /// Parses a polynomial from a string.
    ///
    /// The number of variables is given ahead of time. Parsed variable names will be mapped to
    /// monomial indices in alphabetical order. Any implicit variables (e.g. if only x appears while
    /// parsing Poly::<T, 3>) will be mapped to the leftover high indices and will have, as
    /// expected, exponent 0. Variable names are otherwise arbitrary strings of the following
    /// format [a-zA-Z]{0-9}+
    ///
    /// To force a certain vairable order, a zero-coefficient term can be inserted
    /// ```rust
    /// use pe_lib::math::poly::{Poly,Mono};
    /// let p = Poly::<i32, 3>::parse("5 + y").unwrap();
    /// assert_eq!(p.terms[1].mono, Mono::new([1, 0, 0])); // y is the only var, so it gets index 0
    /// let p = Poly::<i32, 3>::parse("5 + 0x + y").unwrap();
    /// assert_eq!(p.terms[1].mono, Mono::new([0, 1, 0])); // the dummy x forces y to index 1
    /// ```
    pub fn parse(s: &str) -> anyhow::Result<Self>
    where
        T: std::str::FromStr,
    {
        let mut seen_var_names = Vec::<String>::with_capacity(N);
        let term_rx = regex::Regex::new(r"(\p{L}[0-9]*)(?:\^([1-9][0-9]*))?").unwrap();

        // two passes: first to collect + sort variable names
        for term in s.split('+').map(str::trim) {
            if term.is_empty() {
                anyhow::bail!("Empty term in polynomial string");
            }
            let mut cx = 0;
            // skip coefficient
            while cx < term.len() && (term.as_bytes()[cx] as char).is_ascii_digit() {
                cx += 1;
            }
            term_rx.captures_iter(&term[cx..]).for_each(|cap| {
                let var_name = cap.get(1).unwrap().as_str();
                if !seen_var_names.iter().any(|v| v == var_name) {
                    seen_var_names.push(var_name.to_string());
                }
            });
        }

        // no need for a map for any reasonable N, will just look up index in vec
        seen_var_names.sort();

        if seen_var_names.is_empty() {
            return Ok(Self::zero());
        } else if seen_var_names.len() > N {
            anyhow::bail!(
                "Too many variables for Poly<_, <{N}>> (got {}: {:?})",
                seen_var_names.len(),
                seen_var_names
            );
        }

        let mut terms: Vec<Term<T, N>> = Vec::new();
        for term in s.split('+').map(str::trim) {
            // assume non-empty
            let mut coef = String::new();
            let mut cx = 0;
            // parse coefficient
            while cx < term.len() && (term.as_bytes()[cx] as char).is_ascii_digit() {
                coef.push(term.as_bytes()[cx] as char);
                cx += 1;
            }
            let coef: T = if coef.is_empty() {
                T::one()
            } else {
                coef.parse().map_err(|_| {
                    anyhow::anyhow!("Failed to parse coefficient {coef} at index {cx}")
                })?
            };
            let mut mono = Mono::<N>::one();
            term_rx.captures_iter(&term[cx..]).for_each(|cap| {
                let var_name = cap.get(1).unwrap().as_str();
                // SAFETY: we checked above that the variable name exists
                let vx = seen_var_names.iter().position(|v| v == var_name).unwrap();
                let exp = match cap.get(2) {
                    Some(m) => m.as_str().parse().unwrap_or(1),
                    None => 1,
                };
                mono.exps[vx] = exp;
            });
            terms.push((mono, coef).into());
        }
        Ok(Poly::from_raw_terms(terms))
    }
    /// Creates a monomial
    #[inline(always)]
    pub fn monomial(mono: impl Into<Mono<N>>, coef: impl Into<T>) -> Self {
        Poly {
            terms: vec![Term {
                mono: mono.into(),
                coef: coef.into(),
            }],
        }
    }

    /// Creates a constant polynomial
    #[inline(always)]
    pub fn constant(coef: impl Into<T>) -> Self {
        Self::monomial(Mono::one(), coef)
    }

    /// Keeps only those terms whose degree for all variables does not exceed those in the given monomial.
    /// ```rust
    /// use pe_lib::math::poly::{Poly,Mono};
    /// let mut p = Poly::<i32, 2>::parse("5 + x + 2xy + 3x^2y + 4xy^2 + 5x^2y^2").unwrap();
    /// p.retain_terms_leq(&[1,2].into());
    /// assert_eq!(p, Poly::<i32, 2>::parse("5 + x + 2xy + 4xy^2").unwrap());
    /// ```
    #[inline(always)]
    pub fn retain_terms_leq(&mut self, max_mono: &Mono<N>) {
        self.terms.retain(|term| term.mono.all_leq(max_mono));
    }
}

fn canonicalize_terms_inplace<T: PolyCoef, const N: usize>(terms: &mut Vec<Term<T, N>>) {
    terms.sort_unstable_by(|a, b| a.mono.cmp(&b.mono));

    let len = terms.len();
    let mut write = 0usize;
    let mut read = 0usize;

    // raw pointers avoid aliasing UB with &mut
    let base = terms.as_mut_ptr();

    while read < len {
        // move first of the run into `write`
        unsafe {
            let head = core::ptr::read(base.add(read)); // move-out
            core::ptr::write(base.add(write), head); // move-in
        }
        read += 1;

        // merge later duplicates into v[write]
        while read < len {
            let same = unsafe { (&*base.add(read)).mono == (&*base.add(write)).mono };
            if !same {
                break;
            }
            let term = unsafe { core::ptr::read(base.add(read)) }; // move-out coeff
            unsafe {
                (&mut *base.add(write)).coef += &term.coef;
            }
            read += 1;
        }

        // drop run if sum is zero; else keep
        let zero = unsafe { (&*base.add(write)).coef.is_zero() };
        if zero {
            unsafe {
                core::ptr::drop_in_place(base.add(write));
            }
        } else {
            write += 1;
        }
    }

    unsafe {
        terms.set_len(write);
    }
}

// generic helper impls
impl<T: PolyCoef, const N: usize> Deref for Poly<T, N> {
    type Target = Vec<Term<T, N>>;
    fn deref(&self) -> &Self::Target {
        &self.terms
    }
}

impl<T: PolyCoef, const N: usize> Default for Poly<T, N> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<T: PolyCoef + std::fmt::Display, const N: usize> std::fmt::Display for Poly<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const SMALL_VARS: [&str; 4] = ["x", "y", "z", "w"];
        let mut terms: Vec<String> = Vec::new();
        for term in &self.terms {
            let mut s = String::new();
            s.push_str(&term.coef.to_string());
            for (i, &exp) in term.mono.exps.iter().enumerate() {
                if exp > 0 {
                    if N <= SMALL_VARS.len() {
                        s.push_str(SMALL_VARS[i]);
                    } else {
                        s.push('x');
                        s.push_str(&i.to_string());
                    }
                    if exp > 1 {
                        s.push('^');
                        s.push_str(&exp.to_string());
                    }
                }
            }
            terms.push(s);
        }
        terms.reverse(); // grevlex display order: high terms first
        write!(f, "{}", terms.join(" + "))
    }
}

impl<T: PolyCoef, const N: usize, Rhs> Add<Rhs> for Poly<T, N>
where
    Rhs: Borrow<Poly<T, N>>,
{
    type Output = Poly<T, N>;

    #[inline(always)]
    fn add(self, rhs: Rhs) -> Self::Output {
        let mut out = self.clone();
        out += rhs;
        out
    }
}

impl<T: PolyCoef, const N: usize, Rhs> AddAssign<Rhs> for Poly<T, N>
where
    Rhs: Borrow<Poly<T, N>>,
{
    fn add_assign(&mut self, rhs: Rhs) {
        let a = mem::take(&mut self.terms); // old self, sorted
        let b = &rhs.borrow().terms; // sorted
        self.terms = Vec::with_capacity(a.len() + b.len());
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].mono.cmp(&b[j].mono) {
                std::cmp::Ordering::Less => {
                    self.terms.push(a[i].clone());
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    self.terms.push(b[j].clone());
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let mut a_term = a[i].clone();
                    a_term.coef += &b[j].coef;
                    if !a_term.coef.is_zero() {
                        self.terms.push(a_term);
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        self.terms.extend_from_slice(&a[i..]);
        self.terms.extend_from_slice(&b[j..]);
    }
}

impl<T: PolyCoef, const N: usize, Rhs> Mul<Rhs> for Poly<T, N>
where
    Rhs: Borrow<Poly<T, N>>,
{
    type Output = Self;

    fn mul(self, rhs: Rhs) -> Poly<T, N> {
        let rhs: &Poly<T, N> = rhs.borrow();
        let mut buf = Vec::with_capacity(self.terms.len() * rhs.terms.len());
        for term0 in &self.terms {
            for term1 in &rhs.terms {
                buf.push(Term {
                    mono: term0.mono * term1.mono,
                    coef: term0.coef.clone() * term1.coef.clone(),
                });
            }
        }
        Self::from_raw_terms(buf)
    }
}

impl<T: PolyCoef, const N: usize, Rhs> MulAssign<Rhs> for Poly<T, N>
where
    Rhs: Borrow<Poly<T, N>>,
{
    fn mul_assign(&mut self, rhs: Rhs) {
        let lhs = std::mem::take(self);
        *self = lhs * rhs;
    }
}

// forward non-reference arithmetic impls
//

// num_traits impls
impl<T: PolyCoef, const N: usize> Zero for Poly<T, N> {
    fn zero() -> Self {
        Self::new()
    }

    fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }
}

impl<T: PolyCoef, const N: usize> One for Poly<T, N> {
    fn one() -> Self {
        Self::monomial(Mono::one(), T::one())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type P = Poly<i32, 1>;
    type P3 = Poly<u64, 3>;

    #[test]
    fn test_mono_grevlex() {
        let x0x0 = Mono::new([2, 0, 0]); // x0^2
        let x1x1 = Mono::new([0, 2, 0]); // x1^2
        let x2x2 = Mono::new([0, 0, 2]); // x2^2
        let x0x1 = Mono::new([1, 1, 0]); // x0x1
        let x1x2 = Mono::new([0, 1, 1]); // x1x2
        let x0x2 = Mono::new([1, 0, 1]);
        let x0 = Mono::new([1, 0, 0]);
        let x1 = Mono::new([0, 1, 0]);
        let x2 = Mono::new([0, 0, 1]);
        // note that this means that higher degree will be at the back of the list!
        assert!(x0x0 > x0x1);
        assert!(x0x1 > x1x1);
        assert!(x1x1 > x0x2);
        let mut monos = vec![x0x0, x0, x1x1, x2x2, x1, x0x1, x1x2, x2, x0x2];
        monos.sort_unstable();
        monos.reverse();
        assert_eq!(monos, vec![x0x0, x0x1, x1x1, x0x2, x1x2, x2x2, x0, x1, x2]);
    }

    #[test]
    fn test_new_polynomial() {
        let poly: Poly<i32, 1> = Poly::new();
        assert!(poly.is_zero());

        let multivar: Poly<i32, 3> = Poly::new();
        assert!(multivar.is_zero());
    }

    #[test]
    fn test_from_coefficient() {
        let poly = P::constant(5);
        assert_eq!(poly.terms[0].coef, 5);
        assert_eq!(poly.terms.len(), 1);
    }

    #[test]
    fn test_parse_univar() {
        let poly = P::parse("3 + 2x + 5x^2").unwrap();
        assert_eq!(poly.terms.len(), 3);
        assert_eq!(poly.terms[0], (Mono::new([0]), 3).into());
        assert_eq!(poly.terms[1], (Mono::new([1]), 2).into());
        assert_eq!(poly.terms[2], (Mono::new([2]), 5).into());
    }

    #[test]
    fn test_parse_multivar() {
        // we expect terms in grevlex order (reversed in vec)
        let poly = Poly::<i32, 3>::parse("3 + 4y + 2x + 7z^3 + 5x^2y").unwrap();
        assert_eq!(poly.terms.len(), 5);
        assert_eq!(poly.terms[0], (Mono::new([0, 0, 0]), 3).into());
        assert_eq!(poly.terms[1], (Mono::new([0, 1, 0]), 4).into());
        assert_eq!(poly.terms[2], (Mono::new([1, 0, 0]), 2).into());
        assert_eq!(poly.terms[3], (Mono::new([0, 0, 3]), 7).into());
        assert_eq!(poly.terms[4], (Mono::new([2, 1, 0]), 5).into());
    }

    #[test]
    fn test_term_canonicalize() {
        let terms = vec![
            (Mono::new([1]), 2).into(),
            (Mono::new([0]), 3).into(),
            (Mono::new([1]), 4).into(),
            (Mono::new([2]), 5).into(),
            (Mono::new([0]), -3).into(),
        ];
        let poly = Poly::<i32, 1>::from_raw_terms(terms);
        assert_eq!(poly.terms.len(), 2);
        assert_eq!(poly.terms[0], (Mono::new([1]), 6).into());
        assert_eq!(poly.terms[1], (Mono::new([2]), 5).into());
    }

    #[test]
    fn test_add_univar() {
        let poly1 = P::parse("3 + 2x").unwrap();
        let poly2 = P::parse("1 + 4x").unwrap();

        let result = poly1 + poly2; // Should be 4 + 6x
        assert_eq!(result, P::parse("4 + 6x").unwrap());
    }

    #[test]
    fn test_add_multivar() {
        // we need the zeros to force the polynomials to be distinct --
        // there is no shared symbolic processing, and parse is just a shorthand!
        let poly1 = P3::parse("3 + 2x + 4y + 0z").unwrap();
        let poly2 = P3::parse("1 + 4x + 0y + 5z").unwrap();
        assert_eq!(poly1 + poly2, P3::parse("4 + 6x + 4y + 5z").unwrap());
    }

    #[test]
    fn test_add_assign_multivar() {
        let mut poly1 = P3::parse("3 + 2x + 4y + 0z").unwrap();
        let poly2 = P3::parse("1 + 4x + 0y + 5z").unwrap();
        poly1 += poly2;
        assert_eq!(poly1, P3::parse("4 + 6x + 4y + 5z").unwrap());
    }

    #[test]
    fn test_mul_univar() {
        let poly1 = P::parse("2 + 3x").unwrap();
        let poly2 = P::parse("1 + x").unwrap();
        assert_eq!(poly1 * poly2, P::parse("2 + 5x + 3x^2").unwrap());
    }

    #[test]
    fn test_mul_assign_univar() {
        let mut poly1 = P::parse("2 + 3x").unwrap();
        let poly2 = P::parse("1 + x").unwrap();
        poly1 *= poly2;
        assert_eq!(poly1, P::parse("2 + 5x + 3x^2").unwrap());
    }

    #[test]
    fn test_mul_multivar() {
        let poly1 = P3::parse("2 + 3x + y + 0z").unwrap();
        let poly2 = P3::parse("1 + x + 0y + z").unwrap();
        let out = poly1 * poly2;
        let expect = P3::parse("2 + 5x + 3x^2 + y + xy + 2z + 3xz + yz").unwrap();
        out.terms
            .iter()
            .zip(expect.terms.iter())
            .for_each(|(a, b)| {
                assert_eq!(a, b);
            });
    }

    /// Checks that the zero-polynomial is valid
    #[test]
    fn test_zero_polynomial() {
        let poly = P::parse("0").unwrap();
        assert!(poly.is_zero());
    }

    #[test]
    fn test_display_short() {
        let poly = P3::parse("3 + 2x + 4y + 5x^2y + 7z^3").unwrap();
        assert_eq!(poly.to_string(), "5x^2y + 7z^3 + 2x + 4y + 3");
    }

    #[test]
    fn test_display_long() {
        let p = Poly::<i32, 5>::from_raw_terms(vec![
            (Mono::new([2, 0, 1, 0, 0]), 3).into(),
            (Mono::new([0, 1, 0, 0, 0]), 4).into(),
            (Mono::new([1, 0, 0, 2, 0]), 5).into(),
            (Mono::new([0, 0, 0, 0, 4]), 7).into(),
        ]);
        assert_eq!(p.to_string(), "7x4^4 + 3x0^2x2 + 5x0x3^2 + 4x1");
    }
}
