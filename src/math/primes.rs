// Most of this module is going to be a shameless ripoff of
// labmath (https://pypi.org/project/labmath/)

use core::ops::{Add, Mul, Neg, Sub};
use fixedbitset::FixedBitSet;
use num_traits::{Euclid, One, Zero};

const INIT_PRIMES: &[u32] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
const BASE_PRIMES: &[u32] = &[3, 5, 7];

pub trait PG:
    Euclid + Zero + Clone + Mul<Output = Self> + Sub<Output = Self> + TryInto<u32> + From<u32>
{
}
impl<T: Euclid + Zero + Clone + Mul<Output = Self> + Sub<Output = Self> + TryInto<u32> + From<u32>>
    PG for T
{
}

// first, we're going to steal primegen. as a token effort, we will compress the bitvector by 2
// and we'll keep index state between segments
struct PrimeGenSieve<T: PG> {
    pl: Vec<u32>,
    // we persist per-prime indices to only do the modulo computation
    // once per prime
    ks: Vec<u32>,
    pg: Box<PrimeGen<T>>,
    ll: T,
    n: T,
    nn: T,
    sieve_vec: FixedBitSet,
    ix: usize,
}

impl<T: PG> PrimeGenSieve<T> {
    pub fn new() -> Self {
        let mut pg = Box::new(PrimeGen::<T>::new());
        for _ in 0..BASE_PRIMES.len() {
            let _ = pg.next();
        }
        let n0 = pg.next().unwrap();
        // we don't know the segment end, but do know the segment start
        let ll0 = n0.clone() * n0.clone();

        let ks0: Vec<u32> = BASE_PRIMES
            .iter()
            .map(|&p| Self::compute_init_k(p.into(), ll0.clone()))
            .collect();

        let mut dummy = PrimeGenSieve {
            pl: BASE_PRIMES.to_vec().into_iter().collect(),
            ks: ks0,
            pg,
            ll: ll0.clone(),
            nn: ll0, // dummy value to be instarotated
            n: n0,   // dummy value to be instarotated
            ix: 0,
            // the fields below are dummies and are only filled on the first rotate
            sieve_vec: FixedBitSet::with_capacity(0),
        };

        dummy.rotate_segments();
        dummy
    }

    fn compute_init_k(p: T, ll: T) -> u32 {
        let p_u32: u32 = p.clone().try_into().ok().unwrap();
        let mut k = (p_u32 - (ll % p).try_into().ok().unwrap()) % p_u32;
        // if we have an odd naive index that means our starting point was even
        // so we go to the next one. look ma, no branches!
        k += (k % 2) * p_u32;
        k / 2
    }

    fn rotate_segments(&mut self) {
        self.ll = self.nn.clone();
        self.n = self.pg.next().unwrap();
        self.nn = self.n.clone() * self.n.clone();

        // unlike in the original primegen we do implement the compression by 2
        let sl_u32 = (self.nn.clone() - self.ll.clone()).try_into().ok().unwrap() / 2;
        self.sieve_vec = FixedBitSet::with_capacity(sl_u32 as usize);

        // do the sieving! differences from labmmath: 1 = composite, we've culled events
        // TODO wheel30 this mofo
        let bits = self.sieve_vec.as_mut_slice();
        let word_bits = usize::BITS as usize;
        for i in 0..self.ks.len() {
            let mut k = self.ks[i];
            let p_u32: u32 = self.pl[i];
            while k < sl_u32 {
                let w = (k as usize) / word_bits;
                let b = (k as usize) % word_bits;
                unsafe {
                    *bits.get_unchecked_mut(w) |= 1usize << b;
                }
                k += p_u32;
            }
            self.ks[i] = k; // store the next index for this prime
        }
    }

    #[inline(always)]
    fn next(&mut self) -> Option<T> {
        while self.ix < self.sieve_vec.len() && self.sieve_vec[self.ix] {
            self.ix += 1;
        }
        if self.ix >= self.sieve_vec.len() {
            return None;
        }
        // assumes each individual segment length will never be over 2^32, pretty safe
        let res = self.ll.clone() + T::from((self.ix * 2) as u32);
        self.ix += 1;
        Some(res)
    }

    fn finalize_segment(&mut self) {
        let sl_usize = (self.nn.clone() - self.ll.clone()).try_into().ok().unwrap() / 2;
        // adjust our prime indices to be valid for the next segment
        for i in 0..self.ks.len() {
            self.ks[i] -= sl_usize;
        }
        // push the next prime and compute its starting index.
        // self.nn will be the ll of the next segment, so we use it
        self.pl.push(self.n.clone().try_into().ok().unwrap());
        self.ks
            .push(Self::compute_init_k(self.n.clone(), self.nn.clone()));
        self.ix = 0;
    }
}

enum PrimegenState<T: PG> {
    Init(usize),
    Segment(PrimeGenSieve<T>),
}

/// More or less a line-for-line re-implementation of labmath's primegen
/// we'll tweak and benchmark it later
pub struct PrimeGen<T: PG> {
    state: PrimegenState<T>,
}

impl<T: PG> PrimeGen<T> {
    pub fn new() -> Self {
        PrimeGen {
            state: PrimegenState::Init(0),
        }
    }
}

impl<T: PG> Default for PrimeGen<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PG> Iterator for PrimeGen<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match &mut self.state {
                PrimegenState::Init(ix) => {
                    if *ix >= INIT_PRIMES.len() {
                        let segment = PrimeGenSieve::new();
                        self.state = PrimegenState::Segment(segment);
                        continue;
                    } else {
                        let out = INIT_PRIMES[*ix].into();
                        *ix += 1; // increment the index
                        return Some(out);
                    }
                }
                PrimegenState::Segment(sieve) => match sieve.next() {
                    Some(prime) => {
                        return Some(prime);
                    }
                    None => {
                        sieve.finalize_segment();
                        sieve.rotate_segments();
                        continue;
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_primegen() {
        let pg: Vec<u32> = PrimeGen::<u32>::new().take(100).collect();
        let cmp: Vec<u32> = vec![
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
            181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271,
            277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
            383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479,
            487, 491, 499, 503, 509, 521, 523, 541,
        ];
        assert_eq!(pg, cmp);
    }
}

// primefac and friends -- you guessed it, stolen from labmath
// not only that -- but ported as AI slop. the horror.

pub trait Int:
    Clone
    + Ord
    + Zero
    + One
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Euclid
    + From<u32>
{
}
impl<T> Int for T where
    T: Clone
        + Ord
        + Zero
        + One
        + Neg<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Euclid
        + From<u32>
{
}

// ---------- small helpers ----------

#[inline]
fn c<I: From<u32>>(u: u32) -> I {
    I::from(u)
}
#[inline]
fn is_neg<I: Int>(x: &I) -> bool {
    *x < I::zero()
}
#[inline]
fn abs<I: Int>(x: I) -> I {
    if is_neg(&x) { -x } else { x }
}
#[inline]
fn is_zero<I: Int>(x: &I) -> bool {
    x.clone() == I::zero()
}
#[inline]
fn is_one<I: Int>(x: &I) -> bool {
    x.clone() == I::one()
}
#[inline]
fn is_even<I: Int>(x: &I) -> bool {
    let two = c::<I>(2);
    x.rem_euclid(&two) == I::zero()
}
#[inline]
fn is_odd<I: Int>(x: &I) -> bool {
    !is_even(x)
}

fn ipow<I: Int>(mut base: I, mut exp: u32) -> I {
    let mut acc = I::one();
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc * base.clone();
        }
        exp >>= 1;
        if exp > 0 {
            base = base.clone() * base;
        }
    }
    acc
}

fn pow_mod<I: Int>(base: &I, exp: &I, modulus: &I) -> I {
    let mut acc = I::one();
    let two = c::<I>(2);
    let mut base = base.rem_euclid(modulus);
    let mut exp = exp.clone();
    while !is_zero(&exp) {
        if is_odd(&exp) {
            acc = {
                let a: &I = &(acc * base.clone());
                a.rem_euclid(modulus)
            };
        }
        exp = exp.div_euclid(&two);
        let bb = base.clone() * base;
        base = {
            let a: &I = &bb;
            a.rem_euclid(modulus)
        };
    }
    acc
}

pub fn gcd<I: Int>(mut a: I, mut b: I) -> I {
    while !is_zero(&b) {
        let r = a.rem_euclid(&b);
        a = b;
        b = r;
    }
    abs(a)
}

pub fn modinv<I: Int>(a: &I, M: &I) -> Option<I> {
    if *M <= I::zero() {
        return None;
    }

    let (mut a, mut m, mut r, mut x) = (a.clone(), M.clone(), I::zero(), I::one());
    while !is_zero(&m) {
        let q = a.div_euclid(&m);
        (a, m, r, x) = (m.clone(), a.rem_euclid(&m), x.clone() - q * r.clone(), r)
    }
    if is_one(&a) {
        Some(x.rem_euclid(M))
    } else {
        None
    }
}

/// Exact port of Python `introot`.
pub fn introot<I: Int>(n: &I, r: u32) -> Option<I> {
    if is_neg(n) {
        if r % 2 == 0 {
            return None;
        }
        let x = introot(&abs(n.clone()), r)?;
        return Some(-x);
    }
    if *n < c(2) {
        return Some(n.clone());
    }
    if r == 1 {
        return Some(n.clone());
    }
    if r == 2 {
        // isqrt by doubling bound + binary search
        let two = c::<I>(2);
        let mut upper = I::one();
        while ipow(upper.clone(), 2) <= n.clone() {
            upper = upper * two.clone();
        }
        let mut lower = upper.clone().div_euclid(&two);
        while lower.clone() + I::one() < upper {
            let mid = (lower.clone() + upper.clone()).div_euclid(&two);
            let m2 = ipow(mid.clone(), 2);
            if m2 == *n {
                return Some(mid);
            }
            if m2 < *n {
                lower = mid;
            } else {
                upper = mid;
            }
        }
        return Some(lower);
    }
    // generic r >= 3
    let two = c::<I>(2);
    let mut upper = I::one();
    while ipow(upper.clone(), r) <= n.clone() {
        upper = upper * two.clone();
    }
    let mut lower = upper.clone().div_euclid(&two);
    while lower.clone() + I::one() < upper {
        let mid = (lower.clone() + upper.clone()).div_euclid(&two);
        let m = ipow(mid.clone(), r);
        if m == *n {
            return Some(mid);
        }
        if m < *n {
            lower = mid;
        } else {
            upper = mid;
        }
    }
    Some(lower)
}

/// Exact port of Python `ispower` for r>0.
pub fn ispower_r<I: Int>(n: &I, r: u32) -> Option<I> {
    if r == 2
        && ({
            let n: &I = &c(4);
            n.rem_euclid(n)
        }) == c(2)
    {
        return None;
    }
    if r == 3 {
        let m7 = {
            let n: &I = &c(7);
            n.rem_euclid(n)
        };
        if m7 == c(2) || m7 == c(4) || m7 == c(6) {
            return None;
        }
    }
    let x = introot(n, r)?;
    if ipow(x.clone(), r) == *n {
        Some(x)
    } else {
        None
    }
}

/// Python `ispower(n)` mode using your `PrimeGen<T: PG>`.
pub fn ispower<I: Int, J: PG>(n: &I) -> Option<(I, u32)> {
    if *n == I::zero() || *n == I::one() || *n == -I::one() {
        return Some((n.clone(), 1));
    }
    // bit_length(|n|) TODO kinda hacky
    let two = c::<I>(2);
    let mut bits = 0usize;
    let mut t = abs(n.clone());
    while t > I::zero() {
        t = t.div_euclid(&two);
        bits += 1;
    }
    let bits = bits as u32;
    let mut pg = PrimeGen::<J>::new();
    while let Some(p) = pg.next().and_then(|x| x.try_into().ok()) {
        if p > bits + 1 {
            break;
        }
        if let Some(root) = ispower_r(n, p) {
            return Some((root, p));
        }
    }
    None
}

/// Exact port of Python `jacobi(a, n)`.
/// Returns None if n is even or n < 0; else Some(-1|0|1).
pub fn jacobi<I: Int>(mut a: I, mut n: I) -> Option<i8> {
    if is_even(&n) || is_neg(&n) {
        return None;
    }
    if is_zero(&a) {
        return Some(0);
    }
    if is_one(&a) {
        return Some(1);
    }

    a = {
        let a: &I = &a;
        let n: &I = &n;
        a.rem_euclid(n)
    };
    let mut t: i8 = 1;

    let two = c::<I>(2);
    let three = c::<I>(3);
    let four = c::<I>(4);
    let five = c::<I>(5);
    let eight = c::<I>(8);

    while !is_zero(&a) {
        while is_even(&a) {
            a = a.div_euclid(&two);
            let nm8 = {
                let a: &I = &n;
                let n: &I = &eight;
                a.rem_euclid(n)
            };
            if nm8 == three || nm8 == five {
                t = -t;
            }
        }
        core::mem::swap(&mut a, &mut n);
        let am4 = {
            let a: &I = &a;
            let n: &I = &four;
            a.rem_euclid(n)
        };
        let nm4 = {
            let a: &I = &n;
            let n: &I = &four;
            a.rem_euclid(n)
        };
        if am4 == three && nm4 == three {
            t = -t;
        }
        a = {
            let a: &I = &a;
            let n: &I = &n;
            a.rem_euclid(n)
        };
    }
    if is_one(&n) { Some(t) } else { Some(0) }
}

pub const TB_DEFAULT: &[u32] = &[3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59];

/// Faithful port of the given Python `isprime`.
pub fn isprime<I: Int>(n: I, tb: Option<&[u32]>) -> bool {
    let tb = tb.unwrap_or(TB_DEFAULT);

    // 1) trial division and small cases
    if n.clone().rem_euclid(&c::<I>(2)) == I::zero() || n.clone() < c::<I>(3) {
        return n == c(2);
    }
    for &p in tb {
        let p = c::<I>(p);
        if n.clone().rem_euclid(&p) == I::zero() {
            return n == p;
        }
    }

    // 2) base-2 strong probable prime
    let one = I::one();
    let two = c::<I>(2);
    let n_minus_1 = n.clone() - one.clone();
    let mut t = n_minus_1.clone().div_euclid(&two);
    let mut s = 1usize;
    while is_even(&t) {
        t = t.div_euclid(&two);
        s += 1;
    }
    let mut x = pow_mod(&c::<I>(2), &t, &n);
    if x != one && x != n.clone() - one.clone() {
        let mut j = 1usize;
        while j < s {
            x = {
                let a = x.clone() * x.clone();
                let n: &I = &n;
                a.rem_euclid(n)
            };
            if x == one {
                return false;
            }
            if x == n.clone() - one.clone() {
                break;
            }
            j += 1;
        }
        if j == s {
            return false;
        }
    }

    // 3) select D for strong Lucas PRP
    let mut D = c::<I>(5);
    loop {
        match jacobi(D.clone(), n.clone()) {
            Some(0) => return D == n,
            Some(-1) => break,
            Some(1) => {}
            None => unreachable!(),
            _ => {}
        }
        D = -(c::<I>(2) + D.clone());
        match jacobi(D.clone(), n.clone()) {
            Some(0) => return (-D.clone()) == n,
            Some(-1) => break,
            Some(1) => {}
            None => unreachable!(),
            _ => {}
        }
        if D == -c::<I>(13) && ispower_r(&n, 2).is_some() {
            return false;
        }
        D = -(D + c::<I>(2)); // revert
        D = D + c::<I>(4); // advance +4
    }

    // run slprp(n, 1, (1 - D) // 4)
    let b = (I::one() - D.clone()).div_euclid(&c::<I>(4));
    let g = gcd(n.clone(), b.clone());
    if g > I::one() && g < n {
        return false;
    }

    let mut s_l = 1usize;
    let mut t_l = (n.clone() + I::one()).div_euclid(&c::<I>(2));
    while is_even(&t_l) {
        s_l += 1;
        t_l = t_l.div_euclid(&c::<I>(2));
    }

    let mut v = c::<I>(2);
    let mut w = I::one();
    let mut q = I::one();
    let mut Q = I::one();

    // iterate all bits of t_l: bin(t)[2:]
    let mut bits: Vec<bool> = {
        let mut v = Vec::new();
        let mut e = t_l.clone();
        while e > I::zero() {
            v.push(is_odd(&e));
            e = e.div_euclid(&c::<I>(2));
        }
        v.reverse();
        v
    };

    for bit in bits.drain(..) {
        q = {
            let a = q * Q.clone();
            let n: &I = &n;
            a.rem_euclid(n)
        };
        if bit {
            Q = {
                let a = q.clone() * b.clone();
                let n: &I = &n;
                a.rem_euclid(n)
            };
            let wv = w.clone() * v.clone();
            v = {
                let a = wv.clone() - q.clone();
                let n: &I = &n;
                a.rem_euclid(n)
            };
            w = {
                let a = w.clone() * w.clone() - c::<I>(2) * q.clone() * b.clone();
                let n: &I = &n;
                a.rem_euclid(n)
            };
        } else {
            Q = q.clone();
            let wv = w.clone() * v.clone();
            w = {
                let a = wv.clone() - q.clone();
                let n: &I = &n;
                a.rem_euclid(n)
            };
            v = {
                let a = v.clone() * v.clone() - c::<I>(2) * q.clone();
                let n: &I = &n;
                a.rem_euclid(n)
            };
        }
    }

    if v == I::zero() {
        return true;
    }
    let inv_d = modinv(&D, &n).expect("D and n coprime after selection");
    let chk = {
        let a = (c::<I>(2) * w.clone() - v.clone()) * inv_d;
        let n: &I = &n;
        a.rem_euclid(n)
    };
    if chk == I::zero() {
        return true;
    }

    q = pow_mod(&b, &t_l, &n);
    for _ in 1..s_l {
        v = {
            let a = v.clone() * v.clone() - c::<I>(2) * q.clone();
            let n: &I = &n;
            a.rem_euclid(n)
        };
        if v == I::zero() {
            return true;
        }
        q = {
            let a = q.clone() * q.clone();
            let n: &I = &n;
            a.rem_euclid(n)
        };
    }
    false
}

#[cfg(test)]
mod tests_fac {
    use super::*;

    // ---- jacobi ----
    #[test]
    fn jacobi_tables_15() {
        // Python:
        // [-10,-7,-4,-2,-1,0,1,2,4,7,10] -> [0,1,-1,-1,-1,0,1,1,1,-1,0]
        let a_vals = [-10, -7, -4, -2, -1, 0, 1, 2, 4, 7, 10];
        let expect = [0, 1, -1, -1, -1, 0, 1, 1, 1, -1, 0];
        for (a, e) in a_vals.into_iter().zip(expect) {
            assert_eq!(jacobi::<i128>(a.into(), 15i128), Some(e));
        }
    }

    #[test]
    fn jacobi_tables_13() {
        // [-10,-9,-4,-2,-1,0,1,2,4,9,10] -> [1,1,1,-1,1,0,1,-1,1,1,1]
        let a_vals = [-10, -9, -4, -2, -1, 0, 1, 2, 4, 9, 10];
        let expect = [1, 1, 1, -1, 1, 0, 1, -1, 1, 1, 1];
        for (a, e) in a_vals.into_iter().zip(expect) {
            assert_eq!(jacobi::<i128>(a.into(), 13i128), Some(e));
        }
    }

    #[test]
    fn jacobi_tables_11() {
        // [-10,-9,-4,-2,-1,0,1,2,4,9,10] -> [1,-1,-1,1,-1,0,1,-1,1,1,-1]
        let a_vals = [-10, -9, -4, -2, -1, 0, 1, 2, 4, 9, 10];
        let expect = [1, -1, -1, 1, -1, 0, 1, -1, 1, 1, -1];
        for (a, e) in a_vals.into_iter().zip(expect) {
            assert_eq!(jacobi::<i128>(a.into(), 11i128), Some(e));
        }
    }

    #[test]
    fn jacobi_none_for_bad_n() {
        assert!(jacobi::<i128>(3.into(), 10i128).is_none()); // n even
        assert!(jacobi::<i128>(3.into(), -11i128).is_none()); // n < 0
    }

    // ---- introot ----

    #[test]
    fn introot_examples() {
        assert_eq!(introot::<i128>(&(-729).into(), 3), Some((-9).into()));
        assert_eq!(introot::<i128>(&(-728).into(), 3), Some((-8).into()));
        assert_eq!(introot::<i128>(&1023i128, 2), Some(31i128));
        assert_eq!(introot::<i128>(&1024i128, 2), Some(32i128));
        assert_eq!(introot::<i128>(&(-8).into(), 2), None); // even r, negative n
    }

    // ---- ispower ----

    #[test]
    fn ispower_auto_examples() {
        // [(8,2), (5,2), (-9,3), None]
        let cases = [64i128, 25, -729, 1729];
        let want = [Some((8i128, 2u32)), Some((5, 2)), Some((-9, 3)), None];
        for (n, w) in cases.into_iter().zip(want) {
            let got = ispower::<i128, u32>(&n);
            assert_eq!(got, w);
        }
    }

    #[test]
    fn ispower_r_examples_for_64() {
        // Python: [ispower(64,r) for r in range(7)] -> [(8,2), 64, 8, 4, None, None, 2]
        // We test the r>0 slots.
        assert_eq!(ispower_r::<i128>(&64i128, 1), Some(64i128)); // matches Python branch r==1
        assert_eq!(ispower_r::<i128>(&64i128, 2), Some(8i128));
        assert_eq!(ispower_r::<i128>(&64i128, 3), Some(4i128));
        assert_eq!(ispower_r::<i128>(&64i128, 4), None);
        assert_eq!(ispower_r::<i128>(&64i128, 5), None);
        assert_eq!(ispower_r::<i128>(&64i128, 6), Some(2i128));
    }

    #[test]
    fn ispower_residue_shortcuts() {
        // r=2 shortcut: n % 4 == 2 can never be square
        assert_eq!(ispower_r::<i128>(&6i128, 2), None);
        // r=3 shortcut: n % 7 in {2,4,6} can never be cube
        for &n in &[2i128, 4, 6] {
            assert_eq!(ispower_r::<i128>(&n, 3), None);
        }
    }

    // ---- gcd ----

    #[test]
    fn gcd_examples() {
        // (24,42,78,93) -> 3
        let g1 = gcd::<i128>(24.into(), 42.into());
        let g2 = gcd::<i128>(g1, 78.into());
        let g3 = gcd::<i128>(g2, 93.into());
        assert_eq!(g3, 3);

        // gcd(117, -17883411) = 39
        assert_eq!(gcd::<i128>(117.into(), -17883411i128), 39);

        // gcd(3549, 70161, 336882, 702702) -> 273 (chain)
        let g1 = gcd::<i128>(3549.into(), 70161.into());
        let g2 = gcd::<i128>(g1, 336_882.into());
        let g3 = gcd::<i128>(g2, 702_702.into());
        assert_eq!(g3, 273);
    }

    // ---- modinv ----

    #[test]
    fn modinv_examples() {
        assert_eq!(
            modinv::<i128>(&235.into(), &235227.into()),
            Some(147142.into())
        );
        assert_eq!(modinv::<i128>(&1.into(), &1.into()), Some(0.into()));
        assert_eq!(modinv::<i128>(&2.into(), &5.into()), Some(3.into()));
        assert_eq!(modinv::<i128>(&5.into(), &8.into()), Some(5.into()));
        assert_eq!(modinv::<i128>(&37.into(), &100.into()), Some(73.into()));
        assert_eq!(modinv::<i128>(&6.into(), &8.into()), None);
        assert_eq!(modinv::<i128>(&3.into(), &0.into()), None);
        assert_eq!(modinv::<i128>(&3.into(), &-7i128), None);
    }

    // ---- isprime ----

    #[test]
    fn isprime_doc_list() {
        // [n for n in range(91) if isprime(1000*n+1)]
        let mut got = Vec::<i32>::new();
        for n in 0..91 {
            let x = 1000i128 * n as i128 + 1;
            if isprime::<i128>(x, None) {
                got.push(n);
            }
        }
        let expect = [
            3, 4, 7, 9, 13, 16, 19, 21, 24, 28, 51, 54, 55, 61, 69, 70, 76, 81, 88, 90,
        ];
        assert_eq!(got, expect);
    }

    #[test]
    fn isprime_basic_and_carmichael() {
        // smalls
        assert!(isprime::<i128>(2.into(), None));
        assert!(isprime::<i128>(3.into(), None));
        assert!(!isprime::<i128>(1.into(), None));
        assert!(!isprime::<i128>(9.into(), None));
        assert!(!isprime::<i128>(100.into(), None));
        assert!(isprime::<i128>(59.into(), None)); // in TB_DEFAULT
        // composites including Carmichael numbers
        for &n in &[561i128, 1105, 1729, 2465, 2821, 6601] {
            assert!(!isprime::<i128>(n, None));
        }
        // a few primes around SPRP/Lucas edges
        for &p in &[
            2_147_483_647i128,
            2_147_483_629i128,
            9_007_199_254_740_881i128,
        ] {
            assert!(isprime::<i128>(p, None));
        }
    }

    #[test]
    fn isprime_trial_basis_hits() {
        // divisible by a trial basis prime
        for &p in TB_DEFAULT {
            let n = (p as i128) * 101i128;
            assert!(!isprime::<i128>(n, None));
            assert!(isprime::<i128>(p as i128, None));
        }
    }
}
