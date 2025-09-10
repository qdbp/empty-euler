// note: not num, we don't require div or mul here
use core::ops::{Add, Sub};
use facto::Factoring;
use num_traits::{Euclid, One, Signed, Unsigned, Zero};
use std::{borrow::Borrow, collections::HashMap, ops::Mul};

use crate::{
    algo::expsq,
    primes::{PG, PrimeGen},
};
pub trait BaseInt:
    Clone
    + Ord
    + Zero
    + One
    // too many algorithms assume the ability to negate for faffing around with 
    // unsigned to be worth it.
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Euclid
    + std::fmt::Debug
{
}
impl<T> BaseInt for T where
    T: Clone
        + Ord
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Euclid
        + std::fmt::Debug
{
}

pub trait Int: BaseInt + Signed + From<i32> {}
impl<T> Int for T where T: BaseInt + Signed + From<i32> {}

pub trait SignedClosure
where
    Self: Unsigned + Sized + Clone,
{
    type Signed: Int + From<Self> + TryInto<Self> + Sized;

    fn to_signed(&self) -> Self::Signed {
        Self::Signed::from(self.clone())
    }
}

impl SignedClosure for u32 {
    type Signed = i64;
}

impl SignedClosure for u64 {
    type Signed = i128;
}

impl SignedClosure for u128 {
    type Signed = rug::Integer;
}

#[inline]
fn c<I: From<i32>>(u: i32) -> I {
    I::from(u)
}
#[inline]
fn is_neg<I: Int>(x: impl Borrow<I>) -> bool {
    *x.borrow() < I::zero()
}

#[inline]
fn abs<I: Int>(x: impl Borrow<I>) -> I {
    if is_neg::<I>(x.borrow()) {
        -x.borrow().clone()
    } else {
        x.borrow().clone()
    }
}
#[inline]
fn is_zero<I: Int>(x: impl Borrow<I>) -> bool {
    *x.borrow() == I::zero()
}
#[inline]
fn is_one<I: Int>(x: impl Borrow<I>) -> bool {
    *x.borrow() == I::one()
}
#[inline]
fn is_even<I: Int>(x: impl Borrow<I>) -> bool {
    let two = c::<I>(2);
    x.borrow().rem_euclid(&two) == I::zero()
}
#[inline]
fn is_odd<I: Int>(x: impl Borrow<I>) -> bool {
    !is_even(x)
}

fn ipow<I: BaseInt>(base: impl Borrow<I>, exp: u32) -> I {
    expsq::<I, u32>(base, &exp)
}

fn pow_mod<I: Int>(base: impl Borrow<I>, exp: impl Borrow<I>, modulus: impl Borrow<I>) -> I {
    let mut acc = I::one();
    let two = c::<I>(2);
    let mut base = base.borrow().rem_euclid(modulus.borrow());
    let mut exp = exp.borrow().clone();
    while !is_zero::<I>(&exp) {
        if is_odd::<I>(&exp) {
            acc = {
                let a: &I = &(acc * base.clone());
                a.rem_euclid(modulus.borrow())
            };
        }
        exp = exp.div_euclid(&two);
        let bb = base.clone() * base;
        base = {
            let a: &I = &bb;
            a.rem_euclid(modulus.borrow())
        };
    }
    acc
}

pub fn gcd<I: Int>(a: impl Borrow<I>, b: impl Borrow<I>) -> I {
    let (mut a, mut b) = (a.borrow().clone(), b.borrow().clone());
    while !is_zero::<I>(&b) {
        let r = a.rem_euclid(&b);
        a = b;
        b = r;
    }
    abs(&a)
}

pub fn modinv_u<I: BaseInt + SignedClosure>(a: impl Borrow<I>, M: impl Borrow<I>) -> Option<I> {
    let a_signed: I::Signed = a.borrow().to_signed();
    let M_signed: I::Signed = M.borrow().to_signed();
    modinv(a_signed, M_signed).map(|x| {
        let x: I = x.try_into().ok().unwrap();
        x
    })
}

pub fn modinv<I: Int>(a: impl Borrow<I>, M: impl Borrow<I>) -> Option<I> {
    if *M.borrow() <= I::zero() {
        return None;
    }

    let mut r0 = a.borrow().rem_euclid(M.borrow());
    let mut r1 = M.borrow().clone();

    // todo i128 is a huge wart here
    let (mut s0, mut s1) = (I::one(), I::zero());
    let (mut t0, mut t1) = (I::zero(), I::one());

    while !is_zero::<I>(&r1) {
        let (q, r2) = r0.div_rem_euclid(&r1);
        let s2 = s0.clone() - q.clone() * s1.clone();
        let t2 = t0.clone() - q * t1.clone();

        // rotate
        (r0, s0, t0) = (r1, s1, t1);
        (r1, s1, t1) = (r2, s2, t2);
    }

    if is_one::<I>(&r0) {
        let out = s0.rem_euclid(M.borrow());
        Some(out)
    } else {
        None
    }
}

pub fn introot<I: Int>(n: impl Borrow<I>, r: u32) -> Option<I> {
    if is_neg::<I>(n.borrow()) {
        if r % 2 == 0 {
            return None;
        }
        let x = introot::<I>(&abs(n), r)?;
        return Some(-x);
    }

    let n: &I = n.borrow();

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
        while ipow::<I>(&upper, 2) <= *n {
            upper = upper * two.clone();
        }
        let mut lower = upper.div_euclid(&two);
        while lower.clone() + I::one() < upper {
            let mid = (lower.clone() + upper.clone()).div_euclid(&two);
            let m2 = ipow::<I>(&mid, 2);
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
    while ipow::<I>(&upper, r) <= *n {
        upper = upper * two.clone();
    }
    let mut lower = upper.clone().div_euclid(&two);
    while lower.clone() + I::one() < upper {
        let mid = (lower.clone() + upper.clone()).div_euclid(&two);
        let m = ipow::<I>(&mid, r);
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

pub fn ispower_r<I: Int>(n: impl Borrow<I>, r: u32) -> Option<I> {
    let x = introot(n.borrow(), r)?;
    if ipow::<I>(&x, r) == *n.borrow() {
        Some(x)
    } else {
        None
    }
}

pub fn ispower<I: Int, J: PG>(n: &I) -> Option<(I, u32)> {
    if *n == I::zero() || *n == I::one() || *n == -I::one() {
        return Some((n.clone(), 1));
    }
    // bit_length(|n|) TODO kinda hacky
    let two = c::<I>(2);
    let mut bits = 0usize;
    let mut t = abs::<I>(n);
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

/// Returns None if n is even or n < 0; else Some(-1|0|1).
pub fn jacobi<I: Int>(a: impl Borrow<I>, n: impl Borrow<I>) -> Option<i8> {
    let n = n.borrow();
    if is_even::<I>(n) || is_neg::<I>(n) {
        return None;
    }
    if *n == <I>::zero() {
        return Some(0);
    }
    if *n == <I>::one() {
        return Some(1);
    }

    let mut a = {
        // let a: &I = a.borrow();
        a.borrow().rem_euclid(n.borrow())
    };

    let mut t: i8 = 1;

    let two = c::<I>(2);
    let three = c::<I>(3);
    let four = c::<I>(4);
    let five = c::<I>(5);
    let eight = c::<I>(8);

    let mut n = n.clone();
    while !is_zero::<I>(&a) {
        while is_even::<I>(&a) {
            a = a.borrow().div_euclid(&two);
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
    if is_one(n) { Some(t) } else { Some(0) }
}

pub const TB_DEFAULT: &[i32] = &[3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59];

pub fn isprime_u<I: BaseInt + SignedClosure>(n: impl Borrow<I>) -> bool {
    let n_signed: I::Signed = n.borrow().to_signed();
    isprime::<I::Signed>(&n_signed)
}

pub fn isprime<I: Int>(n: impl Borrow<I>) -> bool {
    let tb = TB_DEFAULT;

    // 1) trial division and small cases
    let n = n.borrow().clone();
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
    while is_even::<I>(&t) {
        t = t.div_euclid(&two);
        s += 1;
    }
    let mut x = pow_mod::<I>(&c::<I>(2), &t, &n);
    if x != one && x != n.clone() - one.clone() {
        let mut j = 1usize;
        while j < s {
            x = {
                let a = x.clone() * x.clone();
                a.rem_euclid(&n)
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
        match jacobi::<I>(&D, &n) {
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
        if D == -c::<I>(13) && ispower_r::<I>(&n, 2).is_some() {
            return false;
        }
        D = -(D + c::<I>(2)); // revert
        D = D + c::<I>(4); // advance +4
    }

    // run slprp(n, 1, (1 - D) // 4)
    let b = (I::one() - D.clone()).div_euclid(&c::<I>(4));
    let g = gcd::<I>(&n, &b);
    if g > I::one() && g < n {
        return false;
    }

    let mut s_l = 1usize;
    let mut t_l = (n.clone() + I::one()).div_euclid(&c::<I>(2));
    while is_even::<I>(&t_l) {
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
            v.push(is_odd::<I>(&e));
            e = e.div_euclid(&c::<I>(2));
        }
        v.reverse();
        v
    };

    for bit in bits.drain(..) {
        q = {
            let a = q * Q.clone();
            a.rem_euclid(&n)
        };
        if bit {
            Q = {
                let a = q.clone() * b.clone();
                a.rem_euclid(&n)
            };
            let wv = w.clone() * v.clone();
            v = {
                let a = wv.clone() - q.clone();
                a.rem_euclid(&n)
            };
            w = {
                let a = w.clone() * w.clone() - c::<I>(2) * q.clone() * b.clone();
                a.rem_euclid(&n)
            };
        } else {
            Q = q.clone();
            let wv = w.clone() * v.clone();
            w = {
                let a = wv.clone() - q.clone();
                a.rem_euclid(&n)
            };
            v = {
                let a = v.clone() * v.clone() - c::<I>(2) * q.clone();
                a.rem_euclid(&n)
            };
        }
    }

    if v == I::zero() {
        return true;
    }
    let inv_d = modinv(&D, &n).expect("D and n coprime after selection");
    let chk = {
        let a = (c::<I>(2) * w.clone() - v.clone()) * inv_d;
        a.rem_euclid(&n)
    };
    if chk == I::zero() {
        return true;
    }

    q = pow_mod(&b, &t_l, &n);
    for _ in 1..s_l {
        v = {
            let a = v.clone() * v.clone() - c::<I>(2) * q.clone();
            a.rem_euclid(&n)
        };
        if v == I::zero() {
            return true;
        }
        q = {
            let a = q.clone() * q.clone();
            a.rem_euclid(&n)
        };
    }
    false
}

pub fn factorint<I: BaseInt + Factoring + std::hash::Hash>(n: impl Borrow<I>) -> HashMap<I, u32> {
    let n: &I = n.borrow();
    if *n == I::zero() {
        return HashMap::new();
    } else if *n == I::one() {
        return HashMap::from([(I::one(), 1)]);
    }
    let facs: Vec<I> = n.clone().factor();
    let mut out = HashMap::new();
    if facs.is_empty() {
        out.insert(n.clone(), 1);
        return out;
    }
    let mut cur = facs[0].clone();
    let mut count = 1u32;
    for f in facs.iter().skip(1) {
        if *f == cur {
            count += 1;
        } else {
            out.insert(cur, count);
            cur = f.clone();
            count = 1;
        }
    }
    out.insert(cur, count);
    out
}

pub fn divisors<I: BaseInt + Factoring + std::hash::Hash>(n: impl Borrow<I>) -> Vec<I> {
    let n = n.borrow();
    let mut out = vec![];
    if *n == I::zero() {
        return out;
    } else if *n == I::one() {
        out.push(I::one());
        return out;
    }

    let facs = factorint::<I>(n);
    let mut cur_es = vec![0u32; facs.len()];

    loop {
        out.push(
            facs.keys()
                .zip(cur_es.iter())
                .map(|(p, e)| ipow::<I>(p, *e))
                .fold(I::one(), |a, b| a * b),
        );
        let mut i = 0;
        loop {
            if i >= facs.len() {
                return out;
            }
            if cur_es[i] < *facs.values().nth(i).unwrap() {
                cur_es[i] += 1;
                break;
            } else {
                cur_es[i] = 0;
                i += 1;
            }
        }
    }
}

pub fn proper_divisors<I: BaseInt + Factoring + std::hash::Hash>(n: impl Borrow<I>) -> Vec<I> {
    let mut divs = divisors(n);
    if divs.len() <= 1 {
        return vec![];
    }
    divs.drain((divs.len() - 1)..);
    divs
}

pub fn μ<I: BaseInt + Factoring>(n: impl Borrow<I>) -> i32 {
    let mut facs: Vec<I> = n.borrow().clone().factor();
    facs.sort();
    if n.borrow() == &I::one() {
        return 1;
    }
    let mut n_unique = 1;
    // this handles the len=1 case since in that case we have no windows
    for w in facs.windows(2) {
        if w[0] == w[1] {
            return 0;
        }
        n_unique += 1;
    }
    if n_unique % 2 == 1 { -1 } else { 1 }
}

#[cfg(test)]
mod tests_fac {
    use super::*;

    #[test]
    fn test_ipow() {
        assert_eq!(ipow::<i128>(&3, 4), 81);
        assert_eq!(ipow::<i64>(&2, 10), 1024);
        assert_eq!(ipow::<i32>(&5, 0), 1);
        assert_eq!(ipow::<i32>(&7, 2), 49);
    }

    #[test]
    fn jacobi_tables_15() {
        // Python:
        // [-10,-7,-4,-2,-1,0,1,2,4,7,10] -> [0,1,-1,-1,-1,0,1,1,1,-1,0]
        let a_vals = [-10, -7, -4, -2, -1, 0, 1, 2, 4, 7, 10];
        let expect = [0, 1, -1, -1, -1, 0, 1, 1, 1, -1, 0];
        for (a, e) in a_vals.into_iter().zip(expect) {
            assert_eq!(jacobi::<i128>(&a, &15i128), Some(e));
        }
    }

    #[test]
    fn jacobi_tables_13() {
        // [-10,-9,-4,-2,-1,0,1,2,4,9,10] -> [1,1,1,-1,1,0,1,-1,1,1,1]
        let a_vals = [-10, -9, -4, -2, -1, 0, 1, 2, 4, 9, 10];
        let expect = [1, 1, 1, -1, 1, 0, 1, -1, 1, 1, 1];
        for (a, e) in a_vals.into_iter().zip(expect) {
            assert_eq!(jacobi::<i128>(&a, &13i128), Some(e));
        }
    }

    #[test]
    fn jacobi_tables_11() {
        // [-10,-9,-4,-2,-1,0,1,2,4,9,10] -> [1,-1,-1,1,-1,0,1,-1,1,1,-1]
        let a_vals = [-10, -9, -4, -2, -1, 0, 1, 2, 4, 9, 10];
        let expect = [1, -1, -1, 1, -1, 0, 1, -1, 1, 1, -1];
        for (a, e) in a_vals.into_iter().zip(expect) {
            assert_eq!(jacobi::<i128>(&a, &11i128), Some(e));
        }
    }

    #[test]
    fn jacobi_none_for_bad_n() {
        assert!(jacobi::<i128>(&3, &10i128).is_none()); // n even
        assert!(jacobi::<i128>(&3, &-11i128).is_none()); // n < 0
    }

    #[test]
    fn introot_examples() {
        assert_eq!(introot::<i128>(-729, 3), Some(-9));
        assert_eq!(introot::<i128>(-728, 3), Some(-8));
        assert_eq!(introot::<i128>(1023i128, 2), Some(31i128));
        assert_eq!(introot::<i128>(1024i128, 2), Some(32i128));
        assert_eq!(introot::<i128>(-8, 2), None); // even r, negative n
    }

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
        assert_eq!(ispower_r::<i128>(64i128, 1), Some(64i128)); // matches Python branch r==1
        assert_eq!(ispower_r::<i128>(64i128, 2), Some(8i128));
        assert_eq!(ispower_r::<i128>(64i128, 3), Some(4i128));
        assert_eq!(ispower_r::<i64>(64, 4), None);
        assert_eq!(ispower_r::<i64>(64, 5), None);
        assert_eq!(ispower_r::<i64>(64, 6), Some(2i64));
    }

    #[test]
    fn ispower_residue_shortcuts() {
        // r=2 shortcut: n % 4 == 2 can never be square
        assert_eq!(ispower_r::<i128>(6i128, 2), None);
        // r=3 shortcut: n % 7 in {2,4,6} can never be cube
        for &n in &[2i128, 4, 6] {
            assert_eq!(ispower_r::<i128>(n, 3), None);
        }
    }

    #[test]
    fn gcd_examples() {
        // (24,42,78,93) -> 3
        let g1 = gcd::<i128>(24, 42);
        let g2 = gcd::<i128>(g1, 78);
        let g3 = gcd::<i128>(g2, 93);
        assert_eq!(g3, 3);

        // gcd(117, -17883411) = 39
        assert_eq!(gcd::<i128>(117, -17883411i128), 39);

        // gcd(3549, 70161, 336882, 702702) -> 273 (chain)
        let g1 = gcd::<i128>(3549, 70161);
        let g2 = gcd::<i128>(g1, 336_882);
        let g3 = gcd::<i128>(g2, 702_702);
        assert_eq!(g3, 273);
    }

    #[test]
    fn modinv_examples() {
        assert_eq!(modinv::<i128>(235, 235227), Some(147142));
        assert_eq!(modinv::<i128>(2, 5), Some(3));
        assert_eq!(modinv::<i128>(5, 8), Some(5));
        assert_eq!(modinv_u::<u64>(37, 100), Some(73));
        assert_eq!(modinv::<i64>(6, 8), None);
        assert_eq!(modinv_u::<u32>(3, 0), None);
        assert_eq!(modinv::<i64>(3, -7i64), None);
    }

    #[test]
    fn isprime_doc_list() {
        // [n for n in range(91) if isprime(1000*n+1)]
        let mut got = Vec::<i32>::new();
        for n in 0..91 {
            let x = 1000i128 * n as i128 + 1;
            if isprime::<i128>(x) {
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
        assert!(isprime::<i128>(2));
        assert!(isprime::<i128>(3));
        assert!(!isprime::<i64>(1));
        assert!(!isprime::<i64>(9));
        assert!(!isprime::<i64>(100));
        assert!(isprime::<i64>(59)); // in TB_DEFAULT
        // composites including Carmichael numbers
        for &n in &[561i128, 1105, 1729, 2465, 2821, 6601] {
            assert!(!isprime::<i128>(n));
        }
        // a few primes around SPRP/Lucas edges
        for &p in &[
            2_147_483_647i128,
            2_147_483_629i128,
            9_007_199_254_740_881i128,
        ] {
            assert!(isprime::<i128>(p));
        }
    }

    #[test]
    fn isprime_trial_basis_hits() {
        // divisible by a trial basis prime
        for &p in TB_DEFAULT {
            let n = (p as i128) * 101i128;
            assert!(!isprime::<i128>(n));
            assert!(isprime::<i128>(p as i128));
        }
    }

    #[test]
    fn test_factorint() {
        assert_eq!(
            factorint::<u64>(360),
            HashMap::from([(2, 3), (3, 2), (5, 1)])
        );
        assert_eq!(factorint::<u64>(1), HashMap::from([(1, 1)]));
        assert_eq!(factorint::<u64>(0), HashMap::new());
        assert_eq!(factorint::<u64>(37), HashMap::from([(37, 1)]));
    }

    #[test]
    fn test_divisors() {
        let mut divs = divisors::<u64>(28);
        divs.sort();
        assert_eq!(divs, vec![1, 2, 4, 7, 14, 28]);
        let mut divs = divisors::<u64>(1);
        divs.sort();
        assert_eq!(divs, vec![1]);
        let divs: Vec<u64> = divisors::<u64>(0);
        assert_eq!(divs, Vec::<u64>::new());
        let mut divs = divisors::<u64>(37);
        divs.sort();
        assert_eq!(divs, vec![1, 37]);
    }

    #[test]
    fn test_proper_divisors() {
        let mut divs = proper_divisors::<u64>(28);
        divs.sort();
        assert_eq!(divs, vec![1, 2, 4, 7, 14]);
        let divs: Vec<u64> = proper_divisors::<u64>(1);
        assert_eq!(divs, Vec::<u64>::new());
        let divs: Vec<u64> = proper_divisors::<u64>(0);
        assert_eq!(divs, Vec::<u64>::new());
        let mut divs = proper_divisors::<u64>(37);
        divs.sort();
        assert_eq!(divs, vec![1]);
    }

    #[test]
    fn test_μ() {
        assert_eq!(μ::<u64>(1), 1);
        assert_eq!(μ::<u64>(2), -1);
        assert_eq!(μ::<u64>(3), -1);
        assert_eq!(μ::<u64>(4), 0);
        assert_eq!(μ::<u64>(5), -1);
        assert_eq!(μ::<u64>(6), 1);
        assert_eq!(μ::<u64>(7), -1);
        assert_eq!(μ::<u64>(8), 0);
        assert_eq!(μ::<u64>(9), 0);
        assert_eq!(μ::<u64>(10), 1);
        assert_eq!(μ::<u64>(30), -1); // 2*3*5
        assert_eq!(μ::<u64>(60), 0); // 2^2*3*5
        assert_eq!(μ::<u64>(2 * 3 * 5 * 7), 1);
        assert_eq!(μ::<u64>(3 * 5 * 7 * 11 * 13), -1);
    }
}
