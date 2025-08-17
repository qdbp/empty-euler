#[inline(always)]
pub fn bit_width(n: impl Into<u128>) -> u32 {
    let mut out = 0u32;
    let mut n = n.into();
    while n > 0 {
        out += 1;
        n >>= 1;
    }
    out
}
