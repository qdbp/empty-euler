use rug::Integer as Int;

pub fn int<T: Into<Int>>(x: T) -> Int {
    Into::into(x)
}
