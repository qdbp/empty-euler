use good_lp::{Expression, Variable};

pub fn lp_dot<T>(vars: &[Variable], vals: &[T]) -> Expression
where
    // TODO copy is sloppy
    T: Into<f64> + Copy,
{
    vars.iter()
        .zip(vals.iter())
        .map(|(&var, &val)| var * val)
        .sum()
}
