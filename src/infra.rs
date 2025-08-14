//! Satanic machinery in the back end keeps the front rooms spick and span
//! We want a magic lightness. A childlike wonder. Freedom.
//! The price is paid here -- enjoy your stay.
use anyhow::{Result, anyhow};

mod registry {
    include!(concat!(env!("OUT_DIR"), "/registry.rs"));
}

mod solutions {
    include!(concat!(env!("OUT_DIR"), "/solutions_mod.rs"));
}

pub fn dispatch(id: &str, args: &[String]) -> Result<String> {
    let f = registry::REGISTRY
        .iter()
        .find(|(k, _)| *k == id)
        .map(|(_, f)| *f)
        .ok_or_else(|| anyhow!("unknown problem `{id}`"))?;
    f(args)
}

#[macro_export]
macro_rules! soln {
    (  $( $name:ident : $ty:ty $( = $def:expr )? ),* $(,)? => { $($body:tt)* } ) => {

        use ::clap::Parser;
        #[derive(::clap::Parser, ::std::fmt::Debug)]
        #[command(version, about)]
        struct SolnArgs {
            $(
                #[arg(short, long $(, default_value_t = $def )?)]
                $name: $ty,
            )*
        }

        #[allow(non_snake_case, unused)]
        pub fn solve_raw( $( $name : $ty ),* ) -> ::std::string::String {
            $($body)*

            // this silences the annoying warning in a new solution that made us
            // do "foo".to_string() and similar
            #[allow(unreachable_code)]
            return "unsolved!".to_string()
        }

        pub fn solve(argv: &[::std::string::String]) -> ::anyhow::Result<::std::string::String> {
            #[allow(unused)]
            let parsed = SolnArgs::parse_from(argv);
            // do some magic logging
            let __problem: &str = {
                let __m = module_path!();
                match __m.rsplit("::").next() { Some(s) => s, None => __m }
            };
            let __span = ::tracing::info_span!(
                "Solved ",
                problem = %__problem,
                $( $name = ?&parsed.$name ),*
            );

            let now = ::std::time::Instant::now();
            let __out: ::std::string::String = solve_raw( $( parsed.$name ),*);
            let elapsed = now.elapsed();

            Ok(format!("Solution: {__out} | took {:.6} seconds",
                elapsed.as_secs_f64(),
            ))
        }
    };
}

#[macro_export]
macro_rules! examples {
    ( $( | $( $it:tt ),* $(,)? | => $val:tt ;)+) => {
        #[cfg(test)]
        mod tests {
            use super::*;
            #[test]
            fn test_examples() { $( assert_eq!(solve_raw($($it, )*), $val) ;)* }
        }
    };
}
