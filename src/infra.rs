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
    (  ( $( $name:ident : $ty:tt ),* ) $(,)? { $($body:tt)* } ) => {

        #[allow(non_snake_case)]
        #[cfg(test)]
        pub fn solve_raw( $( $name : $ty ),* ) -> ::std::string::String { $($body)* }

        pub fn solve(argv: &[::std::string::String]) -> ::anyhow::Result<::std::string::String> {
            use ::anyhow::anyhow;
            use ::std::collections::HashMap;

            // Build multimap: key -> Vec<str>
            let mut map: HashMap<&str, Vec<&str>> = HashMap::new();
            let mut i = 0;
            while i < argv.len() {
                let a = &argv[i];
                if let Some(rest) = a.strip_prefix("--") {
                    if let Some((k, v)) = rest.split_once('=') {
                        map.entry(k).or_default().push(v);
                    } else if i + 1 < argv.len() && !argv[i + 1].starts_with('-') {
                        i += 1;
                        map.entry(rest).or_default().push(argv[i].as_str());
                    } else {
                        // presence-only flag (for bool)
                        map.entry(rest).or_default().push("true");
                    }
                } else {
                    return Err(anyhow!("positional not allowed: {}", a));
                }
                i += 1;
            }

            // Unknowns error
            {
                let allowed: &[&str] = &[$( stringify!($name) ),*];
                for k in map.keys() {
                    if !allowed.contains(k) {
                        return Err(anyhow!("unknown arg `--{}`", k));
                    }
                }
            }

            // Bind each declared parameter
            $(
                soln!(@bind $name : $ty, &map);
            )*

            // do some magic logging
            let __problem: &str = {
                let __m = module_path!();
                match __m.rsplit("::").next() { Some(s) => s, None => __m }
            };
            let __span = ::tracing::info_span!(
                "Solved ",
                problem = %__problem,
                $( $name = ?&$name ),*
            );

            let now = ::std::time::Instant::now();

            let __out: ::std::string::String = { $($body)* };
            let elapsed = now.elapsed();
            Ok(format!("Solution: {__out} | took {:.6} seconds",
                elapsed.as_secs_f64(),
            ))
        }
    };

    // required single: T
    (@bind $name:ident : $t:ty, $map:expr) => {
        let $name: $t = {
            let key = stringify!($name);
            let vals = $map.get(key).ok_or_else(|| ::anyhow::anyhow!("missing `--{}`", key))?;
            if vals.len() != 1 { return Err(::anyhow::anyhow!("`--{}` expects one value", key)); }
            vals[0].parse::<$t>().map_err(|e| ::anyhow::anyhow!("bad `--{}`: {}", key, e))?
        };
    };
    // Option<T>
    (@bind $name:ident : Option<$t:ty>, $map:expr) => {
        let $name: Option<$t> = {
            let key = stringify!($name);
            match $map.get(key) {
                None => None,
                Some(vals) => {
                    if vals.len() != 1 { return Err(::anyhow::anyhow!("`--{}` expects one value", key)); }
                    Some(vals[0].parse::<$t>().map_err(|e| ::anyhow::anyhow!("bad `--{}`: {}", key, e))?)
                }
            }
        };
    };
    // Vec<T>
    (@bind $name:ident : Vec<$t:ty>, $map:expr) => {
        let $name: ::std::vec::Vec<$t> = {
            let key = stringify!($name);
            match $map.get(key) {
                None => ::std::vec::Vec::new(),
                Some(vals) => {
                    let mut out = ::std::vec::Vec::with_capacity(vals.len());
                    for s in vals {
                        out.push(s.parse::<$t>().map_err(|e| ::anyhow::anyhow!("bad `--{}`: {}", key, e))?);
                    }
                    out
                }
            }
        };
    };
    // bool (presence sets true; "--flag=false" also allowed)
    (@bind $name:ident : bool, $map:expr) => {
        let $name: bool = {
            let key = stringify!($name);
            match $map.get(key) {
                None => false,
                Some(vals) => {
                    if vals.is_empty() { false }
                    else if vals.iter().all(|v| *v == "true") { true }
                    else if vals.len() == 1 && (vals[0] == "true" || vals[0] == "false") { vals[0] == "true" }
                    else { return Err(::anyhow::anyhow!("`--{}` expects no value or =true/false", key)); }
                }
            }
        };
    };
}
