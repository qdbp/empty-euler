#![allow(non_snake_case)]
use clap::Parser;
use clap_verbosity_flag::{InfoLevel, Verbosity};

#[derive(Parser)]
struct Cli {
    /// Problem id, e.g. p804
    id: String,
    /// Remaining args passed to the problem
    args: Vec<String>,
    /// Debugging verbosity
    #[command(flatten)]
    verbosity: Verbosity<InfoLevel>,
}

fn enable_tracing(args: &Cli) {
    use tracing_subscriber::fmt::format::FmtSpan;

    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .without_time()
        .with_target(false)
        .with_max_level(args.verbosity)
        .init();
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    enable_tracing(&cli);
    let with_name: Vec<String> = std::iter::once(cli.id.clone())
        .chain(cli.args.iter().cloned())
        .collect();
    println!("{}", pe_lib::infra::dispatch(&cli.id, &with_name)?);
    Ok(())
}
