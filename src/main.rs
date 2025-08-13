#![allow(non_snake_case)]
use clap::Parser;

#[derive(Parser)]
struct Cli {
    /// Problem id, e.g. p804
    id: String,
    /// Remaining args passed to the problem
    args: Vec<String>,
}

fn enable_tracing() {
    use tracing_subscriber::fmt::format::FmtSpan;

    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .without_time()
        .with_target(false)
        .init();
}

fn main() -> anyhow::Result<()> {
    enable_tracing();
    let cli = Cli::parse();
    let with_name: Vec<String> = std::iter::once(cli.id.clone())
        .chain(cli.args.iter().cloned())
        .collect();
    println!("{}", pe::infra::dispatch(&cli.id, &with_name)?);
    Ok(())
}
