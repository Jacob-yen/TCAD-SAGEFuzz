mod cli_main_impl;

fn main() -> cli_main_impl::Result<()> {
    cli_main_impl::CliApp.run()
}
