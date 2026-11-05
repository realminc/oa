//! `oa` — command-line application over the OA library.
//!
//! One binary with subcommands. This shell owns argument parsing and file
//! orchestration only; all engine behavior lives in `oa`.

use std::error::Error;

use oa::{Cli, Viewer};

#[derive(Default)]
struct Args {
	path: String,
}

fn main() -> Result<(), Box<dyn Error>> {
	let mut cli: Cli<Args> = Cli::new("oa", "OA GPU compute engine");
	{
		let view = cli.add_subcommand("view", "open an image, video, or audio file in the viewer");
		view.add_positional(
			"path",
			|a: &mut Args| &mut a.path,
			"media file to open",
			true,
		);
	}
	cli.require_subcommand(1, 1);
	cli.set_epilog("`oa view` probes each file as image, then video, then audio.");

	let argv: Vec<String> = std::env::args().collect();
	if !cli.parse_args(&argv[1..]) {
		if cli.help_requested() {
			return Ok(());
		}
		return Err("invalid command line (see `oa --help`)".into());
	}
	match cli.active_subcommand() {
		"view" => Ok(Viewer::preview(&cli.config().path)?),
		name => Err(format!("unknown subcommand `{name}`").into()),
	}
}
