//! `oa` — command-line application over the OA library.
//!
//! One binary with subcommands. This shell owns argument parsing and file
//! orchestration only; all engine behavior lives in `oa`.

use std::error::Error;

use oa::{Cli, Viewer};

#[derive(Default)]
struct Args {
	path: String,
	layout: String,
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
	{
		let workspace = cli.add_subcommand(
			"workspace",
			"open the viewer's workspace mode with two demo figure panes",
		);
		workspace.add_option(
			"--layout",
			|a: &mut Args| &mut a.layout,
			"restore a dock layout JSON produced by Workspace::to_json",
		);
	}
	cli.require_subcommand(1, 1);
	cli.set_epilog(
		"`oa view` probes each file as image, then video, then audio; `oa workspace` docks two demo figures and optionally restores a saved layout.",
	);

	let argv: Vec<String> = std::env::args().collect();
	if !cli.parse_args(&argv[1..]) {
		if cli.help_requested() {
			return Ok(());
		}
		return Err("invalid command line (see `oa --help`)".into());
	}
	match cli.active_subcommand() {
		"view" => Ok(Viewer::preview(&cli.config().path)?),
		"workspace" => run_workspace(&cli.config().layout),
		name => Err(format!("unknown subcommand `{name}`").into()),
	}
}

/// Open `ViewerMode::Workspace` over two demo `FigureView` panes, optionally
/// restoring a serialized dock layout. Item ids `signal` and `throughput`
/// match the default dock; a `--layout` document can rearrange them and add
/// empty panes, which render as disabled tabs.
fn run_workspace(layout: &str) -> Result<(), Box<dyn Error>> {
	use oa::plot::{Figure, FigureConfig, FigureView};
	use oa::ui::UiDirection;
	use oa::viewer::{Workspace, WorkspaceNode};
	use oa::{ViewerConfig, ViewerMode};

	let samples: Vec<f32> = (0..240).map(|i| (i as f32 * 0.05).sin()).collect();
	let mut signal = Figure::new(FigureConfig::default());
	signal.title("signal");
	signal.ax(0, 0).plot(&samples, Default::default());
	let mut signal = FigureView::new(signal);
	signal.set_name("signal");

	let mut throughput = Figure::new(FigureConfig::default());
	throughput.title("throughput");
	throughput
		.ax(0, 0)
		.bar(&[0.4, 0.8, 0.65, 0.9], Default::default());
	let mut throughput = FigureView::new(throughput);
	throughput.set_name("throughput");

	let mut workspace = if layout.is_empty() {
		Workspace::new(WorkspaceNode::split(
			UiDirection::Row,
			0.55,
			WorkspaceNode::tabs(&["signal"]),
			WorkspaceNode::tabs(&["throughput"]),
		))
	} else {
		Workspace::from_json(&std::fs::read_to_string(layout)?)?
	};
	workspace.add_item(Box::new(signal));
	workspace.add_item(Box::new(throughput));

	let mut viewer = Viewer::new(ViewerConfig {
		mode: ViewerMode::Workspace,
		title: "oa workspace".to_owned(),
		..ViewerConfig::default()
	});
	viewer.set_workspace(&mut workspace);
	Ok(viewer.run_standalone()?)
}
