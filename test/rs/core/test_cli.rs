//! Contract tests for oa::Cli

use oa::Cli;

// ─── helpers ─────────────────────────────────────────────────────────────────

fn args(v: &[&str]) -> Vec<String> {
	v.iter().map(|s| s.to_string()).collect()
}

// ─── basic option parsing ─────────────────────────────────────────────────────

#[derive(Default)]
struct Cfg {
	steps: u64,
	lr: f32,
	name: String,
	flag: bool,
	extra: Vec<String>,
}

fn make_cli() -> Cli<Cfg> {
	let mut cli = Cli::new("test", "test program");
	cli.add_option("-s,--steps", |c: &mut Cfg| &mut c.steps, "number of steps");
	cli.add_option("-l,--lr", |c: &mut Cfg| &mut c.lr, "learning rate");
	cli.add_option("-n,--name", |c: &mut Cfg| &mut c.name, "run name");
	cli.add_flag(
		"--flag,--no-flag{false}",
		|c: &mut Cfg, v| c.flag = v,
		"a flag",
	);
	cli.add_multi("--extra", |c: &mut Cfg| &mut c.extra, "extra values");
	cli
}

#[test]
fn long_option_long_form() {
	let mut cli = make_cli();
	assert!(cli.parse_args(&args(&["--steps", "100"])));
	assert_eq!(cli.config().steps, 100);
}

#[test]
fn short_option() {
	let mut cli = make_cli();
	assert!(cli.parse_args(&args(&["-s", "50"])));
	assert_eq!(cli.config().steps, 50);
}

#[test]
fn inline_equals_form() {
	let mut cli = make_cli();
	assert!(cli.parse_args(&args(&["--steps=200", "--lr=0.001"])));
	assert_eq!(cli.config().steps, 200);
	assert!((cli.config().lr - 0.001_f32).abs() < 1e-6);
}

#[test]
fn multiple_options() {
	let mut cli = make_cli();
	assert!(cli.parse_args(&args(&["--steps", "10", "--lr", "0.01", "--name", "run1"])));
	assert_eq!(cli.config().steps, 10);
	assert_eq!(cli.config().name, "run1");
}

#[test]
fn flag_presence_sets_true() {
	let mut cli = make_cli();
	assert!(cli.parse_args(&args(&["--flag"])));
	assert!(cli.config().flag);
}

#[test]
fn flag_false_suffix() {
	let mut cli = make_cli();
	assert!(cli.parse_args(&args(&["--flag", "--no-flag"])));
	assert!(!cli.config().flag);
}

#[test]
fn multi_option_appends() {
	let mut cli = make_cli();
	assert!(cli.parse_args(&args(&["--extra", "a", "--extra", "b"])));
	assert_eq!(cli.config().extra, vec!["a", "b"]);
}

#[test]
fn defaults_unchanged_when_not_supplied() {
	let mut cli = make_cli();
	assert!(cli.parse_args(&args(&[])));
	let cfg = cli.config();
	assert_eq!(cfg.steps, 0);
	assert!((cfg.lr - 0.0_f32).abs() < f32::EPSILON);
	assert!(cfg.name.is_empty());
	assert!(!cfg.flag);
}

#[test]
fn unknown_option_returns_false() {
	let mut cli = make_cli();
	assert!(!cli.parse_args(&args(&["--unknown"])));
}

#[test]
fn option_missing_value_returns_false() {
	let mut cli = make_cli();
	assert!(!cli.parse_args(&args(&["--steps"])));
}

#[test]
fn invalid_integer_returns_false() {
	let mut cli = make_cli();
	assert!(!cli.parse_args(&args(&["--steps", "notanint"])));
}

#[test]
fn help_flag_returns_false_and_marks_request() {
	let mut cli = make_cli();
	assert!(!cli.parse_args(&args(&["--help"])));
	assert!(cli.help_requested());
}

// ─── verbose ─────────────────────────────────────────────────────────────────

#[test]
fn verbose_short() {
	let mut cli = make_cli();
	assert!(cli.parse_args(&args(&["-v", "2"])));
	assert_eq!(cli.verbose(), 2);
}

#[test]
fn verbose_equals() {
	let mut cli = make_cli();
	assert!(cli.parse_args(&args(&["--verbose=3"])));
	assert_eq!(cli.verbose(), 3);
}

// ─── positional arguments ─────────────────────────────────────────────────────

#[derive(Default)]
struct Pos {
	path: String,
	count: u32,
}

fn make_pos_cli() -> Cli<Pos> {
	let mut cli = Cli::new("pos", "positional test");
	cli.add_positional("path", |c: &mut Pos| &mut c.path, "input path", true);
	cli.add_positional("count", |c: &mut Pos| &mut c.count, "count", false);
	cli
}

#[test]
fn positional_required_set() {
	let mut cli = make_pos_cli();
	assert!(cli.parse_args(&args(&["file.txt"])));
	assert_eq!(cli.config().path, "file.txt");
}

#[test]
fn positional_optional_unset() {
	let mut cli = make_pos_cli();
	assert!(cli.parse_args(&args(&["file.txt"])));
	assert_eq!(cli.config().count, 0);
}

#[test]
fn positional_optional_set() {
	let mut cli = make_pos_cli();
	assert!(cli.parse_args(&args(&["file.txt", "42"])));
	assert_eq!(cli.config().count, 42);
}

#[test]
fn positional_required_missing_returns_false() {
	let mut cli = make_pos_cli();
	assert!(!cli.parse_args(&args(&[])));
}

// ─── subcommands ─────────────────────────────────────────────────────────────

#[derive(Default)]
struct SubCfg {
	mode: String,
	steps: u64,
}

fn make_sub_cli() -> Cli<SubCfg> {
	let mut cli = Cli::new("app", "sub test");
	cli.add_option("--steps", |c: &mut SubCfg| &mut c.steps, "steps");
	{
		let train = cli.add_subcommand("train", "run training");
		train.add_option("--mode", |c: &mut SubCfg| &mut c.mode, "mode");
	}
	cli
}

#[test]
fn subcommand_selected() {
	let mut cli = make_sub_cli();
	assert!(cli.parse_args(&args(&["train"])));
	assert!(cli.got_subcommand("train"));
	assert_eq!(cli.active_subcommand(), "train");
}

#[test]
fn subcommand_not_selected() {
	let mut cli = make_sub_cli();
	assert!(cli.parse_args(&args(&["--steps", "5"])));
	assert!(!cli.got_subcommand("train"));
	assert_eq!(cli.active_subcommand(), "");
}

#[test]
fn subcommand_with_option() {
	let mut cli = make_sub_cli();
	assert!(cli.parse_args(&args(&["train", "--mode", "eager"])));
	assert_eq!(cli.config().mode, "eager");
}

#[test]
fn root_option_after_subcommand_fallthrough() {
	let mut cli = make_sub_cli();
	assert!(cli.parse_args(&args(&["train", "--steps", "99"])));
	assert_eq!(cli.config().steps, 99);
}

// ─── config access helpers ────────────────────────────────────────────────────

#[test]
fn into_config_consumes() {
	let mut cli = make_cli();
	cli.parse_args(&args(&["--steps", "7"]));
	let cfg = cli.into_config();
	assert_eq!(cfg.steps, 7);
}

#[test]
fn config_path_empty_when_not_set() {
	let mut cli = make_cli();
	cli.parse_args(&args(&[]));
	assert!(cli.config_path().is_empty());
}

#[test]
fn config_path_extracted() {
	let mut cli = make_cli();
	cli.parse_args(&args(&["-c", "/tmp/my.yaml", "--steps", "3"]));
	assert_eq!(cli.config_path(), "/tmp/my.yaml");
}
