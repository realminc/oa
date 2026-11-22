//! Command-line configuration with YAML-file + argv precedence.
//!
//! Port provenance: `oa/core/cli.h`. Rust translates the `Cli<TConfig>`
//! template class to `Cli<C>` with caller-provided closures for YAML loading
//! and option application.
//!
//! # Precedence
//!
//! `struct defaults < YAML file (-c / --config) < explicit argv flags`
//!
//! # Usage
//!
//! ```rust,ignore
//! use oa::Cli;
//!
//! #[derive(Default)]
//! struct Config { steps: u64, lr: f32, output: String }
//!
//! struct MyApp { cli: Cli<Config> }
//!
//! impl MyApp {
//!     fn new() -> Self {
//!         let mut cli = Cli::new("myapp", "Train a model");
//!         {
//!             let cfg = cli.config_mut();
//!             let _ = cfg; // borrow for defaults; options registered below
//!         }
//!         let app_ptr = /* pointer trick not needed — use a builder pattern: */
//!             MyApp { cli };
//!         app_ptr
//!     }
//! }
//!
//! // Simpler usage — register options before calling parse:
//! let mut config = Config::default();
//! let mut cli: Cli<Config> = Cli::new("myapp", "Train a model");
//! cli.add_option("-s,--steps", |c| &mut c.steps, "number of steps");
//! cli.add_option("-l,--lr",    |c| &mut c.lr,    "learning rate");
//! let args: Vec<String> = std::env::args().collect();
//! if !cli.parse_args(&args[1..]) { return Ok(()); }
//! let cfg = cli.config();
//! ```

use crate::Path;
use crate::core::filesystem::Filesystem;

// ─── internal parse trait ────────────────────────────────────────────────────

/// Parse a string slice into a target value.
///
/// Implemented for all types that implement [`FromStr`] plus a specialised
/// `bool` arm that accepts `true/1/on/yes` and `false/0/off/no`.
fn parse_value<T>(text: &str, out: &mut T) -> bool
where
	T: ParseCliValue,
{
	T::parse_cli(text, out)
}

/// Sealed helper for CLI value parsing.
pub trait ParseCliValue: Sized {
	fn parse_cli(text: &str, out: &mut Self) -> bool;
}

impl ParseCliValue for bool {
	fn parse_cli(text: &str, out: &mut Self) -> bool {
		match text {
			"true" | "1" | "on" | "yes" => {
				*out = true;
				true
			}
			"false" | "0" | "off" | "no" => {
				*out = false;
				true
			}
			_ => false,
		}
	}
}

macro_rules! impl_parse_cli_via_from_str {
	($($t:ty),*) => {
		$(
			impl ParseCliValue for $t {
				fn parse_cli(text: &str, out: &mut Self) -> bool {
					match text.parse::<$t>() {
						Ok(v) => { *out = v; true }
						Err(_) => false,
					}
				}
			}
		)*
	}
}

impl_parse_cli_via_from_str!(i8, i16, i32, i64, i128, isize);
impl_parse_cli_via_from_str!(u8, u16, u32, u64, u128, usize);
impl_parse_cli_via_from_str!(f32, f64);

impl ParseCliValue for String {
	fn parse_cli(text: &str, out: &mut Self) -> bool {
		*out = text.to_owned();
		true
	}
}

// ─── Opt ─────────────────────────────────────────────────────────────────────

struct Alias {
	name: String,
	flag_value: bool,
}

type ParseFn<C> = Box<dyn Fn(&str, &mut C) -> bool>;
type FlagTarget<C> = Box<dyn Fn(&mut C, bool)>;

struct Opt<C> {
	aliases: Vec<Alias>,
	description: String,
	parse: ParseFn<C>,
	positional: bool,
	flag: bool,
	flag_target: Option<FlagTarget<C>>,
	multi: bool,
	required: bool,
	count: u32,
}

impl<C> Opt<C> {
	fn matches(&self, name: &str) -> Option<bool> {
		for alias in &self.aliases {
			if alias.name == name {
				return Some(alias.flag_value);
			}
		}
		None
	}
}

// ─── Cmd ─────────────────────────────────────────────────────────────────────

/// A named subcommand (or the implicit root command).
///
/// Options registered on a `Cmd` are only matched when that subcommand is
/// active. The root `Cmd` is always active.
pub struct Cmd<C> {
	name: String,
	description: String,
	opts: Vec<Opt<C>>,
	parsed: bool,
}

impl<C: 'static> Cmd<C> {
	fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
		Self {
			name: name.into(),
			description: description.into(),
			opts: Vec::new(),
			parsed: false,
		}
	}

	/// Register a named option that writes through a field accessor.
	///
	/// `names` is a comma-separated list of aliases, e.g. `"-s,--steps"`.
	pub fn add_option<T, F>(&mut self, names: &str, field: F, description: &str)
	where
		T: ParseCliValue + 'static,
		F: Fn(&mut C) -> &mut T + Clone + 'static,
	{
		let parse: ParseFn<C> = Box::new(move |text, cfg| parse_value(text, field(cfg)));
		let mut opt = Opt {
			aliases: Self::parse_aliases(names),
			description: description.to_owned(),
			parse,
			positional: false,
			flag: false,
			flag_target: None,
			multi: false,
			required: false,
			count: 0,
		};
		opt.positional = opt
			.aliases
			.first()
			.map(|a| !a.name.starts_with('-'))
			.unwrap_or(false);
		self.opts.push(opt);
	}

	/// Register a `bool` flag (no value argument; presence sets `true`).
	pub fn add_flag<F>(&mut self, names: &str, field: F, description: &str)
	where
		F: Fn(&mut C, bool) + Clone + 'static,
	{
		let ft: FlagTarget<C> = Box::new(field.clone());
		// dummy parse; flags never consume a value token
		let parse: ParseFn<C> = Box::new(move |_text, _cfg| true);
		self.opts.push(Opt {
			aliases: Self::parse_aliases(names),
			description: description.to_owned(),
			parse,
			positional: false,
			flag: true,
			flag_target: Some(ft),
			multi: false,
			required: false,
			count: 0,
		});
	}

	/// Register a multi-option that appends to a `Vec<String>`.
	pub fn add_multi<F>(&mut self, names: &str, field: F, description: &str)
	where
		F: Fn(&mut C) -> &mut Vec<String> + Clone + 'static,
	{
		let parse: ParseFn<C> = Box::new(move |text, cfg| {
			field(cfg).push(text.to_owned());
			true
		});
		self.opts.push(Opt {
			aliases: Self::parse_aliases(names),
			description: description.to_owned(),
			parse,
			positional: false,
			flag: false,
			flag_target: None,
			multi: true,
			required: false,
			count: 0,
		});
	}

	/// Register a positional argument.
	pub fn add_positional<T, F>(&mut self, name: &str, field: F, description: &str, required: bool)
	where
		T: ParseCliValue + 'static,
		F: Fn(&mut C) -> &mut T + Clone + 'static,
	{
		let parse: ParseFn<C> = Box::new(move |text, cfg| parse_value(text, field(cfg)));
		self.opts.push(Opt {
			aliases: vec![Alias {
				name: name.to_owned(),
				flag_value: true,
			}],
			description: description.to_owned(),
			parse,
			positional: true,
			flag: false,
			flag_target: None,
			multi: false,
			required,
			count: 0,
		});
	}

	fn parse_aliases(names: &str) -> Vec<Alias> {
		let mut out = Vec::new();
		for part in names.split(',') {
			let part = part.trim();
			if part.is_empty() {
				continue;
			}
			let (name, flag_value) = if let Some(stripped) = part.strip_suffix("{false}") {
				(stripped, false)
			} else {
				(part, true)
			};
			out.push(Alias {
				name: name.to_owned(),
				flag_value,
			});
		}
		out
	}

	fn find_named(&mut self, name: &str) -> Option<(&mut Opt<C>, bool)> {
		for opt in &mut self.opts {
			if !opt.positional
				&& let Some(flag_value) = opt.matches(name)
			{
				return Some((opt, flag_value));
			}
		}
		None
	}

	fn next_positional(&mut self) -> Option<&mut Opt<C>> {
		self
			.opts
			.iter_mut()
			.find(|opt| opt.positional && (opt.multi || opt.count == 0))
	}

	fn validate_required(&self) -> Option<String> {
		for opt in &self.opts {
			if opt.required && opt.count == 0 {
				return Some(format!(
					"missing required option: {}",
					opt.aliases.first().map(|a| a.name.as_str()).unwrap_or("?")
				));
			}
		}
		None
	}
}

// ─── Cli ─────────────────────────────────────────────────────────────────────

/// Command-line configuration with YAML + argv precedence.
///
/// # Type parameter
///
/// `C` is any config struct that implements [`Default`]. Defaults are the
/// base layer; a YAML file loaded via `-c`/`--config` overrides them; explicit
/// argv flags override those.
///
/// # YAML support
///
/// YAML loading is the caller's responsibility. Pass a loader closure to
/// [`Cli::parse_args_with_yaml`] to apply YAML fields before argv parsing when
/// a `-c`/`--config` path is present.
pub struct Cli<C> {
	root: Cmd<C>,
	subcommands: Vec<Cmd<C>>,
	config: C,
	config_path: String,
	verbose: i32,
	epilog: String,
	fallthrough: bool,
	help_request: bool,
	req_sub_min: i32,
	req_sub_max: i32,
}

impl<C: Default + 'static> Cli<C> {
	/// Create a new CLI with the given program name and description.
	pub fn new(name: &str, description: &str) -> Self {
		let mut root = Cmd::new(name, description);
		// Built-in options present on every OA CLI.
		// These are matched manually; we don't register them as Opt entries
		// because the config path scan must happen before any option parse.
		let _ = &mut root; // keep for structure; config/verbose handled inline
		Self {
			root,
			subcommands: Vec::new(),
			config: C::default(),
			config_path: String::new(),
			verbose: 0,
			epilog: String::new(),
			fallthrough: true,
			help_request: false,
			req_sub_min: 0,
			req_sub_max: 0,
		}
	}

	// ─── Option registration ─────────────────────────────────────────────

	/// Register a named option on the root command.
	pub fn add_option<T, F>(&mut self, names: &str, field: F, description: &str)
	where
		T: ParseCliValue + 'static,
		F: Fn(&mut C) -> &mut T + Clone + 'static,
	{
		self.root.add_option(names, field, description);
	}

	/// Register a bool flag on the root command.
	pub fn add_flag<F>(&mut self, names: &str, field: F, description: &str)
	where
		F: Fn(&mut C, bool) + Clone + 'static,
	{
		self.root.add_flag(names, field, description);
	}

	/// Register a multi-option on the root command.
	pub fn add_multi<F>(&mut self, names: &str, field: F, description: &str)
	where
		F: Fn(&mut C) -> &mut Vec<String> + Clone + 'static,
	{
		self.root.add_multi(names, field, description);
	}

	/// Register a positional argument on the root command.
	pub fn add_positional<T, F>(&mut self, name: &str, field: F, description: &str, required: bool)
	where
		T: ParseCliValue + 'static,
		F: Fn(&mut C) -> &mut T + Clone + 'static,
	{
		self.root.add_positional(name, field, description, required);
	}

	/// Add a named subcommand and return a mutable reference to register options on it.
	pub fn add_subcommand(&mut self, name: &str, description: &str) -> &mut Cmd<C> {
		self.subcommands.push(Cmd::new(name, description));
		self.subcommands.last_mut().unwrap()
	}

	/// Require between `min` and `max` subcommands (0 = unlimited).
	pub fn require_subcommand(&mut self, min: i32, max: i32) {
		self.req_sub_min = min;
		self.req_sub_max = max;
	}

	/// Set the epilog shown after options in `--help`.
	pub fn set_epilog(&mut self, text: &str) {
		self.epilog = text.to_owned();
	}

	/// Control whether unknown flags fall through to the root command when a
	/// subcommand is active. Default: `true`.
	pub fn set_fallthrough(&mut self, enable: bool) {
		self.fallthrough = enable;
	}

	// ─── Config access ───────────────────────────────────────────────────

	/// Borrow the current config value.
	pub fn config(&self) -> &C {
		&self.config
	}

	/// Mutably borrow the current config value (for programmatic override).
	pub fn config_mut(&mut self) -> &mut C {
		&mut self.config
	}

	/// Consume the CLI and return the parsed config.
	pub fn into_config(self) -> C {
		self.config
	}

	/// Path provided via `-c`/`--config`, empty if not supplied.
	pub fn config_path(&self) -> &str {
		&self.config_path
	}

	/// Verbose level provided via `-v`/`--verbose`.
	pub fn verbose(&self) -> i32 {
		self.verbose
	}

	/// True when `--help` or `-h` was the only meaningful argument.
	pub fn help_requested(&self) -> bool {
		self.help_request
	}

	/// True when the named subcommand was parsed.
	pub fn got_subcommand(&self, name: &str) -> bool {
		self.subcommands.iter().any(|c| c.name == name && c.parsed)
	}

	/// Name of the active subcommand, or empty string.
	pub fn active_subcommand(&self) -> &str {
		self
			.subcommands
			.iter()
			.find(|c| c.parsed)
			.map(|c| c.name.as_str())
			.unwrap_or("")
	}

	// ─── Parse ───────────────────────────────────────────────────────────

	/// Parse `argv` (without the program name).
	///
	/// Returns `false` if `--help` was requested or an error occurred.
	/// The caller should exit cleanly on `false` when `help_requested()` is
	/// true, and print an error / exit with failure otherwise.
	///
	/// Precedence: `struct defaults < load_yaml callback < explicit flags`.
	pub fn parse_args(&mut self, args: &[String]) -> bool {
		self.parse_args_with_yaml(args, None::<fn(&str, &mut C)>)
	}

	/// Parse with an optional YAML loader hook.
	///
	/// `yaml_loader` is called with the raw file content (if `-c`/`--config` is
	/// present) before argv flags are applied. The caller owns the YAML
	/// parsing dependency.
	pub fn parse_args_with_yaml<F>(&mut self, args: &[String], yaml_loader: Option<F>) -> bool
	where
		F: FnOnce(&str, &mut C),
	{
		// Pass 1: scan for -c / --config before anything else.
		self.scan_config_path(args);

		// Pass 2: YAML file (if present and loader provided).
		if !self.config_path.is_empty()
			&& let Some(loader) = yaml_loader
		{
			let result = Filesystem::read_text(&Path::from(self.config_path.as_str()));
			match result {
				Ok(text) => loader(&text, &mut self.config),
				Err(e) => eprintln!("[OA CONFIG] YAML load failed: {e} (using defaults)"),
			}
		}

		// Pass 3: parse argv flags.
		self.parse_argv(args)
	}

	/// Convenience: parse `std::env::args()` (skips `argv[0]`).
	pub fn parse(&mut self) -> bool {
		let args: Vec<String> = std::env::args().skip(1).collect();
		self.parse_args(&args)
	}

	// ─── internals ───────────────────────────────────────────────────────

	fn scan_config_path(&mut self, args: &[String]) {
		let mut i = 0;
		while i < args.len() {
			let a = &args[i];
			if (a == "--config" || a == "-c") && i + 1 < args.len() {
				self.config_path = args[i + 1].clone();
				return;
			}
			if let Some(rest) = a.strip_prefix("--config=") {
				self.config_path = rest.to_owned();
				return;
			}
			if let Some(rest) = a.strip_prefix("-c=") {
				self.config_path = rest.to_owned();
				return;
			}
			i += 1;
		}
	}

	fn parse_argv(&mut self, args: &[String]) -> bool {
		self.help_request = false;
		let mut selected: Option<usize> = None; // index into self.subcommands
		let mut i = 0;

		while i < args.len() {
			let arg = &args[i];

			// Help flags.
			if arg == "--help" || arg == "-h" || arg == "--help-all" {
				self.help_request = true;
				self.print_help(selected, arg == "--help-all");
				return false;
			}

			// Built-in: -v / --verbose
			if arg == "-v" || arg == "--verbose" {
				if i + 1 < args.len() {
					if let Ok(v) = args[i + 1].parse::<i32>() {
						self.verbose = v;
						i += 1;
					}
				} else {
					self.verbose += 1;
				}
				i += 1;
				continue;
			}
			if let Some(rest) = arg.strip_prefix("--verbose=") {
				if let Ok(v) = rest.parse::<i32>() {
					self.verbose = v;
				}
				i += 1;
				continue;
			}

			// Built-in: -c / --config (already consumed in scan; skip here)
			if (arg == "-c" || arg == "--config") && i + 1 < args.len() {
				i += 2;
				continue;
			}
			if arg.starts_with("--config=") || arg.starts_with("-c=") {
				i += 1;
				continue;
			}

			// Subcommand detection (first non-flag token).
			if selected.is_none() && !starts_with_dash(arg) {
				let found_idx = self.subcommands.iter().position(|c| c.name == *arg);
				if let Some(idx) = found_idx {
					self.subcommands[idx].parsed = true;
					selected = Some(idx);
					i += 1;
					continue;
				}
			}

			if starts_with_dash(arg) {
				// Named option / flag.
				let (name, inline_value) = if let Some(eq) = arg.find('=') {
					(arg[..eq].to_owned(), Some(arg[eq + 1..].to_owned()))
				} else {
					(arg.clone(), None)
				};

				// Look up in selected subcommand first, then root if fallthrough.
				let found = if let Some(idx) = selected {
					self.subcommands[idx]
						.find_named(&name)
						.map(|(opt, fv)| (opt as *mut Opt<C>, fv))
				} else {
					None
				};
				let found = found.or_else(|| {
					if selected.is_none() || self.fallthrough {
						self
							.root
							.find_named(&name)
							.map(|(opt, fv)| (opt as *mut Opt<C>, fv))
					} else {
						None
					}
				});

				let (opt_ptr, flag_value) = match found {
					Some(p) => p,
					None => return self.fail(&format!("unknown option: {name}"), ""),
				};
				// SAFETY: we hold &mut self and no other borrow to opts exists.
				let opt = unsafe { &mut *opt_ptr };

				if opt.flag {
					if inline_value.is_some() {
						return self.fail(&format!("flag does not take a value: {name}"), "");
					}
					if let Some(ref ft) = opt.flag_target {
						ft(&mut self.config, flag_value);
					}
					opt.count += 1;
					i += 1;
					continue;
				}

				let value = if let Some(v) = inline_value {
					i += 1;
					v
				} else {
					if i + 1 >= args.len() {
						return self.fail(&format!("option requires a value: {name}"), "");
					}
					i += 2;
					args[i - 1].clone()
				};

				if !(opt.parse)(&value, &mut self.config) {
					return self.fail(&format!("invalid value for {name}: {value}"), "");
				}
				opt.count += 1;
				continue;
			}

			// Positional.
			let cmd: *mut Cmd<C> = if let Some(idx) = selected {
				&mut self.subcommands[idx]
			} else {
				&mut self.root
			};
			let opt = unsafe { (*cmd).next_positional() };
			match opt {
				None => return self.fail(&format!("unexpected positional argument: {arg}"), ""),
				Some(opt) => {
					if !(opt.parse)(arg, &mut self.config) {
						return self.fail(&format!("invalid positional argument: {arg}"), "");
					}
					opt.count += 1;
				}
			}
			i += 1;
		}

		// Validate subcommand count.
		let sub_count = if selected.is_some() { 1 } else { 0 };
		if sub_count < self.req_sub_min || (self.req_sub_max > 0 && sub_count > self.req_sub_max) {
			return self.fail("required subcommand missing", "");
		}

		// Validate required options.
		if let Some(err) = self.root.validate_required() {
			return self.fail(&err, "");
		}
		if let (Some(idx), Some(err)) = (
			selected,
			selected.and_then(|i| self.subcommands[i].validate_required()),
		) {
			let _ = idx;
			return self.fail(&err, "");
		}
		// (selected validated above)
		true
	}

	fn fail(&self, message: &str, detail: &str) -> bool {
		eprint!("{message}");
		if !detail.is_empty() {
			eprint!(": {detail}");
		}
		eprintln!("\nUse --help for usage.");
		false
	}

	fn print_help(&self, selected: Option<usize>, all: bool) {
		let cmd = if let Some(idx) = selected {
			&self.subcommands[idx]
		} else {
			&self.root
		};
		eprintln!("{} — {}", cmd.name, cmd.description);
		eprintln!("  -c, --config <path>    YAML config file");
		eprintln!("  -v, --verbose [level]  verbose level (0-3)");
		for opt in &cmd.opts {
			Self::print_opt(opt);
		}
		if selected.is_none() && !self.subcommands.is_empty() {
			eprintln!("\nCommands:");
			for sub in &self.subcommands {
				eprintln!("  {}\n      {}", sub.name, sub.description);
				if all {
					for opt in &sub.opts {
						Self::print_opt(opt);
					}
				}
			}
		}
		if !self.epilog.is_empty() {
			eprintln!("\n{}", self.epilog);
		}
	}

	fn print_opt(opt: &Opt<C>) {
		let names: Vec<&str> = opt.aliases.iter().map(|a| a.name.as_str()).collect();
		let lhs = names.join(", ");
		let value_hint = if opt.flag || opt.positional {
			""
		} else {
			" <value>"
		};
		let req = if opt.required { " (required)" } else { "" };
		eprintln!("  {lhs}{value_hint}\n      {}{req}", opt.description);
	}
}

fn starts_with_dash(s: &str) -> bool {
	s.starts_with('-')
}
