//! MCP control surface over a live training session.
//!
//! Donor: `oa/ml/mcpTraining.h` and its implementation.

use super::training::{TrainingCommandDisposition, TrainingSession, TrainingState, TrainingValue};
use crate::network::mcp::{McpArguments, McpServer, McpTool, McpToolResult};
use crate::{Error, Result, core::push_json_string};

/// Gates controlling which MCP tools are registered for a training session.
///
/// Calling [`McpTraining::register_tools`] is the opt-in boundary.
#[derive(Clone, Debug)]
pub struct McpTrainingConfig {
	/// Enable safe-point pause/resume/checkpoint/evaluate/parameter/recapture commands.
	pub enable_commands: bool,
	/// Enable `training_pause` and `training_resume`.
	pub enable_pause_resume: bool,
	/// Enable `training_checkpoint`.
	pub enable_checkpoint: bool,
	/// Enable `training_evaluate`.
	pub enable_evaluate: bool,
	/// Enable `training_set_parameter`.
	pub enable_set_parameter: bool,
	/// Enable `training_request_recapture`.
	pub enable_recapture: bool,
	/// Enable `training_stop` (separately gated because it changes the terminal path).
	pub enable_stop: bool,
}

impl Default for McpTrainingConfig {
	fn default() -> Self {
		Self {
			enable_commands: true,
			enable_pause_resume: true,
			enable_checkpoint: true,
			enable_evaluate: true,
			enable_set_parameter: true,
			enable_recapture: true,
			enable_stop: false,
		}
	}
}

/// Registers an explicitly included MCP view over an existing training session.
///
/// The server borrows `session` through its handlers and must be closed or
/// destroyed first. `oa/ml` deliberately does not opt applications into the
/// network control surface. Rebuild and preview remain absent until their
/// checkpoint and completion contracts are admitted.
///
/// Donor: `oa::McpTraining`.
pub struct McpTraining;

impl McpTraining {
	/// Register the training MCP tools on `server`, borrowing `session`.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` if the server has an invalid configuration
	/// or has already been started or closed. Returns `AlreadyExists` if any
	/// of the training tool names conflict with previously registered tools.
	pub fn register_tools(
		server: &mut McpServer,
		session: TrainingSession,
		config: McpTrainingConfig,
	) -> Result<()> {
		if let Some(e) = server.configuration_error() {
			return Err(Error::invalid_argument(e.message().to_owned()));
		}
		if server.is_started() || server.is_closed() {
			return Err(Error::failed_precondition(
				"training MCP tools must be registered before server start",
			));
		}

		let mut tools: Vec<McpTool> = Vec::new();

		// ── read tools ───────────────────────────────────────────────────────
		{
			let s = session.clone();
			tools.push(read_tool(
				"training_status",
				"Read the latest immutable training state and scalar timing snapshot.",
				r#"{"type":"object","properties":{"revision":{"type":"integer","minimum":0},"state":{"type":"string"},"step":{"type":"integer","minimum":0},"epoch":{"type":"integer","minimum":0},"learningRate":{"type":["number","null"]},"loss":{"type":["number","null"]},"gpuMs":{"type":["number","null"]},"wallMs":{"type":["number","null"]}},"required":["revision","state","step","epoch","learningRate","loss","gpuMs","wallMs"],"additionalProperties":false}"#,
				move |args: &McpArguments| {
					if !args.is_empty() {
						return Err(Error::invalid_argument("training_status accepts no arguments"));
					}
					let json = snapshot_json(&s);
					let mut result = McpToolResult::success(json.clone());
					result.structured_content_json = json;
					Ok(result)
				},
			));
		}

		{
			let s = session.clone();
			tools.push(read_tool(
				"training_metrics",
				"Read metric samples from the latest immutable training snapshot.",
				r#"{"type":"object","properties":{"metrics":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"value":{"type":["number","null"]},"step":{"type":"integer","minimum":0}},"required":["name","value","step"],"additionalProperties":false}}},"required":["metrics"],"additionalProperties":false}"#,
				move |args: &McpArguments| {
					if !args.is_empty() {
						return Err(Error::invalid_argument(
							"training_metrics accepts no arguments",
						));
					}
					let json = metrics_json(&s);
					let mut result = McpToolResult::success(json.clone());
					result.structured_content_json = json;
					Ok(result)
				},
			));
		}

		{
			let s = session.clone();
			let mut results_tool = read_tool(
				"training_results",
				"Read the bounded command audit stream after a caller-owned sequence cursor.",
				r#"{"type":"object","properties":{"results":{"type":"array","items":{"type":"object","properties":{"sequence":{"type":"integer","minimum":1},"revision":{"type":"integer","minimum":0},"disposition":{"type":"string"},"state":{"type":"string"},"status":{"type":"string"}},"required":["sequence","revision","disposition","state","status"],"additionalProperties":false}}},"required":["results"],"additionalProperties":false}"#,
				move |args: &McpArguments| {
					let count = args.len();
					if count > 1 || (count == 1 && !args.contains("afterSequence")) {
						return Err(Error::invalid_argument(
							"training_results accepts only afterSequence",
						));
					}
					let after: u64 = if args.contains("afterSequence") {
						args.unsigned_integer("afterSequence")?
					} else {
						0
					};
					let results = s.results_after(after);
					let mut json = String::from(r#"{"results":["#);
					for (i, entry) in results.iter().enumerate() {
						if i != 0 {
							json.push(',');
						}
						json.push_str(r#"{"sequence":"#);
						json.push_str(&entry.sequence.to_string());
						json.push_str(r#","revision":"#);
						json.push_str(&entry.revision.to_string());
						json.push_str(r#","disposition":"#);
						push_json_string(&mut json, disposition_name(entry.disposition));
						json.push_str(r#","state":"#);
						push_json_string(&mut json, state_name(entry.state));
						json.push_str(r#","status":"#);
						let status_text = if let Some(ref e) = entry.error {
							e.message().to_owned()
						} else {
							String::from("ok")
						};
						push_json_string(&mut json, &status_text);
						json.push('}');
					}
					json.push_str("]}");
					let mut result = McpToolResult::success(json.clone());
					result.structured_content_json = json;
					Ok(result)
				},
			);
			results_tool.input_schema_json = r#"{"type":"object","properties":{"afterSequence":{"type":"integer","minimum":0}},"additionalProperties":false}"#.to_owned();
			tools.push(results_tool);
		}

		// ── command tools ────────────────────────────────────────────────────
		if config.enable_commands {
			let revision_schema = r#"{"type":"object","properties":{"expectedRevision":{"type":"integer","minimum":0}},"additionalProperties":false}"#;

			if config.enable_pause_resume {
				let s = session.clone();
				tools.push(command_tool(
					"training_pause",
					"Pause at the next training safe point.",
					revision_schema,
					false,
					move |args: &McpArguments| {
						let revision = expected_revision(args)?;
						accepted(s.pause(revision))
					},
				));

				let s = session.clone();
				tools.push(command_tool(
					"training_resume",
					"Resume a paused training session at its next safe point.",
					revision_schema,
					false,
					move |args: &McpArguments| {
						let revision = expected_revision(args)?;
						accepted(s.resume(revision))
					},
				));
			}

			if config.enable_checkpoint {
				let s = session.clone();
				tools.push(command_tool(
					"training_checkpoint",
					"Request a checkpoint at the next training safe point.",
					revision_schema,
					false,
					move |args: &McpArguments| {
						let revision = expected_revision(args)?;
						accepted(s.checkpoint(revision))
					},
				));
			}

			if config.enable_evaluate {
				let s = session.clone();
				tools.push(command_tool(
					"training_evaluate",
					"Request evaluation at the next training safe point.",
					revision_schema,
					false,
					move |args: &McpArguments| {
						let revision = expected_revision(args)?;
						accepted(s.evaluate(revision))
					},
				));
			}

			if config.enable_set_parameter {
				let s = session.clone();
				tools.push(command_tool(
					"training_set_parameter",
					"Queue one allowlisted typed parameter update for a training safe point.",
					r#"{"type":"object","properties":{"name":{"type":"string"},"value":{"anyOf":[{"type":"boolean"},{"type":"integer"},{"type":"number"},{"type":"string"}]},"expectedRevision":{"type":"integer","minimum":0}},"required":["name","value"],"additionalProperties":false}"#,
					false,
					move |args: &McpArguments| {
						let count = args.len();
						if (count != 2 && count != 3)
							|| !args.contains("name")
							|| !args.contains("value")
							|| (count == 3 && !args.contains("expectedRevision"))
						{
							return Err(Error::invalid_argument(
								"training_set_parameter accepts only name, value, and expectedRevision",
							));
						}
						let name = args.string("name")?;
						let value = training_value(args)?;
						let revision = revision_value(args)?;
						accepted(s.set_parameter(name, value, revision))
					},
				));
			}

			if config.enable_recapture {
				let s = session.clone();
				tools.push(command_tool(
					"training_request_recapture",
					"Request graph recapture for a paused training session.",
					revision_schema,
					false,
					move |args: &McpArguments| {
						let revision = expected_revision(args)?;
						accepted(s.request_recapture(revision))
					},
				));
			}
		}

		if config.enable_stop {
			let s = session.clone();
			tools.push(command_tool(
				"training_stop",
				"Request terminal training stop at the next safe point.",
				r#"{"type":"object","properties":{"expectedRevision":{"type":"integer","minimum":0}},"additionalProperties":false}"#,
				true,
				move |args: &McpArguments| {
					let revision = expected_revision(args)?;
					accepted(s.stop(revision))
				},
			));
		}

		// Duplicate-name check before any mutation of the server.
		for tool in &tools {
			if server.has_tool(&tool.name) {
				return Err(Error::already_exists(format!(
					"training MCP tool already exists: {}",
					tool.name
				)));
			}
		}
		for tool in tools {
			server.add_tool(tool)?;
		}
		Ok(())
	}
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn state_name(state: TrainingState) -> &'static str {
	match state {
		TrainingState::Running => "running",
		TrainingState::Paused => "paused",
		TrainingState::Stopping => "stopping",
		TrainingState::Completed => "completed",
		TrainingState::Failed => "failed",
	}
}

fn disposition_name(disposition: TrainingCommandDisposition) -> &'static str {
	if disposition == TrainingCommandDisposition::Applied {
		"applied"
	} else {
		"rejected"
	}
}

fn write_finite(out: &mut String, value: f64) {
	if value.is_finite() {
		let text = format!("{value}");
		out.push_str(&text);
	} else {
		out.push_str("null");
	}
}

fn snapshot_json(session: &TrainingSession) -> String {
	let snapshot = session.current_snapshot();
	let mut json = String::from(r#"{"revision":"#);
	json.push_str(&snapshot.revision.to_string());
	json.push_str(r#","state":"#);
	push_json_string(&mut json, state_name(snapshot.state));
	json.push_str(r#","step":"#);
	json.push_str(&snapshot.step.to_string());
	json.push_str(r#","epoch":"#);
	json.push_str(&snapshot.epoch.to_string());
	json.push_str(r#","learningRate":"#);
	write_finite(&mut json, f64::from(snapshot.learning_rate));
	json.push_str(r#","loss":"#);
	write_finite(&mut json, f64::from(snapshot.loss));
	json.push_str(r#","gpuMs":"#);
	write_finite(&mut json, snapshot.gpu_ms);
	json.push_str(r#","wallMs":"#);
	write_finite(&mut json, snapshot.wall_ms);
	json.push('}');
	json
}

fn metrics_json(session: &TrainingSession) -> String {
	let latest = session.latest_snapshot();
	let mut json = String::from(r#"{"metrics":["#);
	if let Some(snapshot) = latest {
		for (i, metric) in snapshot.metrics.iter().enumerate() {
			if i != 0 {
				json.push(',');
			}
			json.push_str(r#"{"name":"#);
			push_json_string(&mut json, &metric.name);
			json.push_str(r#","value":"#);
			write_finite(&mut json, metric.value);
			json.push_str(r#","step":"#);
			json.push_str(&metric.step.to_string());
			json.push('}');
		}
	}
	json.push_str("]}");
	json
}

fn revision_value(args: &McpArguments) -> Result<u64> {
	if !args.contains("expectedRevision") {
		return Ok(0);
	}
	args.unsigned_integer("expectedRevision")
}

fn expected_revision(args: &McpArguments) -> Result<u64> {
	let count = args.len();
	if count > 1 || (count == 1 && !args.contains("expectedRevision")) {
		return Err(Error::invalid_argument("only expectedRevision is accepted"));
	}
	revision_value(args)
}

fn training_value(args: &McpArguments) -> Result<TrainingValue> {
	if !args.contains("value") {
		return Err(Error::invalid_argument("value is required"));
	}
	if let Ok(v) = args.boolean("value") {
		return Ok(TrainingValue::Boolean(v));
	}
	if let Ok(v) = args.integer("value") {
		return Ok(TrainingValue::Integer(v));
	}
	if let Ok(v) = args.number("value") {
		return Ok(TrainingValue::Float(v));
	}
	if let Ok(v) = args.string("value") {
		return Ok(TrainingValue::String(v));
	}
	Err(Error::invalid_argument(
		"value must be a boolean, integer, finite number or string",
	))
}

fn accepted(sequence: Result<u64>) -> Result<McpToolResult> {
	let seq = sequence?;
	let text = format!("training command accepted with sequence {seq}");
	let mut result = McpToolResult::success(text);
	result.structured_content_json = format!(r#"{{"sequence":{seq}}}"#);
	Ok(result)
}

fn read_tool(
	name: &'static str,
	description: &'static str,
	output_schema: &'static str,
	call: impl Fn(&McpArguments) -> Result<McpToolResult> + Send + Sync + 'static,
) -> McpTool {
	McpTool {
		name: name.to_owned(),
		title: String::new(),
		description: description.to_owned(),
		input_schema_json: r#"{"type":"object","properties":{},"additionalProperties":false}"#
			.to_owned(),
		output_schema_json: output_schema.to_owned(),
		read_only: true,
		destructive: false,
		idempotent: true,
		open_world: false,
		call: Box::new(call),
	}
}

fn command_tool(
	name: &'static str,
	description: &'static str,
	input_schema: &'static str,
	destructive: bool,
	call: impl Fn(&McpArguments) -> Result<McpToolResult> + Send + Sync + 'static,
) -> McpTool {
	McpTool {
		name: name.to_owned(),
		title: String::new(),
		description: description.to_owned(),
		input_schema_json: input_schema.to_owned(),
		output_schema_json: r#"{"type":"object","properties":{"sequence":{"type":"integer","minimum":1}},"required":["sequence"],"additionalProperties":false}"#.to_owned(),
		read_only: false,
		destructive,
		idempotent: false,
		open_world: false,
		call: Box::new(call),
	}
}
