//! Transport-independent Model Context Protocol control plane.
//!
//! Donor: `oa/network/mcp.h` and its implementation.

use std::io::{BufRead, Write};

use crate::{Error, Result, core::push_json_string};

/// Latest protocol version advertised and required for modern (non-legacy) requests.
pub const MCP_LATEST_PROTOCOL_VERSION: &str = "2026-07-28";

/// Cache scope hint for MCP discovery and list responses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum McpCacheScope {
	/// Responses may only be cached by the requesting client.
	#[default]
	Private,
	/// Responses may be shared across clients.
	Public,
}

/// Configuration for one [`McpServer`] instance.
#[derive(Clone, Debug)]
pub struct McpServerConfig {
	/// Server name reported during initialization and discovery.
	pub name: String,
	/// Server version string.
	pub version: String,
	/// Optional human-readable instructions for the model.
	pub instructions: String,
	/// Maximum accepted and emitted message size in bytes (minimum 1024).
	pub max_message_bytes: usize,
	/// Maximum JSON nesting depth accepted during parsing (1–256).
	pub max_nesting_depth: u32,
	/// Maximum JSON node count accepted during parsing (minimum 1).
	pub max_json_nodes: u32,
	/// Cache TTL in milliseconds for discovery/list responses.
	pub cache_ttl_ms: u32,
	/// Cache scope for discovery/list responses.
	pub cache_scope: McpCacheScope,
}

impl Default for McpServerConfig {
	fn default() -> Self {
		Self {
			name: String::from("oa"),
			version: String::new(),
			instructions: String::new(),
			max_message_bytes: 1024 * 1024,
			max_nesting_depth: 64,
			max_json_nodes: 65536,
			cache_ttl_ms: 0,
			cache_scope: McpCacheScope::Private,
		}
	}
}

// ── private JSON parser ────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JsonKind {
	Null,
	Boolean,
	Number,
	String,
	Array,
	Object,
}

struct JsonMember {
	name: std::string::String,
	value: Box<JsonValue>,
}

/// Type alias for the complex tool-call handler type.
type McpToolCallFn = Box<dyn Fn(&McpArguments<'_>) -> Result<McpToolResult> + Send + Sync>;

struct JsonValue {
	kind: JsonKind,
	boolean: bool,
	text: std::string::String,
	array: Vec<JsonValue>,
	object: Vec<JsonMember>,
}

impl Default for JsonValue {
	fn default() -> Self {
		Self {
			kind: JsonKind::Null,
			boolean: false,
			text: std::string::String::new(),
			array: Vec::new(),
			object: Vec::new(),
		}
	}
}

struct JsonParser<'a> {
	text: &'a [u8],
	pos: usize,
	max_depth: u32,
	max_nodes: u32,
	nodes: u32,
	error: std::string::String,
}

impl<'a> JsonParser<'a> {
	fn new(text: &'a str, max_depth: u32, max_nodes: u32) -> Self {
		Self {
			text: text.as_bytes(),
			pos: 0,
			max_depth,
			max_nodes,
			nodes: 0,
			error: std::string::String::new(),
		}
	}

	fn parse(&mut self) -> Option<JsonValue> {
		self.skip_whitespace();
		let value = self.parse_value(0)?;
		self.skip_whitespace();
		if self.pos != self.text.len() {
			self.fail("unexpected trailing data");
			return None;
		}
		Some(value)
	}

	fn parse_value(&mut self, depth: u32) -> Option<JsonValue> {
		if self.nodes >= self.max_nodes {
			self.fail("JSON node limit exceeded");
			return None;
		}
		self.nodes += 1;
		if self.pos >= self.text.len() {
			self.fail("expected a JSON value");
			return None;
		}
		let c = self.text[self.pos];
		match c {
			b'n' => self.parse_literal(b"null", JsonKind::Null),
			b't' => {
				let mut v = self.parse_literal(b"true", JsonKind::Boolean)?;
				v.boolean = true;
				Some(v)
			}
			b'f' => self.parse_literal(b"false", JsonKind::Boolean),
			b'"' => {
				let text = self.parse_string()?;
				Some(JsonValue {
					kind: JsonKind::String,
					text,
					..Default::default()
				})
			}
			b'[' => self.parse_array(depth),
			b'{' => self.parse_object(depth),
			b'-' | b'0'..=b'9' => self.parse_number(),
			_ => {
				self.fail("invalid JSON value");
				None
			}
		}
	}

	fn parse_literal(&mut self, literal: &[u8], kind: JsonKind) -> Option<JsonValue> {
		if self.text.get(self.pos..self.pos + literal.len()) != Some(literal) {
			self.fail("invalid JSON literal");
			return None;
		}
		self.pos += literal.len();
		Some(JsonValue {
			kind,
			..Default::default()
		})
	}

	fn parse_array(&mut self, depth: u32) -> Option<JsonValue> {
		if depth >= self.max_depth {
			self.fail("JSON nesting depth exceeded");
			return None;
		}
		let mut v = JsonValue {
			kind: JsonKind::Array,
			..Default::default()
		};
		self.pos += 1;
		self.skip_whitespace();
		if self.consume(b']') {
			return Some(v);
		}
		loop {
			let item = self.parse_value(depth + 1)?;
			v.array.push(item);
			self.skip_whitespace();
			if self.consume(b']') {
				return Some(v);
			}
			if !self.consume(b',') {
				self.fail("expected ',' or ']' in array");
				return None;
			}
			self.skip_whitespace();
		}
	}

	fn parse_object(&mut self, depth: u32) -> Option<JsonValue> {
		if depth >= self.max_depth {
			self.fail("JSON nesting depth exceeded");
			return None;
		}
		let mut v = JsonValue {
			kind: JsonKind::Object,
			..Default::default()
		};
		self.pos += 1;
		self.skip_whitespace();
		if self.consume(b'}') {
			return Some(v);
		}
		loop {
			if self.pos >= self.text.len() || self.text[self.pos] != b'"' {
				self.fail("expected object member name");
				return None;
			}
			let name = self.parse_string()?;
			if v.object.iter().any(|m| m.name == name) {
				self.fail("duplicate object member");
				return None;
			}
			self.skip_whitespace();
			if !self.consume(b':') {
				self.fail("expected ':' after object member name");
				return None;
			}
			self.skip_whitespace();
			let value = self.parse_value(depth + 1)?;
			v.object.push(JsonMember {
				name,
				value: Box::new(value),
			});
			self.skip_whitespace();
			if self.consume(b'}') {
				return Some(v);
			}
			if !self.consume(b',') {
				self.fail("expected ',' or '}' in object");
				return None;
			}
			self.skip_whitespace();
		}
	}

	fn parse_string(&mut self) -> Option<std::string::String> {
		if !self.consume(b'"') {
			self.fail("expected JSON string");
			return None;
		}
		let mut out = std::string::String::new();
		while self.pos < self.text.len() {
			let c = self.text[self.pos];
			self.pos += 1;
			if c == b'"' {
				return Some(out);
			}
			if c == b'\\' {
				if self.pos >= self.text.len() {
					self.fail("incomplete JSON escape");
					return None;
				}
				let esc = self.text[self.pos];
				self.pos += 1;
				match esc {
					b'"' => out.push('"'),
					b'\\' => out.push('\\'),
					b'/' => out.push('/'),
					b'b' => out.push('\u{0008}'),
					b'f' => out.push('\u{000c}'),
					b'n' => out.push('\n'),
					b'r' => out.push('\r'),
					b't' => out.push('\t'),
					b'u' => {
						let mut codepoint = self.parse_hex4()?;
						if (0xD800..=0xDBFF).contains(&codepoint) {
							if self.pos + 2 > self.text.len()
								|| self.text[self.pos] != b'\\'
								|| self.text[self.pos + 1] != b'u'
							{
								self.fail("high surrogate without low surrogate");
								return None;
							}
							self.pos += 2;
							let low = self.parse_hex4()?;
							if !(0xDC00..=0xDFFF).contains(&low) {
								self.fail("invalid low surrogate");
								return None;
							}
							codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
						} else if (0xDC00..=0xDFFF).contains(&codepoint) {
							self.fail("unpaired low surrogate");
							return None;
						}
						match char::from_u32(codepoint) {
							Some(ch) => out.push(ch),
							None => {
								self.fail("invalid Unicode codepoint");
								return None;
							}
						}
					}
					_ => {
						self.fail("invalid JSON escape");
						return None;
					}
				}
				continue;
			}
			if c < 0x20 {
				self.fail("unescaped control character in string");
				return None;
			}
			// Accept the full raw UTF-8 byte(s) — Rust strings are already
			// guaranteed valid UTF-8, so no further validation is required here.
			if c < 0x80 {
				out.push(c as char);
			} else {
				// Multi-byte UTF-8: collect bytes until end of sequence.
				let first = c;
				let (extra, mut codepoint): (usize, u32) = if (0xC2..=0xDF).contains(&first) {
					(1, u32::from(first & 0x1F))
				} else if (0xE0..=0xEF).contains(&first) {
					(2, u32::from(first & 0x0F))
				} else if (0xF0..=0xF4).contains(&first) {
					(3, u32::from(first & 0x07))
				} else {
					self.fail("invalid UTF-8 leading byte");
					return None;
				};
				if self.pos + extra > self.text.len() {
					self.fail("incomplete UTF-8 sequence");
					return None;
				}
				for i in 0..extra {
					let cont = self.text[self.pos + i];
					if (cont & 0xC0) != 0x80 {
						self.fail("invalid UTF-8 continuation byte");
						return None;
					}
					codepoint = (codepoint << 6) | u32::from(cont & 0x3F);
				}
				if (extra == 2 && codepoint < 0x800)
					|| (extra == 3 && codepoint < 0x10000)
					|| (0xD800..=0xDFFF).contains(&codepoint)
					|| codepoint > 0x10FFFF
				{
					self.fail("invalid UTF-8 code point");
					return None;
				}
				self.pos += extra;
				match char::from_u32(codepoint) {
					Some(ch) => out.push(ch),
					None => {
						self.fail("invalid UTF-8 code point");
						return None;
					}
				}
			}
		}
		self.fail("unterminated JSON string");
		None
	}

	fn parse_hex4(&mut self) -> Option<u32> {
		if self.pos + 4 > self.text.len() {
			self.fail("incomplete Unicode escape");
			return None;
		}
		let mut value = 0u32;
		for _ in 0..4 {
			let c = self.text[self.pos];
			self.pos += 1;
			value <<= 4;
			match c {
				b'0'..=b'9' => value |= u32::from(c - b'0'),
				b'a'..=b'f' => value |= u32::from(c - b'a' + 10),
				b'A'..=b'F' => value |= u32::from(c - b'A' + 10),
				_ => {
					self.fail("invalid Unicode escape");
					return None;
				}
			}
		}
		Some(value)
	}

	fn parse_number(&mut self) -> Option<JsonValue> {
		let start = self.pos;
		self.consume(b'-');
		if self.pos >= self.text.len() {
			self.fail("incomplete JSON number");
			return None;
		}
		if self.text[self.pos] == b'0' {
			self.pos += 1;
			if self.pos < self.text.len() && self.text[self.pos] >= b'0' && self.text[self.pos] <= b'9' {
				self.fail("leading zero in JSON number");
				return None;
			}
		} else {
			if !(b'1'..=b'9').contains(&self.text[self.pos]) {
				self.fail("invalid JSON number");
				return None;
			}
			while self.pos < self.text.len() && self.text[self.pos] >= b'0' && self.text[self.pos] <= b'9'
			{
				self.pos += 1;
			}
		}
		if self.pos < self.text.len() && self.text[self.pos] == b'.' {
			self.pos += 1;
			if self.pos >= self.text.len() || !self.text[self.pos].is_ascii_digit() {
				self.fail("missing fraction digits in JSON number");
				return None;
			}
			while self.pos < self.text.len() && self.text[self.pos].is_ascii_digit() {
				self.pos += 1;
			}
		}
		if self.pos < self.text.len() && (self.text[self.pos] == b'e' || self.text[self.pos] == b'E') {
			self.pos += 1;
			if self.pos < self.text.len() && (self.text[self.pos] == b'+' || self.text[self.pos] == b'-')
			{
				self.pos += 1;
			}
			if self.pos >= self.text.len() || !self.text[self.pos].is_ascii_digit() {
				self.fail("missing exponent digits in JSON number");
				return None;
			}
			while self.pos < self.text.len() && self.text[self.pos].is_ascii_digit() {
				self.pos += 1;
			}
		}
		let text = std::str::from_utf8(&self.text[start..self.pos])
			.unwrap_or_default()
			.to_owned();
		Some(JsonValue {
			kind: JsonKind::Number,
			text,
			..Default::default()
		})
	}

	fn skip_whitespace(&mut self) {
		while self.pos < self.text.len() && matches!(self.text[self.pos], b' ' | b'\t' | b'\r' | b'\n')
		{
			self.pos += 1;
		}
	}

	fn consume(&mut self, c: u8) -> bool {
		if self.pos < self.text.len() && self.text[self.pos] == c {
			self.pos += 1;
			true
		} else {
			false
		}
	}

	fn fail(&mut self, message: &str) {
		if self.error.is_empty() {
			self.error = format!("{message} at byte {}", self.pos);
		}
	}
}

fn parse_json(
	text: &str,
	max_depth: u32,
	max_nodes: u32,
) -> std::result::Result<JsonValue, std::string::String> {
	let mut parser = JsonParser::new(text, max_depth, max_nodes);
	match parser.parse() {
		Some(value) => Ok(value),
		None => Err(parser.error),
	}
}

fn write_json(out: &mut String, value: &JsonValue) {
	match value.kind {
		JsonKind::Null => out.push_str("null"),
		JsonKind::Boolean => out.push_str(if value.boolean { "true" } else { "false" }),
		JsonKind::Number => out.push_str(&value.text),
		JsonKind::String => push_json_string(out, &value.text),
		JsonKind::Array => {
			out.push('[');
			for (i, item) in value.array.iter().enumerate() {
				if i != 0 {
					out.push(',');
				}
				write_json(out, item);
			}
			out.push(']');
		}
		JsonKind::Object => {
			out.push('{');
			for (i, member) in value.object.iter().enumerate() {
				if i != 0 {
					out.push(',');
				}
				push_json_string(out, &member.name);
				out.push(':');
				write_json(out, &member.value);
			}
			out.push('}');
		}
	}
}

fn serialize_json(value: &JsonValue) -> String {
	let mut out = String::new();
	write_json(&mut out, value);
	out
}

fn find_member<'a>(object: &'a JsonValue, name: &str) -> Option<&'a JsonValue> {
	if object.kind != JsonKind::Object {
		return None;
	}
	object
		.object
		.iter()
		.find_map(|m| (m.name == name).then(|| m.value.as_ref()))
}

fn canonical_schema(
	schema: &str,
	require_object_type: bool,
	max_depth: u32,
	max_nodes: u32,
) -> Result<String> {
	let parsed = parse_json(schema, max_depth, max_nodes)
		.map_err(|e| Error::invalid_argument(format!("invalid MCP JSON schema: {e}")))?;
	if parsed.kind != JsonKind::Object {
		return Err(Error::invalid_argument(
			"MCP JSON schema root must be an object",
		));
	}
	if require_object_type {
		match find_member(&parsed, "type") {
			Some(t) if t.kind == JsonKind::String && t.text == "object" => {}
			_ => {
				return Err(Error::invalid_argument(
					"MCP tool input schema requires root type 'object'",
				));
			}
		}
	}
	Ok(serialize_json(&parsed))
}

fn is_valid_tool_name(name: &str) -> bool {
	!name.is_empty()
		&& name.len() <= 128
		&& name
			.bytes()
			.all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.')
}

fn is_valid_uri(uri: &str) -> bool {
	if uri.is_empty() {
		return false;
	}
	let bytes = uri.as_bytes();
	let colon = bytes.iter().position(|&c| c == b':');
	let Some(colon) = colon else {
		return false;
	};
	if colon == 0 {
		return false;
	}
	if bytes.iter().any(|&c| c <= 0x20 || c == 0x7F) {
		return false;
	}
	// Scheme must start with a letter then letters/digits/+/-/.
	bytes[0].is_ascii_alphabetic()
		&& bytes[1..colon]
			.iter()
			.all(|&c| c.is_ascii_alphanumeric() || c == b'+' || c == b'-' || c == b'.')
}

fn is_legacy_version(version: &str) -> bool {
	matches!(
		version,
		"2025-11-25" | "2025-06-18" | "2025-03-26" | "2024-11-05"
	)
}

fn make_error(id_json: &str, code: i32, message: &str, data_json: &str) -> String {
	let mut json = format!(r#"{{"jsonrpc":"2.0","id":{id_json},"error":{{"code":{code},"message":"#);
	push_json_string(&mut json, message);
	if !data_json.is_empty() {
		json.push_str(r#","data":"#);
		json.push_str(data_json);
	}
	json.push_str("}}");
	json
}

fn make_success(id_json: &str, result_json: &str) -> String {
	format!(r#"{{"jsonrpc":"2.0","id":{id_json},"result":{result_json}}}"#)
}

fn server_info_json(config: &McpServerConfig) -> String {
	let mut json = r#"{"name":"#.to_owned();
	push_json_string(&mut json, &config.name);
	json.push_str(r#","version":"#);
	push_json_string(&mut json, &config.version);
	json.push('}');
	json
}

fn capabilities_json(has_tools: bool, has_resources: bool) -> String {
	let mut json = String::from("{");
	let mut comma = false;
	if has_resources {
		json.push_str(r#""resources":{"subscribe":false,"listChanged":false}"#);
		comma = true;
	}
	if has_tools {
		if comma {
			json.push(',');
		}
		json.push_str(r#""tools":{"listChanged":false}"#);
	}
	json.push('}');
	json
}

fn append_modern_result_fields(out: &mut String, config: &McpServerConfig) {
	out.push_str(r#""resultType":"complete","_meta":{"io.modelcontextprotocol/serverInfo":"#);
	out.push_str(&server_info_json(config));
	out.push('}');
}

fn append_cache_fields(out: &mut String, config: &McpServerConfig) {
	out.push_str(&format!(r#","ttlMs":{}"#, config.cache_ttl_ms));
	out.push_str(r#","cacheScope":"#);
	out.push_str(if config.cache_scope == McpCacheScope::Public {
		r#""public""#
	} else {
		r#""private""#
	});
}

fn supported_version_data(requested: &str) -> String {
	let mut data = r#"{"supported":["2026-07-28","2025-11-25","2025-06-18","2025-03-26","2024-11-05"],"requested":"#.to_owned();
	push_json_string(&mut data, requested);
	data.push('}');
	data
}

struct RequestEnvelope {
	id_json: String,
	method: String,
	is_notification: bool,
}

fn parse_envelope(root: &JsonValue) -> std::result::Result<RequestEnvelope, &'static str> {
	if root.kind != JsonKind::Object {
		return Err("JSON-RPC message must be an object");
	}
	let jsonrpc = find_member(root, "jsonrpc");
	let method = find_member(root, "method");
	match (jsonrpc, method) {
		(Some(jrpc), Some(m))
			if jrpc.kind == JsonKind::String
				&& jrpc.text == "2.0"
				&& m.kind == JsonKind::String
				&& !m.text.is_empty() => {}
		_ => return Err("invalid JSON-RPC request envelope"),
	}
	let method_text = method.ok_or("missing JSON-RPC method")?.text.clone();
	let id = find_member(root, "id");
	let (id_json, is_notification) = match id {
		None => ("null".to_owned(), true),
		Some(id_value) => {
			if id_value.kind != JsonKind::String && id_value.kind != JsonKind::Number {
				return Err("JSON-RPC id must be a string or number");
			}
			(serialize_json(id_value), false)
		}
	};
	// Validate params
	if find_member(root, "params").is_some_and(|params| params.kind != JsonKind::Object) {
		return Err("JSON-RPC params must be an object");
	}
	// Params presence is validated here; callers access params directly from the
	// root JsonValue reference via find_member(root, "params").
	Ok(RequestEnvelope {
		id_json,
		method: method_text,
		is_notification,
	})
}

fn request_error(request: &RequestEnvelope, code: i32, message: &str, data: &str) -> String {
	if request.is_notification {
		return String::new();
	}
	make_error(&request.id_json, code, message, data)
}

fn validate_implementation(value: &JsonValue, context: &str) -> Result<()> {
	if value.kind != JsonKind::Object {
		return Err(Error::invalid_argument(format!(
			"{context} must be an object"
		)));
	}
	let name = find_member(value, "name");
	let version = find_member(value, "version");
	match (name, version) {
		(Some(n), Some(v))
			if n.kind == JsonKind::String
				&& !n.text.is_empty()
				&& v.kind == JsonKind::String
				&& !v.text.is_empty() => {}
		_ => {
			return Err(Error::invalid_argument(format!(
				"{context} requires non-empty name and version strings"
			)));
		}
	}
	Ok(())
}

fn validate_modern_metadata(
	params: Option<&JsonValue>,
	requested_version: &mut String,
) -> Result<()> {
	let params =
		params.ok_or_else(|| Error::invalid_argument("current MCP requests require params"))?;
	if params.kind != JsonKind::Object {
		return Err(Error::invalid_argument(
			"current MCP requests require params",
		));
	}
	let meta = find_member(params, "_meta")
		.ok_or_else(|| Error::invalid_argument("current MCP requests require params._meta"))?;
	if meta.kind != JsonKind::Object {
		return Err(Error::invalid_argument(
			"current MCP requests require params._meta",
		));
	}
	let version = find_member(meta, "io.modelcontextprotocol/protocolVersion").ok_or_else(|| {
		Error::invalid_argument("current MCP request metadata requires protocolVersion")
	})?;
	if version.kind != JsonKind::String {
		return Err(Error::invalid_argument(
			"current MCP request metadata requires protocolVersion",
		));
	}
	*requested_version = version.text.clone();
	let capabilities =
		find_member(meta, "io.modelcontextprotocol/clientCapabilities").ok_or_else(|| {
			Error::invalid_argument("current MCP request metadata requires clientCapabilities")
		})?;
	if capabilities.kind != JsonKind::Object {
		return Err(Error::invalid_argument(
			"current MCP request metadata requires clientCapabilities",
		));
	}
	if let Some(client_info) = find_member(meta, "io.modelcontextprotocol/clientInfo") {
		validate_implementation(client_info, "current MCP request clientInfo")?;
	}
	Ok(())
}

// ── public types ────────────────────────────────────────────────────────────

/// Validated JSON object passed to every tool call handler.
///
/// Scalar accessors provide the common application boundary.
/// [`Self::json`] and [`Self::json_field`] preserve nested values without
/// exposing OA's private protocol parser as a public JSON framework.
pub struct McpArguments<'request> {
	json: String,
	value: &'request JsonValue,
}

impl<'request> McpArguments<'request> {
	fn new(value: &'request JsonValue) -> Self {
		Self {
			json: serialize_json(value),
			value,
		}
	}

	/// Canonical JSON serialization of the validated argument object.
	#[must_use]
	pub fn json(&self) -> &str {
		&self.json
	}

	/// Whether the argument object is empty.
	#[must_use]
	pub fn is_empty(&self) -> bool {
		self.value.object.is_empty()
	}

	/// Number of top-level argument fields.
	#[must_use]
	pub fn len(&self) -> usize {
		self.value.object.len()
	}

	/// Whether the named argument is present.
	#[must_use]
	pub fn contains(&self, name: &str) -> bool {
		find_member(self.value, name).is_some()
	}

	/// Serialize the named argument value as a JSON string.
	///
	/// # Errors
	///
	/// Returns `NotFound` when the argument is absent.
	pub fn json_field(&self, name: &str) -> Result<String> {
		let value = find_member(self.value, name)
			.ok_or_else(|| Error::not_found(format!("MCP argument not found: {name}")))?;
		Ok(serialize_json(value))
	}

	/// String argument.
	///
	/// # Errors
	///
	/// Returns `NotFound` when absent, `InvalidArgument` when the type mismatches.
	pub fn string(&self, name: &str) -> Result<String> {
		let value = find_member(self.value, name)
			.ok_or_else(|| Error::not_found(format!("MCP argument not found: {name}")))?;
		if value.kind != JsonKind::String {
			return Err(Error::invalid_argument(format!(
				"MCP argument must be a string: {name}"
			)));
		}
		Ok(value.text.clone())
	}

	/// Signed 64-bit integer argument parsed from a JSON number token.
	///
	/// # Errors
	///
	/// Returns `NotFound` when absent, `InvalidArgument` when the type or
	/// representability check fails.
	pub fn integer(&self, name: &str) -> Result<i64> {
		let value = find_member(self.value, name)
			.ok_or_else(|| Error::not_found(format!("MCP argument not found: {name}")))?;
		if value.kind != JsonKind::Number {
			return Err(Error::invalid_argument(format!(
				"MCP argument must be an integer: {name}"
			)));
		}
		value.text.trim().parse::<i64>().map_err(|_| {
			Error::invalid_argument(format!(
				"MCP argument is not a representable integer: {name}"
			))
		})
	}

	/// Unsigned 64-bit integer argument parsed from a JSON number token.
	///
	/// # Errors
	///
	/// Returns `NotFound` when absent, `InvalidArgument` when the type or
	/// representability check fails.
	pub fn unsigned_integer(&self, name: &str) -> Result<u64> {
		let value = find_member(self.value, name)
			.ok_or_else(|| Error::not_found(format!("MCP argument not found: {name}")))?;
		if value.kind != JsonKind::Number {
			return Err(Error::invalid_argument(format!(
				"MCP argument must be an unsigned integer: {name}"
			)));
		}
		value.text.trim().parse::<u64>().map_err(|_| {
			Error::invalid_argument(format!(
				"MCP argument is not a representable unsigned integer: {name}"
			))
		})
	}

	/// Finite `f64` argument parsed from a JSON number token.
	///
	/// # Errors
	///
	/// Returns `NotFound` when absent, `InvalidArgument` when the type is wrong
	/// or the value is non-finite.
	pub fn number(&self, name: &str) -> Result<f64> {
		let value = find_member(self.value, name)
			.ok_or_else(|| Error::not_found(format!("MCP argument not found: {name}")))?;
		if value.kind != JsonKind::Number {
			return Err(Error::invalid_argument(format!(
				"MCP argument must be a number: {name}"
			)));
		}
		let f: f64 = value.text.trim().parse().map_err(|_| {
			Error::invalid_argument(format!("MCP argument is not a finite number: {name}"))
		})?;
		if !f.is_finite() {
			return Err(Error::invalid_argument(format!(
				"MCP argument is not a finite number: {name}"
			)));
		}
		Ok(f)
	}

	/// Boolean argument.
	///
	/// # Errors
	///
	/// Returns `NotFound` when absent, `InvalidArgument` when the type mismatches.
	pub fn boolean(&self, name: &str) -> Result<bool> {
		let value = find_member(self.value, name)
			.ok_or_else(|| Error::not_found(format!("MCP argument not found: {name}")))?;
		if value.kind != JsonKind::Boolean {
			return Err(Error::invalid_argument(format!(
				"MCP argument must be a boolean: {name}"
			)));
		}
		Ok(value.boolean)
	}
}

/// Result returned by a tool call handler.
#[derive(Clone, Debug, Default)]
pub struct McpToolResult {
	/// Human-readable or machine-readable text content.
	pub text: String,
	/// Optional structured JSON content (must be one valid JSON value when set).
	pub structured_content_json: String,
	/// Whether this represents an error condition.
	pub is_error: bool,
}

impl McpToolResult {
	/// Construct a successful result with the given text.
	#[must_use]
	pub fn success(text: impl Into<String>) -> Self {
		Self {
			text: text.into(),
			structured_content_json: String::new(),
			is_error: false,
		}
	}

	/// Construct an error result with the given text.
	#[must_use]
	pub fn error(text: impl Into<String>) -> Self {
		Self {
			text: text.into(),
			structured_content_json: String::new(),
			is_error: true,
		}
	}
}

/// Registered MCP tool definition.
pub struct McpTool {
	/// Tool name (1–128 alphanumeric, `_`, `-`, or `.`).
	pub name: String,
	/// Optional display title.
	pub title: String,
	/// Human-readable description for the model.
	pub description: String,
	/// JSON Schema for the input object.
	pub input_schema_json: String,
	/// Optional JSON Schema for the structured output.
	pub output_schema_json: String,
	/// Hint: this tool does not modify state.
	pub read_only: bool,
	/// Hint: this tool may have irreversible effects.
	pub destructive: bool,
	/// Hint: repeated calls with the same arguments produce the same result.
	pub idempotent: bool,
	/// Hint: the tool may have side effects not described in its schema.
	pub open_world: bool,
	/// The handler invoked for every `tools/call` request.
	pub call: McpToolCallFn,
}

impl McpTool {
	/// Construct a tool with default annotation hints and a handler closure.
	pub fn new(
		name: impl Into<String>,
		call: impl Fn(&McpArguments<'_>) -> Result<McpToolResult> + Send + Sync + 'static,
	) -> Self {
		Self {
			name: name.into(),
			title: String::new(),
			description: String::new(),
			input_schema_json: r#"{"type":"object"}"#.to_owned(),
			output_schema_json: String::new(),
			read_only: false,
			destructive: true,
			idempotent: false,
			open_world: true,
			call: Box::new(call),
		}
	}
}

/// Registered MCP text resource.
pub struct McpTextResource {
	/// Absolute URI.
	pub uri: String,
	/// Display name.
	pub name: String,
	/// Optional display title.
	pub title: String,
	/// Optional description.
	pub description: String,
	/// MIME type (defaults to `text/plain`).
	pub mime_type: String,
	/// The handler invoked for every `resources/read` request.
	pub read: Box<dyn Fn() -> Result<String> + Send + Sync>,
}

impl McpTextResource {
	/// Construct a text resource with a handler closure.
	pub fn new(
		uri: impl Into<String>,
		name: impl Into<String>,
		read: impl Fn() -> Result<String> + Send + Sync + 'static,
	) -> Self {
		Self {
			uri: uri.into(),
			name: name.into(),
			title: String::new(),
			description: String::new(),
			mime_type: String::from("text/plain"),
			read: Box::new(read),
		}
	}
}

/// Stateful MCP protocol session.
///
/// Registration is frozen by the first handled message. [`Self::handle_message`]
/// is transport-independent; [`Self::run_stdio`] adds the standard
/// newline-delimited subprocess binding. Calls are externally synchronized.
/// [`Self::close`] never waits for or drains external I/O.
///
/// Donor: `oa::McpServer`.
pub struct McpServer {
	config: McpServerConfig,
	configuration_status: Option<Error>,
	tools: Vec<McpTool>,
	resources: Vec<McpTextResource>,
	legacy_protocol_version: String,
	legacy_initialize_seen: bool,
	legacy_ready: bool,
	started: bool,
	closed: bool,
}

impl McpServer {
	/// Construct a new server with the given configuration.
	///
	/// Any invalid configuration is captured and returned by the first
	/// operation that requires a valid server.
	#[must_use]
	pub fn new(mut config: McpServerConfig) -> Self {
		if config.name.is_empty() {
			config.name = "oa".to_owned();
		}
		if config.version.is_empty() {
			config.version = env!("CARGO_PKG_VERSION").to_owned();
		}
		let err = validate_config(&config);
		Self {
			config,
			configuration_status: err,
			tools: Vec::new(),
			resources: Vec::new(),
			legacy_protocol_version: String::new(),
			legacy_initialize_seen: false,
			legacy_ready: false,
			started: false,
			closed: false,
		}
	}

	/// Register a tool. Registration is rejected once the server is started or closed.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` if the server is in an invalid, started, or
	/// closed state. Returns `InvalidArgument` for invalid tool metadata or
	/// `AlreadyExists` for a duplicate name.
	pub fn add_tool(&mut self, mut tool: McpTool) -> Result<()> {
		self.check_config()?;
		if self.started || self.closed {
			return Err(Error::failed_precondition(
				"MCP registration is frozen after start or close",
			));
		}
		if !is_valid_tool_name(&tool.name) {
			return Err(Error::invalid_argument(
				"MCP tool name must be 1-128 letters, digits, '_', '-' or '.'",
			));
		}
		if tool.read_only && tool.destructive {
			return Err(Error::invalid_argument(
				"read-only MCP tools cannot carry a destructive hint",
			));
		}
		tool.input_schema_json = canonical_schema(
			&tool.input_schema_json,
			true,
			self.config.max_nesting_depth,
			self.config.max_json_nodes,
		)?;
		if !tool.output_schema_json.is_empty() {
			tool.output_schema_json = canonical_schema(
				&tool.output_schema_json,
				false,
				self.config.max_nesting_depth,
				self.config.max_json_nodes,
			)?;
		}
		if self.tools.iter().any(|t| t.name == tool.name) {
			return Err(Error::already_exists(format!(
				"MCP tool already registered: {}",
				tool.name
			)));
		}
		let pos = self.tools.partition_point(|t| t.name < tool.name);
		self.tools.insert(pos, tool);
		Ok(())
	}

	/// Register a text resource. Registration is rejected once the server is started or closed.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` if the server is in an invalid, started, or
	/// closed state. Returns `InvalidArgument` for invalid resource metadata or
	/// `AlreadyExists` for a duplicate URI.
	pub fn add_text_resource(&mut self, resource: McpTextResource) -> Result<()> {
		self.check_config()?;
		if self.started || self.closed {
			return Err(Error::failed_precondition(
				"MCP registration is frozen after start or close",
			));
		}
		if !is_valid_uri(&resource.uri) {
			return Err(Error::invalid_argument(
				"MCP text resource requires a valid absolute URI",
			));
		}
		if resource.name.is_empty() {
			return Err(Error::invalid_argument(
				"MCP text resource requires a name and read handler",
			));
		}
		if self.resources.iter().any(|r| r.uri == resource.uri) {
			return Err(Error::already_exists(format!(
				"MCP resource already registered: {}",
				resource.uri
			)));
		}
		let pos = self.resources.partition_point(|r| r.uri < resource.uri);
		self.resources.insert(pos, resource);
		Ok(())
	}

	/// Whether a tool with the given name has been registered.
	#[must_use]
	pub fn has_tool(&self, name: &str) -> bool {
		self.tools.iter().any(|t| t.name == name)
	}

	/// Whether a text resource with the given URI has been registered.
	#[must_use]
	pub fn has_text_resource(&self, uri: &str) -> bool {
		self.resources.iter().any(|r| r.uri == uri)
	}

	/// Process one complete JSON-RPC message and return the response.
	///
	/// Notifications return an empty string. Protocol errors are encoded as
	/// successful JSON-RPC error objects; [`Error`] is reserved for local
	/// lifecycle or I/O failures.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` when the server has an invalid configuration
	/// or has been closed.
	pub fn handle_message(&mut self, message: &str) -> Result<String> {
		let response = self.handle_message_inner(message)?;
		Ok(self.bound_response(response, "null"))
	}

	fn handle_message_inner(&mut self, message: &str) -> Result<String> {
		self.check_config()?;
		if self.closed {
			return Err(Error::failed_precondition("MCP server is closed"));
		}
		self.started = true;

		if message.len() > self.config.max_message_bytes {
			return Ok(make_error(
				"null",
				-32600,
				"MCP message exceeds configured byte limit",
				"",
			));
		}

		let root = match parse_json(
			message,
			self.config.max_nesting_depth,
			self.config.max_json_nodes,
		) {
			Ok(r) => r,
			Err(_) => return Ok(make_error("null", -32700, "parse error", "")),
		};

		let request = match parse_envelope(&root) {
			Ok(r) => r,
			Err(_) => return Ok(make_error("null", -32600, "Invalid Request", "")),
		};

		if request.method == "initialize" {
			return Ok(self.handle_initialize(&root, &request));
		}

		if request.method == "notifications/initialized" {
			if !request.is_notification {
				return Ok(request_error(
					&request,
					-32600,
					"notifications/initialized must be a notification",
					"",
				));
			}
			if self.legacy_initialize_seen {
				self.legacy_ready = true;
			}
			return Ok(String::new());
		}

		if request.method == "notifications/cancelled" {
			return Ok(String::new());
		}

		let modern = if self.legacy_ready {
			false
		} else {
			let mut requested_version = String::new();
			let params = find_member(&root, "params");
			if let Err(e) = validate_modern_metadata(params, &mut requested_version) {
				return Ok(request_error(&request, -32602, e.message(), ""));
			}
			if requested_version != MCP_LATEST_PROTOCOL_VERSION {
				let data = supported_version_data(&requested_version);
				return Ok(request_error(
					&request,
					-32022,
					"Unsupported MCP protocol version",
					&data,
				));
			}
			true
		};

		if request.method == "ping" {
			if request.is_notification {
				return Ok(String::new());
			}
			let mut result = String::from("{");
			if modern {
				append_modern_result_fields(&mut result, &self.config);
			}
			result.push('}');
			return Ok(self.bound_response(make_success(&request.id_json, &result), &request.id_json));
		}

		if modern && request.method == "server/discover" {
			if request.is_notification {
				return Ok(String::new());
			}
			let mut result = String::from("{");
			append_modern_result_fields(&mut result, &self.config);
			append_cache_fields(&mut result, &self.config);
			result.push_str(
				r#","supportedVersions":["2026-07-28","2025-11-25","2025-06-18","2025-03-26","2024-11-05"],"capabilities":"#,
			);
			result.push_str(&capabilities_json(
				!self.tools.is_empty(),
				!self.resources.is_empty(),
			));
			if !self.config.instructions.is_empty() {
				result.push_str(r#","instructions":"#);
				push_json_string(&mut result, &self.config.instructions);
			}
			result.push('}');
			return Ok(self.bound_response(make_success(&request.id_json, &result), &request.id_json));
		}

		if request.method == "tools/list" {
			if request.is_notification {
				return Ok(String::new());
			}
			return Ok(self.handle_tools_list(&request, modern));
		}

		if request.method == "tools/call" {
			if request.is_notification {
				return Ok(String::new());
			}
			return Ok(self.handle_tools_call(&root, &request, modern));
		}

		if request.method == "resources/list" {
			if request.is_notification {
				return Ok(String::new());
			}
			return Ok(self.handle_resources_list(&request, modern));
		}

		if request.method == "resources/read" {
			if request.is_notification {
				return Ok(String::new());
			}
			return Ok(self.handle_resources_read(&root, &request, modern));
		}

		Ok(request_error(&request, -32601, "method not found", ""))
	}

	/// Run the server on stdin/stdout using newline-delimited JSON-RPC.
	///
	/// Reads lines until EOF, dispatches each through [`Self::handle_message`],
	/// and writes non-empty responses followed by a newline and flush.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` for an invalid or closed server. Returns
	/// `Io` if stdio read or write fails.
	pub fn run_stdio(&mut self) -> Result<()> {
		self.run_io(std::io::stdin().lock(), std::io::stdout().lock())
	}

	fn run_io(&mut self, mut input: impl BufRead, mut output: impl Write) -> Result<()> {
		self.check_config()?;
		if self.closed {
			return Err(Error::failed_precondition("MCP server is closed"));
		}
		let mut line = Vec::with_capacity(self.config.max_message_bytes.min(64 * 1024));
		loop {
			line.clear();
			let mut oversized = false;
			let eof = loop {
				let bytes = input
					.fill_buf()
					.map_err(|error| Error::io("MCP stdio read", error))?;
				if bytes.is_empty() {
					break true;
				}
				let newline = bytes.iter().position(|&byte| byte == b'\n');
				let count = newline.unwrap_or(bytes.len());
				let remaining = self.config.max_message_bytes - line.len();
				let retained = count.min(remaining);
				line.extend_from_slice(&bytes[..retained]);
				oversized |= count > remaining;
				input.consume(count + usize::from(newline.is_some()));
				if newline.is_some() {
					break false;
				}
			};
			if eof && line.is_empty() && !oversized {
				return Ok(());
			}
			self.started = true;
			if line.last() == Some(&b'\r') {
				line.pop();
			}
			let response = if oversized {
				make_error(
					"null",
					-32600,
					"MCP message exceeds configured byte limit",
					"",
				)
			} else {
				match std::str::from_utf8(&line) {
					Ok(message) => self.handle_message(message)?,
					Err(_) => make_error("null", -32700, "parse error", ""),
				}
			};
			if !response.is_empty() {
				output
					.write_all(response.as_bytes())
					.and_then(|_| output.write_all(b"\n"))
					.and_then(|_| output.flush())
					.map_err(|error| Error::io("MCP stdio write", error))?;
			}
			if eof {
				return Ok(());
			}
		}
	}

	/// Mark the server as closed. No further messages are accepted.
	pub fn close(&mut self) {
		self.closed = true;
	}

	/// Whether a message has been received, freezing registration.
	#[must_use]
	pub fn is_started(&self) -> bool {
		self.started
	}

	/// Whether the server has been closed.
	#[must_use]
	pub fn is_closed(&self) -> bool {
		self.closed
	}

	/// Any error captured at construction due to invalid configuration.
	#[must_use]
	pub fn configuration_error(&self) -> Option<&Error> {
		self.configuration_status.as_ref()
	}

	// ── private helpers ──────────────────────────────────────────────────

	fn check_config(&self) -> Result<()> {
		match &self.configuration_status {
			Some(e) => Err(Error::invalid_argument(e.message().to_owned())),
			None => Ok(()),
		}
	}

	fn bound_response(&self, response: String, id_json: &str) -> String {
		if response.len() <= self.config.max_message_bytes {
			return response;
		}
		let error = make_error(
			id_json,
			-32603,
			"MCP response exceeds configured byte limit",
			"",
		);
		if error.len() <= self.config.max_message_bytes {
			error
		} else {
			make_error(
				"null",
				-32603,
				"MCP response exceeds configured byte limit",
				"",
			)
		}
	}

	fn handle_initialize(&mut self, root: &JsonValue, request: &RequestEnvelope) -> String {
		if request.is_notification {
			return String::new();
		}
		if self.legacy_initialize_seen {
			return request_error(request, -32600, "MCP session is already initialized", "");
		}
		let params = match find_member(root, "params") {
			None => {
				return request_error(request, -32602, "request params are required", "");
			}
			Some(p) => p,
		};
		let version = match find_member(params, "protocolVersion") {
			Some(v) if v.kind == JsonKind::String => &v.text,
			_ => {
				return request_error(
					request,
					-32602,
					"missing or invalid request parameter: protocolVersion",
					"",
				);
			}
		};
		let version_text = version.clone();
		if find_member(params, "capabilities")
			.filter(|v| v.kind == JsonKind::Object)
			.is_none()
		{
			return request_error(
				request,
				-32602,
				"missing or invalid request parameter: capabilities",
				"",
			);
		}
		let client_info = match find_member(params, "clientInfo") {
			Some(v) if v.kind == JsonKind::Object => v,
			_ => {
				return request_error(
					request,
					-32602,
					"missing or invalid request parameter: clientInfo",
					"",
				);
			}
		};
		if let Err(e) = validate_implementation(client_info, "legacy MCP clientInfo") {
			return request_error(request, -32602, e.message(), "");
		}
		self.legacy_protocol_version = if is_legacy_version(&version_text) {
			version_text
		} else {
			"2025-11-25".to_owned()
		};
		self.legacy_initialize_seen = true;
		self.legacy_ready = false;

		let mut result = String::from(r#"{"protocolVersion":"#);
		push_json_string(&mut result, &self.legacy_protocol_version);
		result.push_str(r#","capabilities":"#);
		result.push_str(&capabilities_json(
			!self.tools.is_empty(),
			!self.resources.is_empty(),
		));
		result.push_str(r#","serverInfo":"#);
		result.push_str(&server_info_json(&self.config));
		if !self.config.instructions.is_empty() {
			result.push_str(r#","instructions":"#);
			push_json_string(&mut result, &self.config.instructions);
		}
		result.push('}');
		let response = make_success(&request.id_json, &result);
		self.bound_response(response, &request.id_json)
	}

	fn handle_tools_list(&self, request: &RequestEnvelope, modern: bool) -> String {
		let mut result = String::from("{");
		if modern {
			append_modern_result_fields(&mut result, &self.config);
			append_cache_fields(&mut result, &self.config);
		}
		if result.len() > 1 {
			result.push(',');
		}
		result.push_str(r#""tools":["#);
		for (i, tool) in self.tools.iter().enumerate() {
			if i != 0 {
				result.push(',');
			}
			result.push_str(r#"{"name":"#);
			push_json_string(&mut result, &tool.name);
			if !tool.title.is_empty() {
				result.push_str(r#","title":"#);
				push_json_string(&mut result, &tool.title);
			}
			if !tool.description.is_empty() {
				result.push_str(r#","description":"#);
				push_json_string(&mut result, &tool.description);
			}
			result.push_str(r#","inputSchema":"#);
			result.push_str(&tool.input_schema_json);
			if !tool.output_schema_json.is_empty() {
				result.push_str(r#","outputSchema":"#);
				result.push_str(&tool.output_schema_json);
			}
			result.push_str(r#","annotations":{"readOnlyHint":"#);
			result.push_str(if tool.read_only { "true" } else { "false" });
			result.push_str(r#","destructiveHint":"#);
			result.push_str(if tool.destructive { "true" } else { "false" });
			result.push_str(r#","idempotentHint":"#);
			result.push_str(if tool.idempotent { "true" } else { "false" });
			result.push_str(r#","openWorldHint":"#);
			result.push_str(if tool.open_world { "true" } else { "false" });
			result.push_str("}}");
		}
		result.push_str("]}");
		let response = make_success(&request.id_json, &result);
		self.bound_response(response, &request.id_json)
	}

	fn handle_tools_call(&self, root: &JsonValue, request: &RequestEnvelope, modern: bool) -> String {
		let params = match find_member(root, "params") {
			None => {
				return request_error(request, -32602, "request params are required", "");
			}
			Some(p) => p,
		};
		let name = match find_member(params, "name") {
			Some(v) if v.kind == JsonKind::String => v.text.clone(),
			_ => {
				return request_error(
					request,
					-32602,
					"missing or invalid request parameter: name",
					"",
				);
			}
		};
		let selected = match self.tools.iter().find(|t| t.name == name) {
			Some(t) => t,
			None => return request_error(request, -32602, "Unknown MCP tool", ""),
		};
		let empty_arguments = JsonValue {
			kind: JsonKind::Object,
			..Default::default()
		};
		let arguments = match find_member(params, "arguments") {
			None => &empty_arguments,
			Some(value) if value.kind == JsonKind::Object => value,
			Some(_) => return request_error(request, -32602, "MCP tool arguments must be an object", ""),
		};
		let call_arguments = McpArguments::new(arguments);
		let tool_result = match (selected.call)(&call_arguments) {
			Ok(r) => r,
			Err(e) => McpToolResult::error(e.to_string()),
		};
		let structured = if tool_result.structured_content_json.is_empty() {
			None
		} else {
			match parse_json(
				&tool_result.structured_content_json,
				self.config.max_nesting_depth,
				self.config.max_json_nodes,
			) {
				Ok(v) => Some(serialize_json(&v)),
				Err(_) => {
					return request_error(
						request,
						-32603,
						"MCP tool returned invalid structured JSON",
						"",
					);
				}
			}
		};
		let mut result = String::from("{");
		if modern {
			append_modern_result_fields(&mut result, &self.config);
		}
		if result.len() > 1 {
			result.push(',');
		}
		result.push_str(r#""content":[{"type":"text","text":"#);
		push_json_string(&mut result, &tool_result.text);
		result.push_str("}]");
		if let Some(s) = structured {
			result.push_str(r#","structuredContent":"#);
			result.push_str(&s);
		}
		if tool_result.is_error {
			result.push_str(r#","isError":true"#);
		}
		result.push('}');
		let response = make_success(&request.id_json, &result);
		self.bound_response(response, &request.id_json)
	}

	fn handle_resources_list(&self, request: &RequestEnvelope, modern: bool) -> String {
		let mut result = String::from("{");
		if modern {
			append_modern_result_fields(&mut result, &self.config);
			append_cache_fields(&mut result, &self.config);
		}
		if result.len() > 1 {
			result.push(',');
		}
		result.push_str(r#""resources":["#);
		for (i, resource) in self.resources.iter().enumerate() {
			if i != 0 {
				result.push(',');
			}
			result.push_str(r#"{"uri":"#);
			push_json_string(&mut result, &resource.uri);
			result.push_str(r#","name":"#);
			push_json_string(&mut result, &resource.name);
			if !resource.title.is_empty() {
				result.push_str(r#","title":"#);
				push_json_string(&mut result, &resource.title);
			}
			if !resource.description.is_empty() {
				result.push_str(r#","description":"#);
				push_json_string(&mut result, &resource.description);
			}
			if !resource.mime_type.is_empty() {
				result.push_str(r#","mimeType":"#);
				push_json_string(&mut result, &resource.mime_type);
			}
			result.push('}');
		}
		result.push_str("]}");
		let response = make_success(&request.id_json, &result);
		self.bound_response(response, &request.id_json)
	}

	fn handle_resources_read(
		&self,
		root: &JsonValue,
		request: &RequestEnvelope,
		modern: bool,
	) -> String {
		let params = match find_member(root, "params") {
			None => {
				return request_error(request, -32602, "request params are required", "");
			}
			Some(p) => p,
		};
		let uri = match find_member(params, "uri") {
			Some(v) if v.kind == JsonKind::String => v.text.clone(),
			_ => {
				return request_error(
					request,
					-32602,
					"missing or invalid request parameter: uri",
					"",
				);
			}
		};
		let selected = match self.resources.iter().find(|r| r.uri == uri) {
			Some(r) => r,
			None => return request_error(request, -32602, "Unknown MCP resource URI", ""),
		};
		let content = match (selected.read)() {
			Ok(c) => c,
			Err(e) => {
				return request_error(request, -32603, &e.to_string(), "");
			}
		};
		let mut result = String::from("{");
		if modern {
			append_modern_result_fields(&mut result, &self.config);
			append_cache_fields(&mut result, &self.config);
		}
		if result.len() > 1 {
			result.push(',');
		}
		result.push_str(r#""contents":[{"uri":"#);
		push_json_string(&mut result, &selected.uri);
		if !selected.mime_type.is_empty() {
			result.push_str(r#","mimeType":"#);
			push_json_string(&mut result, &selected.mime_type);
		}
		result.push_str(r#","text":"#);
		push_json_string(&mut result, &content);
		result.push_str("}]}");
		let response = make_success(&request.id_json, &result);
		self.bound_response(response, &request.id_json)
	}
}

impl Default for McpServer {
	fn default() -> Self {
		Self::new(McpServerConfig::default())
	}
}

fn validate_config(config: &McpServerConfig) -> Option<Error> {
	if config.max_message_bytes < 1024 {
		return Some(Error::invalid_argument(
			"MCP maxMessageBytes must be at least 1024",
		));
	}
	if config.max_nesting_depth == 0 || config.max_nesting_depth > 256 {
		return Some(Error::invalid_argument(
			"MCP maxNestingDepth must be in [1, 256]",
		));
	}
	if config.max_json_nodes == 0 {
		return Some(Error::invalid_argument(
			"MCP maxJsonNodes must be greater than zero",
		));
	}
	None
}
