//! Donor: test/cpp/network/mcp/testMcp.cpp. Rust ownership and boundary regressions.
use oa::{McpServer, McpServerConfig, McpTextResource, McpTool, McpToolResult};
use serde_json::{Value, json};

fn request(method: &str, mut params: Value) -> String {
	params["_meta"] = json!({
		"io.modelcontextprotocol/protocolVersion": oa::MCP_LATEST_PROTOCOL_VERSION,
		"io.modelcontextprotocol/clientCapabilities": {}
	});
	json!({"jsonrpc":"2.0", "id":1, "method":method, "params":params}).to_string()
}

fn handle(server: &mut McpServer, message: &str) -> Value {
	serde_json::from_str(&server.handle_message(message).expect("local server"))
		.expect("valid JSON response")
}

#[test]
fn typed_arguments_preserve_unicode_nested_values_and_integer_extremes() -> oa::Result<()> {
	let mut server = McpServer::default();
	server.add_tool(McpTool::new("echo", |args| {
		assert_eq!(args.len(), 6);
		assert!(!args.is_empty());
		assert!(args.contains("text"));
		assert_eq!(args.string("text")?, "Zażółć 🚀");
		assert_eq!(args.integer("signed")?, i64::MIN);
		assert_eq!(args.unsigned_integer("unsigned")?, u64::MAX);
		assert_eq!(args.number("number")?, 1.25);
		assert!(args.boolean("enabled")?);
		assert_eq!(
			serde_json::from_str::<Value>(&args.json_field("nested")?).unwrap(),
			json!([null, {"x":1}])
		);
		assert_eq!(
			args.string("absent").err().unwrap().kind(),
			oa::ErrorKind::NotFound
		);
		assert_eq!(
			args.integer("unsigned").err().unwrap().kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(McpToolResult::success(args.string("text")?))
	}))?;
	let response = handle(
		&mut server,
		&request(
			"tools/call",
			json!({"name":"echo", "arguments":{
				"text":"Zażółć 🚀", "signed":i64::MIN, "unsigned":u64::MAX,
				"number":1.25, "enabled":true, "nested":[null, {"x":1}]
			}}),
		),
	);
	assert_eq!(response["result"]["content"][0]["text"], "Zażółć 🚀");
	Ok(())
}

#[test]
fn registration_freezes_and_legacy_session_roundtrips() -> oa::Result<()> {
	let mut server = McpServer::default();
	for name in ["zeta", "alpha"] {
		server.add_tool(McpTool::new(name, |_| Ok(McpToolResult::success("ok"))))?;
	}
	assert!(
		server
			.add_tool(McpTool::new("alpha", |_| Ok(Default::default())))
			.is_err()
	);
	server.add_text_resource(McpTextResource::new("oa://status", "status", || {
		Ok("ready".into())
	}))?;
	let response = handle(
		&mut server,
		r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}"#,
	);
	assert_eq!(response["result"]["protocolVersion"], "2025-11-25");
	assert!(server.is_started());
	assert!(
		server
			.add_tool(McpTool::new("late", |_| Ok(Default::default())))
			.is_err()
	);
	assert!(
		server
			.handle_message(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)?
			.is_empty()
	);
	let response = handle(
		&mut server,
		r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
	);
	assert_eq!(response["result"]["tools"][0]["name"], "alpha");
	assert_eq!(response["result"]["tools"][1]["name"], "zeta");
	let response = handle(
		&mut server,
		r#"{"jsonrpc":"2.0","id":3,"method":"resources/read","params":{"uri":"oa://status"}}"#,
	);
	assert_eq!(response["result"]["contents"][0]["text"], "ready");
	server.close();
	assert_eq!(
		server.handle_message("{}").err().unwrap().kind(),
		oa::ErrorKind::FailedPrecondition
	);
	Ok(())
}

#[test]
fn invalid_json_and_parser_budgets_fail_closed() {
	for invalid in [
		"{",
		"[]",
		r#"{"jsonrpc":"2.0","jsonrpc":"2.0"}"#,
		r#"{"x":"\uD800"}"#,
	] {
		assert!(
			handle(&mut McpServer::default(), invalid)
				.get("error")
				.is_some()
		);
	}
	for config in [
		McpServerConfig {
			max_nesting_depth: 1,
			..Default::default()
		},
		McpServerConfig {
			max_json_nodes: 1,
			..Default::default()
		},
	] {
		assert_eq!(
			handle(&mut McpServer::new(config), &request("ping", json!({})))["error"]["code"],
			-32700
		);
	}
	let mut server = McpServer::new(McpServerConfig {
		max_message_bytes: 100,
		..Default::default()
	});
	assert_eq!(
		server.handle_message("{}").err().unwrap().kind(),
		oa::ErrorKind::InvalidArgument
	);
}

#[test]
fn response_limit_applies_to_error_ids_and_handler_output() -> oa::Result<()> {
	let mut server = McpServer::new(McpServerConfig {
		max_message_bytes: 1024,
		..Default::default()
	});
	server.add_tool(McpTool::new("large", |_| {
		Ok(McpToolResult::success("x".repeat(4096)))
	}))?;
	let response = handle(&mut server, &request("tools/call", json!({"name":"large"})));
	assert_eq!(response["error"]["code"], -32603);
	// The request fits but echoing its large id into an error would exceed the ceiling.
	let message = json!({"jsonrpc":"2.0", "id":"x".repeat(950), "method":"unknown"}).to_string();
	assert!(message.len() <= 1024);
	let response = server.handle_message(&message)?;
	assert!(response.len() <= 1024);
	let response: Value = serde_json::from_str(&response).unwrap();
	assert!(response["id"].is_null());
	assert_eq!(response["error"]["code"], -32603);
	Ok(())
}

#[test]
fn modern_discovery_and_notifications_preserve_handler_boundary() -> oa::Result<()> {
	use std::sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	};
	let calls = Arc::new(AtomicUsize::new(0));
	let captured = calls.clone();
	let mut server = McpServer::default();
	server.add_tool(McpTool::new("count", move |_| {
		captured.fetch_add(1, Ordering::Relaxed);
		Ok(McpToolResult::success("ok"))
	}))?;
	let response = handle(&mut server, &request("server/discover", json!({})));
	assert_eq!(response["result"]["resultType"], "complete");
	let mut message: Value =
		serde_json::from_str(&request("tools/call", json!({"name":"count"}))).unwrap();
	message.as_object_mut().unwrap().remove("id");
	assert!(server.handle_message(&message.to_string())?.is_empty());
	assert_eq!(calls.load(Ordering::Relaxed), 0);
	Ok(())
}

// Exercise the actual stdin/stdout adapter in a child process. The test harness
// owns stdout outside the call, so retain only JSON response lines below.
#[test]
fn stdio_child() -> oa::Result<()> {
	if std::env::var_os("OA_MCP_STDIO_CHILD").is_none() {
		return Ok(());
	}
	let mut server = McpServer::new(McpServerConfig {
		max_message_bytes: 1024,
		..Default::default()
	});
	server.add_tool(McpTool::new("echo", |args| {
		Ok(McpToolResult::success(args.string("text")?))
	}))?;
	server.run_stdio()
}

fn run_stdio(input: &[u8], chunk_size: usize) -> Vec<Value> {
	use std::{
		io::Write,
		process::{Command, Stdio},
	};
	let mut child = Command::new(std::env::current_exe().unwrap())
		.args(["--exact", "mcp::stdio_child", "--nocapture", "--quiet"])
		.env("OA_MCP_STDIO_CHILD", "1")
		.stdin(Stdio::piped())
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.spawn()
		.unwrap();
	let mut stdin = child.stdin.take().unwrap();
	for chunk in input.chunks(chunk_size) {
		stdin.write_all(chunk).unwrap();
		stdin.flush().unwrap();
	}
	drop(stdin);
	let output = child.wait_with_output().unwrap();
	assert!(
		output.status.success(),
		"{}",
		String::from_utf8_lossy(&output.stderr)
	);
	String::from_utf8(output.stdout)
		.unwrap()
		.lines()
		.filter(|line| line.starts_with('{'))
		.map(|line| serde_json::from_str(line).expect("valid JSON response"))
		.collect()
}

#[test]
fn stdio_preserves_utf8_and_clean_eof() {
	assert!(run_stdio(b"", 1).is_empty());
	let text = "Zażółć 🚀";
	for suffix in ["", "\n", "\r\n"] {
		let input = request(
			"tools/call",
			json!({"name":"echo", "arguments":{"text":text}}),
		) + suffix;
		for chunk_size in [1, 128] {
			let responses = run_stdio(input.as_bytes(), chunk_size);
			assert_eq!(responses.len(), 1);
			assert_eq!(responses[0]["result"]["content"][0]["text"], text);
		}
	}
}

#[test]
fn stdio_drains_oversize_and_rejects_invalid_utf8_then_recovers() {
	for invalid in [vec![b'x'; 4096], vec![0xff, 0xfe]] {
		let mut input = invalid;
		input.push(b'\n');
		input.extend_from_slice(
			request(
				"tools/call",
				json!({"name":"echo", "arguments":{"text":"ok"}}),
			)
			.as_bytes(),
		);
		let responses = run_stdio(&input, 7);
		assert_eq!(responses.len(), 2);
		assert!(responses[0].get("error").is_some());
		assert_eq!(responses[1]["result"]["content"][0]["text"], "ok");
	}
}
