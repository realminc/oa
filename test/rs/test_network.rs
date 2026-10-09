use std::{any::TypeId, thread, time::Duration};

use oa::{ErrorKind, TcpFramed, TcpListener, TcpStream};

#[test]
fn root_exports_are_the_network_types() {
	assert_eq!(
		TypeId::of::<oa::TcpStream>(),
		TypeId::of::<oa::network::TcpStream>()
	);
	assert_eq!(
		TypeId::of::<oa::TcpListener>(),
		TypeId::of::<oa::network::TcpListener>()
	);
}

#[test]
fn framed_loopback_roundtrip_and_empty_message() -> oa::Result<()> {
	let listener = TcpListener::bind("127.0.0.1", 0, 8)?;
	let port = listener.port();
	let server = thread::spawn(move || -> oa::Result<()> {
		let mut stream = listener.accept()?;
		stream.set_io_timeout(Duration::from_secs(2))?;
		let mut buffer = Vec::with_capacity(16);
		let allocation = buffer.as_ptr();
		TcpFramed::read_message_into(&mut stream, &mut buffer, 16)?;
		assert_eq!(buffer, b"hi");
		assert_eq!(buffer.as_ptr(), allocation);
		TcpFramed::read_message_into(&mut stream, &mut buffer, 16)?;
		assert!(buffer.is_empty());
		assert_eq!(buffer.as_ptr(), allocation);
		TcpFramed::write_message(&mut stream, &[1, 2, 3])?;
		Ok(())
	});
	let mut client = TcpStream::connect("127.0.0.1", port)?;
	client.set_io_timeout(Duration::from_secs(2))?;
	assert_eq!(client.remote_port(), port);
	TcpFramed::write_message(&mut client, b"hi")?;
	TcpFramed::write_message(&mut client, &[])?;
	assert_eq!(TcpFramed::read_message(&mut client, 16)?, [1, 2, 3]);
	server.join().expect("server thread")?;
	Ok(())
}

#[test]
fn rejected_length_closes_stream_without_allocating_body() -> oa::Result<()> {
	let listener = TcpListener::bind("127.0.0.1", 0, 8)?;
	let port = listener.port();
	let server = thread::spawn(move || -> oa::Result<()> {
		let mut stream = listener.accept()?;
		stream.set_io_timeout(Duration::from_secs(2))?;
		let error = TcpFramed::read_message(&mut stream, 4).expect_err("length must be rejected");
		assert_eq!(error.kind(), ErrorKind::DataLoss);
		assert!(!stream.is_open());
		Ok(())
	});
	let mut client = TcpStream::connect("127.0.0.1", port)?;
	client.write_all(&5_u32.to_le_bytes())?;
	server.join().expect("server thread")?;
	Ok(())
}

#[test]
fn incomplete_body_closes_stream() -> oa::Result<()> {
	let listener = TcpListener::bind("127.0.0.1", 0, 8)?;
	let port = listener.port();
	let server = thread::spawn(move || -> oa::Result<()> {
		let mut stream = listener.accept()?;
		stream.set_io_timeout(Duration::from_secs(2))?;
		let error = TcpFramed::read_message(&mut stream, 8).expect_err("truncated body");
		assert_eq!(error.kind(), ErrorKind::Io);
		assert!(!stream.is_open());
		Ok(())
	});
	let mut client = TcpStream::connect("127.0.0.1", port)?;
	client.write_all(&4_u32.to_le_bytes())?;
	client.write_all(&[1, 2])?;
	client.close();
	server.join().expect("server thread")?;
	Ok(())
}

#[test]
fn timeout_and_closed_socket_contract() -> oa::Result<()> {
	let listener = TcpListener::bind("127.0.0.1", 0, 8)?;
	let port = listener.port();
	let server = thread::spawn(move || -> oa::Result<()> {
		let mut stream = listener.accept()?;
		stream.set_io_timeout(Duration::from_millis(30))?;
		let error = TcpFramed::read_message(&mut stream, 8).expect_err("idle peer must time out");
		assert_eq!(error.kind(), ErrorKind::Io);
		assert!(!stream.is_open());
		Ok(())
	});
	let mut client = TcpStream::connect("127.0.0.1", port)?;
	server.join().expect("server thread")?;
	client.close();
	assert_eq!(
		client.read(&mut [0]).expect_err("closed socket").kind(),
		ErrorKind::FailedPrecondition
	);
	Ok(())
}

#[test]
fn invalid_limits_and_bind_arguments_fail_before_io() -> oa::Result<()> {
	assert_eq!(
		TcpListener::bind("127.0.0.1", 0, 0)
			.err()
			.expect("backlog")
			.kind(),
		ErrorKind::InvalidArgument
	);
	assert_eq!(
		TcpListener::bind("not-an-ip", 0, 8)
			.err()
			.expect("IP address")
			.kind(),
		ErrorKind::InvalidArgument
	);
	let listener = TcpListener::bind("127.0.0.1", 0, 8)?;
	let mut client = TcpStream::connect("127.0.0.1", listener.port())?;
	let error = TcpFramed::read_message(&mut client, TcpFramed::MAX_PAYLOAD_BYTES + 1)
		.expect_err("invalid limit");
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	assert!(client.is_open());
	let oversized = vec![0; TcpFramed::MAX_PAYLOAD_BYTES + 1];
	let error = TcpFramed::write_message(&mut client, &oversized).expect_err("oversized payload");
	assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	assert!(client.is_open());
	Ok(())
}

#[path = "network/test_mcp.rs"]
mod mcp;
