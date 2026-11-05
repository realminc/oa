//! Donor `testSatelliteSession.cpp` and `testSatelliteVulkan.cpp` coverage for
//! the private authenticated session layer. The server serves on the test
//! thread because [`SatelliteServerSession`] borrows a `!Send` engine; client
//! sessions move to a spawned thread instead, the inverse of the donor's
//! thread split and the only arrangement the Rust ownership model admits.

use std::thread;
use std::time::Duration;

use super::*;

use crate::TcpListener;
use crate::crypto::kmac256;

fn satellite_mac(secret: &[u8], data: &[u8], custom: &str, out: &mut [u8; 32]) -> Result<()> {
	out.copy_from_slice(&kmac256(secret, data, custom.as_bytes(), 32)?);
	Ok(())
}

fn make_secret(seed: u8) -> SatelliteSecret {
	let bytes: Vec<u8> = (0..32).map(|i| seed.wrapping_add(i)).collect();
	SatelliteSecret::from_bytes(&bytes).expect("32-byte secret")
}

fn make_config(seed: u8, timeout_ms: u32) -> SatelliteSessionConfig {
	let mut config = SatelliteSessionConfig::new(make_secret(seed), satellite_mac);
	config.io_timeout_ms = timeout_ms;
	config.limits.max_payload_bytes = 64 * 1024;
	config.limits.max_resident_bytes = 1024 * 1024;
	config.limits.max_objects = 8;
	config.limits.max_inflight = 1;
	config.device_name = "loopback-oracle".to_owned();
	config
}

fn bind_loopback() -> (TcpListener, u16) {
	let listener = TcpListener::bind("127.0.0.1", 0, 8).expect("loopback listener");
	let port = listener.port();
	(listener, port)
}

fn result_status(message: &SatelliteMessage) -> u32 {
	let field = SatelliteProtocol::find_field(message, field_id::RESULT_STATUS_CODE)
		.expect("result status field");
	SatelliteProtocol::read_u32(field).expect("result status value")
}

fn result_complete(message: &SatelliteMessage) -> u8 {
	let field = SatelliteProtocol::find_field(message, field_id::RESULT_COMPLETE)
		.expect("result complete field");
	SatelliteProtocol::read_u8(field).expect("result complete value")
}

/// Serve one connection on the calling thread while `client` runs on a
/// spawned thread. Returns the server result, the returned session (for
/// GPU-event provenance assertions), and the client thread's value.
fn serve_with_client<'a, T: Send + 'static>(
	listener: &TcpListener,
	mut server: SatelliteServerSession<'a>,
	client: impl FnOnce(u16) -> T + Send + 'static,
) -> (Result<()>, SatelliteServerSession<'a>, T) {
	let port = listener.port();
	let client_thread = thread::spawn(move || client(port));
	let accepted = listener.accept().expect("accepted stream");
	let status = server.serve(accepted);
	(status, server, client_thread.join().expect("client thread"))
}

#[test]
fn authenticated_probe_negotiates_limits_and_runs_cpu_oracle() {
	let (listener, _port) = bind_loopback();
	let (server_status, _server, client_status) = serve_with_client(
		&listener,
		SatelliteServerSession::new(make_config(11, 500)),
		|port| -> Result<()> {
			let mut config = make_config(11, 500);
			config.limits.max_payload_bytes = 128 * 1024;
			config.limits.max_resident_bytes = 2 * 1024 * 1024;
			config.limits.max_objects = 16;
			let mut client = SatelliteClientSession::connect("127.0.0.1", port, config)?;
			assert_ne!(client.probe().session_epoch, 0);
			assert_eq!(client.probe().device_name, "loopback-oracle");
			assert_eq!(client.probe().limits.max_payload_bytes, 64 * 1024);
			assert_eq!(client.probe().limits.max_resident_bytes, 1024 * 1024);
			assert_eq!(client.probe().limits.max_objects, 8);

			let started = client.start_cpu_add(19, 23)?;
			let polled = client.poll(started)?;
			assert_eq!(result_complete(&polled), 0);
			let waited = client.wait(started)?;
			assert_eq!(SatelliteClientSession::read_cpu_add_result(&waited)?, 42);
			let retained = client.get_result(started)?;
			assert_eq!(SatelliteClientSession::read_cpu_add_result(&retained)?, 42);
			client.close()
		},
	);
	client_status.expect("client session");
	server_status.expect("server session");
}

#[test]
fn cancellation_before_execution_leaves_no_result() {
	let (listener, _port) = bind_loopback();
	let (server_status, _server, client_status) = serve_with_client(
		&listener,
		SatelliteServerSession::new(make_config(21, 500)),
		|port| -> Result<()> {
			let mut client = SatelliteClientSession::connect("127.0.0.1", port, make_config(21, 500))?;
			let started = client.start_cpu_add(0xffffffff, 1)?;
			let cancelled = client.cancel(started)?;
			assert_eq!(result_complete(&cancelled), 1);
			assert_eq!(result_status(&cancelled), status_code::CANCELLED);
			assert_eq!(
				SatelliteClientSession::read_cpu_add_result(&cancelled)
					.unwrap_err()
					.kind(),
				ErrorKind::Cancelled
			);
			let waited = client.wait(started)?;
			assert_eq!(result_status(&waited), status_code::CANCELLED);
			client.close()
		},
	);
	client_status.expect("client session");
	server_status.expect("server session");
}

#[test]
fn bounded_inflight_admission_rejects_a_second_pending_request() {
	let (listener, _port) = bind_loopback();
	let (server_status, _server, client_status) = serve_with_client(
		&listener,
		SatelliteServerSession::new(make_config(26, 500)),
		|port| -> Result<()> {
			let mut client = SatelliteClientSession::connect("127.0.0.1", port, make_config(26, 500))?;
			let first = client.start_cpu_add(1, 2)?;
			let second = client.start_cpu_add(3, 4).unwrap_err();
			assert_eq!(second.kind(), ErrorKind::ResourceExhausted);
			client.cancel(first)?;
			client.close()
		},
	);
	client_status.expect("client session");
	server_status.expect("server session");
}

#[test]
fn object_and_resident_quotas_reject_before_mutation() {
	let (listener, _port) = bind_loopback();
	let (server_status, _server, client_status) = serve_with_client(
		&listener,
		SatelliteServerSession::new({
			let mut config = make_config(28, 500);
			config.limits.max_objects = 1;
			config.limits.max_resident_bytes = 4;
			config
		}),
		|port| -> Result<()> {
			let mut config = make_config(28, 500);
			config.limits.max_objects = 1;
			config.limits.max_resident_bytes = 4;
			let mut client = SatelliteClientSession::connect("127.0.0.1", port, config)?;
			client.put_f32(1, &[1], &[2.0])?;
			assert_eq!(
				client.put_f32(2, &[1], &[2.0]).unwrap_err().kind(),
				ErrorKind::ResourceExhausted
			);
			client.drop_object(1)?;
			assert_eq!(
				client.put_f32(2, &[2], &[2.0, 3.0]).unwrap_err().kind(),
				ErrorKind::ResourceExhausted
			);
			assert_eq!(
				client.put_f32(3, &[], &[]).unwrap_err().kind(),
				ErrorKind::InvalidArgument
			);
			client.close()
		},
	);
	client_status.expect("client session");
	server_status.expect("server session");
}

#[test]
fn envelope_rejects_duplicate_and_stale_epoch_before_mutation() {
	let request = SatelliteMessage {
		message_type: SatelliteMessageType::Close,
		request_id: 9,
		session_epoch: 17,
		..SatelliteMessage::default()
	};
	satellite_validate_request_envelope(&request, 17, 8).expect("fresh envelope");
	assert_eq!(
		satellite_validate_request_envelope(&request, 17, 9)
			.unwrap_err()
			.kind(),
		ErrorKind::Aborted
	);
	let stale = SatelliteMessage {
		session_epoch: 16,
		..request
	};
	assert_eq!(
		satellite_validate_request_envelope(&stale, 17, 8)
			.unwrap_err()
			.kind(),
		ErrorKind::Aborted
	);
}

#[test]
fn wrong_secret_is_rejected_before_application_mutation() {
	let (listener, _port) = bind_loopback();
	let (server_status, _server, client_status) = serve_with_client(
		&listener,
		SatelliteServerSession::new(make_config(31, 500)),
		|port| {
			SatelliteClientSession::connect("127.0.0.1", port, make_config(32, 500))
				.err()
				.expect("wrong-secret connect")
				.kind()
		},
	);
	assert_eq!(client_status, ErrorKind::Unauthenticated);
	assert_eq!(
		server_status.unwrap_err().kind(),
		ErrorKind::Unauthenticated
	);
}

#[test]
fn idle_handshake_times_out() {
	let (listener, _port) = bind_loopback();
	let (server_status, _server, _client) = serve_with_client(
		&listener,
		SatelliteServerSession::new(make_config(41, 25)),
		|port| {
			let _stream = TcpStream::connect("127.0.0.1", port).expect("raw connect");
			thread::sleep(Duration::from_millis(400));
		},
	);
	assert_eq!(server_status.unwrap_err().kind(), ErrorKind::Unavailable);
}

#[test]
fn disconnect_abandons_pending_work_and_reconnect_gets_fresh_epoch() {
	let (listener, _port) = bind_loopback();
	let port = listener.port();
	let client_thread = thread::spawn(move || -> Result<()> {
		{
			let mut client = SatelliteClientSession::connect("127.0.0.1", port, make_config(51, 500))?;
			client.start_cpu_add(1, 2)?;
			// No explicit Close: dropping the session closes the socket only. It
			// does not execute, submit, wait, or drain the accepted pending op.
		}
		let mut reconnected = SatelliteClientSession::connect("127.0.0.1", port, make_config(51, 500))?;
		reconnected.close()
	});

	let mut server = SatelliteServerSession::new(make_config(51, 500));
	let first = listener.accept().expect("first accept");
	let first_status = server.serve(first);
	let first_epoch = server.last_session_epoch();
	let second = listener.accept().expect("second accept");
	let second_status = server.serve(second);
	let second_epoch = server.last_session_epoch();
	client_thread
		.join()
		.expect("client thread")
		.expect("client");

	assert_eq!(first_status.unwrap_err().kind(), ErrorKind::Unavailable);
	second_status.expect("second serve");
	assert_ne!(first_epoch, 0);
	assert_ne!(second_epoch, 0);
	assert_ne!(first_epoch, second_epoch);
}

#[test]
fn explicit_abort_terminates_the_epoch() {
	let (listener, _port) = bind_loopback();
	let (server_status, _server, client_status) = serve_with_client(
		&listener,
		SatelliteServerSession::new(make_config(61, 500)),
		|port| -> Result<()> {
			let mut client = SatelliteClientSession::connect("127.0.0.1", port, make_config(61, 500))?;
			client.abort("test abort")
		},
	);
	client_status.expect("client abort");
	assert_eq!(server_status.unwrap_err().kind(), ErrorKind::Aborted);
}

fn vulkan_config(seed: u8, device_name: String) -> SatelliteSessionConfig {
	let mut config = make_config(seed, 1000);
	config.device_name = device_name;
	config
}

#[test]
#[ignore = "requires an admitted Vulkan compute device"]
fn satellite_matrix_add_uses_exact_owned_event() -> Result<()> {
	let engine = Engine::builder()
		.devices(crate::DeviceSelection::Index(0))
		.build()?;
	let (listener, _port) = bind_loopback();
	let device_name = engine.device_info().name().to_owned();
	let (server_status, server, client_status) = serve_with_client(
		&listener,
		SatelliteServerSession::new_with_engine(&engine, vulkan_config(0xa0, device_name.clone())),
		move |port| -> Result<()> {
			let mut client =
				SatelliteClientSession::connect("127.0.0.1", port, vulkan_config(0xa0, device_name))?;
			let a = [1.0_f32, -2.0, 3.5, 4.0, -6.0];
			let b = [0.5_f32, 2.0, -1.5, 8.0, 7.0];
			client.put_f32(1, &[5], &a)?;
			client.put_f32(2, &[5], &b)?;

			// Cancellation is admitted before submission, and the destination
			// object is still absent. Reusing the same destination then proves
			// no hidden mutation.
			let cancelled_request = client.start_matrix_add_f32(1, 2, 3)?;
			let cancelled = client.cancel(cancelled_request)?;
			assert_eq!(result_complete(&cancelled), 1);
			assert_eq!(
				client.drop_object(3).unwrap_err().kind(),
				ErrorKind::NotFound
			);

			let request = client.start_matrix_add_f32(1, 2, 3)?;
			let waited = client.wait(request)?;
			let values = SatelliteClientSession::read_f32_result(&waited)?;
			let expected = [1.5_f32, 0.0, 2.0, 12.0, 1.0];
			assert_eq!(values.len(), expected.len());
			for (value, expected) in values.iter().zip(expected) {
				assert_eq!(*value, expected);
			}
			let retained = client.get_result(request)?;
			assert_eq!(
				SatelliteClientSession::read_f32_result(&retained)?.len(),
				values.len()
			);
			client.drop_object(1)?;
			client.drop_object(2)?;
			client.drop_object(3)?;
			client.close()
		},
	);
	client_status.expect("client session");
	server_status.expect("server session");
	assert!(server.last_gpu_event_was_owned());
	assert!(server.last_gpu_event_completed());
	assert_ne!(server.last_gpu_event_value(), 0);
	Ok(())
}

#[test]
#[ignore = "requires an admitted Vulkan compute device"]
fn abandon_pending_matrix_request_does_not_submit_or_wait() -> Result<()> {
	let engine = Engine::builder()
		.devices(crate::DeviceSelection::Index(0))
		.build()?;
	let (listener, _port) = bind_loopback();
	let device_name = engine.device_info().name().to_owned();
	let server =
		SatelliteServerSession::new_with_engine(&engine, vulkan_config(0xa0, device_name.clone()));
	let (server_status, server, client_status) =
		serve_with_client(&listener, server, move |port| -> Result<()> {
			let mut client =
				SatelliteClientSession::connect("127.0.0.1", port, vulkan_config(0xa0, device_name))?;
			client.put_f32(11, &[3], &[1.0, 2.0, 3.0])?;
			client.put_f32(12, &[3], &[4.0, 5.0, 6.0])?;
			client.start_matrix_add_f32(11, 12, 13)?;
			// Socket-only destruction abandons the admitted-but-unsubmitted
			// request.
			Ok(())
		});
	client_status.expect("client session");
	assert_eq!(server_status.unwrap_err().kind(), ErrorKind::Unavailable);
	assert!(!server.last_gpu_event_was_owned());
	assert!(!server.last_gpu_event_completed());
	assert_eq!(server.last_gpu_event_value(), 0);
	Ok(())
}
