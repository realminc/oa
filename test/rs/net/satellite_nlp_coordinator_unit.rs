//! Donor `testSatelliteNlp.cpp` coordinator coverage: the split-batch and
//! standalone frozen byte-Transformer coordinators. The donor shares one
//! engine between the serving thread and the coordinator thread; the Rust
//! `Engine` is `!Send`, so the coordinator constructs its own engine inside
//! the client thread while the server keeps the test thread's engine.

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;
use std::thread;

use super::super::SatelliteProtocol;
use super::super::nlp_step::{
	NLP_GRADIENT_OPERATION, NLP_STEP_OPERATION, satellite_build_nlp_batch, satellite_execute_nlp_work,
};
use super::super::session::*;
use super::*;

use crate::TcpListener;
use crate::sdk::ml::nlp::{GENERATION_LENGTH, GENERATION_PROMPT, TRAINING_STEPS};
use crate::{Engine, ErrorKind, Result};

fn satellite_mac(secret: &[u8], data: &[u8], custom: &str, out: &mut [u8; 32]) -> Result<()> {
	out.copy_from_slice(&crate::crypto::kmac256(
		secret,
		data,
		custom.as_bytes(),
		32,
	)?);
	Ok(())
}

fn make_secret(seed: u8) -> SatelliteSecret {
	let bytes: Vec<u8> = (0..32).map(|i| seed.wrapping_add(i)).collect();
	SatelliteSecret::from_bytes(&bytes).expect("32-byte secret")
}

fn make_config(seed: u8, device_name: String) -> SatelliteSessionConfig {
	let mut config = SatelliteSessionConfig::new(make_secret(seed), satellite_mac);
	config.io_timeout_ms = 30_000;
	config.limits.max_payload_bytes = SatelliteProtocol::MAX_PAYLOAD_BYTES as u32;
	config.limits.max_resident_bytes = 16 * 1024 * 1024;
	config.limits.max_objects = 16;
	config.limits.max_inflight = 1;
	config.device_name = device_name;
	config.named_operations = vec![
		NLP_STEP_OPERATION.to_owned(),
		NLP_GRADIENT_OPERATION.to_owned(),
	];
	config
}

fn nlp_config(seed: u8, device_name: String, calls: Rc<Cell<u32>>) -> SatelliteSessionConfig {
	let mut config = make_config(seed, device_name);
	config.named_work = Some(Box::new(move |engine, request| {
		calls.set(calls.get() + 1);
		satellite_execute_nlp_work(engine, request)
	}));
	config
}

fn test_engine() -> Result<Engine> {
	Engine::builder()
		.devices(crate::DeviceSelection::Index(0))
		.build()
}

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

fn checkpoint_path(tag: &str) -> PathBuf {
	std::env::temp_dir().join(format!(
		"oars-satellite-nlp-{tag}-{}.oam",
		std::process::id()
	))
}

#[test]
fn batch_builder_source_contract_holds_for_coordinator() -> Result<()> {
	let first = satellite_build_nlp_batch(0)?;
	assert_eq!(first.sample_indices.len(), 64);
	assert_eq!(first.inputs.len(), 64 * 16);
	assert_eq!(first.targets.len(), first.inputs.len());
	assert_eq!(
		satellite_build_nlp_batch(TRAINING_STEPS as u64)
			.err()
			.unwrap()
			.kind(),
		ErrorKind::OutOfRange
	);
	Ok(())
}

#[test]
#[ignore = "requires an admitted Vulkan compute device"]
fn satellite_nlp_coordinator_persists_optimizer_and_acknowledged_snapshots() -> Result<()> {
	let engine = test_engine()?;
	let listener = TcpListener::bind("127.0.0.1", 0, 8)?;
	let calls = Rc::new(Cell::new(0_u32));
	let device_name = engine.device_info().name().to_owned();
	let server = SatelliteServerSession::new_with_engine(
		&engine,
		nlp_config(0xa0, device_name.clone(), Rc::clone(&calls)),
	);
	let checkpoint = checkpoint_path("coordinator");
	let (server_status, server, client_status) = serve_with_client(
		&listener,
		server,
		move |port| -> Result<SatelliteNlpCoordinatorReport> {
			let coordinator_engine = test_engine()?;
			let mut client =
				SatelliteClientSession::connect("127.0.0.1", port, make_config(0xa0, device_name))?;
			let config = SatelliteNlpCoordinatorConfig {
				steps: 2,
				local_batch_count: 32,
				initial_version: 41,
				checkpoint_path: checkpoint,
				..SatelliteNlpCoordinatorConfig::default()
			};
			let report = satellite_run_split_batch_nlp(&coordinator_engine, &mut client, &config)?;
			client.close()?;
			Ok(report)
		},
	);
	let report = client_status.expect("split-batch coordinator");
	server_status.expect("server session");
	assert_eq!(report.completed_steps, 2);
	assert_eq!(report.optimizer_step, 2);
	assert!(report.initial_loss.is_finite());
	assert!(report.final_loss.is_finite());
	assert!(report.final_accuracy.is_finite());
	assert_ne!(report.initial_parameter_hash, report.final_parameter_hash);
	assert!(report.local_kernel_selections > 0);
	assert_eq!(report.local_kernel_fallbacks, 0);
	assert!(report.remote_kernel_selections > 0);
	assert_eq!(report.remote_kernel_fallbacks, 0);
	assert!(report.checkpoint_round_trip);
	assert_eq!(
		report.generated.len(),
		GENERATION_PROMPT.len() + GENERATION_LENGTH
	);
	assert_eq!(calls.get(), 2);
	assert!(server.last_gpu_event_was_owned());
	assert!(server.last_gpu_event_completed());
	assert_ne!(server.last_gpu_event_value(), 0);
	std::fs::remove_file(checkpoint_path("coordinator"))
		.expect("remove completed split coordinator checkpoint");
	Ok(())
}

#[test]
#[ignore = "requires an admitted Vulkan compute device"]
fn satellite_nlp_coordinator_completes_canonical_300_step_quality_gate() -> Result<()> {
	let engine = test_engine()?;
	let listener = TcpListener::bind("127.0.0.1", 0, 8)?;
	let calls = Rc::new(Cell::new(0_u32));
	let device_name = engine.device_info().name().to_owned();
	let server = SatelliteServerSession::new_with_engine(
		&engine,
		nlp_config(0xa0, device_name.clone(), Rc::clone(&calls)),
	);
	let checkpoint = checkpoint_path("coordinator-full");
	let (server_status, server, client_status) = serve_with_client(
		&listener,
		server,
		move |port| -> Result<SatelliteNlpCoordinatorReport> {
			let coordinator_engine = test_engine()?;
			let mut client =
				SatelliteClientSession::connect("127.0.0.1", port, make_config(0xa0, device_name))?;
			let config = SatelliteNlpCoordinatorConfig {
				checkpoint_path: checkpoint,
				..SatelliteNlpCoordinatorConfig::default()
			};
			let report = satellite_run_split_batch_nlp(&coordinator_engine, &mut client, &config)?;
			client.close()?;
			Ok(report)
		},
	);
	let report = client_status.expect("canonical split-batch coordinator");
	server_status.expect("server session");
	assert_eq!(report.completed_steps, TRAINING_STEPS as u32);
	assert_eq!(report.optimizer_step, TRAINING_STEPS as u64);
	assert!(report.final_loss < report.initial_loss);
	assert!(report.final_accuracy > 0.85);
	assert!(report.generation_quality_passed);
	assert!(report.checkpoint_round_trip);
	assert_eq!(report.local_kernel_fallbacks, 0);
	assert!(report.local_kernel_selections > 0);
	assert_eq!(report.remote_kernel_fallbacks, 0);
	assert!(report.remote_kernel_selections > 0);
	assert_eq!(calls.get(), TRAINING_STEPS as u32);
	assert!(server.last_gpu_event_was_owned());
	assert!(server.last_gpu_event_completed());
	std::fs::remove_file(checkpoint_path("coordinator-full"))
		.expect("remove completed canonical coordinator checkpoint");
	Ok(())
}

#[test]
#[ignore = "requires an admitted Vulkan compute device"]
fn satellite_nlp_standalone_completes_canonical_300_step_quality_gate() -> Result<()> {
	let engine = test_engine()?;
	let checkpoint = checkpoint_path("standalone-full");
	let config = SatelliteNlpCoordinatorConfig {
		checkpoint_path: checkpoint.clone(),
		..SatelliteNlpCoordinatorConfig::default()
	};
	let report = satellite_run_standalone_nlp(&engine, &config)?;
	assert_eq!(report.completed_steps, TRAINING_STEPS as u32);
	assert_eq!(report.optimizer_step, TRAINING_STEPS as u64);
	assert!(report.final_loss < report.initial_loss);
	assert!(report.final_accuracy > 0.85);
	assert!(report.generation_quality_passed);
	assert!(report.checkpoint_round_trip);
	assert!(report.local_kernel_selections > 0);
	assert_eq!(report.local_kernel_fallbacks, 0);
	assert_eq!(report.remote_kernel_selections, 0);
	assert_eq!(report.remote_kernel_fallbacks, 0);
	std::fs::remove_file(checkpoint).expect("remove completed standalone coordinator checkpoint");
	Ok(())
}
