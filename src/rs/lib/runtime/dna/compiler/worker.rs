//! Bounded compiler worker; owns its Slang session and no Device or GPU resource.
//! Donor: device/jit.rs; channel bounds and detached-owner lifecycle unchanged.
use super::{CompiledChain, SlangCompiler};
use crate::runtime::dna::{fusion::census::JitOutcome, ir::elementwise::Program};

/// Compilations waiting on the single background worker; together with the
/// admission bound this caps total compile work per device lifetime.
const MAX_QUEUED_COMPILATIONS: usize = 16;
/// One queued specialization: everything the worker needs to compile the
/// program for the device's descriptor profile.
pub(in crate::runtime) struct CompileRequest {
	pub(in crate::runtime) program: Program,
	pub(in crate::runtime) bounded: bool,
}

/// Folded back under the jit lock by the next admission, record lookup or
/// persistence snapshot. A failed outcome becomes a negative admission.
pub(in crate::runtime) struct CompileResult {
	pub(in crate::runtime) key: Vec<u8>,
	pub(in crate::runtime) compiler_version: String,
	pub(in crate::runtime) result: Result<CompiledChain, JitOutcome>,
}

/// Bounded background compiler: Slang compilation, `spirv-val` and ABI
/// reflection run off the admission path on one worker that owns the compiler
/// session. The worker never borrows the thread-affine Runtime and never
/// touches command recording, device state or GPU resources.
pub(in crate::runtime) struct CompileWorker {
	pub(in crate::runtime) requests: std::sync::mpsc::SyncSender<CompileRequest>,
	pub(in crate::runtime) results: std::sync::mpsc::Receiver<CompileResult>,
	/// The detached worker exits once the request channel closes; joining is
	/// never required because it owns every resource it touches.
	_thread: std::thread::JoinHandle<()>,
}

impl CompileWorker {
	pub(in crate::runtime) fn spawn(compiler: SlangCompiler) -> std::io::Result<Self> {
		let (requests, inbox) = std::sync::mpsc::sync_channel(MAX_QUEUED_COMPILATIONS);
		let (results, outbox) = std::sync::mpsc::channel();
		let thread = std::thread::Builder::new()
			.name("oa-jit-compile".into())
			.spawn(move || compile_worker_main(compiler, inbox, results))?;
		Ok(Self {
			requests,
			results: outbox,
			_thread: thread,
		})
	}
}

fn compile_worker_main(
	compiler: SlangCompiler,
	inbox: std::sync::mpsc::Receiver<CompileRequest>,
	outbox: std::sync::mpsc::Sender<CompileResult>,
) {
	while let Ok(request) = inbox.recv() {
		let result = CompileResult {
			key: request.program.artifact_key(),
			compiler_version: compiler.version.clone(),
			result: compiler
				.compile_program(&request.program, request.bounded)
				.map_err(|error| {
					crate::log_warn!(
						crate::LogComponent::RUNTIME,
						"JIT specialization admission rejected: {error:?}"
					);
					JitOutcome::CompileRejected
				}),
		};
		if outbox.send(result).is_err() {
			return;
		}
	}
}
