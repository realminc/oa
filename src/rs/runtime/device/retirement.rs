use std::{
	mem,
	sync::{
		Arc, Condvar, Mutex,
		atomic::{AtomicBool, AtomicUsize, Ordering},
		mpsc::{self, Sender},
	},
	thread::{self, JoinHandle},
};

use crate::{Error, Result};

use crate::runtime::{Device, RecordedCommandBuffer};

pub(in crate::runtime) struct RetirementService {
	sender: Option<Sender<Retirement>>,
	control: Arc<WorkerControl>,
	device: Arc<Device>,
}

#[derive(Clone)]
pub(in crate::runtime) struct RetirementTicket {
	completion: Arc<RetirementCompletion>,
	control: Arc<WorkerControl>,
}

struct Retirement {
	command: RecordedCommandBuffer,
	epoch: u64,
	completion: Arc<RetirementCompletion>,
}

struct RetirementCompletion {
	state: Mutex<RetirementStateData>,
	changed: Condvar,
}

struct RetirementStateData {
	state: RetirementState,
	retained: Vec<Box<dyn Send>>,
}

struct WorkerControl {
	pending: AtomicUsize,
	failed: AtomicBool,
	failure: Mutex<Option<Arc<Error>>>,
	disconnected: AtomicBool,
	worker: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RetirementState {
	Pending,
	Retired,
	Failed,
}

impl RetirementService {
	pub(in crate::runtime) fn new(device: &Arc<Device>) -> Self {
		Self {
			sender: None,
			control: Arc::new(WorkerControl::new()),
			device: device.clone(),
		}
	}

	pub(in crate::runtime) fn prepare(&mut self) -> Result<()> {
		if self.control.failed.load(Ordering::Acquire) {
			return Err(self.control.failure_error());
		}
		if self.sender.is_some() {
			return Ok(());
		}

		let (sender, receiver) = mpsc::channel::<Retirement>();
		let worker_device = self.device.clone();
		let worker_control = self.control.clone();
		let worker = thread::Builder::new()
			.name("oa-vk-retirement".to_owned())
			.spawn(move || {
				let mut leaked_device = false;
				drain_retirement_queue(
					receiver,
					&worker_control,
					|retirement| worker_device.wait(retirement.epoch),
					|retirement| {
						retirement.command.confirm_secret_erasures();
						worker_device.free(retirement.command);
						retirement.completion.finish(RetirementState::Retired);
					},
					|retirement| {
						if !leaked_device {
							mem::forget(worker_device.clone());
							leaked_device = true;
						}
						retirement.command.fail_secret_erasures();
						retirement.completion.finish(RetirementState::Failed);
						mem::forget(retirement);
					},
				);
			})
			.map_err(|source| Error::backend_failure("Vulkan", "retirement-worker creation", source))?;

		self.sender = Some(sender);
		self.control.install(worker);
		Ok(())
	}

	pub(in crate::runtime) fn retire(
		&self,
		command: RecordedCommandBuffer,
		epoch: u64,
	) -> Result<RetirementTicket> {
		let completion = Arc::new(RetirementCompletion::new());
		let retirement = Retirement {
			command,
			epoch,
			completion: completion.clone(),
		};
		let Some(sender) = &self.sender else {
			self.control.fail(retirement_unavailable());
			retirement.command.fail_secret_erasures();
			self.leak_device_graph();
			mem::forget(retirement);
			return Err(retirement_unavailable());
		};
		self.control.pending.fetch_add(1, Ordering::AcqRel);
		if let Err(mpsc::SendError(retirement)) = sender.send(retirement) {
			self.control.fail(retirement_unavailable());
			retirement.command.fail_secret_erasures();
			self.control.pending.fetch_sub(1, Ordering::AcqRel);
			self.leak_device_graph();
			mem::forget(retirement);
			return Err(retirement_unavailable());
		}
		Ok(RetirementTicket {
			completion,
			control: self.control.clone(),
		})
	}

	fn leak_device_graph(&self) {
		// The submitted command buffer may still be pending. Retaining one permanent
		// device reference is the safe terminal fallback after retirement failure.
		mem::forget(self.device.clone());
	}
}

impl Drop for RetirementService {
	fn drop(&mut self) {
		// Disconnect first so the worker drains every queued retirement and exits.
		drop(self.sender.take());
		self.control.disconnected.store(true, Ordering::Release);
		// Join only after every queued item was retired or quarantined. This releases
		// an idle host thread; it never turns engine drop into a GPU wait.
		self.control.join_if_drained();
	}
}

impl RetirementTicket {
	pub(in crate::runtime) fn check_failure(&self) -> Result<()> {
		let state = match self.completion.state.lock() {
			Ok(state) => state,
			Err(poisoned) => poisoned.into_inner(),
		};
		if state.state == RetirementState::Failed
			|| (state.state == RetirementState::Pending && self.control.failed.load(Ordering::Acquire))
		{
			return Err(self.control.failure_error());
		}
		Ok(())
	}

	pub(in crate::runtime) fn wait(&self) -> Result<()> {
		self.check_failure()?;
		self
			.completion
			.wait()
			.map_err(|_| self.control.failure_error())?;
		self.control.join_if_drained();
		Ok(())
	}

	pub(in crate::runtime) fn retain_until_complete<T>(&self, value: T)
	where
		T: Send + 'static,
	{
		let mut state = match self.completion.state.lock() {
			Ok(state) => state,
			Err(poisoned) => poisoned.into_inner(),
		};
		if state.state == RetirementState::Pending {
			state.retained.push(Box::new(value));
			return;
		}
		let failed = state.state == RetirementState::Failed;
		drop(state);
		if failed {
			// A late attachment has the same unknown GPU lifetime as an early one.
			mem::forget(value);
		}
	}
}

impl WorkerControl {
	fn fail(&self, error: Error) {
		let mut failure = match self.failure.lock() {
			Ok(failure) => failure,
			Err(poisoned) => poisoned.into_inner(),
		};
		if failure.is_none() {
			*failure = Some(Arc::new(error));
		}
		self.failed.store(true, Ordering::Release);
	}

	fn failure_error(&self) -> Error {
		let failure = match self.failure.lock() {
			Ok(failure) => failure,
			Err(poisoned) => poisoned.into_inner(),
		};
		match failure.as_ref() {
			Some(source) => Error::backend_failure("Vulkan", "submission retirement", source.clone()),
			None => retirement_unavailable(),
		}
	}

	fn new() -> Self {
		Self {
			pending: AtomicUsize::new(0),
			failed: AtomicBool::new(false),
			failure: Mutex::new(None),
			disconnected: AtomicBool::new(false),
			worker: Mutex::new(None),
		}
	}

	fn install(&self, worker: JoinHandle<()>) {
		let mut slot = match self.worker.lock() {
			Ok(slot) => slot,
			Err(poisoned) => poisoned.into_inner(),
		};
		*slot = Some(worker);
	}

	fn join_if_drained(&self) {
		if !self.disconnected.load(Ordering::Acquire) || self.pending.load(Ordering::Acquire) != 0 {
			return;
		}
		let worker = {
			let mut slot = match self.worker.lock() {
				Ok(slot) => slot,
				Err(poisoned) => poisoned.into_inner(),
			};
			// Completion-attached destructors can inspect their Event on this
			// worker. Never turn that host observation into a self-join.
			if slot
				.as_ref()
				.is_some_and(|worker| worker.thread().id() == thread::current().id())
			{
				return;
			}
			slot.take()
		};
		if let Some(worker) = worker {
			drop(worker.join());
		}
	}
}

impl RetirementCompletion {
	fn new() -> Self {
		Self {
			state: Mutex::new(RetirementStateData {
				state: RetirementState::Pending,
				retained: Vec::new(),
			}),
			changed: Condvar::new(),
		}
	}

	fn finish(&self, finished: RetirementState) {
		let mut state = match self.state.lock() {
			Ok(state) => state,
			Err(poisoned) => poisoned.into_inner(),
		};
		if state.state != RetirementState::Pending {
			return;
		}
		state.state = finished;
		let retained = mem::take(&mut state.retained);
		self.changed.notify_all();
		drop(state);
		if finished == RetirementState::Failed {
			// Failure is not completion evidence. Never recycle attached resources.
			mem::forget(retained);
		} else {
			// User-owned destructors may inspect this completion; release the lock first.
			drop(retained);
		}
	}

	fn wait(&self) -> Result<()> {
		let mut state = match self.state.lock() {
			Ok(state) => state,
			Err(poisoned) => poisoned.into_inner(),
		};
		while state.state == RetirementState::Pending {
			state = match self.changed.wait(state) {
				Ok(state) => state,
				Err(poisoned) => poisoned.into_inner(),
			};
		}
		match state.state {
			RetirementState::Retired => Ok(()),
			RetirementState::Failed => Err(retirement_unavailable()),
			RetirementState::Pending => Err(retirement_unavailable()),
		}
	}
}

// Keep receiving after unknown completion: queued and racing submissions also
// require quarantine. The generic ownership loop permits CPU-only failure proofs
// without fake Vulkan handles; production supplies the exact device wait/free.
fn drain_retirement_queue<T>(
	receiver: mpsc::Receiver<T>,
	control: &WorkerControl,
	mut wait: impl FnMut(&T) -> Result<()>,
	mut retire: impl FnMut(T),
	mut quarantine: impl FnMut(T),
) {
	while let Ok(retirement) = receiver.recv() {
		let completed = if control.failed.load(Ordering::Acquire) {
			false
		} else {
			match wait(&retirement) {
				Ok(()) => true,
				Err(error) => {
					control.fail(error);
					false
				}
			}
		};
		if completed {
			retire(retirement);
		} else {
			quarantine(retirement);
		}
		control.pending.fetch_sub(1, Ordering::AcqRel);
	}
}

fn retirement_unavailable() -> Error {
	Error::backend_failure(
		"Vulkan",
		"submission retirement",
		std::io::Error::new(
			std::io::ErrorKind::BrokenPipe,
			"retirement worker is unavailable",
		),
	)
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/retirement_unit.rs"]
mod tests;
