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

use super::{Device, RecordedCommandBuffer};

pub(in crate::runtime) struct RetirementService {
	sender: Option<Sender<Retirement>>,
	control: Arc<WorkerControl>,
	device: Device,
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
	state: Mutex<RetirementState>,
	changed: Condvar,
}

struct WorkerControl {
	pending: AtomicUsize,
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
	pub(in crate::runtime) fn new(device: &Device) -> Self {
		Self {
			sender: None,
			control: Arc::new(WorkerControl::new()),
			device: device.clone(),
		}
	}

	pub(in crate::runtime) fn prepare(&mut self) -> Result<()> {
		if self.sender.is_some() {
			return Ok(());
		}

		let (sender, receiver) = mpsc::channel::<Retirement>();
		let worker_device = self.device.clone();
		let worker_control = self.control.clone();
		let worker = thread::Builder::new()
			.name("oa-vk-retirement".to_owned())
			.spawn(move || {
				while let Ok(retirement) = receiver.recv() {
					if worker_device.wait(retirement.epoch).is_err() {
						worker_control.pending.fetch_sub(1, Ordering::AcqRel);
						retirement.completion.finish(RetirementState::Failed);
						// Completion failure cannot prove the command buffer is no longer
						// pending. Leak its retained resources and device graph instead of
						// destroying live Vulkan state.
						mem::forget(worker_device);
						mem::forget(retirement);
						return;
					}
					worker_device.free(retirement.command);
					worker_control.pending.fetch_sub(1, Ordering::AcqRel);
					retirement.completion.finish(RetirementState::Retired);
				}
			})
			.map_err(|source| {
				Error::backend_failure("Vulkan", "retirement-worker creation", source)
			})?;

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
			self.leak_device_graph();
			return Err(retirement_unavailable());
		};
		self.control.pending.fetch_add(1, Ordering::AcqRel);
		if let Err(mpsc::SendError(retirement)) = sender.send(retirement) {
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
		// Joining is permitted only after every GPU retirement completed. This releases
		// an idle host thread; it never turns engine drop into a GPU wait.
		self.control.join_if_drained();
	}
}

impl RetirementTicket {
	pub(in crate::runtime) fn wait(&self) -> Result<()> {
		self.completion.wait()?;
		self.control.join_if_drained();
		Ok(())
	}
}

impl WorkerControl {
	fn new() -> Self {
		Self {
			pending: AtomicUsize::new(0),
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
			state: Mutex::new(RetirementState::Pending),
			changed: Condvar::new(),
		}
	}

	fn finish(&self, finished: RetirementState) {
		let mut state = match self.state.lock() {
			Ok(state) => state,
			Err(poisoned) => poisoned.into_inner(),
		};
		*state = finished;
		self.changed.notify_all();
	}

	fn wait(&self) -> Result<()> {
		let mut state = match self.state.lock() {
			Ok(state) => state,
			Err(poisoned) => poisoned.into_inner(),
		};
		while *state == RetirementState::Pending {
			state = match self.changed.wait(state) {
				Ok(state) => state,
				Err(poisoned) => poisoned.into_inner(),
			};
		}
		match *state {
			RetirementState::Retired => Ok(()),
			RetirementState::Failed => Err(retirement_unavailable()),
			RetirementState::Pending => Err(retirement_unavailable()),
		}
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
