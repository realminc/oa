use std::sync::atomic::AtomicUsize;

use super::*;

#[derive(Debug)]
struct DropProbe(Arc<AtomicUsize>);

impl Drop for DropProbe {
	fn drop(&mut self) {
		self.0.fetch_add(1, Ordering::Relaxed);
	}
}

fn ticket(completion: &Arc<RetirementCompletion>) -> RetirementTicket {
	RetirementTicket {
		completion: completion.clone(),
		control: Arc::new(WorkerControl::new()),
	}
}

#[test]
fn failed_completion_quarantines_early_and_late_attachments() {
	let completion = Arc::new(RetirementCompletion::new());
	let ticket = ticket(&completion);
	let drops = Arc::new(AtomicUsize::new(0));
	ticket.retain_until_complete(DropProbe(drops.clone()));
	completion.finish(RetirementState::Failed);
	assert!(ticket.wait().is_err());
	assert!(ticket.check_failure().is_err());
	ticket.retain_until_complete(DropProbe(drops.clone()));
	// A failed ticket cannot later claim successful completion.
	completion.finish(RetirementState::Retired);
	assert!(ticket.wait().is_err());
	assert!(ticket.check_failure().is_err());
	drop(ticket);
	drop(completion);
	assert_eq!(drops.load(Ordering::Relaxed), 0);
}

#[test]
fn successful_completion_releases_early_and_late_attachments() {
	let completion = Arc::new(RetirementCompletion::new());
	let ticket = ticket(&completion);
	let drops = Arc::new(AtomicUsize::new(0));
	ticket.retain_until_complete(DropProbe(drops.clone()));
	completion.finish(RetirementState::Retired);
	ticket.wait().unwrap();
	assert_eq!(drops.load(Ordering::Relaxed), 1);
	ticket.retain_until_complete(DropProbe(drops.clone()));
	assert_eq!(drops.load(Ordering::Relaxed), 2);
}

#[test]
fn successful_attachment_destructor_can_inspect_completion() {
	struct InspectCompletion(Arc<RetirementCompletion>);
	impl Drop for InspectCompletion {
		fn drop(&mut self) {
			assert!(
				self.0.state.try_lock().is_ok(),
				"destructor ran under completion lock"
			);
		}
	}
	let completion = Arc::new(RetirementCompletion::new());
	let ticket = ticket(&completion);
	ticket.retain_until_complete(InspectCompletion(completion.clone()));
	completion.finish(RetirementState::Retired);
	ticket.retain_until_complete(InspectCompletion(completion.clone()));
}

#[test]
fn failed_wait_quarantines_queued_and_racing_work_without_another_wait() {
	let control = WorkerControl::new();
	let (sender, receiver) = mpsc::channel();
	control.pending.store(4, Ordering::Release);
	let drops = Arc::new(AtomicUsize::new(0));
	for epoch in 1..=3 {
		sender.send((epoch, DropProbe(drops.clone()))).unwrap();
	}
	let waited = Mutex::new(Vec::new());
	let retired = Mutex::new(Vec::new());
	let quarantined = Mutex::new(Vec::new());
	let mut racing_sender = Some(sender);
	drain_retirement_queue(
		receiver,
		&control,
		|work| {
			waited.lock().unwrap().push(work.0);
			if work.0 == 2 {
				Err(Error::failed_precondition("injected unknown completion"))
			} else {
				Ok(())
			}
		},
		|work| {
			retired.lock().unwrap().push(work.0);
			drop(work);
		},
		|work| {
			assert!(control.failed.load(Ordering::Acquire));
			quarantined.lock().unwrap().push(work.0);
			if let Some(sender) = racing_sender.take() {
				sender.send((4, DropProbe(drops.clone()))).unwrap();
			}
			mem::forget(work);
		},
	);
	assert_eq!(*waited.lock().unwrap(), [1, 2]);
	assert_eq!(*retired.lock().unwrap(), [1]);
	assert_eq!(*quarantined.lock().unwrap(), [2, 3, 4]);
	assert_eq!(control.pending.load(Ordering::Acquire), 0);
	assert_eq!(
		drops.load(Ordering::Relaxed),
		1,
		"only the completed work may release resources"
	);
}

#[test]
fn retirement_worker_cannot_join_itself_during_completion_observation() {
	let control = Arc::new(WorkerControl::new());
	control.disconnected.store(true, Ordering::Release);
	let worker_control = control.clone();
	let (start, ready) = mpsc::channel();
	let (done, completed) = mpsc::channel();
	let worker = thread::spawn(move || {
		ready.recv().unwrap();
		worker_control.join_if_drained();
		done.send(()).unwrap();
	});
	control.install(worker);
	start.send(()).unwrap();
	completed
		.recv_timeout(std::time::Duration::from_secs(5))
		.unwrap();
	control.join_if_drained();
	assert!(control.worker.lock().unwrap().is_none());
}

#[test]
fn failed_domain_rejects_pending_observation_but_preserves_retired_evidence() {
	let pending = Arc::new(RetirementCompletion::new());
	let pending_ticket = ticket(&pending);
	assert!(pending_ticket.check_failure().is_ok());
	pending_ticket.control.failed.store(true, Ordering::Release);
	assert!(pending_ticket.check_failure().is_err());
	assert!(pending_ticket.wait().is_err());
	let completed = Arc::new(RetirementCompletion::new());
	let completed_ticket = ticket(&completed);
	completed.finish(RetirementState::Retired);
	completed_ticket
		.control
		.failed
		.store(true, Ordering::Release);
	assert!(completed_ticket.check_failure().is_ok());
	completed_ticket.wait().unwrap();
}

#[test]
fn terminal_failure_preserves_the_first_backend_cause() {
	use std::error::Error as _;
	let completion = Arc::new(RetirementCompletion::new());
	let ticket = ticket(&completion);
	ticket
		.control
		.fail(Error::failed_precondition("first wait failure"));
	ticket
		.control
		.fail(Error::failed_precondition("later failure"));
	let error = ticket.wait().unwrap_err();
	assert!(
		error
			.source()
			.unwrap()
			.to_string()
			.contains("first wait failure")
	);
}
