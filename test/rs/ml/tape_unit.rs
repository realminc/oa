//! Recording selection and tape lifetime contracts; no GPU work.

use super::*;

#[test]
fn closing_nested_tape_restores_the_outer_recording() {
	let outer = GradientTape::new();
	let nested = GradientTape::new();
	ACTIVE_TAPES.with(|tapes| {
		assert!(
			tapes
				.borrow()
				.last()
				.is_some_and(|selected| { selected.ptr_eq(&Rc::downgrade(&nested.recording)) })
		);
	});
	nested.close();
	ACTIVE_TAPES.with(|tapes| {
		assert!(
			tapes
				.borrow()
				.last()
				.is_some_and(|selected| { selected.ptr_eq(&Rc::downgrade(&outer.recording)) })
		);
	});
	outer.close();
	assert!(!recording_active());
}

#[test]
fn closing_outer_tape_does_not_remove_the_nested_selection() {
	let outer = GradientTape::new();
	let nested = GradientTape::new();
	outer.close();
	outer.close();
	assert!(recording_active());
	ACTIVE_TAPES.with(|tapes| {
		assert!(
			tapes
				.borrow()
				.last()
				.is_some_and(|selected| { selected.ptr_eq(&Rc::downgrade(&nested.recording)) })
		);
	});
	nested.close();
	assert!(!recording_active());
}

#[test]
fn dropping_tape_releases_recording_and_weak_observers() {
	let recording = {
		let tape = GradientTape::new();
		let recording = Rc::downgrade(&tape.recording);
		assert_eq!(recording.strong_count(), 1);
		recording
	};
	assert!(recording.upgrade().is_none());
	assert!(!recording_active());
	ACTIVE_TAPES.with(|tapes| assert!(tapes.borrow().is_empty()));
}
