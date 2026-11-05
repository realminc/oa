//! GPU-consumable object-detection result values.
//!
//! [`Detection`] is the 32-byte packed record layout shared by host reference
//! providers, GPU postprocess kernels, and render consumers. Geometry is
//! normalized to the source image so camera/view transforms remain GPU-side.
//! [`DetectionBuffer`] is a fixed-capacity host-upload ring slot: records are
//! replaced through [`DetectionBuffer::upload`] while
//! [`DetectionBuffer::mark_consumed`] pins the exact [`Event`] a GPU consumer
//! must complete before the next overwrite.
//!
//! Donor: `oacpp/source/cpp/include/oa/vision/detection.h`.

use std::mem::{offset_of, size_of};

use crate::{Engine, Error, Event, Matrix, Result, runtime::EngineHandle};

/// One GPU-consumable detection record, exactly 32 bytes.
///
/// `color_rgba` is an optional per-instance display color in `0xRRGGBBAA`;
/// zero selects the overlay style fallback and preserves compatibility with
/// older producers. `track_id` is a stable identity for trackers; rendering
/// ignores it while postprocess and interaction code retain the same record
/// layout across frames.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Detection {
	pub center_x: f32,
	pub center_y: f32,
	pub width: f32,
	pub height: f32,
	pub confidence: f32,
	pub class_id: u32,
	pub color_rgba: u32,
	pub track_id: u32,
}

const _: () = assert!(size_of::<Detection>() == 32);
const _: () = assert!(offset_of!(Detection, color_rgba) == 24);
const _: () = assert!(offset_of!(Detection, track_id) == 28);

/// A fixed-capacity GPU-consumable detection buffer.
///
/// Construction requires an [`Engine`]; dropping the buffer never waits —
/// the engine's retirement service keeps submitted resources alive through
/// completion.
pub struct DetectionBuffer {
	engine: EngineHandle,
	storage: Matrix,
	consumer: Option<Event>,
	count: u32,
	capacity: u32,
}

impl DetectionBuffer {
	/// Allocate a host-uploadable buffer for `capacity` detection records.
	///
	/// Native GPU postprocess producers use the same record layout and
	/// consumer API.
	///
	/// # Errors
	///
	/// Returns an error when `capacity` is zero, the capacity byte extent
	/// overflows, or allocation fails.
	pub fn create_host_upload(engine: &Engine, capacity: u32) -> Result<Self> {
		if capacity == 0 {
			return Err(Error::invalid_argument(
				"oa::vision::DetectionBuffer::create_host_upload: capacity must be non-zero",
			));
		}
		let byte_len = usize::try_from(capacity)
			.ok()
			.and_then(|count| count.checked_mul(size_of::<Detection>()))
			.ok_or_else(|| {
				Error::invalid_argument(
					"oa::vision::DetectionBuffer::create_host_upload: capacity byte extent overflows usize",
				)
			})?;
		let storage = Matrix::new_from_slice(engine, [byte_len], &vec![0_u8; byte_len])?;
		Ok(Self {
			engine: engine.handle(),
			storage,
			consumer: None,
			count: 0,
			capacity,
		})
	}

	/// Replace the current records.
	///
	/// # Errors
	///
	/// Returns [`crate::ErrorKind::NotReady`] while a previous render
	/// submission still consumes this slot,
	/// [`crate::ErrorKind::ResourceExhausted`] when `detections` exceeds
	/// capacity, or an upload error.
	pub fn upload(&mut self, detections: &[Detection]) -> Result<()> {
		if !self.is_ready()? {
			return Err(Error::not_ready(
				"oa::vision::DetectionBuffer::upload: buffer is still consumed by the GPU",
			));
		}
		if detections.len() > usize::try_from(self.capacity).unwrap_or(usize::MAX) {
			return Err(Error::resource_exhausted(
				"oa::vision::DetectionBuffer::upload: detection count exceeds capacity",
			));
		}
		let used = std::mem::size_of_val(detections);
		let mut bytes = vec![0_u8; self.storage.num_elements()];
		if used > 0 {
			// SAFETY: `Detection` is `repr(C)`, plain data with a statically
			// checked layout; the used prefix never exceeds the byte capacity.
			let source = unsafe { std::slice::from_raw_parts(detections.as_ptr().cast::<u8>(), used) };
			bytes[..used].copy_from_slice(source);
		}
		self.storage.write_values::<u8>(&bytes)?;
		self.count = u32::try_from(detections.len())
			.map_err(|_| Error::invalid_argument("detection count exceeds u32"))?;
		self.consumer = None;
		Ok(())
	}

	/// Pin `completion` as the event that must complete before the next
	/// [`upload`](Self::upload).
	///
	/// # Errors
	///
	/// Returns an error when `completion` does not belong to this buffer's
	/// engine or completion cannot be queried.
	pub fn mark_consumed(&mut self, completion: &Event) -> Result<()> {
		if !self.engine.owns_event(completion) {
			return Err(Error::invalid_argument(
				"oa::vision::DetectionBuffer::mark_consumed requires an event from its engine",
			));
		}
		self.consumer = if completion.is_complete()? {
			None
		} else {
			Some(completion.clone())
		};
		Ok(())
	}

	/// Return whether no pinned consumer event is still in flight.
	///
	/// # Errors
	///
	/// Returns an error when consumer completion cannot be queried.
	pub fn is_ready(&self) -> Result<bool> {
		match &self.consumer {
			Some(completion) => completion.is_complete(),
			None => Ok(true),
		}
	}

	/// Return the number of records written by the last [`upload`](Self::upload).
	pub const fn count(&self) -> u32 {
		self.count
	}

	/// Return the record capacity fixed at creation.
	pub const fn capacity(&self) -> u32 {
		self.capacity
	}

	/// Bindless storage-buffer descriptor index for UI/render consumers.
	pub(crate) fn bindless_index(&self) -> Result<u32> {
		self.engine.storage_descriptor_index(self.storage.storage())
	}

	/// The packed record storage retained by UI draw commands.
	pub(crate) const fn storage(&self) -> &Matrix {
		&self.storage
	}

	/// The engine that owns this buffer's storage.
	pub(crate) const fn engine_handle(&self) -> &EngineHandle {
		&self.engine
	}
}
