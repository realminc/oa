//! Numeric dtype, shape, and runtime storage.

use std::{
	marker::PhantomData,
	rc::Rc,
	sync::atomic::{AtomicU64, Ordering},
};

use crate::{
	Engine, Error, Result,
	runtime::{EngineHandle, Storage},
};

use super::{DType, Element};

/// Device-visible numeric storage with explicit shape and dtype semantics.
#[derive(Clone)]
pub struct Matrix {
	engine: EngineHandle,
	storage: Storage,
	shape: Vec<usize>,
	dtype: DType,
	element_count: usize,
	semantic: Rc<MatrixSemantic>,
	_not_send_sync: PhantomData<Rc<()>>,
}

/// Persistent handle-free provenance shared by clones of one Matrix value.
pub(crate) struct MatrixSemantic {
	id: u64,
	shape: Vec<usize>,
	dtype: DType,
	view_source: Option<Rc<Self>>,
	view_byte_offset: i64,
}

impl MatrixSemantic {
	pub(crate) const fn id(&self) -> u64 {
		self.id
	}

	pub(crate) fn shape(&self) -> &[usize] {
		&self.shape
	}

	pub(crate) const fn dtype(&self) -> DType {
		self.dtype
	}

	pub(crate) fn view_source(&self) -> Option<&Rc<Self>> {
		self.view_source.as_ref()
	}

	pub(crate) const fn view_byte_offset(&self) -> i64 {
		self.view_byte_offset
	}
}

impl Matrix {
	/// Create a dense matrix from an admitted Rust scalar slice.
	///
	/// The element type selects the matrix dtype exactly; this function performs
	/// no numeric conversion.
	///
	/// # Errors
	///
	/// Returns an error when the shape overflows, its element count differs from
	/// `values`, or storage allocation/upload fails.
	pub fn from_slice<T: Element>(
		engine: &Engine,
		shape: impl Into<Vec<usize>>,
		values: &[T],
	) -> Result<Self> {
		Self::from_slice_handle(&engine.handle(), shape.into(), values)
	}

	pub(crate) fn from_slice_handle<T: Element>(
		engine: &EngineHandle,
		shape: Vec<usize>,
		values: &[T],
	) -> Result<Self> {
		let element_count = checked_element_count(&shape)?;
		if element_count != values.len() {
			return Err(Error::invalid_argument(format!(
				"matrix shape contains {element_count} elements but {} values were provided",
				values.len()
			)));
		}

		let byte_len = element_count
			.checked_mul(size_of::<T>())
			.ok_or_else(|| Error::invalid_argument("matrix byte size overflows usize"))?;
		debug_assert_eq!(size_of::<T>(), T::DTYPE.size_bytes());
		// SAFETY: `Element` is sealed to OA-owned implementations whose values have
		// no padding, every bit pattern is valid, and storage width equals
		// `DType::size_bytes`. `values` is an initialized contiguous slice, and the
		// checked calculation proves its exact byte extent.
		let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), byte_len) };
		let storage = engine.create_storage(bytes)?;

		let semantic = matrix_semantic(&shape, T::DTYPE, None)?;
		Ok(Self {
			engine: engine.clone(),
			storage,
			shape,
			dtype: T::DTYPE,
			element_count,
			semantic,
			_not_send_sync: PhantomData,
		})
	}

	/// Create an FP32 matrix and initialize its device-visible storage.
	///
	/// # Errors
	///
	/// Returns an error when the shape overflows, its element count differs from
	/// `values`, or storage allocation/upload fails.
	pub fn from_f32(engine: &Engine, shape: impl Into<Vec<usize>>, values: &[f32]) -> Result<Self> {
		Self::from_slice(engine, shape, values)
	}

	pub(crate) fn filled_f32(
		engine: &Engine,
		shape: impl Into<Vec<usize>>,
		value: f32,
	) -> Result<Self> {
		let shape = shape.into();
		let element_count = checked_element_count(&shape)?;
		Self::from_f32(engine, shape, &vec![value; element_count])
	}

	/// Return the matrix shape.
	pub fn shape(&self) -> &[usize] {
		&self.shape
	}

	/// Return the matrix scalar representation.
	pub const fn dtype(&self) -> DType {
		self.dtype
	}

	/// Return the number of logical dense elements.
	pub const fn num_elements(&self) -> usize {
		self.element_count
	}

	/// Return a zero-copy view with a different dense shape.
	///
	/// This creates a distinct semantic value while retaining the same storage and
	/// readiness state. The element order is unchanged.
	///
	/// # Errors
	///
	/// Returns an error when the new shape overflows or contains a different
	/// number of elements.
	pub fn reshape(&self, shape: impl Into<Vec<usize>>) -> Result<Self> {
		crate::matrix::reshape(self, shape)
	}

	/// Read all FP32 values back to host memory.
	///
	/// # Errors
	///
	/// This is an explicit host-observation boundary: it submits the pending eager
	/// batch when this matrix is still recorded, then waits for the exact GPU event
	/// that produced it before mapping its storage. Returns an error when this is
	/// not an FP32 matrix or when submission, completion, mapping, or cache
	/// invalidation fails.
	pub fn read_f32(&self) -> Result<Vec<f32>> {
		self.read()
	}

	/// Attempt to read all FP32 values without waiting for pending GPU work.
	///
	/// # Errors
	///
	/// Returns an error when this is not an FP32 matrix,
	/// [`crate::ErrorKind::NotReady`] when GPU production is pending, or an error
	/// when mapping or cache invalidation fails.
	pub fn try_read_f32(&self) -> Result<Vec<f32>> {
		self.try_read()
	}

	/// Read all values into their exact admitted Rust scalar representation.
	///
	/// # Errors
	///
	/// Returns an error before submission or waiting when `T` does not match this
	/// matrix's dtype. Otherwise this submits its pending eager batch when needed,
	/// waits for the exact producing event, and returns an error when submission,
	/// completion, mapping, or cache invalidation fails.
	pub fn read<T: Element>(&self) -> Result<Vec<T>> {
		self.validate_element::<T>()?;
		if self.storage.needs_flush() {
			self.engine.flush()?;
		}
		self.storage.wait_ready()?;
		self.read_ready::<T>()
	}

	/// Attempt typed readback without waiting for pending GPU work.
	///
	/// # Errors
	///
	/// Returns an error when `T` does not match this matrix's dtype,
	/// [`crate::ErrorKind::NotReady`] when production is pending, or an error
	/// when mapping or cache invalidation fails.
	pub fn try_read<T: Element>(&self) -> Result<Vec<T>> {
		self.validate_element::<T>()?;
		self.storage.ensure_ready()?;
		self.read_ready::<T>()
	}

	fn validate_element<T: Element>(&self) -> Result<()> {
		if self.dtype != T::DTYPE {
			return Err(Error::invalid_argument(format!(
				"matrix dtype {} cannot be read as {}",
				self.dtype.token(),
				T::DTYPE.token()
			)));
		}
		Ok(())
	}

	fn read_ready<T: Element>(&self) -> Result<Vec<T>> {
		let byte_len = self
			.element_count
			.checked_mul(size_of::<T>())
			.ok_or_else(|| Error::invalid_argument("matrix byte size overflows usize"))?;
		debug_assert_eq!(size_of::<T>(), self.dtype.size_bytes());
		let mut output = vec![T::default(); self.element_count];
		// SAFETY: `Element` is sealed to OA-owned no-padding scalar types for which
		// every bit pattern is valid. `output` is initialized contiguous storage with
		// exactly `byte_len` bytes; the mutable byte view weakens alignment and does
		// not exceed the vector's initialized extent.
		let output_bytes =
			unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<u8>(), byte_len) };
		self.storage.read(output_bytes)?;
		Ok(output)
	}

	pub(crate) const fn engine_handle(&self) -> &EngineHandle {
		&self.engine
	}

	pub(crate) const fn element_count(&self) -> usize {
		self.element_count
	}

	pub(crate) fn value_id(&self) -> u64 {
		self.semantic.id
	}

	pub(crate) fn same_value_as(&self, other: &Self) -> bool {
		self.semantic.id == other.semantic.id
	}

	pub(crate) fn semantic(&self) -> &Rc<MatrixSemantic> {
		&self.semantic
	}

	pub(crate) fn storage(&self) -> &Storage {
		&self.storage
	}

	pub(crate) fn write_values<T: Element>(&self, values: &[T]) -> Result<()> {
		self.validate_element::<T>()?;
		if values.len() != self.element_count {
			return Err(Error::invalid_argument(format!(
				"matrix upload requires {} elements; received {}",
				self.element_count,
				values.len()
			)));
		}
		let byte_len = self
			.element_count
			.checked_mul(size_of::<T>())
			.ok_or_else(|| Error::invalid_argument("matrix byte size overflows usize"))?;
		// SAFETY: `Element` is sealed to initialized, no-padding OA scalar types and
		// the checked length covers exactly the supplied slice.
		let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), byte_len) };
		self.storage.write(bytes)
	}

	pub(crate) fn reshape_view(&self, shape: Vec<usize>) -> Result<Self> {
		let element_count = checked_element_count(&shape)?;
		if element_count != self.element_count {
			return Err(Error::invalid_argument(format!(
				"matrix reshape requires {} elements; requested shape {:?} contains {element_count}",
				self.element_count, shape
			)));
		}
		let semantic = matrix_semantic(&shape, self.dtype, Some(self.semantic.clone()))?;
		Ok(Self {
			engine: self.engine.clone(),
			storage: self.storage.clone(),
			shape,
			dtype: self.dtype,
			element_count,
			semantic,
			_not_send_sync: PhantomData,
		})
	}

	pub(crate) fn allocate(
		engine: &EngineHandle,
		shape: Vec<usize>,
		element_count: usize,
		dtype: DType,
	) -> Result<Self> {
		let byte_len = element_count
			.checked_mul(dtype.size_bytes())
			.ok_or_else(|| Error::invalid_argument("matrix byte size overflows usize"))?;
		let storage = engine.create_storage(&vec![0_u8; byte_len])?;
		let semantic = matrix_semantic(&shape, dtype, None)?;
		Ok(Self {
			engine: engine.clone(),
			storage,
			shape,
			dtype,
			element_count,
			semantic,
			_not_send_sync: PhantomData,
		})
	}
}

fn matrix_semantic(
	shape: &[usize],
	dtype: DType,
	view_source: Option<Rc<MatrixSemantic>>,
) -> Result<Rc<MatrixSemantic>> {
	Ok(Rc::new(MatrixSemantic {
		id: next_value_id()?,
		shape: shape.to_vec(),
		dtype,
		view_source,
		view_byte_offset: 0,
	}))
}

fn next_value_id() -> Result<u64> {
	static NEXT_VALUE_ID: AtomicU64 = AtomicU64::new(1);
	NEXT_VALUE_ID
		.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
			(value != u64::MAX).then_some(value + 1)
		})
		.map_err(|_| Error::resource_exhausted("matrix semantic value identity exhausted"))
}

fn checked_element_count(shape: &[usize]) -> Result<usize> {
	shape.iter().copied().try_fold(1_usize, |count, extent| {
		count
			.checked_mul(extent)
			.ok_or_else(|| Error::invalid_argument("matrix element count overflows usize"))
	})
}

#[cfg(test)]
mod tests {
	use super::checked_element_count;
	use crate::ErrorKind;

	#[test]
	fn scalar_and_zero_extent_counts_are_explicit() {
		assert_eq!(checked_element_count(&[]).ok(), Some(1));
		assert_eq!(checked_element_count(&[2, 0, usize::MAX]).ok(), Some(0));
	}

	#[test]
	fn element_count_overflow_is_rejected() {
		assert_eq!(
			checked_element_count(&[usize::MAX, 2]).map_err(|error| error.kind()),
			Err(ErrorKind::InvalidArgument)
		);
	}
}
