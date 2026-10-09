//! Numeric dtype, shape, and runtime storage.

use std::{
	marker::PhantomData,
	ops::{Add, Div, Mul, Neg, Sub},
	rc::Rc,
	sync::atomic::{AtomicU64, Ordering},
};

use crate::{
	Engine, Error, Result,
	runtime::{EngineHandle, Storage},
};

use crate::core::{DType, Element};

/// Device-visible numeric storage with explicit shape and dtype semantics.
///
/// Arithmetic operators delegate to [`crate::matrix`] and return `Result<Matrix>`:
/// use `let sum = (&left + &right)?;`. They preserve validation, allocation, and
/// recording failures without panicking. Multiplication is elementwise; use
/// [`crate::matrix::mat_mul_nt`] for matrix multiplication.
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
		let storage = if T::DTYPE == DType::U8 {
			let mut padded = vec![0_u8; byte_storage_len(byte_len, T::DTYPE)?];
			padded[..byte_len].copy_from_slice(bytes);
			engine.create_storage(&padded)?
		} else {
			engine.create_storage(bytes)?
		};

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

	/// Return a zero-copy one-dimensional view over all elements.
	///
	/// Equivalent to `self.reshape([self.num_elements()])`.
	///
	/// # Errors
	///
	/// Returns an error when the element count overflows or storage allocation
	/// fails.
	pub fn flatten(&self) -> Result<Self> {
		crate::matrix::reshape(self, vec![self.element_count])
	}

	/// Return a zero-copy view with a new size-one axis inserted at `dim`.
	///
	/// `dim` must be in `0..=self.shape().len()`. Negative indices are not
	/// supported; use `self.shape().len()` to append at the end.
	///
	/// # Errors
	///
	/// Returns an error when `dim` is out of range.
	pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
		let rank = self.shape.len();
		if dim > rank {
			return Err(Error::invalid_argument(format!(
				"unsqueeze dim {dim} is out of range for rank-{rank} matrix"
			)));
		}
		let mut shape = self.shape.clone();
		shape.insert(dim, 1);
		crate::matrix::reshape(self, shape)
	}

	/// Return a zero-copy view with one size-one axis removed.
	///
	/// `dim` must be in `0..self.shape().len()` and the selected axis must have
	/// extent one. Negative indices are not supported.
	///
	/// # Errors
	///
	/// Returns an error when `dim` is out of range or the selected axis is not
	/// size one.
	pub fn squeeze(&self, dim: usize) -> Result<Self> {
		let rank = self.shape.len();
		if dim >= rank {
			return Err(Error::invalid_argument(format!(
				"squeeze dim {dim} is out of range for rank-{rank} matrix"
			)));
		}
		if self.shape[dim] != 1 {
			return Err(Error::invalid_argument(format!(
				"squeeze dim {dim} has extent {}; only size-one axes may be squeezed",
				self.shape[dim]
			)));
		}
		let mut shape = self.shape.clone();
		shape.remove(dim);
		crate::matrix::reshape(self, shape)
	}

	/// Elementwise add, equivalent to [`crate::matrix::add`].
	///
	/// # Errors
	///
	/// Returns the canonical operation's validation, allocation or recording error.
	pub fn add(&self, rhs: &Self) -> Result<Self> {
		crate::matrix::add(self, rhs)
	}

	/// Elementwise sub, equivalent to [`crate::matrix::sub`].
	///
	/// # Errors
	///
	/// Returns the canonical operation's validation, allocation or recording error.
	pub fn sub(&self, rhs: &Self) -> Result<Self> {
		crate::matrix::sub(self, rhs)
	}

	/// Elementwise mul, equivalent to [`crate::matrix::mul`].
	///
	/// # Errors
	///
	/// Returns the canonical operation's validation, allocation or recording error.
	pub fn mul(&self, rhs: &Self) -> Result<Self> {
		crate::matrix::mul(self, rhs)
	}

	/// Elementwise div, equivalent to [`crate::matrix::div`].
	///
	/// # Errors
	///
	/// Returns the canonical operation's validation, allocation or recording error.
	pub fn div(&self, rhs: &Self) -> Result<Self> {
		crate::matrix::div(self, rhs)
	}

	/// Matrix multiply `self @ rhs.T` — equivalent to `matrix::mat_mul_nt(self, rhs)`.
	///
	/// # Errors
	///
	/// Returns an error when inputs are not rank-two FP32 matrices, K extents
	/// disagree, element counts exceed u32, or the runtime recording contract fails.
	pub fn mat_mul(&self, rhs: &Self) -> Result<Self> {
		crate::matrix::mat_mul_nt(self, rhs)
	}

	/// Exchange two axes, equivalent to `matrix::transpose(self, dim0, dim1)`.
	///
	/// # Errors
	///
	/// Returns an error when either dim is out of range or when the runtime
	/// recording contract fails.
	pub fn transpose(&self, dim0: i32, dim1: i32) -> Result<Self> {
		crate::matrix::transpose(self, dim0, dim1)
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
		self.storage.read_prefix(output_bytes)?;
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
		if self.dtype == DType::U8 {
			let mut padded = vec![0_u8; byte_storage_len(byte_len, self.dtype)?];
			padded[..byte_len].copy_from_slice(bytes);
			self.storage.write(&padded)
		} else {
			self.storage.write(bytes)
		}
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

	pub(crate) fn reshape_semantic_output(&self, shape: Vec<usize>) -> Result<Self> {
		let element_count = checked_element_count(&shape)?;
		if element_count != self.element_count {
			return Err(Error::invalid_argument(format!(
				"matrix reshape requires {} elements; requested shape {:?} contains {element_count}",
				self.element_count, shape
			)));
		}
		let semantic = matrix_semantic(&shape, self.dtype, None)?;
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

	/// Allocate initialized zero storage for persistent state or producers that
	/// intentionally depend on a zeroed destination.
	pub(crate) fn allocate(
		engine: &EngineHandle,
		shape: Vec<usize>,
		element_count: usize,
		dtype: DType,
	) -> Result<Self> {
		let logical_byte_len = element_count
			.checked_mul(dtype.size_bytes())
			.ok_or_else(|| Error::invalid_argument("matrix byte size overflows usize"))?;
		let byte_len = byte_storage_len(logical_byte_len, dtype)?;
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

	/// Allocate a Matrix that must be fully produced by recorded GPU work.
	/// No host placeholder is uploaded; readiness rejects observation and GPU
	/// reads until a producer dispatch has been recorded and completed.
	pub(crate) fn allocate_output(
		engine: &EngineHandle,
		shape: Vec<usize>,
		element_count: usize,
		dtype: DType,
	) -> Result<Self> {
		let logical_byte_len = element_count
			.checked_mul(dtype.size_bytes())
			.ok_or_else(|| Error::invalid_argument("matrix byte size overflows usize"))?;
		let storage = if logical_byte_len == 0 {
			engine.create_storage(&[])?
		} else {
			let byte_len = byte_storage_len(logical_byte_len, dtype)?;
			engine.create_output_storage(byte_len)?
		};
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

fn byte_storage_len(logical_byte_len: usize, dtype: DType) -> Result<usize> {
	if dtype != DType::U8 {
		return Ok(logical_byte_len);
	}
	logical_byte_len
		.max(1)
		.checked_add(3)
		.map(|length| length & !3)
		.ok_or_else(|| Error::invalid_argument("byte matrix storage size overflows usize"))
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
			value.checked_add(1)
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

// Operators delegate to the same fallible semantic operations. Result preserves
// validation, allocation, capability, and recording errors without hidden panics.

/// `&mat_a + &mat_b` — elementwise add with broadcasting.
impl Add for &Matrix {
	type Output = Result<Matrix>;
	fn add(self, rhs: &Matrix) -> Result<Matrix> {
		crate::matrix::add(self, rhs)
	}
}

/// `&mat_a - &mat_b` — elementwise subtract with broadcasting.
impl Sub for &Matrix {
	type Output = Result<Matrix>;
	fn sub(self, rhs: &Matrix) -> Result<Matrix> {
		crate::matrix::sub(self, rhs)
	}
}

/// `&mat_a * &mat_b` — elementwise multiply (not matmul).
impl Mul for &Matrix {
	type Output = Result<Matrix>;
	fn mul(self, rhs: &Matrix) -> Result<Matrix> {
		crate::matrix::mul(self, rhs)
	}
}

/// `&mat_a / &mat_b` — elementwise divide with broadcasting.
impl Div for &Matrix {
	type Output = Result<Matrix>;
	fn div(self, rhs: &Matrix) -> Result<Matrix> {
		crate::matrix::div(self, rhs)
	}
}

/// `-&mat` — elementwise negate.
impl Neg for &Matrix {
	type Output = Result<Matrix>;
	fn neg(self) -> Result<Matrix> {
		crate::matrix::neg(self)
	}
}

/// `-mat` — elementwise negate of an owned matrix (avoids a `&` at call sites
/// where the value is already a temporary, e.g. `-(&a * &b)?`).
impl Neg for Matrix {
	type Output = Result<Matrix>;
	fn neg(self) -> Result<Matrix> {
		crate::matrix::neg(&self)
	}
}

/// `&mat + scalar` — add a scalar to every element.
impl Add<f32> for &Matrix {
	type Output = Result<Matrix>;
	fn add(self, rhs: f32) -> Result<Matrix> {
		crate::matrix::add_scalar(self, rhs)
	}
}

/// `&mat - scalar` — subtract a scalar from every element.
impl Sub<f32> for &Matrix {
	type Output = Result<Matrix>;
	fn sub(self, rhs: f32) -> Result<Matrix> {
		crate::matrix::sub_scalar(self, rhs)
	}
}

/// `&mat * scalar` — scale every element.
impl Mul<f32> for &Matrix {
	type Output = Result<Matrix>;
	fn mul(self, rhs: f32) -> Result<Matrix> {
		crate::matrix::scale(self, rhs)
	}
}

/// `&mat / scalar` — divide every element by a scalar.
impl Div<f32> for &Matrix {
	type Output = Result<Matrix>;
	fn div(self, rhs: f32) -> Result<Matrix> {
		crate::matrix::div_scalar(self, rhs)
	}
}

// scalar-on-left variants: `2.0_f32 * &mat`, `2.0_f32 + &mat`
// Rust does not automatically reverse binary operators, so these need
// explicit impls to match C++ implicit-conversion behaviour.

/// `scalar * &mat` — scale every element (left-hand scalar form).
impl Mul<&Matrix> for f32 {
	type Output = Result<Matrix>;
	fn mul(self, rhs: &Matrix) -> Result<Matrix> {
		crate::matrix::scale(rhs, self)
	}
}

/// `scalar + &mat` — add a scalar to every element (left-hand scalar form).
impl Add<&Matrix> for f32 {
	type Output = Result<Matrix>;
	fn add(self, rhs: &Matrix) -> Result<Matrix> {
		crate::matrix::add_scalar(rhs, self)
	}
}
