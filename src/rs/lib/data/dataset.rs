//! Dataset values, batch iteration, and stateless dataset operations.
//!
//! Donor: `oa/data/dataset.h`, `oa/data/dsSubset.h`, `oa/data/fnDataset.h`

pub use super::types::{DataLoaderConfig, SplitResult};
pub use crate::data::types::Batch;
use crate::{DType, Error, Matrix, Result, Runtime, matrix};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;

/// Multi-tensor sample: input `x` and optional label `y`.
///
/// Corresponds to `oa::Dataset::Sample`.
pub struct Sample {
	/// Input value.
	pub x: Matrix,
	/// Label / target value.  `None` when the dataset has no labels.
	pub y: Option<Matrix>,
}

impl Sample {
	/// Construct an unlabeled sample.
	pub fn new(x: Matrix) -> Self {
		Self { x, y: None }
	}

	/// Construct a labeled sample.
	pub fn new_with_label(x: Matrix, y: Matrix) -> Self {
		Self { x, y: Some(y) }
	}

	/// Return `true` when a label is present.
	pub fn has_label(&self) -> bool {
		self.y.is_some()
	}
}

/// Object-safe dataset contract: indexed access and a total count.
///
/// Corresponds to the C++ pure-virtual `oa::Dataset` base class.
pub trait Dataset {
	/// Total number of items in the dataset.
	fn len(&self) -> i64;

	/// Return `true` when the dataset contains no items.
	fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Retrieve one sample by index.
	///
	/// The default implementation wraps [`Dataset::get_item`] for single-output
	/// datasets.  Override to return a labeled sample.
	fn get_sample(&self, index: i64) -> Result<Sample> {
		Ok(Sample::new(self.get_item(index)?))
	}

	/// Retrieve one input `Matrix` by index.
	fn get_item(&self, index: i64) -> Result<Matrix>;

	/// Return a zero-copy indexed view over this dataset.
	///
	/// Corresponds to `oa::DsSubset`.
	///
	/// # Errors
	///
	/// Returns an error for a negative parent length or an out-of-range index.
	fn subset(&self, indices: Vec<i64>) -> Result<Subset<'_>>
	where
		Self: Sized,
	{
		Subset::new(self, indices)
	}

	/// Return a zero-copy view over every item in deterministic [`shuffle`]
	/// order. A nonzero seed reproduces the same permutation on each call.
	///
	/// # Errors
	///
	/// Returns the index-building, shuffle, and subset errors of the pieces
	/// it composes.
	fn shuffled_subset(&self, seed: u64) -> Result<Subset<'_>>
	where
		Self: Sized,
	{
		let mut indices = make_indices(checked_len(self)?)?;
		shuffle(&mut indices, seed)?;
		self.subset(indices)
	}
}

/// Dense labeled feature/label pair implementing [`Dataset`].
///
/// `x` is a rank-≥1 `[N, ...]` F32 feature matrix and `y` a `[N]` F32
/// label vector on the same runtime. `get_item` returns the rank-(N-1)
/// feature slice; `get_sample` pairs it with the `[1]` label slice —
/// the contract [`DataLoader`]/[`collate`] expect for labeled corpora.
/// Dataset packs and synthetic benchmarks build their `Dataset` impl on
/// this type instead of repeating bounds, slice, and pairing logic.
pub struct MatrixDataset {
	x: Matrix,
	y: Matrix,
}

impl MatrixDataset {
	/// Wrap caller-owned matrices as a labeled dataset.
	///
	/// `x` must have a leading sample dimension and `y` must be rank-1
	/// with the same leading count; both must be F32 on one runtime —
	/// [`matrix::slice`] is F32-only.
	///
	/// # Errors
	///
	/// Returns an error for a rank-0 feature matrix, non-vector labels,
	/// mismatched leading counts, non-F32 dtypes, or matrices owned by
	/// different runtimes.
	pub fn new(x: Matrix, y: Matrix) -> Result<Self> {
		if x.shape().is_empty() || y.shape().len() != 1 || x.shape()[0] != y.shape()[0] {
			return Err(Error::invalid_argument(
				"matrix dataset requires [N, ...] features and [N] labels with equal counts",
			));
		}
		if x.dtype() != crate::DType::F32 || y.dtype() != crate::DType::F32 {
			return Err(Error::invalid_argument(
				"matrix dataset requires F32 features and labels",
			));
		}
		if !x.same_runtime(&y) {
			return Err(Error::invalid_argument(
				"matrix dataset features and labels must share one runtime",
			));
		}
		Ok(Self { x, y })
	}

	/// Full `[N, ...]` feature matrix on the owning runtime.
	#[must_use]
	pub const fn x(&self) -> &Matrix {
		&self.x
	}

	/// Full `[N]` label matrix on the owning runtime.
	#[must_use]
	pub const fn y(&self) -> &Matrix {
		&self.y
	}
}

impl Dataset for MatrixDataset {
	fn len(&self) -> i64 {
		i64::try_from(self.x.shape()[0]).unwrap_or(i64::MAX)
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		if index < 0 || index >= self.len() {
			return Err(Error::out_of_range(
				"matrix dataset index is outside the dataset",
			));
		}
		matrix::slice(&self.x, 0, index, index + 1)?.squeeze(0)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		let x = self.get_item(index)?;
		let y = matrix::slice(&self.y, 0, index, index + 1)?;
		Ok(Sample::new_with_label(x, y))
	}
}

/// Zero-copy indexed view returned by [`Dataset::subset`].
///
/// Used for train/val/test splits, subsampling, and index filtering.
/// `get_item(i)` forwards to `parent.get_item(indices[i])` without copying data.
///
/// Corresponds to `oa::DsSubset`.
pub struct Subset<'ds> {
	parent: &'ds dyn Dataset,
	indices: Vec<i64>,
}

impl<'ds> Subset<'ds> {
	/// Borrow a dataset, including a trait object, with checked parent indices.
	///
	/// # Errors
	///
	/// Returns an error for a negative parent length or an out-of-range index.
	pub fn new(parent: &'ds dyn Dataset, indices: Vec<i64>) -> Result<Self> {
		let count = checked_len(parent)?;
		if indices.iter().any(|&index| index < 0 || index >= count) {
			return Err(Error::out_of_range(
				"subset index is outside the parent dataset",
			));
		}
		Ok(Self { parent, indices })
	}

	fn parent_index(&self, index: i64) -> Result<i64> {
		usize::try_from(index)
			.ok()
			.and_then(|index| self.indices.get(index))
			.copied()
			.ok_or_else(|| Error::out_of_range("index is outside the subset"))
	}

	/// Read-only view of the stored index list.
	pub fn indices(&self) -> &[i64] {
		&self.indices
	}

	/// Borrow the parent dataset.
	pub fn parent(&self) -> &dyn Dataset {
		self.parent
	}
}

impl Dataset for Subset<'_> {
	fn len(&self) -> i64 {
		self.indices.len() as i64
	}

	fn get_item(&self, index: i64) -> Result<Matrix> {
		self.parent.get_item(self.parent_index(index)?)
	}

	fn get_sample(&self, index: i64) -> Result<Sample> {
		self.parent.get_sample(self.parent_index(index)?)
	}
}

impl Batch {
	/// Return `true` when the batch contains at least one sample.
	pub fn is_valid(&self) -> bool {
		self.x.num_elements() > 0
	}
}

/// Stateful batch iterator over a borrowed [`Dataset`].
///
/// Corresponds to `oa::DataLoader`.
pub struct DataLoader<'ds> {
	dataset: &'ds dyn Dataset,
	config: DataLoaderConfig,
	indices: Vec<i64>,
	current_batch: i64,
}

impl<'ds> DataLoader<'ds> {
	/// Construct a data loader and build the initial index order.
	///
	/// # Errors
	///
	/// Returns an error for a non-positive batch size, a negative dataset length,
	/// index allocation failure, or an unavailable clock for seed zero.
	pub fn new(dataset: &'ds dyn Dataset, config: DataLoaderConfig) -> Result<Self> {
		if config.batch_size <= 0 {
			return Err(Error::invalid_argument(
				"data loader batch size must be positive",
			));
		}
		let mut loader = Self {
			dataset,
			config,
			indices: Vec::new(),
			current_batch: 0,
		};
		loader.reset()?;
		Ok(loader)
	}

	/// Advance to the next batch; `None` means the epoch is exhausted.
	///
	/// # Errors
	///
	/// Sample and collation failures are propagated without dropping samples or
	/// advancing the batch cursor. A caller may retry or explicitly reset.
	pub fn next_batch(&mut self) -> Result<Option<Batch>> {
		if self.current_batch >= self.num_batches() {
			return Ok(None);
		}
		// The cursor is below num_batches, so start is inside the allocated index list.
		let start = self.current_batch as usize * self.config.batch_size as usize;
		let count = (self.config.batch_size as usize).min(self.indices.len() - start);
		let samples = self.indices[start..start + count]
			.iter()
			.map(|&index| self.dataset.get_sample(index))
			.collect::<Result<Vec<_>>>()?;
		let batch = collate(&samples)?;
		self.current_batch += 1;
		Ok(Some(batch))
	}

	/// Reset the epoch, rebuilding its index order from the configured seed.
	///
	/// Nonzero seeds reproduce the same order on every reset, matching the donor.
	///
	/// # Errors
	///
	/// Returns the index-building errors described by [`Self::new`]. A failed
	/// reset leaves the existing order and cursor unchanged.
	pub fn reset(&mut self) -> Result<()> {
		let mut indices = make_indices(checked_len(self.dataset)?)?;
		if self.config.shuffle {
			shuffle(&mut indices, self.config.seed)?;
		}
		self.indices = indices;
		self.current_batch = 0;
		Ok(())
	}

	/// Reset for a new data pass, deriving the shuffle order from the
	/// configured seed mixed with the epoch number so each epoch iterates a
	/// different deterministic order — the equivalent of reshuffling a
	/// dataset between epochs. `reset_epoch(0)` reproduces [`Self::reset`].
	///
	/// # Errors
	///
	/// Returns the index-building errors described by [`Self::new`]. A failed
	/// reset leaves the existing order and cursor unchanged.
	pub fn reset_epoch(&mut self, epoch: u64) -> Result<()> {
		let mut indices = make_indices(checked_len(self.dataset)?)?;
		if self.config.shuffle {
			shuffle(&mut indices, self.config.seed.wrapping_add(epoch))?;
		}
		self.indices = indices;
		self.current_batch = 0;
		Ok(())
	}

	/// Number of batches in the current epoch.
	pub fn num_batches(&self) -> i64 {
		let n = self.indices.len() as i64;
		let b = i64::from(self.config.batch_size);
		n / b + i64::from(!self.config.drop_last && n % b != 0)
	}

	/// Index of the next batch that will be returned.
	pub fn current_batch(&self) -> i64 {
		self.current_batch
	}
}

/// Fisher-Yates shuffle using the donor's PCG32 stream and unbiased rejection.
///
/// A nonzero seed reproduces the C++ `oa::Random(seed)` permutation. Zero uses
/// the system clock. This generator is for sampling, not cryptography.
///
/// # Errors
///
/// Returns an error when seed zero cannot be derived from the system clock.
pub fn shuffle(indices: &mut [i64], seed: u64) -> Result<()> {
	if indices.len() < 2 {
		return Ok(());
	}
	let seed = if seed == 0 {
		use std::time::{SystemTime, UNIX_EPOCH};
		SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.map_err(|_| Error::failed_precondition("dataset shuffle clock precedes Unix epoch"))?
			.as_nanos() as u64
	} else {
		seed
	};
	let mut rng = Pcg32::new(seed);
	for count in (2..=indices.len()).rev() {
		let range = count as u64;
		let reject = range.wrapping_neg() % range;
		let value = loop {
			let value = rng.next_u64();
			if value >= reject {
				break value;
			}
		};
		indices.swap(count - 1, (value % range) as usize);
	}
	Ok(())
}

/// Randomly split indices into train, validation, and test subsets.
///
/// Matches donor `FnDataset::randomSplit`: clamp train to `[0, 1]` and validation
/// to `[0, 1 - train]`, truncate their FP32 counts, and assign the remainder to
/// test. The donor header's claim that the remainder goes to train contradicts
/// its implementation; this port follows the implementation.
///
/// # Errors
///
/// Returns an error for a negative size, non-finite ratios, allocation failure,
/// or failure to derive a clock seed when seed is zero.
pub fn random_split(
	total_size: i64,
	train_ratio: f32,
	val_ratio: f32,
	seed: u64,
) -> Result<SplitResult> {
	if !train_ratio.is_finite() || !val_ratio.is_finite() {
		return Err(Error::invalid_argument(
			"dataset split ratios must be finite",
		));
	}
	let mut all = make_indices(total_size)?;
	let train_ratio = train_ratio.clamp(0.0, 1.0);
	let val_ratio = val_ratio.clamp(0.0, 1.0 - train_ratio);
	let n = all.len();
	let train_count = ((n as f32 * train_ratio) as usize).min(n);
	let val_count = ((n as f32 * val_ratio) as usize).min(n - train_count);
	shuffle(&mut all, seed)?;
	let test = all.split_off(train_count + val_count);
	let val = all.split_off(train_count);
	Ok(SplitResult {
		train: all,
		val,
		test,
	})
}

/// Stack samples into a leading batch dimension using device-side concatenation.
///
/// Inputs must be nonempty matrices of rank at most three, with any admitted
/// Matrix dtype (U8, F32, I32, or U32). Labels must be present on every sample or
/// absent on every sample. Each component must have equal shapes, dtypes, and
/// one runtime owner. A one-sample batch reuses its Matrix storage through
/// zero-copy views. No default Runtime or host readback is used.
///
/// # Errors
///
/// Returns an error for empty input, inconsistent samples, unsupported rank,
/// allocation failure, or GPU recording failure.
pub fn collate(samples: &[Sample]) -> Result<Batch> {
	let first = samples
		.first()
		.ok_or_else(|| Error::invalid_argument("collate requires at least one sample"))?;
	for sample in samples {
		validate_component(&sample.x, &first.x)?;
		match (&sample.y, &first.y) {
			(Some(value), Some(reference)) => {
				validate_component(value, reference)?;
				if !value.runtime().same_as(first.x.runtime()) {
					return Err(Error::invalid_argument(
						"collate inputs and labels must share one runtime",
					));
				}
			}
			(None, None) => {}
			_ => {
				return Err(Error::invalid_argument(
					"collate label presence must match for every sample",
				));
			}
		}
	}
	if let [only] = samples {
		return Ok(Batch {
			x: only.x.unsqueeze(0)?,
			y: only
				.y
				.as_ref()
				.map(|value| value.unsqueeze(0))
				.transpose()?,
		});
	}
	let x_views = samples
		.iter()
		.map(|sample| sample.x.unsqueeze(0))
		.collect::<Result<Vec<_>>>()?;
	let y_views = samples
		.iter()
		.filter_map(|sample| sample.y.as_ref())
		.map(|value| value.unsqueeze(0))
		.collect::<Result<Vec<_>>>()?;
	let x = matrix::concat(&x_views, 0)?;
	let y = if y_views.is_empty() {
		None
	} else {
		Some(matrix::concat(&y_views, 0)?)
	};
	Ok(Batch { x, y })
}

fn validate_component(value: &Matrix, reference: &Matrix) -> Result<()> {
	if value.shape().len() > 3 {
		return Err(Error::invalid_argument(
			"collate requires samples of rank at most three",
		));
	}
	if value.num_elements() == 0
		|| value.shape() != reference.shape()
		|| value.dtype() != reference.dtype()
		|| !value.runtime().same_as(reference.runtime())
	{
		return Err(Error::invalid_argument(
			"collate requires nonempty, same-shape, same-dtype, same-runtime components",
		));
	}
	Ok(())
}

fn checked_len(dataset: &dyn Dataset) -> Result<i64> {
	let count = dataset.len();
	if count < 0 {
		return Err(Error::invalid_argument(
			"dataset length must be nonnegative",
		));
	}
	Ok(count)
}

fn make_indices(count: i64) -> Result<Vec<i64>> {
	let length = usize::try_from(count)
		.map_err(|_| Error::invalid_argument("dataset size must fit usize and be nonnegative"))?;
	let mut indices = Vec::new();
	indices
		.try_reserve_exact(length)
		.map_err(|_| Error::resource_exhausted("dataset index allocation failed"))?;
	indices.extend(0..count);
	Ok(indices)
}

// Donor: oa/core/std/random.h, Random. Mechanical integer/ownership adaptation;
// retain its PCG32 sequence=1 and two-draw nextU64 order for seed compatibility.
struct Pcg32 {
	state: u64,
}

impl Pcg32 {
	fn new(seed: u64) -> Self {
		let mut rng = Self { state: 0 };
		rng.next_u32();
		rng.state = rng.state.wrapping_add(seed);
		rng.next_u32();
		rng
	}

	fn next_u32(&mut self) -> u32 {
		let old = self.state;
		self.state = old.wrapping_mul(6364136223846793005).wrapping_add(3);
		// Truncation to 32 bits is part of the donor PCG permutation.
		let shifted = (((old >> 18) ^ old) >> 27) as u32;
		shifted.rotate_right((old >> 59) as u32)
	}

	fn next_u64(&mut self) -> u64 {
		(u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())
	}
}

// ---- Async prefetch ---------------------------------------------------------

/// Typed host values carried by worker-produced samples and batches.
///
/// The runtime and `Matrix` are `!Send`, so prefetch production happens on
/// host values; the consumer uploads a collated [`HostBatch`] on its own
/// thread. Variants cover the admitted dataset element types; `F16` host
/// staging is not admitted because host readback never produces half values.
#[derive(Clone, Debug)]
pub enum HostValues {
	/// F32 elements.
	F32(Vec<f32>),
	/// U32 elements.
	U32(Vec<u32>),
	/// I32 elements.
	I32(Vec<i32>),
	/// U8 elements.
	U8(Vec<u8>),
}

impl HostValues {
	fn dtype(&self) -> DType {
		match self {
			Self::F32(_) => DType::F32,
			Self::U32(_) => DType::U32,
			Self::I32(_) => DType::I32,
			Self::U8(_) => DType::U8,
		}
	}

	fn len(&self) -> usize {
		match self {
			Self::F32(values) => values.len(),
			Self::U32(values) => values.len(),
			Self::I32(values) => values.len(),
			Self::U8(values) => values.len(),
		}
	}

	fn slice(&self, start: usize, count: usize) -> Self {
		match self {
			Self::F32(values) => Self::F32(values[start..start + count].to_vec()),
			Self::U32(values) => Self::U32(values[start..start + count].to_vec()),
			Self::I32(values) => Self::I32(values[start..start + count].to_vec()),
			Self::U8(values) => Self::U8(values[start..start + count].to_vec()),
		}
	}

	fn concat(parts: &[&Self]) -> Result<Self> {
		let first = parts
			.first()
			.ok_or_else(|| Error::invalid_argument("host concat requires a part"))?;
		Ok(match first {
			Self::F32(_) => {
				let mut values = Vec::new();
				for part in parts {
					let Self::F32(part) = part else {
						return Err(Error::invalid_argument(
							"host concat requires one dtype across parts",
						));
					};
					values.extend_from_slice(part);
				}
				Self::F32(values)
			}
			Self::U32(_) => {
				let mut values = Vec::new();
				for part in parts {
					let Self::U32(part) = part else {
						return Err(Error::invalid_argument(
							"host concat requires one dtype across parts",
						));
					};
					values.extend_from_slice(part);
				}
				Self::U32(values)
			}
			Self::I32(_) => {
				let mut values = Vec::new();
				for part in parts {
					let Self::I32(part) = part else {
						return Err(Error::invalid_argument(
							"host concat requires one dtype across parts",
						));
					};
					values.extend_from_slice(part);
				}
				Self::I32(values)
			}
			Self::U8(_) => {
				let mut values = Vec::new();
				for part in parts {
					let Self::U8(part) = part else {
						return Err(Error::invalid_argument(
							"host concat requires one dtype across parts",
						));
					};
					values.extend_from_slice(part);
				}
				Self::U8(values)
			}
		})
	}
}

/// A dense tensor living in host memory.
///
/// Shape semantics match `Matrix`: `num_elements` is the shape product and
/// `shape[0]` is the leading (sample or batch) dimension.
#[derive(Clone, Debug)]
pub struct HostTensor {
	/// Logical shape.
	pub shape: Vec<usize>,
	/// Element values in row-major order.
	pub values: HostValues,
}

impl HostTensor {
	/// Construct a host tensor, checking the element count against the shape.
	///
	/// # Errors
	///
	/// Returns an error when the shape product overflows or differs from the
	/// value count.
	pub fn new(shape: impl Into<Vec<usize>>, values: HostValues) -> Result<Self> {
		let tensor = Self {
			shape: shape.into(),
			values,
		};
		tensor.check_count()?;
		Ok(tensor)
	}

	/// Element count implied by the shape.
	pub fn num_elements(&self) -> usize {
		self.shape.iter().product()
	}

	/// Element dtype.
	pub fn dtype(&self) -> DType {
		self.values.dtype()
	}

	fn check_count(&self) -> Result<()> {
		if self.num_elements() != self.values.len() {
			return Err(Error::invalid_argument(
				"host tensor shape and value count disagree",
			));
		}
		Ok(())
	}

	/// Upload to `runtime`, producing a device `Matrix`.
	///
	/// # Errors
	///
	/// Returns an error when storage allocation or upload fails.
	pub fn to_matrix_on(&self, runtime: &Runtime) -> Result<Matrix> {
		self.check_count()?;
		match &self.values {
			HostValues::F32(values) => Matrix::new_from_slice_on(runtime, self.shape.clone(), values),
			HostValues::U32(values) => Matrix::new_from_slice_on(runtime, self.shape.clone(), values),
			HostValues::I32(values) => Matrix::new_from_slice_on(runtime, self.shape.clone(), values),
			HostValues::U8(values) => Matrix::new_from_slice_on(runtime, self.shape.clone(), values),
		}
	}
}

/// Host-side sample: input `x` and optional label `y`, as produced by
/// [`HostDataset::get_host_sample`].
#[derive(Clone, Debug)]
pub struct HostSample {
	/// Input value.
	pub x: HostTensor,
	/// Label / target value, absent for unlabeled datasets.
	pub y: Option<HostTensor>,
}

/// Collated host batch: `[B, ...]` tensors stacked along a new leading axis.
#[derive(Clone, Debug)]
pub struct HostBatch {
	/// Batched inputs.
	pub x: HostTensor,
	/// Batched labels, present iff every sample carried one.
	pub y: Option<HostTensor>,
}

impl HostBatch {
	/// Upload both components to `runtime`, producing a device [`Batch`].
	///
	/// # Errors
	///
	/// Returns an error when storage allocation or upload fails.
	pub fn upload_on(&self, runtime: &Runtime) -> Result<Batch> {
		Ok(Batch {
			x: self.x.to_matrix_on(runtime)?,
			y: self
				.y
				.as_ref()
				.map(|label| label.to_matrix_on(runtime))
				.transpose()?,
		})
	}
}

/// Collate host samples into one `[B, ...]` batch.
///
/// Mirrors [`collate`]: identical per-sample shapes and dtypes are required,
/// and label presence must match across the batch.
///
/// # Errors
///
/// Returns an error for an empty sample list or mismatched components.
pub fn collate_host(samples: &[HostSample]) -> Result<HostBatch> {
	let first = samples
		.first()
		.ok_or_else(|| Error::invalid_argument("collate_host requires at least one sample"))?;
	for sample in samples {
		validate_host_component(&sample.x, &first.x)?;
		match (&sample.y, &first.y) {
			(Some(value), Some(reference)) => validate_host_component(value, reference)?,
			(None, None) => {}
			_ => {
				return Err(Error::invalid_argument(
					"collate_host label presence must match for every sample",
				));
			}
		}
	}
	let stack = |tensors: Vec<&HostTensor>| -> Result<HostTensor> {
		let mut shape = tensors[0].shape.clone();
		shape.insert(0, tensors.len());
		HostTensor::new(
			shape,
			HostValues::concat(
				&tensors
					.iter()
					.map(|tensor| &tensor.values)
					.collect::<Vec<_>>(),
			)?,
		)
	};
	Ok(HostBatch {
		x: stack(samples.iter().map(|sample| &sample.x).collect())?,
		y: samples
			.iter()
			.filter_map(|sample| sample.y.as_ref())
			.collect::<Vec<_>>()
			.split_first()
			.map(|_| stack(samples.iter().filter_map(|s| s.y.as_ref()).collect()))
			.transpose()?,
	})
}

fn validate_host_component(value: &HostTensor, reference: &HostTensor) -> Result<()> {
	if value.num_elements() == 0
		|| value.shape != reference.shape
		|| value.dtype() != reference.dtype()
	{
		return Err(Error::invalid_argument(
			"collate_host requires nonempty, same-shape, same-dtype components",
		));
	}
	Ok(())
}

/// Thread-shareable host dataset contract for [`PrefetchDataLoader`].
///
/// `Runtime` is `!Send`, so async prefetch runs on host values: worker
/// threads index a `HostDataset`, collate host-side, and the consumer
/// thread uploads each batch to the device — the same split as
/// `DataLoader(num_workers>0, pin_memory=True)` in torch, where workers
/// also produce host tensors.
pub trait HostDataset: Send + Sync {
	/// Total number of items in the dataset.
	fn len(&self) -> i64;

	/// Return `true` when the dataset contains no items.
	fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Retrieve one host sample by index.
	///
	/// # Errors
	///
	/// Returns an error for an out-of-range index or a failed read.
	fn get_host_sample(&self, index: i64) -> Result<HostSample>;
}

/// Dense labeled feature/label pair implementing [`HostDataset`].
///
/// `x` is a rank-≥1 `[N, ...]` host tensor and `y` a `[N]` label vector
/// with the same leading count and dtype class as [`collate_host`]
/// accepts. `get_host_sample` slices along the leading axis.
pub struct HostTensorDataset {
	x: HostTensor,
	y: HostTensor,
}

impl HostTensorDataset {
	/// Wrap caller-owned host tensors as a labeled dataset.
	///
	/// # Errors
	///
	/// Returns an error for a rank-0 feature tensor, non-vector labels, or
	/// mismatched leading counts.
	pub fn new(x: HostTensor, y: HostTensor) -> Result<Self> {
		if x.shape.is_empty() || y.shape.len() != 1 || x.shape[0] != y.shape[0] {
			return Err(Error::invalid_argument(
				"host tensor dataset requires [N, ...] features and [N] labels with equal counts",
			));
		}
		Ok(Self { x, y })
	}

	/// Materialize a host mirror of F32 device matrices.
	///
	/// Reads `x` and `y` back once on the calling thread; afterwards the
	/// dataset is pure host data that prefetch workers may index without
	/// touching the runtime.
	///
	/// # Errors
	///
	/// Returns an error for a rank-0 feature matrix, non-vector labels,
	/// mismatched leading counts, or a failed readback.
	pub fn from_f32_matrices(x: &Matrix, y: &Matrix) -> Result<Self> {
		if x.shape().is_empty() || y.shape().len() != 1 || x.shape()[0] != y.shape()[0] {
			return Err(Error::invalid_argument(
				"host tensor dataset requires [N, ...] features and [N] labels with equal counts",
			));
		}
		Self::new(
			HostTensor::new(x.shape().to_vec(), HostValues::F32(x.read_f32()?))?,
			HostTensor::new(y.shape().to_vec(), HostValues::F32(y.read_f32()?))?,
		)
	}
}

impl HostDataset for HostTensorDataset {
	fn len(&self) -> i64 {
		i64::try_from(self.x.shape[0]).unwrap_or(i64::MAX)
	}

	fn get_host_sample(&self, index: i64) -> Result<HostSample> {
		let count = self.len();
		if index < 0 || index >= count {
			return Err(Error::invalid_argument(
				"host tensor dataset index out of range",
			));
		}
		let index = usize::try_from(index)
			.map_err(|_| Error::invalid_argument("host tensor dataset index out of range"))?;
		let stride: usize = self.x.shape[1..].iter().product();
		Ok(HostSample {
			x: HostTensor::new(
				self.x.shape[1..].to_vec(),
				self.x.values.slice(index * stride, stride),
			)?,
			y: Some(HostTensor::new(vec![1], self.y.values.slice(index, 1))?),
		})
	}
}

/// Asynchronous batch iterator over a shared [`HostDataset`].
///
/// A single worker thread walks the epoch's index order, calls
/// `get_host_sample`, collates, and pushes [`HostBatch`] values into a
/// queue bounded at `prefetch_batches`. `next_batch` pops on the caller
/// thread, so sample production overlaps consumption; `next_batch_on`
/// additionally uploads the collated batch to the runtime. Shuffle order,
/// `drop_last`, and epoch resets share the [`DataLoader`] seed contract.
///
/// One worker is the admitted contract: the queue depth already decouples
/// production from consumption, and multi-worker partitioning would add
/// index ownership without changing the device boundary.
pub struct PrefetchDataLoader {
	dataset: Arc<dyn HostDataset>,
	config: DataLoaderConfig,
	prefetch_batches: usize,
	receiver: Option<mpsc::Receiver<Result<HostBatch>>>,
	worker: Option<JoinHandle<()>>,
	epoch: u64,
}

impl PrefetchDataLoader {
	/// Construct a prefetching loader and start the epoch-0 worker.
	///
	/// # Errors
	///
	/// Returns an error for a non-positive batch size, a zero prefetch
	/// depth, a negative dataset length, or index allocation failure.
	pub fn new(
		dataset: Arc<dyn HostDataset>,
		config: DataLoaderConfig,
		prefetch_batches: usize,
	) -> Result<Self> {
		if config.batch_size <= 0 {
			return Err(Error::invalid_argument(
				"prefetch data loader batch size must be positive",
			));
		}
		if prefetch_batches == 0 {
			return Err(Error::invalid_argument(
				"prefetch data loader requires at least one queued batch",
			));
		}
		let mut loader = Self {
			dataset,
			config,
			prefetch_batches,
			receiver: None,
			worker: None,
			epoch: 0,
		};
		loader.spawn_epoch(0)?;
		Ok(loader)
	}

	/// Advance to the next host batch; `None` means the epoch is exhausted.
	///
	/// The call blocks only when the queue is empty and the worker is still
	/// producing; a worker-side failure is delivered once and ends the
	/// epoch.
	///
	/// # Errors
	///
	/// Returns the first worker-side sample or collation error.
	pub fn next_batch(&mut self) -> Result<Option<HostBatch>> {
		let Some(receiver) = &self.receiver else {
			return Ok(None);
		};
		match receiver.recv() {
			Ok(Ok(batch)) => Ok(Some(batch)),
			Ok(Err(error)) => {
				self.receiver = None;
				Err(error)
			}
			Err(mpsc::RecvError) => {
				self.receiver = None;
				Ok(None)
			}
		}
	}

	/// Pop the next batch and upload it to `runtime`.
	///
	/// # Errors
	///
	/// Returns worker errors plus upload failures.
	pub fn next_batch_on(&mut self, runtime: &Runtime) -> Result<Option<Batch>> {
		self
			.next_batch()?
			.map(|batch| batch.upload_on(runtime))
			.transpose()
	}

	/// Reset for a new data pass, deriving the shuffle order from the
	/// configured seed mixed with the epoch number — matching
	/// [`DataLoader::reset_epoch`]. A new worker replaces the old one;
	/// queued batches from the previous epoch are discarded.
	///
	/// # Errors
	///
	/// Returns the index-building errors described by [`Self::new`].
	pub fn reset_epoch(&mut self, epoch: u64) -> Result<()> {
		self.spawn_epoch(epoch)
	}

	/// Number of batches in the current epoch.
	pub fn num_batches(&self) -> i64 {
		let n = self.dataset.len().max(0);
		let b = i64::from(self.config.batch_size);
		n / b + i64::from(!self.config.drop_last && n % b != 0)
	}

	fn spawn_epoch(&mut self, epoch: u64) -> Result<()> {
		let mut indices = make_indices_host(self.dataset.len())?;
		if self.config.shuffle {
			shuffle(&mut indices, self.config.seed.wrapping_add(epoch))?;
		}
		let (sender, receiver) = mpsc::sync_channel(self.prefetch_batches);
		let dataset = Arc::clone(&self.dataset);
		let batch_size = usize::try_from(self.config.batch_size)
			.map_err(|_| Error::invalid_argument("batch size must fit usize"))?;
		let drop_last = self.config.drop_last;
		self.receiver = Some(receiver);
		self.worker = Some(std::thread::spawn(move || {
			let mut start = 0_usize;
			while start < indices.len() {
				let count = batch_size.min(indices.len() - start);
				if drop_last && count < batch_size {
					break;
				}
				let produced = indices[start..start + count]
					.iter()
					.map(|&index| dataset.get_host_sample(index))
					.collect::<Result<Vec<_>>>()
					.and_then(|samples| collate_host(&samples));
				let keep_going = produced.is_ok();
				if sender.send(produced).is_err() || !keep_going {
					return;
				}
				start += count;
			}
		}));
		self.epoch = epoch;
		Ok(())
	}
}

fn make_indices_host(count: i64) -> Result<Vec<i64>> {
	let length = usize::try_from(count)
		.map_err(|_| Error::invalid_argument("dataset size must fit usize and be nonnegative"))?;
	let mut indices = Vec::new();
	indices
		.try_reserve_exact(length)
		.map_err(|_| Error::resource_exhausted("dataset index allocation failed"))?;
	indices.extend(0..count);
	Ok(indices)
}
