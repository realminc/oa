//! Dataset values, batch iteration, and stateless dataset operations.
//!
//! Donor: `oa/data/dataset.h`, `oa/data/dsSubset.h`, `oa/data/fnDataset.h`

use crate::{Error, Matrix, Result, matrix};

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
	pub fn with_label(x: Matrix, y: Matrix) -> Self {
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

/// Batched output: `x` \[B, …\] and optional `y` \[B, …\].
///
/// Corresponds to `oa::DataLoader::Batch`.
pub struct Batch {
	/// Batched input matrix with a leading batch dimension.
	pub x: Matrix,
	/// Batched label matrix with a leading batch dimension, or `None`.
	pub y: Option<Matrix>,
}

impl Batch {
	/// Return `true` when the batch contains at least one sample.
	pub fn is_valid(&self) -> bool {
		self.x.num_elements() > 0
	}
}

/// Configuration for [`DataLoader`].
///
/// Corresponds to `oa::DataLoaderConfig`.
#[derive(Clone, Copy, Debug)]
pub struct DataLoaderConfig {
	/// Number of samples per batch.
	pub batch_size: i32,
	/// Whether to shuffle indices before each epoch.
	pub shuffle: bool,
	/// Seed for shuffling.  `0` derives a seed from the system clock, as in the donor.
	pub seed: u64,
	/// Drop the last incomplete batch when `true`.
	pub drop_last: bool,
}

impl Default for DataLoaderConfig {
	fn default() -> Self {
		Self {
			batch_size: 32,
			shuffle: true,
			seed: 0,
			drop_last: false,
		}
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

/// Result of a train / validation / test split.
///
/// Corresponds to `oa::FnDataset::SplitResult`.
pub struct SplitResult {
	/// Training indices.
	pub train: Vec<i64>,
	/// Validation indices.
	pub val: Vec<i64>,
	/// Test indices.
	pub test: Vec<i64>,
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
/// one engine owner. A one-sample batch reuses its Matrix storage through
/// zero-copy views. No default Engine or host readback is used.
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
				if !value.engine_handle().same_as(first.x.engine_handle()) {
					return Err(Error::invalid_argument(
						"collate inputs and labels must share one engine",
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
		|| !value.engine_handle().same_as(reference.engine_handle())
	{
		return Err(Error::invalid_argument(
			"collate requires nonempty, same-shape, same-dtype, same-engine components",
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
