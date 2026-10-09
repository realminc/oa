//! Dataset, DataLoader, Subset, and stateless operation contracts.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

// ---- shuffle ----------------------------------------------------------------

#[test]
fn shuffle_is_deterministic_for_nonzero_seed() {
	let mut a: Vec<i64> = (0..8).collect();
	let mut b: Vec<i64> = (0..8).collect();
	oa::data::shuffle(&mut a, 42).expect("shuffle");
	oa::data::shuffle(&mut b, 42).expect("shuffle");
	assert_eq!(a, b);
	assert_ne!(a, (0..8_i64).collect::<Vec<_>>());
}

#[test]
fn shuffle_produces_a_permutation() {
	let mut indices: Vec<i64> = (0..16).collect();
	oa::data::shuffle(&mut indices, 7).expect("shuffle");
	let mut sorted = indices.clone();
	sorted.sort_unstable();
	assert_eq!(sorted, (0..16_i64).collect::<Vec<_>>());
}

// ---- random_split -----------------------------------------------------------

#[test]
fn random_split_sizes_match_ratios() {
	let split = oa::data::random_split(100, 0.8, 0.1, 42).expect("split");
	// Donor implementation floors train and val; the remainder goes to test.
	assert_eq!(split.val.len(), 10);
	assert_eq!(split.train.len() + split.val.len() + split.test.len(), 100);
	assert_eq!(split.train.len(), 80);
	assert_eq!(split.test.len(), 10);
}

#[test]
fn random_split_all_indices_accounted_for() {
	let split = oa::data::random_split(50, 0.7, 0.15, 1).expect("split");
	let mut all: Vec<i64> = split
		.train
		.iter()
		.chain(&split.val)
		.chain(&split.test)
		.copied()
		.collect();
	all.sort_unstable();
	assert_eq!(all, (0..50_i64).collect::<Vec<_>>());
}

#[test]
fn random_split_empty_input_returns_empty() {
	let split = oa::data::random_split(0, 0.8, 0.1, 42).expect("split");
	assert!(split.train.is_empty());
	assert!(split.val.is_empty());
	assert!(split.test.is_empty());
}

// ---- Sample -----------------------------------------------------------------

test_vk!(sample_has_label_reflects_constructor, engine, {
	let x = oa::Matrix::from_f32(&engine, [2], &[1.0, 2.0])?;
	let y = oa::Matrix::from_f32(&engine, [2], &[0.0, 1.0])?;
	let unlabeled = oa::Sample::new(x.clone());
	assert!(!unlabeled.has_label());
	let labeled = oa::Sample::with_label(x, y);
	assert!(labeled.has_label());
	Ok(())
});

// ---- Dataset::subset --------------------------------------------------------

struct VecDataset<'e> {
	data: Vec<Vec<f32>>,
	engine: &'e oa::Engine,
}

impl oa::Dataset for VecDataset<'_> {
	fn len(&self) -> i64 {
		self.data.len() as i64
	}

	fn get_item(&self, index: i64) -> oa::Result<oa::Matrix> {
		let row = &self.data[index as usize];
		oa::Matrix::from_f32(self.engine, [row.len()], row)
	}
}

test_vk!(ds_subset_wraps_parent_correctly, engine, {
	use oa::Dataset;
	let parent = VecDataset {
		data: vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
		engine: &engine,
	};
	let subset = parent.subset(vec![3, 1])?;
	assert_eq!(subset.len(), 2);
	assert_eq!(subset.get_item(0)?.read_f32()?, vec![4.0]);
	assert_eq!(subset.get_item(1)?.read_f32()?, vec![2.0]);
	Ok(())
});

// ---- DataLoader -------------------------------------------------------------

test_vk!(data_loader_exhausts_epoch_and_resets, engine, {
	let parent = VecDataset {
		data: (0..10).map(|i| vec![i as f32]).collect(),
		engine: &engine,
	};
	let config = oa::DataLoaderConfig {
		batch_size: 3,
		shuffle: false,
		seed: 0,
		drop_last: false,
	};
	let mut loader = oa::DataLoader::new(&parent, config)?;
	assert_eq!(loader.num_batches(), 4); // ceil(10 / 3)
	let mut count = 0;
	while loader.next_batch()?.is_some() {
		count += 1;
	}
	assert_eq!(count, 4);
	assert!(loader.next_batch()?.is_none());
	loader.reset()?;
	assert_eq!(loader.current_batch(), 0);
	let batch = loader.next_batch()?;
	assert!(batch.is_some());
	assert!(batch.unwrap().is_valid());
	Ok(())
});

test_vk!(data_loader_drop_last_drops_incomplete_batch, engine, {
	let parent = VecDataset {
		data: (0..10).map(|i| vec![i as f32]).collect(),
		engine: &engine,
	};
	let config = oa::DataLoaderConfig {
		batch_size: 3,
		shuffle: false,
		seed: 0,
		drop_last: true,
	};
	let loader = oa::DataLoader::new(&parent, config)?;
	assert_eq!(loader.num_batches(), 3); // floor(10 / 3)
	Ok(())
});

// ---- collate ----------------------------------------------------------------

test_vk!(collate_stacks_x_along_batch_dim, engine, {
	let samples: Vec<oa::Sample> = (0..4_i64)
		.map(|i| {
			let x = oa::Matrix::from_f32(&engine, [3], &[i as f32; 3]).expect("from_f32 in collate test");
			oa::Sample::new(x)
		})
		.collect();
	let batch = oa::data::collate(&samples)?;
	assert!(batch.is_valid());
	assert_eq!(batch.x.shape(), [4, 3]);
	assert!(batch.y.is_none());
	Ok(())
});

test_vk!(collate_stacks_y_when_all_labeled, engine, {
	let samples: Vec<oa::Sample> = (0..3_i64)
		.map(|i| {
			let x =
				oa::Matrix::from_f32(&engine, [2], &[i as f32; 2]).expect("from_f32 x in labeled collate");
			let y =
				oa::Matrix::from_f32(&engine, [1], &[i as f32]).expect("from_f32 y in labeled collate");
			oa::Sample::with_label(x, y)
		})
		.collect();
	let batch = oa::data::collate(&samples)?;
	assert!(batch.is_valid());
	assert_eq!(batch.x.shape(), [3, 2]);
	assert!(batch.y.is_some());
	assert_eq!(batch.y.unwrap().shape(), [3, 1]);
	Ok(())
});

test_vk!(collate_stacks_integer_labels_and_byte_samples, engine, {
	let first = oa::Sample::with_label(
		oa::Matrix::from_slice(&engine, [3], &[1_u8, 2, 3])?,
		oa::Matrix::from_slice(&engine, [], &[-4_i32])?,
	);
	let second = oa::Sample::with_label(
		oa::Matrix::from_slice(&engine, [3], &[4_u8, 5, 6])?,
		oa::Matrix::from_slice(&engine, [], &[7_i32])?,
	);
	let batch = oa::data::collate(&[first, second])?;
	assert_eq!(batch.x.shape(), [2, 3]);
	assert_eq!(batch.x.read::<u8>()?, vec![1, 2, 3, 4, 5, 6]);
	let labels = batch.y.expect("both samples have labels");
	assert_eq!(labels.shape(), [2]);
	assert_eq!(labels.read::<i32>()?, vec![-4, 7]);
	Ok(())
});

// These regressions are host-only: invalid data must not initialize Vulkan.
struct FailingDataset {
	length: i64,
	calls: std::cell::Cell<usize>,
}

impl oa::Dataset for FailingDataset {
	fn len(&self) -> i64 {
		self.length
	}
	fn get_item(&self, _: i64) -> oa::Result<oa::Matrix> {
		self.calls.set(self.calls.get() + 1);
		Err(oa::Error::callback("sample decode failed"))
	}
}

#[test]
fn invalid_loader_and_subset_inputs_fail_without_gpu() -> oa::Result<()> {
	use oa::Dataset;
	let dataset = FailingDataset {
		length: 2,
		calls: Default::default(),
	};
	for batch_size in [0, -1, i32::MIN] {
		let error = oa::DataLoader::new(
			&dataset,
			oa::DataLoaderConfig {
				batch_size,
				..Default::default()
			},
		)
		.err()
		.expect("invalid size");
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}
	for indices in [vec![-1], vec![2], vec![i64::MAX]] {
		assert_eq!(
			dataset.subset(indices).err().expect("invalid index").kind(),
			oa::ErrorKind::OutOfRange
		);
	}
	let subset = dataset.subset(vec![1])?;
	for index in [-1, 1, i64::MAX] {
		assert_eq!(
			subset.get_item(index).err().expect("invalid index").kind(),
			oa::ErrorKind::OutOfRange
		);
		assert_eq!(
			subset
				.get_sample(index)
				.err()
				.expect("invalid index")
				.kind(),
			oa::ErrorKind::OutOfRange
		);
	}
	assert_eq!(dataset.calls.get(), 0);
	let invalid = FailingDataset {
		length: -1,
		calls: Default::default(),
	};
	assert!(oa::DataLoader::new(&invalid, Default::default()).is_err());
	assert!(invalid.subset(vec![]).is_err());
	assert_eq!(
		oa::data::collate(&[]).err().expect("empty batch").kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
}

#[test]
fn loader_preserves_sample_errors_and_cursor() -> oa::Result<()> {
	let dataset = FailingDataset {
		length: 5,
		calls: Default::default(),
	};
	let mut loader = oa::DataLoader::new(
		&dataset,
		oa::DataLoaderConfig {
			batch_size: 2,
			shuffle: false,
			..Default::default()
		},
	)?;
	assert_eq!(loader.num_batches(), 3);
	for attempt in 1..=2 {
		let error = loader
			.next_batch()
			.err()
			.expect("sample failure must propagate");
		assert_eq!(error.kind(), oa::ErrorKind::CallbackFailure);
		assert_eq!(error.message(), "sample decode failed");
		assert_eq!(loader.current_batch(), 0);
		assert_eq!(dataset.calls.get(), attempt);
	}
	Ok(())
}

#[test]
fn empty_loader_is_exhausted_without_gpu() -> oa::Result<()> {
	let dataset = FailingDataset {
		length: 0,
		calls: Default::default(),
	};
	let mut loader = oa::DataLoader::new(&dataset, Default::default())?;
	assert_eq!(loader.num_batches(), 0);
	assert!(loader.next_batch()?.is_none());
	loader.reset()?;
	assert!(loader.next_batch()?.is_none());
	assert_eq!(dataset.calls.get(), 0);
	Ok(())
}

#[test]
fn split_clamps_validation_after_train_and_rejects_nonfinite_ratios() -> oa::Result<()> {
	let split = oa::data::random_split(10, 0.8, 0.8, 42)?;
	assert_eq!(split.train.len(), 8);
	// FP32 donor multiplication truncates 10 * (1 - 0.8) to one.
	assert_eq!(split.val.len(), 1);
	assert_eq!(split.test.len(), 1);
	let split = oa::data::random_split(3, 0.5, 0.0, 42)?;
	assert_eq!(
		(split.train.len(), split.val.len(), split.test.len()),
		(1, 0, 2)
	);
	for ratio in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
		assert!(oa::data::random_split(10, ratio, 0.1, 42).is_err());
		assert!(oa::data::random_split(10, 0.8, ratio, 42).is_err());
	}
	assert!(oa::data::random_split(-1, 0.8, 0.1, 42).is_err());
	assert_eq!(
		oa::data::random_split(i64::MAX, 0.8, 0.1, 42)
			.err()
			.expect("allocation bound")
			.kind(),
		oa::ErrorKind::ResourceExhausted
	);
	Ok(())
}

test_vk!(
	collate_rejects_inconsistent_labels_shapes_and_dtypes,
	engine,
	{
		let x = oa::Matrix::from_f32(&engine, [2], &[1.0, 2.0])?;
		let y = oa::Matrix::from_f32(&engine, [1], &[1.0])?;
		assert!(
			oa::data::collate(&[
				oa::Sample::new(x.clone()),
				oa::Sample::with_label(x.clone(), y.clone())
			])
			.is_err()
		);
		assert!(oa::data::collate(&[oa::Sample::new(x.clone()), oa::Sample::new(y)]).is_err());
		let integers = oa::Matrix::from_slice(&engine, [2], &[1_i32, 2])?;
		assert_eq!(
			oa::data::collate(&[oa::Sample::new(integers)])
				.err()
				.expect("unsupported dtype")
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		let batch = oa::data::collate(&[oa::Sample::new(x.clone()), oa::Sample::new(x)])?;
		assert_eq!(batch.x.read_f32()?, [1.0, 2.0, 1.0, 2.0]);
		Ok(())
	}
);

#[test]
fn shuffle_matches_compiled_donor_random_header() -> oa::Result<()> {
	// Oracle: compiled oa/core/std/random.h from oacpp with Random(seed).shuffle.
	for (seed, expected) in [
		(1, [7, 15, 3, 5, 0, 2, 10, 14, 1, 13, 6, 11, 8, 9, 12, 4]),
		(7, [8, 6, 4, 15, 10, 1, 3, 7, 5, 9, 2, 0, 12, 11, 13, 14]),
		(42, [15, 11, 1, 7, 13, 4, 10, 5, 14, 9, 0, 8, 6, 12, 3, 2]),
		(
			u64::MAX,
			[8, 12, 9, 15, 13, 2, 7, 14, 0, 3, 5, 10, 4, 11, 1, 6],
		),
	] {
		let mut indices: Vec<i64> = (0..16).collect();
		oa::data::shuffle(&mut indices, seed)?;
		assert_eq!(indices, expected, "seed {seed}");
	}
	Ok(())
}

test_vk!(
	single_sample_collation_preserves_values_and_metadata,
	engine,
	{
		let x = oa::Matrix::from_f32(&engine, [2], &[3.0, 4.0])?;
		let y = oa::Matrix::from_f32(&engine, [1], &[1.0])?;
		let batch = oa::data::collate(&[oa::Sample::with_label(x, y)])?;
		assert_eq!(batch.x.shape(), [1, 2]);
		assert_eq!(batch.x.read_f32()?, [3.0, 4.0]);
		let label = batch.y.expect("labeled batch");
		assert_eq!(label.shape(), [1, 1]);
		assert_eq!(label.read_f32()?, [1.0]);
		Ok(())
	}
);
