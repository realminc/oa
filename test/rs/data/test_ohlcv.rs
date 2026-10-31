use oa::{DataFrame, OhlcvView, OhlcvWindowConfig, OhlcvWindows, Series, SeriesView, UnixTimeUnit};

fn candles() -> oa::Result<DataFrame> {
	let mut frame = DataFrame::new();
	frame.set_column(
		"timestamp",
		Series::I64(
			(0..12)
				.map(|i| 1_700_000_000_000_000_001 + i * 1_000_000)
				.collect(),
		),
	)?;
	for (name, value) in [
		("open", 10.0),
		("high", 12.0),
		("low", 9.0),
		("close", 11.0),
		("volume", 100.0),
	] {
		frame.set_column(name, Series::F64(vec![value; 12]))?;
	}
	Ok(frame)
}

#[test]
fn frame_moves_storage_and_rejects_invalid_replacements() -> oa::Result<()> {
	let values = vec![1.0_f32, 2.0];
	let pointer = values.as_ptr();
	let mut frame = DataFrame::new();
	frame.set_column("price", Series::F32(values))?;
	let SeriesView::F32(column) = frame.column("price")? else {
		panic!("wrong dtype")
	};
	assert_eq!(pointer, column.as_ptr());
	let SeriesView::F32(tail) = frame.slice(1..2)?.column("price")? else {
		panic!("wrong dtype")
	};
	assert_eq!(tail.as_ptr(), column[1..].as_ptr());
	assert_eq!(tail, &[2.0]);
	assert!(frame.set_column("price", Series::F64(vec![3.0])).is_err());
	assert!(frame.set_column("", Series::I64(vec![1, 2])).is_err());
	assert_eq!(frame.len(), 2);
	assert_eq!(frame.column_names().collect::<Vec<_>>(), ["price"]);
	assert!(frame.column("missing").is_err());
	assert!(
		frame
			.column("price")?
			.slice(std::ops::Range { start: 2, end: 1 })
			.is_err()
	);
	assert!(frame.column("price")?.slice(0..3).is_err());
	Ok(())
}

#[test]
fn ohlcv_windows_borrow_exact_time_and_never_cross_splits() -> oa::Result<()> {
	let frame = candles()?;
	let view = OhlcvView::new_from_frame(&frame, UnixTimeUnit::Nanoseconds)?;
	assert_eq!(view.timestamps()[0], 1_700_000_000_000_000_001);
	let [train, val, test] = view.chronological_split(6, 9)?;
	let config = OhlcvWindowConfig {
		lookback: 2,
		horizon: 1,
		stride: 2,
	};
	let windows = OhlcvWindows::new(train, config)?;
	assert_eq!(windows.len(), 2);
	let (input, future) = windows.window(1)?;
	assert_eq!(input.timestamps().as_ptr(), view.timestamps()[2..].as_ptr());
	assert_eq!(input.time_unit(), UnixTimeUnit::Nanoseconds);
	assert_eq!(future.timestamps(), &view.timestamps()[4..5]);
	assert_eq!(OhlcvWindows::new(val, config)?.len(), 1);
	assert_eq!(OhlcvWindows::new(test, config)?.len(), 1);
	assert!(windows.window(-1).is_err());
	assert!(windows.window(2).is_err());
	assert!(view.chronological_split(9, 6).is_err());
	assert!(view.chronological_split(6, 13).is_err());
	Ok(())
}

#[test]
fn ohlcv_rejects_bad_candles_time_types_and_nonfinite_values() -> oa::Result<()> {
	for (name, values) in [
		("low", vec![11.0; 12]),
		("high", vec![10.0; 12]),
		("volume", vec![-1.0; 12]),
		("open", vec![f64::NAN; 12]),
		("close", vec![f64::INFINITY; 12]),
	] {
		let mut frame = candles()?;
		frame.set_column(name, Series::F64(values))?;
		assert!(OhlcvView::new_from_frame(&frame, UnixTimeUnit::Seconds).is_err());
	}
	let mut frame = candles()?;
	frame.set_column("timestamp", Series::I64(vec![0; 12]))?;
	assert!(OhlcvView::new_from_frame(&frame, UnixTimeUnit::Seconds).is_err());
	frame.set_column("timestamp", Series::F64(vec![0.0; 12]))?;
	assert!(OhlcvView::new_from_frame(&frame, UnixTimeUnit::Seconds).is_err());
	assert!(OhlcvView::new_from_frame(&DataFrame::new(), UnixTimeUnit::Seconds).is_err());
	Ok(())
}

#[test]
fn ohlcv_window_extents_are_checked_and_short_sources_are_empty() -> oa::Result<()> {
	let frame = candles()?;
	let source = OhlcvView::new_from_frame(&frame, UnixTimeUnit::Seconds)?;
	for config in [
		OhlcvWindowConfig {
			lookback: 0,
			horizon: 1,
			stride: 1,
		},
		OhlcvWindowConfig {
			lookback: 1,
			horizon: 0,
			stride: 1,
		},
		OhlcvWindowConfig {
			lookback: 1,
			horizon: 1,
			stride: 0,
		},
		OhlcvWindowConfig {
			lookback: usize::MAX,
			horizon: 1,
			stride: 1,
		},
	] {
		assert!(OhlcvWindows::new(source, config).is_err());
	}
	assert!(
		OhlcvWindows::new(
			source.slice(0..0)?,
			OhlcvWindowConfig {
				lookback: 2,
				horizon: 1,
				stride: 1
			}
		)?
		.is_empty()
	);
	let exact = OhlcvWindows::new(
		source,
		OhlcvWindowConfig {
			lookback: 10,
			horizon: 2,
			stride: usize::MAX,
		},
	)?;
	assert_eq!(exact.len(), 1);
	assert_eq!(exact.window(0)?.1.len(), 2);
	Ok(())
}

test_vk!(ohlcv_dataset_packs_features_and_future_labels, engine, {
	use oa::Dataset;
	let mut frame = candles()?;
	frame.set_column(
		"close",
		Series::F64((0..12).map(|row| 10.0 + f64::from(row) * 0.1).collect()),
	)?;
	frame.set_column(
		"volume",
		Series::F32((0..12).map(|row| 100.0 + row as f32).collect()),
	)?;
	let source = OhlcvView::new_from_frame(&frame, UnixTimeUnit::Nanoseconds)?;
	let windows = OhlcvWindows::new(
		source,
		OhlcvWindowConfig {
			lookback: 2,
			horizon: 2,
			stride: 1,
		},
	)?;
	let dataset = oa::DsOhlcv::new(&engine, windows);
	assert_eq!(dataset.len(), 9);
	let sample = dataset.get_sample(0)?;
	assert_eq!(sample.x.shape(), &[2, 5]);
	assert_eq!(
		sample.x.read::<f32>()?,
		vec![10.0, 12.0, 9.0, 10.0, 100.0, 10.0, 12.0, 9.0, 10.1, 101.0]
	);
	let label = sample.y.expect("future labels");
	assert_eq!(label.shape(), &[2]);
	assert_eq!(label.read::<f32>()?, vec![10.2, 10.3]);
	let mut loader = oa::DataLoader::new(
		&dataset,
		oa::DataLoaderConfig {
			batch_size: 2,
			shuffle: false,
			seed: 1,
			drop_last: false,
		},
	)?;
	let batch = loader.next_batch()?.expect("first batch");
	assert_eq!(batch.x.shape(), &[2, 2, 5]);
	assert_eq!(batch.y.expect("batch targets").shape(), &[2, 2]);
	let mut overflow = candles()?;
	overflow.set_column("volume", Series::F64(vec![f64::MAX; 12]))?;
	let windows = OhlcvWindows::new(
		OhlcvView::new_from_frame(&overflow, UnixTimeUnit::Seconds)?,
		OhlcvWindowConfig {
			lookback: 2,
			horizon: 1,
			stride: 1,
		},
	)?;
	let overflow_dataset = oa::DsOhlcv::new(&engine, windows);
	assert!(overflow_dataset.get_sample(0).is_err());

	Ok(())
});

#[test]
fn ohlcv_accepts_mixed_precision_negative_prices_and_pre_epoch_times() -> oa::Result<()> {
	let mut frame = DataFrame::new();
	frame.set_column("timestamp", Series::I64(vec![-2, -1, 0]))?;
	frame.set_column("open", Series::F32(vec![-3.0; 3]))?;
	frame.set_column("high", Series::F64(vec![-1.0; 3]))?;
	frame.set_column("low", Series::F32(vec![-4.0; 3]))?;
	frame.set_column("close", Series::F64(vec![-2.0; 3]))?;
	frame.set_column("volume", Series::F32(vec![0.0; 3]))?;
	let view = OhlcvView::new_from_frame(&frame, UnixTimeUnit::Microseconds)?;
	let SeriesView::F32(original) = frame.column("open")? else {
		panic!("wrong dtype")
	};
	let SeriesView::F32(borrowed) = view.slice(1..3)?.columns()[0] else {
		panic!("wrong dtype")
	};
	assert_eq!(borrowed.as_ptr(), original[1..].as_ptr());
	assert_eq!(view.timestamps(), &[-2, -1, 0]);
	assert_eq!(view.time_unit(), UnixTimeUnit::Microseconds);
	Ok(())
}

#[test]
fn frame_in_place_edits_reuse_buffer_and_ohlcv_is_revalidated() -> oa::Result<()> {
	let mut frame = candles()?;
	let SeriesView::F64(column) = frame.column("volume")? else {
		panic!("wrong dtype")
	};
	let pointer = column.as_ptr();
	let oa::SeriesViewMut::F64(column) = frame.column_mut("volume")? else {
		panic!("wrong dtype")
	};
	assert_eq!(pointer, column.as_ptr());
	column[0] = -1.0;
	assert_eq!(frame.len(), 12);
	assert!(OhlcvView::new_from_frame(&frame, UnixTimeUnit::Seconds).is_err());
	Ok(())
}

#[test]
fn ohlcv_frame_partition_validates_only_selected_rows() {
	let mut frame = oa::DataFrame::new();
	frame
		.set_column("timestamp", oa::Series::I64(vec![0, 1, 2, 3]))
		.unwrap();
	for (name, value) in [
		("open", 10.0),
		("high", 12.0),
		("low", 9.0),
		("close", 11.0),
	] {
		frame
			.set_column(name, oa::Series::F64(vec![value; 4]))
			.unwrap();
	}
	frame
		.set_column("volume", oa::Series::F64(vec![-1.0, -1.0, 1.0, 1.0]))
		.unwrap();
	assert!(oa::OhlcvView::new_from_frame(&frame, oa::UnixTimeUnit::Seconds).is_err());
	let view =
		oa::OhlcvView::new_from_frame_view(frame.slice(2..4).unwrap(), oa::UnixTimeUnit::Seconds)
			.unwrap();
	assert_eq!(view.timestamps(), &[2, 3]);
}
