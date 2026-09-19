#[derive(Clone, Copy, Debug)]
enum Layout {
	Nn,
	Nt,
	Tn,
}

impl Layout {
	fn apply(self, left: &oa::Matrix, right: &oa::Matrix) -> oa::Result<oa::Matrix> {
		match self {
			Self::Nn => oa::ml::matrix::bmm(left, right),
			Self::Nt => oa::ml::matrix::bmm_nt(left, right),
			Self::Tn => oa::ml::matrix::bmm_tn(left, right),
		}
	}

	const fn operation(self) -> &'static str {
		match self {
			Self::Nn => "oa::ml::matrix::bmm",
			Self::Nt => "oa::ml::matrix::bmm_nt",
			Self::Tn => "oa::ml::matrix::bmm_tn",
		}
	}

	const fn generic_kernel(self) -> &'static str {
		match self {
			Self::Nn => "ml.matrix.bmm.f32",
			Self::Nt => "ml.matrix.bmm_nt.f32",
			Self::Tn => "ml.matrix.bmm_tn.f32",
		}
	}

	const fn tiled_kernel(self) -> &'static str {
		match self {
			Self::Nn => "ml.bmm_tiled_16.f32",
			Self::Nt => "ml.bmm_nt_tiled_16.f32",
			Self::Tn => "ml.bmm_tn_tiled_16.f32",
		}
	}
}

fn input_shapes(
	layout: Layout,
	batch: usize,
	rows: usize,
	inner: usize,
	columns: usize,
) -> ([usize; 3], [usize; 3]) {
	match layout {
		Layout::Nn => ([batch, rows, inner], [batch, inner, columns]),
		Layout::Nt => ([batch, rows, inner], [batch, columns, inner]),
		Layout::Tn => ([batch, inner, rows], [batch, inner, columns]),
	}
}

fn host_bmm(
	layout: Layout,
	left: &[f32],
	right: &[f32],
	batch: usize,
	rows: usize,
	inner: usize,
	columns: usize,
) -> Vec<f32> {
	let mut output = vec![0.0; batch * rows * columns];
	for batch_index in 0..batch {
		for row in 0..rows {
			for column in 0..columns {
				let mut value = 0.0;
				for inner_index in 0..inner {
					let left_index = match layout {
						Layout::Nn | Layout::Nt => (batch_index * rows + row) * inner + inner_index,
						Layout::Tn => (batch_index * inner + inner_index) * rows + row,
					};
					let right_index = match layout {
						Layout::Nn | Layout::Tn => (batch_index * inner + inner_index) * columns + column,
						Layout::Nt => (batch_index * columns + column) * inner + inner_index,
					};
					value += left[left_index] * right[right_index];
				}
				output[(batch_index * rows + row) * columns + column] = value;
			}
		}
	}
	output
}

fn values(count: usize, seed: usize) -> Vec<f32> {
	(0..count)
		.map(|index| {
			let centered = ((index * 17 + seed * 11) % 29) as f32 - 14.0;
			centered / 13.0
		})
		.collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= tolerance,
			"element {index}: found {actual}, expected {expected}, tolerance {tolerance}"
		);
	}
}

test_vk!(bmm_family_matches_generic_and_tiled_host_oracles, engine, {
	for layout in [Layout::Nn, Layout::Nt, Layout::Tn] {
		for (batch, rows, inner, columns, expected_kernel, tolerance) in [
			(2, 3, 5, 4, layout.generic_kernel(), 3.0e-5),
			(2, 9, 11, 10, layout.tiled_kernel(), 5.0e-5),
		] {
			let (left_shape, right_shape) = input_shapes(layout, batch, rows, inner, columns);
			let left_values = values(left_shape.iter().product(), 1);
			let right_values = values(right_shape.iter().product(), 2);
			let expected = host_bmm(
				layout,
				&left_values,
				&right_values,
				batch,
				rows,
				inner,
				columns,
			);
			let left = oa::Matrix::from_f32(&engine, left_shape, &left_values)?;
			let right = oa::Matrix::from_f32(&engine, right_shape, &right_values)?;
			let (plan, output) = engine.capture(|| layout.apply(&left, &right))?;

			assert_eq!(output.shape(), [batch, rows, columns]);
			assert_eq!(plan.semantic_graph().operations().len(), 1);
			assert_eq!(
				plan.semantic_graph().operations()[0].name(),
				layout.operation()
			);
			let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("bmm"))
				.expect("BMM report must be valid JSON");
			assert_eq!(report["nodes"][0]["kernel"], expected_kernel);
			assert!(!report["nodes"][0]["physical_write"].is_null());

			let _event = engine.submit(&plan)?;
			plan.wait()?;
			assert_close(&output.read_f32()?, &expected, tolerance);
		}
	}
	Ok(())
});

fn loss_and_gradients(
	engine: &oa::Engine,
	layout: Layout,
	left_value: f32,
	right_value: f32,
	backward: bool,
) -> oa::Result<(f32, [f32; 2])> {
	let left =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(engine, [1, 1], &[left_value])?)?;
	let right =
		oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(engine, [1, 1], &[right_value])?)?;
	let indices = oa::Matrix::from_slice(engine, [1, 1], &[0_u32])?;
	let tape = oa::ml::GradientTape::new();
	let output = layout.apply(&left.forward(&indices)?, &right.forward(&indices)?)?;
	let target = oa::Matrix::from_f32(engine, [1, 1, 1], &[0.35])?;
	let loss = oa::ml::loss::mse(&output, &target)?;
	if backward {
		tape.backward(&loss)?;
	}
	let loss_value = loss.read_f32()?[0];
	let gradients = if backward {
		[
			left
				.weight()
				.gradient()
				.expect("left BMM gradient is missing")
				.read_f32()?[0],
			right
				.weight()
				.gradient()
				.expect("right BMM gradient is missing")
				.read_f32()?[0],
		]
	} else {
		[0.0, 0.0]
	};
	Ok((loss_value, gradients))
}

test_vk!(bmm_family_reverse_matches_finite_differences, engine, {
	const EPSILON: f32 = 1.0e-3;
	for layout in [Layout::Nn, Layout::Nt, Layout::Tn] {
		let values = [0.7_f32, -0.4];
		let (_, actual) = loss_and_gradients(&engine, layout, values[0], values[1], true)?;
		for parameter in 0..2 {
			let mut below = values;
			let mut above = values;
			below[parameter] -= EPSILON;
			above[parameter] += EPSILON;
			let below = loss_and_gradients(&engine, layout, below[0], below[1], false)?.0;
			let above = loss_and_gradients(&engine, layout, above[0], above[1], false)?.0;
			let numerical = (above - below) / (2.0 * EPSILON);
			assert!(
				(actual[parameter] - numerical).abs() <= 8.0e-4,
				"{layout:?} parameter {parameter}: found {}, expected {numerical}",
				actual[parameter]
			);
		}
	}
	Ok(())
});

test_vk!(bmm_family_rejects_invalid_contracts, engine, {
	let valid_left = oa::Matrix::from_f32(&engine, [2, 3, 4], &[0.0; 24])?;
	let valid_right = oa::Matrix::from_f32(&engine, [2, 4, 5], &[0.0; 40])?;
	let rank_two = oa::Matrix::from_f32(&engine, [3, 4], &[0.0; 12])?;
	let wrong_batch = oa::Matrix::from_f32(&engine, [1, 4, 5], &[0.0; 20])?;
	let wrong_inner = oa::Matrix::from_f32(&engine, [2, 6, 5], &[0.0; 60])?;
	let wrong_dtype = oa::Matrix::from_slice(&engine, [2, 4, 5], &[0_i32; 40])?;

	for error in [
		oa::ml::matrix::bmm(&rank_two, &valid_right)
			.err()
			.expect("rank-two BMM input was accepted"),
		oa::ml::matrix::bmm(&valid_left, &wrong_batch)
			.err()
			.expect("mismatched BMM batch was accepted"),
		oa::ml::matrix::bmm(&valid_left, &wrong_inner)
			.err()
			.expect("mismatched BMM inner dimension was accepted"),
		oa::ml::matrix::bmm(&valid_left, &wrong_dtype)
			.err()
			.expect("non-F32 BMM input was accepted"),
	] {
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}
	Ok(())
});
