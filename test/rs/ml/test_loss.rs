#[derive(Clone, Copy)]
enum LossKind {
	SmoothL1,
	Mse,
	L1,
	Bce,
}

impl LossKind {
	fn compute(self, prediction: &oa::Matrix, target: &oa::Matrix) -> oa::Result<oa::Matrix> {
		match self {
			Self::SmoothL1 => oa::ml::loss::smooth_l1(prediction, target),
			Self::Mse => oa::ml::loss::mse(prediction, target),
			Self::L1 => oa::ml::loss::l1(prediction, target),
			Self::Bce => oa::ml::loss::bce(prediction, target),
		}
	}

	fn prediction(self) -> [f32; 6] {
		match self {
			Self::Bce => [0.1, 0.25, 0.4, 0.6, 0.8, 0.9],
			Self::SmoothL1 | Self::Mse | Self::L1 => [0.2, -0.7, 1.4, 0.5, -1.2, 0.9],
		}
	}

	fn target(self) -> [f32; 6] {
		match self {
			Self::Bce => [0.0, 1.0, 0.3, 0.7, 1.0, 0.0],
			Self::SmoothL1 | Self::Mse | Self::L1 => [0.0, -0.2, 0.4, 0.9, -0.2, -0.1],
		}
	}

	fn value(self) -> f32 {
		let prediction = self.prediction();
		let target = self.target();
		prediction
			.into_iter()
			.zip(target)
			.map(|(prediction, target)| match self {
				Self::SmoothL1 => {
					let difference = prediction - target;
					if difference.abs() < 1.0 {
						0.5 * difference * difference
					} else {
						difference.abs() - 0.5
					}
				}
				Self::Mse => (prediction - target).powi(2),
				Self::L1 => (prediction - target).abs(),
				Self::Bce => {
					let prediction = prediction.clamp(1.0e-7, 1.0 - 1.0e-7);
					-(target * prediction.ln() + (1.0 - target) * (1.0 - prediction).ln())
				}
			})
			.sum::<f32>()
			/ prediction.len() as f32
	}

	fn gradient(self) -> [f32; 6] {
		let prediction = self.prediction();
		let target = self.target();
		let mut gradient = [0.0; 6];
		for index in 0..gradient.len() {
			let difference = prediction[index] - target[index];
			gradient[index] = match self {
				Self::SmoothL1 if difference.abs() < 1.0 => difference / 6.0,
				Self::SmoothL1 => difference.signum() / 6.0,
				Self::Mse => 2.0 * difference / 6.0,
				Self::L1 => difference.signum() / 6.0,
				Self::Bce => {
					let prediction = prediction[index].clamp(1.0e-7, 1.0 - 1.0e-7);
					(prediction - target[index]) / (prediction * (1.0 - prediction) * 6.0)
				}
			};
		}
		gradient
	}
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
		let error = (actual - expected).abs();
		assert!(
			error <= tolerance,
			"element {index}: expected {expected}, found {actual}, error {error} exceeds {tolerance}"
		);
	}
}

fn masked_cross_entropy_oracle(
	logits: &[f32],
	targets: &[u32],
	mask: &[f32],
	classes: usize,
	valid_count: usize,
) -> (f32, Vec<f32>) {
	let mut loss = 0.0_f32;
	let mut gradient = vec![0.0_f32; logits.len()];
	for (row, (&target, &selected)) in targets.iter().zip(mask).enumerate() {
		if selected == 0.0 {
			continue;
		}
		let values = &logits[row * classes..(row + 1) * classes];
		let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		let sum = values
			.iter()
			.map(|value| (*value - maximum).exp())
			.sum::<f32>();
		loss += sum.ln() + maximum - values[target as usize];
		for (column, value) in values.iter().enumerate() {
			gradient[row * classes + column] = ((*value - maximum).exp() / sum
				- if column == target as usize { 1.0 } else { 0.0 })
				/ valid_count as f32;
		}
	}
	(loss / valid_count as f32, gradient)
}

test_vk!(
	core_losses_match_donor_forward_and_prediction_adjoint_contracts,
	engine,
	{
		for kind in [
			LossKind::SmoothL1,
			LossKind::Mse,
			LossKind::L1,
			LossKind::Bce,
		] {
			let prediction_values = kind.prediction();
			let target_values = kind.target();
			let embedding = oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(
				&engine,
				[6, 1],
				&prediction_values,
			)?)?;
			let indices = oa::Matrix::from_slice(&engine, [6], &[0_u32, 1, 2, 3, 4, 5])?;
			let target = oa::Matrix::from_f32(&engine, [6, 1], &target_values)?;
			let tape = oa::ml::GradientTape::new();
			let prediction = embedding.forward(&indices)?;
			let loss = kind.compute(&prediction, &target)?;
			assert_close(&loss.read_f32()?, &[kind.value()], 3.0e-6);
			tape.backward(&loss)?;
			let gradient = embedding
				.weight()
				.gradient()
				.expect("loss prediction adjoint must reach the embedding parameter")
				.read_f32()?;
			assert_close(&gradient, &kind.gradient(), 3.0e-6);
		}
		Ok(())
	}
);

test_vk!(
	core_loss_capture_has_one_semantic_op_and_three_classified_dispatches,
	engine,
	{
		let prediction = oa::Matrix::from_f32(&engine, [2, 3], &[0.2, -0.7, 1.4, 0.5, -1.2, 0.9])?;
		let target = oa::Matrix::from_f32(&engine, [2, 3], &[0.0, -0.2, 0.4, 0.9, -0.2, -0.1])?;
		let (plan, output) = engine.capture(|| oa::ml::loss::mse(&prediction, &target))?;
		assert_eq!(plan.semantic_graph().operations().len(), 1);
		let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("MSE"))
			.expect("loss execution report must be valid JSON");
		let nodes = report["nodes"]
			.as_array()
			.expect("loss execution nodes must be an array");
		assert_eq!(nodes.len(), 3);
		let kernels = nodes
			.iter()
			.map(|node| node["kernel"].as_str().expect("kernel name must be text"))
			.collect::<Vec<_>>();
		assert_eq!(
			kernels,
			["ml.loss.mse.f32", "matrix.sum.f32", "matrix.scale.f32"]
		);
		assert!(nodes.iter().all(|node| !node["physical_write"].is_null()));
		engine.submit(&plan)?.wait()?;
		assert_close(&output.read_f32()?, &[LossKind::Mse.value()], 3.0e-6);
		Ok(())
	}
);

test_vk!(smooth_l1_selects_the_donor_fused_mean_candidate, engine, {
	let prediction = oa::Matrix::from_f32(&engine, [2, 3], &[0.2, -0.7, 1.4, 0.5, -1.2, 0.9])?;
	let target = oa::Matrix::from_f32(&engine, [2, 3], &[0.0, -0.2, 0.4, 0.9, -0.2, -0.1])?;
	let (plan, output) = engine.capture(|| oa::ml::loss::smooth_l1(&prediction, &target))?;
	assert_eq!(plan.semantic_graph().operations().len(), 1);
	assert_eq!(
		plan.semantic_graph().operations()[0].name(),
		"oa::ml::loss::smooth_l1"
	);
	let report: serde_json::Value =
		serde_json::from_str(&plan.debug_report_json("Smooth L1 fused mean"))
			.expect("Smooth-L1 execution report must be valid JSON");
	let nodes = report["nodes"]
		.as_array()
		.expect("Smooth-L1 execution nodes must be an array");
	assert_eq!(nodes.len(), 1);
	assert_eq!(nodes[0]["kernel"], "ml.smooth_l1_mean.f32");
	assert!(!nodes[0]["physical_write"].is_null());
	engine.submit(&plan)?.wait()?;
	assert_close(&output.read_f32()?, &[LossKind::SmoothL1.value()], 3.0e-6);
	Ok(())
});

test_vk!(
	core_losses_reject_invalid_inputs_before_recording,
	engine,
	{
		let prediction = oa::Matrix::from_f32(&engine, [2], &[0.2, 0.8])?;
		let wrong_shape = oa::Matrix::from_f32(&engine, [1], &[0.2])?;
		let wrong_dtype = oa::Matrix::from_slice(&engine, [2], &[0_i32, 1])?;
		let empty = oa::Matrix::from_f32(&engine, [0], &[])?;
		for invalid in [&wrong_shape, &wrong_dtype, &empty] {
			let error = match oa::ml::loss::mse(&prediction, invalid) {
				Ok(_) => panic!("invalid loss input was accepted"),
				Err(error) => error,
			};
			assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
		}
		Ok(())
	}
);

test_vk!(
	masked_cross_entropy_matches_donor_mask_and_autograd_contract,
	engine,
	{
		let logits = [1.0, 2.0, 3.0, 3.0, 1.0, 0.0, 9.0, 8.0, 7.0, -2.0, 0.0, 2.0];
		let targets = [2_u32, 0, 0, 1];
		let mask = [1.0_f32, 1.0, 0.0, 0.0];
		let (expected_loss, expected_gradient) =
			masked_cross_entropy_oracle(&logits, &targets, &mask, 3, 2);
		let embedding =
			oa::ml::nn::Embedding::from_matrix(oa::Matrix::from_f32(&engine, [4, 3], &logits)?)?;
		let indices = oa::Matrix::from_slice(&engine, [4], &[0_u32, 1, 2, 3])?;
		let targets = oa::Matrix::from_slice(&engine, [4], &targets)?;
		let mask = oa::Matrix::from_f32(&engine, [4], &mask)?;
		let tape = oa::ml::GradientTape::new();
		let logits = embedding.forward(&indices)?;
		let loss = oa::ml::loss::masked_cross_entropy(&logits, &targets, &mask, 2)?;
		assert_close(&loss.read_f32()?, &[expected_loss], 3.0e-6);
		tape.backward(&loss)?;
		let gradient = embedding
			.weight()
			.gradient()
			.expect("masked loss logits adjoint must reach the embedding parameter")
			.read_f32()?;
		assert_close(&gradient, &expected_gradient, 3.0e-6);
		Ok(())
	}
);

test_vk!(
	masked_cross_entropy_preserves_invalid_target_and_capture_evidence,
	engine,
	{
		let logits = oa::Matrix::from_f32(&engine, [2, 3], &[2.0, -1.0, 0.5, -0.5, 1.25, 0.75])?;
		let targets = oa::Matrix::from_slice(&engine, [2], &[3_i32, -1])?;
		let mask = oa::Matrix::from_f32(&engine, [2], &[1.0, 0.0])?;
		let (plan, loss) =
			engine.capture(|| oa::ml::loss::masked_cross_entropy(&logits, &targets, &mask, 1))?;
		let operation = &plan.semantic_graph().operations()[0];
		assert_eq!(operation.name(), "oa::ml::loss::masked_cross_entropy");
		assert_eq!(
			operation.attributes(),
			[oa::OpAttribute::SignedInteger {
				name: "valid_count".into(),
				value: 1,
			}]
		);
		let report: serde_json::Value =
			serde_json::from_str(&plan.debug_report_json("masked cross entropy"))
				.expect("masked-loss execution report must be valid JSON");
		let nodes = report["nodes"]
			.as_array()
			.expect("masked-loss execution nodes must be an array");
		assert_eq!(nodes.len(), 3);
		assert_eq!(nodes[0]["kernel"], "ml.loss.masked_cross_entropy.f32");
		assert!(nodes.iter().all(|node| !node["physical_write"].is_null()));
		engine.submit(&plan)?.wait()?;
		assert!(loss.read_f32()?[0].is_nan());

		let embedding = oa::ml::nn::Embedding::from_matrix(logits)?;
		let indices = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let tape = oa::ml::GradientTape::new();
		let logits = embedding.forward(&indices)?;
		let loss = oa::ml::loss::masked_cross_entropy(&logits, &targets, &mask, 1)?;
		tape.backward(&loss)?;
		let gradient = embedding
			.weight()
			.gradient()
			.expect("masked loss logits adjoint must reach the embedding parameter")
			.read_f32()
			.expect("masked-loss gradient must be readable");
		assert!(gradient[..3].iter().all(|value| value.is_nan()));
		assert_eq!(&gradient[3..], &[0.0, 0.0, 0.0]);
		Ok(())
	}
);

test_vk!(
	masked_cross_entropy_rejects_invalid_mask_and_count,
	engine,
	{
		let logits = oa::Matrix::from_f32(&engine, [2, 3], &[0.0; 6])?;
		let targets = oa::Matrix::from_slice(&engine, [2], &[0_u32, 1])?;
		let wrong_shape = oa::Matrix::from_f32(&engine, [1], &[1.0])?;
		let wrong_dtype = oa::Matrix::from_slice(&engine, [2], &[1_i32, 1])?;
		let mask = oa::Matrix::from_f32(&engine, [2], &[1.0, 1.0])?;
		for (candidate, count) in [(&wrong_shape, 1), (&wrong_dtype, 1), (&mask, 0), (&mask, 3)] {
			let error =
				match oa::ml::loss::masked_cross_entropy(&logits, &targets, candidate, count) {
					Ok(_) => panic!("invalid masked loss input was accepted"),
					Err(error) => error,
				};
			assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
		}
		Ok(())
	}
);
