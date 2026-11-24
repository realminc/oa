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
			let embedding = oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32(
				[6, 1],
				&prediction_values,
			)?)?;
			let indices = oa::Matrix::new_from_slice([6], &[0_u32, 1, 2, 3, 4, 5])?;
			let target = oa::Matrix::new_from_f32([6, 1], &target_values)?;
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
	core_loss_capture_has_one_semantic_op_and_two_classified_dispatches,
	engine,
	{
		let prediction = oa::Matrix::new_from_f32([2, 3], &[0.2, -0.7, 1.4, 0.5, -1.2, 0.9])?;
		let target = oa::Matrix::new_from_f32([2, 3], &[0.0, -0.2, 0.4, 0.9, -0.2, -0.1])?;
		let (plan, output) = engine.capture(|| oa::ml::loss::mse(&prediction, &target))?;
		assert_eq!(plan.semantic_graph().operations().len(), 1);
		let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("MSE"))
			.expect("loss execution report must be valid JSON");
		let nodes = report["nodes"]
			.as_array()
			.expect("loss execution nodes must be an array");
		assert_eq!(nodes.len(), 2);
		let kernels = nodes
			.iter()
			.map(|node| node["kernel"].as_str().expect("kernel name must be text"))
			.collect::<Vec<_>>();
		assert_eq!(kernels, ["ml.loss.mse.f32", "ml.sum_scale.f32"]);
		assert!(nodes.iter().all(|node| !node["physical_write"].is_null()));
		engine.submit(&plan)?.wait()?;
		assert_close(&output.read_f32()?, &[LossKind::Mse.value()], 3.0e-6);
		Ok(())
	}
);

test_vk!(smooth_l1_selects_the_donor_fused_mean_candidate, engine, {
	let prediction = oa::Matrix::new_from_f32([2, 3], &[0.2, -0.7, 1.4, 0.5, -1.2, 0.9])?;
	let target = oa::Matrix::new_from_f32([2, 3], &[0.0, -0.2, 0.4, 0.9, -0.2, -0.1])?;
	let (plan, output) = engine.capture(|| oa::ml::loss::smooth_l1(&prediction, &target))?;
	assert_eq!(plan.semantic_graph().operations().len(), 1);
	assert_eq!(
		plan.semantic_graph().operations()[0].name(),
		"oa::FnLoss::smoothL1"
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
	composed_scalar_loss_scales_and_accumulates_adjoint,
	engine,
	{
		let embedding =
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([3, 1], &[1.0, -2.0, 0.5])?)?;
		let indices = oa::Matrix::new_from_slice([3], &[0_u32, 1, 2])?;
		let target = oa::Matrix::new_from_f32([3, 1], &[0.0; 3])?;
		let tape = oa::ml::GradientTape::new();
		let prediction = embedding.forward(&indices)?;
		let loss = oa::ml::loss::mse(&prediction, &target)?;
		let scaled = oa::matrix::scale(&loss, 2.0)?;
		let combined = oa::matrix::add(&loss, &scaled)?;
		assert_close(&combined.read_f32()?, &[5.25], 1.0e-6);
		tape.backward(&combined)?;
		assert_close(
			&embedding
				.weight()
				.gradient()
				.expect("composed scalar loss gradient is missing")
				.read_f32()?,
			&[2.0, -4.0, 1.0],
			1.0e-6,
		);
		Ok(())
	}
);

test_vk!(broadcast_mul_reverse_reduces_expanded_axes, engine, {
	let left =
		oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([2, 1], &[2.0, -1.0])?)?;
	let right =
		oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([1, 3], &[3.0, 0.5, -2.0])?)?;
	let left_rows = oa::Matrix::new_from_slice([2], &[0_u32, 1])?;
	let right_rows = oa::Matrix::new_from_slice([1], &[0_u32])?;
	let target = oa::Matrix::new_from_f32([2, 3], &[0.0; 6])?;
	let tape = oa::ml::GradientTape::new();
	let product = oa::matrix::mul(&left.forward(&left_rows)?, &right.forward(&right_rows)?)?;
	let loss = oa::ml::loss::mse(&product, &target)?;
	tape.backward(&loss)?;
	assert_close(
		&left
			.weight()
			.gradient()
			.expect("broadcast left gradient is missing")
			.read_f32()?,
		&[8.833_333, -4.416_666_5],
		2.0e-6,
	);
	assert_close(
		&right
			.weight()
			.gradient()
			.expect("broadcast right gradient is missing")
			.read_f32()?,
		&[5.0, 0.833_333_3, -3.333_333_3],
		2.0e-6,
	);
	Ok(())
});

test_vk!(
	core_losses_reject_invalid_inputs_before_recording,
	engine,
	{
		let prediction = oa::Matrix::new_from_f32([2], &[0.2, 0.8])?;
		let wrong_shape = oa::Matrix::new_from_f32([1], &[0.2])?;
		let wrong_dtype = oa::Matrix::new_from_slice([2], &[0_i32, 1])?;
		let empty = oa::Matrix::new_from_f32([0], &[])?;
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
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([4, 3], &logits)?)?;
		let indices = oa::Matrix::new_from_slice([4], &[0_u32, 1, 2, 3])?;
		let targets = oa::Matrix::new_from_slice([4], &targets)?;
		let mask = oa::Matrix::new_from_f32([4], &mask)?;
		let tape = oa::ml::GradientTape::new();
		let logits = embedding.forward(&indices)?;
		let loss = oa::ml::loss::cross_entropy_masked(&logits, &targets, &mask, 2)?;
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
		let logits = oa::Matrix::new_from_f32([2, 3], &[2.0, -1.0, 0.5, -0.5, 1.25, 0.75])?;
		let targets = oa::Matrix::new_from_slice([2], &[3_i32, -1])?;
		let mask = oa::Matrix::new_from_f32([2], &[1.0, 0.0])?;
		let (plan, loss) =
			engine.capture(|| oa::ml::loss::cross_entropy_masked(&logits, &targets, &mask, 1))?;
		let operation = &plan.semantic_graph().operations()[0];
		assert_eq!(operation.name(), "oa::FnLoss::maskedCrossEntropy");
		assert_eq!(
			operation.attributes(),
			[oa::OpAttribute::SignedInteger {
				name: "validCount".into(),
				value: 1,
			}]
		);
		let report: serde_json::Value =
			serde_json::from_str(&plan.debug_report_json("masked cross entropy"))
				.expect("masked-loss execution report must be valid JSON");
		let nodes = report["nodes"]
			.as_array()
			.expect("masked-loss execution nodes must be an array");
		assert_eq!(nodes.len(), 2);
		assert_eq!(nodes[0]["kernel"], "ml.loss.masked_cross_entropy.f32");
		assert!(nodes.iter().all(|node| !node["physical_write"].is_null()));
		engine.submit(&plan)?.wait()?;
		assert!(loss.read_f32()?[0].is_nan());

		let embedding = oa::ml::nn::Embedding::new_from_matrix(logits)?;
		let indices = oa::Matrix::new_from_slice([2], &[0_u32, 1])?;
		let tape = oa::ml::GradientTape::new();
		let logits = embedding.forward(&indices)?;
		let loss = oa::ml::loss::cross_entropy_masked(&logits, &targets, &mask, 1)?;
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
		let logits = oa::Matrix::new_from_f32([2, 3], &[0.0; 6])?;
		let targets = oa::Matrix::new_from_slice([2], &[0_u32, 1])?;
		let wrong_shape = oa::Matrix::new_from_f32([1], &[1.0])?;
		let wrong_dtype = oa::Matrix::new_from_slice([2], &[1_i32, 1])?;
		let mask = oa::Matrix::new_from_f32([2], &[1.0, 1.0])?;
		for (candidate, count) in [(&wrong_shape, 1), (&wrong_dtype, 1), (&mask, 0), (&mask, 3)] {
			let error = match oa::ml::loss::cross_entropy_masked(&logits, &targets, candidate, count) {
				Ok(_) => panic!("invalid masked loss input was accepted"),
				Err(error) => error,
			};
			assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
		}
		Ok(())
	}
);

test_vk!(composed_losses_match_host_oracles, engine, {
	let _ = engine;
	// margin_ranking: max(0, -y*(x1-x2)+m); x1=[2,1], x2=[1,3], y=[1,-1], m=0.5
	// row0: -1*(1)+0.5 = -0.5 -> 0; row1: +1*(1-3)+0.5 = -1.5 -> 0? y=-1: -(-1)*(x1-x2)= (x1-x2)= -2 +0.5 -> -1.5 -> 0
	let x1 = oa::Matrix::new_from_f32([1, 2], &[2.0, 1.0])?;
	let x2 = oa::Matrix::new_from_f32([1, 2], &[1.0, 3.0])?;
	let y = oa::Matrix::new_from_f32([1, 2], &[1.0, 1.0])?;
	let loss = oa::ml::loss::margin_ranking(&x1, &x2, &y, 0.5)?;
	// elements: max(0, -1*(2-1)+0.5)=0, max(0, -1*(1-3)+0.5)=2.5 -> mean 1.25
	assert_close(&loss.read_f32()?, &[1.25], 1e-6);

	// soft_margin: mean(log(1+exp(-y*x))); x=[1,-1], y=[1,-1]
	let x = oa::Matrix::new_from_f32([2], &[1.0, -1.0])?;
	let t = oa::Matrix::new_from_f32([2], &[1.0, -1.0])?;
	let loss = oa::ml::loss::soft_margin(&x, &t)?;
	// z = -y*x = [-1,-1] -> softplus(-1) = ln(1+e^-1)*2/2
	let expected = (1.0_f32 + (-1.0_f32).exp()).ln();
	assert_close(&loss.read_f32()?, &[expected], 1e-6);

	// nll: -mean(x[i, y_i]); scores [[1,2,3],[4,5,6]], y=[[1],[2]] -> -(2+6)/2 = -4
	let scores = oa::Matrix::new_from_f32([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
	let labels = oa::Matrix::new_from_slice([2, 1], &[1_i32, 2])?;
	let loss = oa::ml::loss::nll(&scores, &labels)?;
	assert_close(&loss.read_f32()?, &[-4.0], 1e-6);

	// multi_margin: margin=1, same scores/labels
	// row0: s_yi=2; hinges j0: 1-2+1=0, j2: 1-2+3=2 -> sum 2 -1(j=yi term) =1? j=yi term = margin=1
	//   row0 loss = (0+0+2 - 1)/3 = 1/3
	// row1: s_yi=6; hinges: 1-6+4=-1->0, 1-6+5=0->0, 1 -> sum=1-1=0 -> 0
	let loss = oa::ml::loss::multi_margin(&scores, &labels, 1.0)?;
	assert_close(&loss.read_f32()?, &[1.0 / 3.0], 1e-6);
	Ok(())
});

test_vk!(
	composed_distance_and_probabilistic_losses_match_host_oracles,
	engine,
	{
		let _ = engine;
		// triplet_margin: a=[0,0], p=[1,0], n=[3,4]; d_ap=1, d_an=5, m=1 -> max(0,1-5+1)=0? use m=10 -> 6
		let anchor = oa::Matrix::new_from_f32([1, 2], &[0.0, 0.0])?;
		let positive = oa::Matrix::new_from_f32([1, 2], &[1.0, 0.0])?;
		let negative = oa::Matrix::new_from_f32([1, 2], &[3.0, 4.0])?;
		let loss = oa::ml::loss::triplet_margin(&anchor, &positive, &negative, 10.0)?;
		assert_close(&loss.read_f32()?, &[6.0], 1e-5);

		// cosine_embedding: x1=[1,0], x2=[1,0], y=1 -> 1-cos=0; y=-1 -> max(0,1-0)=1
		let x1 = oa::Matrix::new_from_f32([1, 2], &[1.0, 0.0])?;
		let x2 = oa::Matrix::new_from_f32([1, 2], &[1.0, 0.0])?;
		let y1 = oa::Matrix::new_from_f32([1, 1], &[1.0])?;
		let loss = oa::ml::loss::cosine_embedding(&x1, &x2, &y1, 0.0)?;
		assert_close(&loss.read_f32()?, &[0.0], 1e-6);
		let yn = oa::Matrix::new_from_f32([1, 1], &[-1.0])?;
		let loss = oa::ml::loss::cosine_embedding(&x1, &x2, &yn, 0.0)?;
		assert_close(&loss.read_f32()?, &[1.0], 1e-6);

		// kl_div: input log-probs [ln .5, ln .5], target [0.5, 0.5]
		// loss = mean(t*ln t - t*x) = mean(0.5 ln .5 - 0.5 ln .5)*2 = 0
		let logp = oa::Matrix::new_from_f32([2], &[0.5_f32.ln(), 0.5_f32.ln()])?;
		let target = oa::Matrix::new_from_f32([2], &[0.5, 0.5])?;
		let loss = oa::ml::loss::kl_div(&logp, &target)?;
		assert_close(&loss.read_f32()?, &[0.0], 1e-6);
		// diverging: x = ln .25 both, t = [0.5,0.5] -> each term 0.5(ln .5 - ln .25) = 0.5 ln 2
		let logq = oa::Matrix::new_from_f32([2], &[0.25_f32.ln(), 0.25_f32.ln()])?;
		let loss = oa::ml::loss::kl_div(&logq, &target)?;
		assert_close(&loss.read_f32()?, &[0.5 * 2.0_f32.ln()], 1e-6);

		// gaussian_nll: x=[1], t=[0], var=[4], eps=1e-6 -> 0.5(ln4 + 1/4)
		let x = oa::Matrix::new_from_f32([1], &[1.0])?;
		let t = oa::Matrix::new_from_f32([1], &[0.0])?;
		let var = oa::Matrix::new_from_f32([1], &[4.0])?;
		let loss = oa::ml::loss::gaussian_nll(&x, &t, &var, 1e-6, false)?;
		assert_close(&loss.read_f32()?, &[0.5 * (4.0_f32.ln() + 0.25)], 1e-6);

		// poisson_nll: x=ln2 -> rate 2, t=[3] -> 2 - 3*ln2
		let x = oa::Matrix::new_from_f32([1], &[2.0_f32.ln()])?;
		let t = oa::Matrix::new_from_f32([1], &[3.0])?;
		let loss = oa::ml::loss::poisson_nll(&x, &t)?;
		assert_close(&loss.read_f32()?, &[2.0 - 3.0 * 2.0_f32.ln()], 1e-6);

		// multi_label_soft_margin: logits [0,0], targets [1,0] -> mean bce = ln2 per elem
		let logits = oa::Matrix::new_from_f32([1, 2], &[0.0, 0.0])?;
		let targets = oa::Matrix::new_from_f32([1, 2], &[1.0, 0.0])?;
		let loss = oa::ml::loss::multi_label_soft_margin(&logits, &targets, None)?;
		assert_close(&loss.read_f32()?, &[2.0_f32.ln()], 1e-6);
		Ok(())
	}
);

test_vk!(
	composed_loss_adjoints_reach_prediction_parameters,
	engine,
	{
		// kl_div gradient on input: d mean(t*ln t - t*x)/dx = -t/N
		let embedding = oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32(
			[1, 2],
			&[0.5_f32.ln(), 0.25_f32.ln()],
		)?)?;
		let ids = oa::Matrix::new_from_slice([1], &[0_u32])?;
		let target = oa::Matrix::new_from_f32([1, 2], &[0.5, 0.5])?;
		let tape = oa::ml::GradientTape::new();
		let input = embedding.forward(&ids)?;
		let loss = oa::ml::loss::kl_div(&input, &target)?;
		tape.backward(&loss)?;
		let gradient = embedding
			.weight()
			.gradient()
			.expect("kl_div adjoint must reach the input")
			.read_f32()?;
		assert_close(&gradient, &[-0.25, -0.25], 1e-5);

		// soft_margin gradient: sigma(-y*x)*(-y)/N with x=y=1 -> -sigma(-1)/2 each
		let embedding =
			oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32([1, 2], &[1.0, 1.0])?)?;
		let ids = oa::Matrix::new_from_slice([1], &[0_u32])?;
		let target = oa::Matrix::new_from_f32([1, 2], &[1.0, 1.0])?;
		let tape = oa::ml::GradientTape::new();
		let input = embedding.forward(&ids)?;
		let loss = oa::ml::loss::soft_margin(&input, &target)?;
		tape.backward(&loss)?;
		let gradient = embedding
			.weight()
			.gradient()
			.expect("soft_margin adjoint must reach the input")
			.read_f32()?;
		let expected = -1.0 / (1.0 + 1.0_f32.exp()) / 2.0;
		assert_close(&gradient, &[expected, expected], 1e-6);
		Ok(())
	}
);

fn ctc_oracle_item_nll(
	log_probs: &[f64],
	stride: usize,
	times: usize,
	classes: usize,
	target: &[i32],
	blank: i32,
) -> f64 {
	let mut total = 0.0f64;
	let mut sequence = vec![0i32; times];
	'paths: loop {
		let mut collapsed = Vec::new();
		let mut previous = -1i32;
		for &class in &sequence {
			if class != previous {
				collapsed.push(class);
				previous = class;
			}
		}
		let labels: Vec<i32> = collapsed
			.into_iter()
			.filter(|&class| class != blank)
			.collect();
		if labels == target {
			let mut probability = 1.0f64;
			for (step, &class) in sequence.iter().enumerate() {
				probability *= log_probs[step * stride + class as usize].exp();
			}
			total += probability;
		}
		let mut index = times;
		loop {
			if index == 0 {
				break 'paths;
			}
			index -= 1;
			sequence[index] += 1;
			if sequence[index] < classes as i32 {
				break;
			}
			sequence[index] = 0;
		}
	}
	-total.ln()
}

test_vk!(ml_loss_ctc_matches_enumerated_alignments, engine, {
	// [T, N, C] = [3, 2, 4]; arbitrary log-domain values.
	let log_probs = [
		-0.4, -1.1, -2.0, -0.7, // t0 n0
		-1.3, -0.6, -0.9, -1.8, // t0 n1
		-0.9, -0.3, -1.4, -1.0, // t1 n0
		-0.8, -1.7, -0.5, -0.4, // t1 n1
		-1.6, -0.8, -0.7, -1.2, // t2 n0
		-0.5, -1.0, -1.9, -0.2, // t2 n1
	];
	let log_probs = oa::Matrix::new_from_f32([3, 2, 4], &log_probs)?;
	let targets = oa::Matrix::new_from_slice([2, 2], &[1_i32, 2, 3, 0])?;
	let input_lengths = oa::Matrix::new_from_slice([2], &[3_i32, 3])?;
	let target_lengths = oa::Matrix::new_from_slice([2], &[2_i32, 1])?;
	let loss =
		oa::ml::loss::ctc(&log_probs, &targets, &input_lengths, &target_lengths, 0)?.read_f32()?;
	// Stride between consecutive frames for a fixed item is N*C.
	let nll0 = ctc_oracle_item_nll(
		&[
			-0.4, -1.1, -2.0, -0.7, -0.9, -0.3, -1.4, -1.0, -1.6, -0.8, -0.7, -1.2,
		],
		4,
		3,
		4,
		&[1, 2],
		0,
	);
	let nll1 = ctc_oracle_item_nll(
		&[
			-1.3, -0.6, -0.9, -1.8, -0.8, -1.7, -0.5, -0.4, -0.5, -1.0, -1.9, -0.2,
		],
		4,
		3,
		4,
		&[3],
		0,
	);
	let expected = ((nll0 + nll1) / 2.0) as f32;
	assert_close(&loss, &[expected], 1e-4);
	Ok(())
});

fn item_frames(
	values: &[f64],
	item: usize,
	batch: usize,
	times: usize,
	classes: usize,
) -> Vec<f64> {
	(0..times)
		.flat_map(|t| {
			values[(t * batch + item) * classes..(t * batch + item + 1) * classes]
				.iter()
				.copied()
		})
		.collect()
}

test_vk!(ml_loss_ctc_backward_matches_finite_difference, engine, {
	let times = 3usize;
	let batch = 2usize;
	let classes = 4usize;
	let values = [
		-0.4, -1.1, -2.0, -0.7, -1.3, -0.6, -0.9, -1.8, -0.9, -0.3, -1.4, -1.0, -0.8, -1.7, -0.5, -0.4,
		-1.6, -0.8, -0.7, -1.2, -0.5, -1.0, -1.9, -0.2,
	];
	// Rows of the embedding weight are the [T*N, C] frames; reshape
	// restores [T, N, C] while staying on the tape.
	let weight = oa::ml::nn::Embedding::new_from_matrix(oa::Matrix::new_from_f32(
		[times * batch, classes],
		&values,
	)?)?;
	let ids = oa::Matrix::new_from_slice([times * batch], &[0_u32, 1, 2, 3, 4, 5])?;
	let targets = oa::Matrix::new_from_slice([2, 2], &[1_i32, 2, 3, 0])?;
	let input_lengths = oa::Matrix::new_from_slice([2], &[3_i32, 3])?;
	let target_lengths = oa::Matrix::new_from_slice([2], &[2_i32, 1])?;
	let tape = oa::ml::GradientTape::new();
	let log_probs = weight.forward(&ids)?.reshape([times, batch, classes])?;
	let loss = oa::ml::loss::ctc(&log_probs, &targets, &input_lengths, &target_lengths, 0)?;
	tape.backward(&loss)?;
	let gradient = weight
		.weight()
		.gradient()
		.expect("ctc adjoint must reach log_probs")
		.read_f32()?;
	// Finite-difference oracle over the same item layout: row f of the
	// weight is frame (f / N) item (f % N); element stride between
	// frames of one item is N*C.
	let targets_host = [[1, 2].as_slice(), [3].as_slice()];
	let delta = 1e-3f64;
	let mut expected = Vec::with_capacity(values.len());
	for row in 0..times * batch {
		let item = row % batch;
		let target = targets_host[item];
		for class in 0..classes {
			let mut perturbed: Vec<f64> = values.iter().map(|v| *v as f64).collect();
			let slot = row * classes + class;
			perturbed[slot] += delta;
			let plus = ctc_oracle_item_nll(
				&item_frames(&perturbed, item, batch, times, classes),
				classes,
				times,
				classes,
				target,
				0,
			);
			perturbed[slot] -= 2.0 * delta;
			let minus = ctc_oracle_item_nll(
				&item_frames(&perturbed, item, batch, times, classes),
				classes,
				times,
				classes,
				target,
				0,
			);
			expected.push(((plus - minus) / (2.0 * delta) / batch as f64) as f32);
		}
	}
	assert_close(&gradient, &expected, 5e-2);
	Ok(())
});
