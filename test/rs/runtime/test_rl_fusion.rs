use oa::ml::{GradientTape, Module};
use oa::{Matrix, Result, matrix};

fn values(count: usize, phase: usize) -> Vec<f32> {
	(0..count)
		.map(|i| (((i * 17 + phase) % 47) as f32 - 23.0) / 31.0)
		.collect()
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
		assert!(
			(a - b).abs() <= tolerance || (a.is_nan() && b.is_nan()),
			"element {i}: {a} != {b}"
		);
	}
}
fn has_kernel(plan: &oa::ExecutionPlan, suffix: &str) -> bool {
	let report: serde_json::Value =
		serde_json::from_str(&plan.debug_report_json("rl fusion")).unwrap();
	report["nodes"]
		.as_array()
		.unwrap()
		.iter()
		.any(|node| node["kernel"].as_str().unwrap().ends_with(suffix))
}

test_vk!(
	rl_fusion_router_materializes_unbiased_probabilities_and_biased_selection,
	engine,
	{
		for (rows, experts, k) in [(1, 1, 1), (3, 7, 2), (2, 8, 8), (2, 256, 3)] {
			let logits = Matrix::new_from_f32(&engine, [rows, experts], &vec![0.0; rows * experts])?;
			let bias_values = (0..experts)
				.map(|i| if i.is_multiple_of(2) { 1.0 } else { -1.0 })
				.collect::<Vec<_>>();
			let bias = Matrix::new_from_f32(&engine, [1, experts], &bias_values)?;
			for biased in [false, true] {
				let probability = matrix::softmax(&logits, 1)?;
				let reference_probability = probability.read_f32()?;
				let selection = if biased {
					matrix::add(&logits, &bias)?
				} else {
					logits.clone()
				};
				selection.read_f32()?;
				let reference_top = matrix::top_k(&selection, k as i32, 1)?;
				let reference_values = reference_top.values.read_f32()?;
				let reference_indices = reference_top.indices.read::<i32>()?;
				let reference_mask = matrix::top_k_mask(&reference_top.indices, experts)?.read_f32()?;
				let (mut plan, (probability, selection, top, mask)) = engine.capture(|| {
					let p = matrix::softmax(&logits, 1)?;
					let s = if biased {
						matrix::add(&logits, &bias)?
					} else {
						logits.clone()
					};
					let top = matrix::top_k(&s, k as i32, 1)?;
					let mask = matrix::top_k_mask(&top.indices, experts)?;
					Ok((p, s, top, mask))
				})?;
				assert!(has_kernel(&plan, "moe_router_select.f32"));
				for _ in 0..3 {
					engine.submit(&plan)?.wait()?;
					close(&probability.read_f32()?, &reference_probability, 1e-7);
					assert_eq!(top.values.read_f32()?, reference_values);
					assert_eq!(top.indices.read::<i32>()?, reference_indices);
					assert_eq!(mask.read_f32()?, reference_mask);
					if biased {
						assert_eq!(selection.read_f32()?, bias_values.repeat(rows));
					}
				}
				if !biased {
					let replacement =
						Matrix::new_from_f32(&engine, [rows, experts], &vec![2.0; rows * experts])?;
					plan.bind_matrix_input(&logits, &replacement)?;
					engine.submit(&plan)?.wait()?;
					close(&probability.read_f32()?, &reference_probability, 1e-7);
					assert_eq!(top.values.read_f32()?, vec![2.0; rows * k]);
				}
			}
		}
		Ok(())
	}
);

test_vk!(
	rl_fusion_router_keeps_unqualified_route_counts_on_source,
	engine,
	{
		let logits = Matrix::new_from_f32(&engine, [1, 33], &values(33, 0))?;
		let (plan, mask) = engine.capture(|| {
			let _p = matrix::softmax(&logits, 1)?;
			let top = matrix::top_k(&logits, 33, 1)?;
			matrix::top_k_mask(&top.indices, 33)
		})?;
		assert!(!has_kernel(&plan, "moe_router_select.f32"));
		engine.submit(&plan)?.wait()?;
		assert_eq!(mask.read_f32()?, vec![1.0; 33]);
		Ok(())
	}
);

test_vk!(
	rl_fusion_gated_linear_preserves_both_halves_and_backward,
	engine,
	{
		for hidden in [7, 65] {
			let input = Matrix::new_from_f32(&engine, [3, 5], &values(15, 1))?;
			let reference = oa::ml::nn::Linear::new_with_seed(&engine, 5, 2 * hidden, 0x984)?;
			let candidate = oa::ml::nn::Linear::new_with_seed(&engine, 5, 2 * hidden, 0x984)?;
			let target = Matrix::new_from_f32(&engine, [3, hidden], &vec![0.2; 3 * hidden])?;
			let tape = GradientTape::new();
			let raw = reference.forward(&input)?;
			let expected_raw = raw.read_f32()?;
			let output = oa::ml::matrix::silu_mul(&raw, hidden)?;
			let expected = output.read_f32()?;
			let loss = oa::ml::loss::mse(&output, &target)?;
			tape.backward(&loss)?;
			let gradients = reference
				.all_parameters()?
				.iter()
				.map(|p| p.gradient().unwrap().read_f32())
				.collect::<Result<Vec<_>>>()?;
			let (plan, (raw, output)) = engine.capture(|| {
				let tape = GradientTape::new();
				let raw = candidate.forward(&input)?;
				let output = oa::ml::matrix::silu_mul(&raw, hidden)?;
				let loss = oa::ml::loss::mse(&output, &target)?;
				tape.backward(&loss)?;
				Ok((raw, output))
			})?;
			assert!(has_kernel(&plan, "linear_gated_act.f32"));
			engine.submit(&plan)?.wait()?;
			close(&raw.read_f32()?, &expected_raw, 1e-6);
			close(&output.read_f32()?, &expected, 1e-6);
			for (p, g) in candidate.all_parameters()?.iter().zip(&gradients) {
				close(&p.gradient().unwrap().read_f32()?, g, 2e-5);
			}
		}
		Ok(())
	}
);

test_vk!(
	rl_fusion_grouped_gate_up_preserves_boundary_and_homogeneous_tiles,
	engine,
	{
		let rows = 70;
		let inner = 5;
		let hidden = 19;
		let experts = 3;
		let input = Matrix::new_from_f32(&engine, [rows, inner], &values(rows * inner, 1))?;
		let weight = Matrix::new_from_f32(
			&engine,
			[experts, 2 * hidden, inner],
			&values(experts * 2 * hidden * inner, 2),
		)?;
		let bias = Matrix::new_from_f32(
			&engine,
			[experts, 2 * hidden],
			&values(experts * 2 * hidden, 3),
		)?;
		let offsets = Matrix::new_from_slice(&engine, [experts + 1], &[0_u32, 3, 3, 70])?;
		let raw = oa::ml::matrix::grouped_linear_m(&input, &weight, &bias, &offsets)?;
		let expected_raw = raw.read_f32()?;
		let expected = oa::ml::matrix::silu_mul(&raw, hidden)?.read_f32()?;
		let (plan, (raw, output)) = engine.capture(|| {
			let raw = oa::ml::matrix::grouped_linear_m(&input, &weight, &bias, &offsets)?;
			let output = oa::ml::matrix::silu_mul(&raw, hidden)?;
			Ok((raw, output))
		})?;
		assert!(has_kernel(&plan, "grouped_linear_m_act.f32"));
		for _ in 0..3 {
			engine.submit(&plan)?.wait()?;
			close(&raw.read_f32()?, &expected_raw, 1e-6);
			close(&output.read_f32()?, &expected, 1e-6);
		}
		Ok(())
	}
);

test_vk!(
	rl_fusion_concat_linear_keeps_observed_input_and_gradients,
	engine,
	{
		for parts in [2, 3] {
			let inputs = (0..parts)
				.map(|i| Matrix::new_from_f32(&engine, [3, i + 1], &values(3 * (i + 1), i)))
				.collect::<Result<Vec<_>>>()?;
			let width = (1..=parts).sum::<usize>();
			let reference = oa::ml::nn::Linear::new_with_seed(&engine, width, 67, 0x23)?;
			let candidate = oa::ml::nn::Linear::new_with_seed(&engine, width, 67, 0x23)?;
			let target = Matrix::new_from_f32(&engine, [3, 67], &vec![0.0; 201])?;
			let tape = GradientTape::new();
			let combined = matrix::concat(&inputs, 1)?;
			let expected_combined = combined.read_f32()?;
			let output = reference.forward(&combined)?;
			let expected = output.read_f32()?;
			tape.backward(&oa::ml::loss::mse(&output, &target)?)?;
			let gradients = reference
				.all_parameters()?
				.iter()
				.map(|p| p.gradient().unwrap().read_f32())
				.collect::<Result<Vec<_>>>()?;
			let (plan, (combined, output)) = engine.capture(|| {
				let tape = GradientTape::new();
				let combined = matrix::concat(&inputs, 1)?;
				let output = candidate.forward(&combined)?;
				tape.backward(&oa::ml::loss::mse(&output, &target)?)?;
				Ok((combined, output))
			})?;
			assert!(has_kernel(&plan, "linear_concat.f32"));
			engine.submit(&plan)?.wait()?;
			assert_eq!(combined.read_f32()?, expected_combined);
			close(&output.read_f32()?, &expected, 1e-6);
			for (p, g) in candidate.all_parameters()?.iter().zip(&gradients) {
				close(&p.gradient().unwrap().read_f32()?, g, 1e-6);
			}
		}
		Ok(())
	}
);

fn attention_reference(
	q: &[f32],
	k: &[f32],
	v: &[f32],
	batch: usize,
	sequence: usize,
	heads: usize,
	dim: usize,
) -> Vec<f32> {
	let mut result = vec![0.0; q.len()];
	let index = |b, s, h, d| ((b * sequence + s) * heads + h) * dim + d;
	for b in 0..batch {
		for h in 0..heads {
			for row in 0..sequence {
				let mut scores = (0..=row)
					.map(|col| {
						(0..dim)
							.map(|d| q[index(b, row, h, d)] * k[index(b, col, h, d)])
							.sum::<f32>()
							/ (dim as f32).sqrt()
					})
					.collect::<Vec<_>>();
				let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
				scores.iter_mut().for_each(|s| *s = (*s - maximum).exp());
				let sum = scores.iter().sum::<f32>();
				for d in 0..dim {
					result[index(b, row, h, d)] = scores
						.iter()
						.enumerate()
						.map(|(col, p)| p / sum * v[index(b, col, h, d)])
						.sum();
				}
			}
		}
	}
	result
}

test_vk!(
	rl_fusion_small_causal_attention_preserves_heads_and_source_saves,
	engine,
	{
		for (sequence, dim) in [(3, 5), (16, 32), (31, 17), (32, 8), (128, 16)] {
			let batch = 2;
			let heads = 2;
			let count = batch * sequence * heads * dim;
			let qv = values(count, 1);
			let kv = values(count, 2);
			let vv = values(count, 3);
			let q = Matrix::new_from_f32(&engine, [batch * sequence, heads * dim], &qv)?;
			let k = Matrix::new_from_f32(&engine, [batch * sequence, heads * dim], &kv)?;
			let v = Matrix::new_from_f32(&engine, [batch * sequence, heads * dim], &vv)?;
			let expected = attention_reference(&qv, &kv, &vv, batch, sequence, heads, dim);
			let (plan, (sq, sk, sv, context, output)) = engine.capture(|| {
				let sq = oa::ml::matrix::split_heads(&q, batch, sequence, heads)?;
				let sk = oa::ml::matrix::split_heads(&k, batch, sequence, heads)?;
				let sv = oa::ml::matrix::split_heads(&v, batch, sequence, heads)?;
				let context = oa::ml::matrix::scaled_dot_product_attention(
					&sq,
					&sk,
					&sv,
					None,
					1.0 / (dim as f32).sqrt(),
					true,
				)?;
				let output = oa::ml::matrix::merge_heads(&context, batch, sequence, heads)?;
				Ok((sq, sk, sv, context, output))
			})?;
			assert!(
				has_kernel(&plan, "small_causal_sdpa.f32"),
				"{sequence} {dim}"
			);
			for _ in 0..2 {
				engine.submit(&plan)?.wait()?;
				close(&output.read_f32()?, &expected, 2e-5);
				for (split, source) in [(&sq, &qv), (&sk, &kv), (&sv, &vv), (&context, &expected)] {
					let actual = split.read_f32()?;
					for b in 0..batch {
						for h in 0..heads {
							for s in 0..sequence {
								for d in 0..dim {
									close(
										&[actual[((b * heads + h) * sequence + s) * dim + d]],
										&[source[((b * sequence + s) * heads + h) * dim + d]],
										2e-5,
									);
								}
							}
						}
					}
				}
			}
		}
		Ok(())
	}
);

test_vk!(
	rl_fusion_small_causal_attention_supports_single_head_rows,
	engine,
	{
		for (sequence, dim) in [(3, 5), (16, 32), (31, 17), (32, 8), (128, 16)] {
			let batch = 2;
			let heads = 1;
			let count = batch * sequence * dim;
			let qv = values(count, 1);
			let kv = values(count, 2);
			let vv = values(count, 3);
			let q = Matrix::new_from_f32(&engine, [batch * sequence, dim], &qv)?;
			let k = Matrix::new_from_f32(&engine, [batch * sequence, dim], &kv)?;
			let v = Matrix::new_from_f32(&engine, [batch * sequence, dim], &vv)?;
			let expected = attention_reference(&qv, &kv, &vv, batch, sequence, heads, dim);
			let (plan, output) = engine.capture(|| {
				oa::ml::matrix::scaled_dot_product_attention_causal(&q, &k, &v, sequence, heads)
			})?;
			assert!(
				has_kernel(&plan, "small_causal_sdpa.f32"),
				"{sequence} {dim}"
			);
			for _ in 0..2 {
				engine.submit(&plan)?.wait()?;
				close(&output.read_f32()?, &expected, 2e-5);
			}
		}
		Ok(())
	}
);

test_vk!(
	rl_fusion_tape_supports_broadcast_and_typed_gather_leaves,
	engine,
	{
		let input = Matrix::new_from_f32(&engine, [3, 7], &values(21, 1))?;
		let column = Matrix::new_from_f32(&engine, [3, 1], &[1.0, 2.0, 3.0])?;
		let row = Matrix::new_from_f32(&engine, [1, 7], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])?;
		let a = matrix::mul(&input, &column)?;
		let ea = a.read_f32()?;
		let b = matrix::add(&a, &row)?;
		let eb = b.read_f32()?;
		let ec = matrix::reciprocal(&b)?.read_f32()?;
		let (plan, (a, b, c)) = engine.capture(|| {
			let a = matrix::mul(&input, &column)?;
			let b = matrix::add(&a, &row)?;
			let c = matrix::reciprocal(&b)?;
			Ok((a, b, c))
		})?;
		assert_eq!(plan.semantic_lowering().fused_op_count(), 3);
		engine.submit(&plan)?.wait()?;
		close(&a.read_f32()?, &ea, 1e-6);
		close(&b.read_f32()?, &eb, 1e-6);
		close(&c.read_f32()?, &ec, 1e-5);
		for dtype in 0..3 {
			let indices = match dtype {
				0 => Matrix::new_from_slice(&engine, [3], &[2_u8, 0, 9])?,
				1 => Matrix::new_from_slice(&engine, [3], &[2_u32, 0, 9])?,
				_ => Matrix::new_from_slice(&engine, [3], &[2_i32, 0, -1])?,
			};
			let gathered = matrix::gather(&input, &indices)?;
			let expected = gathered.read_f32()?;
			let expected_scaled = matrix::scale(&gathered, 2.0)?.read_f32()?;
			let (plan, (gathered, scaled)) = engine.capture(|| {
				let g = matrix::gather(&input, &indices)?;
				let s = matrix::scale(&g, 2.0)?;
				Ok((g, s))
			})?;
			assert_eq!(plan.semantic_lowering().fused_op_count(), 2);
			engine.submit(&plan)?.wait()?;
			close(&gathered.read_f32()?, &expected, 0.0);
			close(&scaled.read_f32()?, &expected_scaled, 0.0);
		}
		Ok(())
	}
);

test_vk!(
	rl_fusion_tape_activation_adjoint_preserves_parameter_gradients,
	engine,
	{
		let input = Matrix::new_from_f32(&engine, [3, 5], &values(15, 7))?;
		let target = Matrix::new_from_f32(&engine, [3, 7], &[0.0; 21])?;
		let reference = oa::ml::nn::Linear::new_with_seed(&engine, 5, 7, 19)?;
		let candidate = oa::ml::nn::Linear::new_with_seed(&engine, 5, 7, 19)?;
		let tape = GradientTape::new();
		let raw = reference.forward(&input)?;
		raw.read_f32()?;
		let activated = oa::ml::matrix::gelu(&raw)?;
		activated.read_f32()?;
		let scaled = matrix::scale(&activated, 0.7)?;
		scaled.read_f32()?;
		tape.backward(&oa::ml::loss::mse(&scaled, &target)?)?;
		let gradients = reference
			.all_parameters()?
			.iter()
			.map(|p| p.gradient().unwrap().read_f32())
			.collect::<Result<Vec<_>>>()?;
		let (plan, ()) = engine.capture(|| {
			let tape = GradientTape::new();
			let raw = candidate.forward(&input)?;
			let activated = oa::ml::matrix::gelu(&raw)?;
			let scaled = matrix::scale(&activated, 0.7)?;
			tape.backward(&oa::ml::loss::mse(&scaled, &target)?)
		})?;
		assert!(has_kernel(&plan, "fused_chain.f32"));
		engine.submit(&plan)?.wait()?;
		for (p, g) in candidate.all_parameters()?.iter().zip(&gradients) {
			close(&p.gradient().unwrap().read_f32()?, g, 2e-6);
		}
		Ok(())
	}
);

// Fresh-process differential timing: run this one test with fusion enabled,
// then with OA_DISABLE_RL_FUSION=1, retaining the pre-existing fusion passes.
// Timings exclude capture/preparation.
test_vk!(rl_fusion_benchmark_static_workloads, engine, {
	let mut plans = Vec::new();
	let logits = Matrix::new_from_f32(&engine, [64, 8], &values(512, 1))?;
	let bias = Matrix::new_from_f32(&engine, [1, 8], &values(8, 2))?;
	let (router, _) = engine.capture(|| {
		let p = matrix::softmax(&logits, 1)?;
		let s = matrix::add(&logits, &bias)?;
		let t = matrix::top_k(&s, 2, 1)?;
		let m = matrix::top_k_mask(&t.indices, 8)?;
		Ok((p, t, m))
	})?;
	plans.push(("router64x8k2", router));
	let input = Matrix::new_from_f32(&engine, [64, 64], &values(4096, 3))?;
	let linear = oa::ml::nn::Linear::new_with_seed(&engine, 64, 128, 0x24)?;
	let (gated, _) = engine.capture(|| oa::ml::matrix::silu_mul(&linear.forward(&input)?, 64))?;
	plans.push(("gate64x64x64", gated));
	let odd_linear = oa::ml::nn::Linear::new_with_seed(&engine, 64, 130, 0x24)?;
	let (odd_gate, _) =
		engine.capture(|| oa::ml::matrix::silu_mul(&odd_linear.forward(&input)?, 65))?;
	plans.push(("gate64x64x65", odd_gate));
	let grouped_input = Matrix::new_from_f32(&engine, [70, 16], &values(70 * 16, 1))?;
	let grouped_weight = Matrix::new_from_f32(&engine, [3, 64, 16], &values(3 * 64 * 16, 2))?;
	let grouped_bias = Matrix::new_from_f32(&engine, [3, 64], &values(3 * 64, 3))?;
	let grouped_offsets = Matrix::new_from_slice(&engine, [4], &[0_u32, 3, 17, 70])?;
	let (grouped_gate, _) = engine.capture(|| {
		oa::ml::matrix::silu_mul(
			&oa::ml::matrix::grouped_linear_m(
				&grouped_input,
				&grouped_weight,
				&grouped_bias,
				&grouped_offsets,
			)?,
			32,
		)
	})?;
	plans.push(("grouped_gate70x16x32", grouped_gate));
	let query = Matrix::new_from_f32(&engine, [64 * 16, 64], &values(64 * 16 * 64, 1))?;
	let key = Matrix::new_from_f32(&engine, [64 * 16, 64], &values(64 * 16 * 64, 2))?;
	let value = Matrix::new_from_f32(&engine, [64 * 16, 64], &values(64 * 16 * 64, 3))?;
	let (attention, _) = engine
		.capture(|| oa::ml::matrix::scaled_dot_product_attention_causal(&query, &key, &value, 16, 2))?;
	plans.push(("sdpa64x16h2d32", attention));
	let long_query = Matrix::new_from_f32(&engine, [16 * 128, 32], &values(16 * 128 * 32, 1))?;
	let long_key = Matrix::new_from_f32(&engine, [16 * 128, 32], &values(16 * 128 * 32, 2))?;
	let long_value = Matrix::new_from_f32(&engine, [16 * 128, 32], &values(16 * 128 * 32, 3))?;
	let (long_attention, _) = engine.capture(|| {
		oa::ml::matrix::scaled_dot_product_attention_causal(&long_query, &long_key, &long_value, 128, 2)
	})?;
	plans.push(("sdpa16x128h2d16", long_attention));
	let parts = [
		Matrix::new_from_f32(&engine, [64, 3], &values(192, 1))?,
		Matrix::new_from_f32(&engine, [64, 1], &values(64, 2))?,
		Matrix::new_from_f32(&engine, [64, 12], &values(768, 3))?,
	];
	let joint = oa::ml::nn::Linear::new_with_seed(&engine, 16, 64, 0x25)?;
	let (head, _) =
		engine.capture(|| oa::ml::matrix::gelu(&joint.forward(&matrix::concat(&parts, 1)?)?))?;
	plans.push(("head64x16x64", head));
	let a = Matrix::new_from_f32(&engine, [128, 8], &values(1024, 4))?;
	let row = Matrix::new_from_f32(&engine, [1, 8], &[1.0; 8])?;
	let col = Matrix::new_from_f32(&engine, [128, 1], &vec![0.5; 128])?;
	let (broadcast, _) =
		engine.capture(|| matrix::reciprocal(&matrix::add(&matrix::mul(&a, &col)?, &row)?))?;
	plans.push(("broadcast128x8", broadcast));
	for (name, plan) in plans {
		for _ in 0..3 {
			engine.submit(&plan)?.wait()?;
		}
		let mut samples = Vec::new();
		for _ in 0..11 {
			samples.push(engine.submit_timed(&plan)?.device_duration()?.as_secs_f64() * 1000.0);
		}
		samples.sort_by(f64::total_cmp);
		eprintln!(
			"RL_FUSION_BENCH {name} nodes={} median_ms={:.6} min_ms={:.6} max_ms={:.6}",
			plan.diagnostics().node_count(),
			samples[5],
			samples[0],
			samples[10]
		);
	}
	Ok(())
});

test_vk!(
	rl_fusion_nine_activation_adjoints_match_host_finite_differences,
	engine,
	{
		let inputs = [-1.1_f32, 0.3, 0.8, -0.2, -0.7, 0.9, 0.4, 0.5, -0.6];
		let initial = [0.4_f32, -0.3, 0.8, 0.12];
		let targets = [0.2_f32, -0.1, 0.4];
		let input = Matrix::new_from_f32(&engine, [3, 3], &inputs)?;
		let target = Matrix::new_from_f32(&engine, [3, 1], &targets)?;
		for kind in 0..9 {
			let host = |x: f32| match kind {
				0 => x.max(0.0),
				1 => 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh()),
				2 => x / (1.0 + (-x).exp()),
				3 => x.tanh(),
				4 => 1.0 / (1.0 + (-x).exp()),
				5 => x * (1.0 + x.exp()).ln().tanh(),
				6 => x.max(0.0) + (1.0 + (-x.abs()).exp()).ln(),
				7 => {
					if x > 0.0 {
						x
					} else {
						0.2 * (x.exp() - 1.0)
					}
				}
				_ => {
					if x > 0.0 {
						x
					} else {
						0.1 * x
					}
				}
			};
			let loss = |parameters: &[f32; 4]| {
				(0..3)
					.map(|row| {
						let raw = parameters[3]
							+ (0..3)
								.map(|col| inputs[row * 3 + col] * parameters[col])
								.sum::<f32>();
						(0.7 * host(raw) - targets[row]).powi(2) / 3.0
					})
					.sum::<f32>()
			};
			let mut expected = Vec::new();
			for i in 0..4 {
				let mut plus = initial;
				let mut minus = initial;
				plus[i] += 0.002;
				minus[i] -= 0.002;
				expected.push((loss(&plus) - loss(&minus)) / 0.004);
			}
			let linear = oa::ml::nn::Linear::new_from_matrices(
				Matrix::new_from_f32(&engine, [1, 3], &initial[..3])?,
				Matrix::new_from_f32(&engine, [1], &initial[3..])?,
			)?;
			let (plan, ()) = engine.capture(|| {
				let tape = GradientTape::new();
				let raw = linear.forward(&input)?;
				let activated = match kind {
					0 => oa::ml::matrix::relu(&raw)?,
					1 => oa::ml::matrix::gelu(&raw)?,
					2 => oa::ml::matrix::silu(&raw)?,
					3 => oa::ml::matrix::tanh(&raw)?,
					4 => oa::ml::matrix::sigmoid(&raw)?,
					5 => oa::ml::matrix::mish(&raw)?,
					6 => oa::ml::matrix::softplus(&raw)?,
					7 => oa::ml::matrix::elu(&raw, 0.2)?,
					_ => oa::ml::matrix::leaky_relu(&raw, 0.1)?,
				};
				let output = matrix::scale(&activated, 0.7)?;
				tape.backward(&oa::ml::loss::mse(&output, &target)?)
			})?;
			assert!(has_kernel(&plan, "fused_chain.f32"), "activation {kind}");
			engine.submit(&plan)?.wait()?;
			let mut actual = linear.weight().gradient().unwrap().read_f32()?;
			actual.extend(linear.bias().unwrap().gradient().unwrap().read_f32()?);
			close(&actual, &expected, 3e-4);
		}
		Ok(())
	}
);

test_vk!(rl_fusion_typed_gather_reverse_matches_host_rows, engine, {
	let input = Matrix::new_from_f32(&engine, [3, 1], &[1.0, 2.0, 3.0])?;
	let weights = values(7, 1);
	let biases = values(7, 2);
	let target = Matrix::new_from_f32(&engine, [3, 7], &[0.0; 21])?;
	for dtype in 0..3 {
		let indices = match dtype {
			0 => Matrix::new_from_slice(&engine, [3], &[2_u8, 0, 2])?,
			1 => Matrix::new_from_slice(&engine, [3], &[2_u32, 0, 2])?,
			_ => Matrix::new_from_slice(&engine, [3], &[2_i32, 0, 2])?,
		};
		let linear = oa::ml::nn::Linear::new_from_matrices(
			Matrix::new_from_f32(&engine, [7, 1], &weights)?,
			Matrix::new_from_f32(&engine, [7], &biases)?,
		)?;
		let (plan, ()) = engine.capture(|| {
			let tape = GradientTape::new();
			let table = linear.forward(&input)?;
			let selected = matrix::gather(&table, &indices)?;
			let scaled = matrix::scale(&selected, 2.0)?;
			tape.backward(&oa::ml::loss::mse(&scaled, &target)?)
		})?;
		engine.submit(&plan)?.wait()?;
		let mut expected_weight = Vec::new();
		let mut expected_bias = Vec::new();
		for (w, b) in weights.iter().zip(&biases) {
			let first = w + b;
			let last = 3.0 * w + b;
			expected_weight.push(8.0 * (first + 6.0 * last) / 21.0);
			expected_bias.push(8.0 * (first + 2.0 * last) / 21.0);
		}
		close(
			&linear.weight().gradient().unwrap().read_f32()?,
			&expected_weight,
			2e-5,
		);
		close(
			&linear.bias().unwrap().gradient().unwrap().read_f32()?,
			&expected_bias,
			2e-5,
		);
	}
	Ok(())
});
