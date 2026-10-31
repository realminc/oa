// Independent dense-pair F64 oracles, including module projections and adjoints.
use oa::ml::Module;
use std::collections::BTreeMap;

type Weights = BTreeMap<String, (Vec<usize>, Vec<f32>)>;
fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
	assert_eq!(actual.len(), expected.len());
	for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
		assert!(
			a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance * (1.0 + b.abs()),
			"element {i}: {a} != {b}, tol {tolerance}"
		);
	}
}
fn phi(x: f64) -> f64 {
	if x >= 0.0 { x + 1.0 } else { x.exp() }
}
#[allow(
	clippy::too_many_arguments,
	reason = "independent oracle keeps tensor geometry, visibility and floor explicit"
)]
fn oracle(
	q: &[f32],
	k: &[f32],
	v: &[f32],
	heads: usize,
	seq: usize,
	d: usize,
	causal: bool,
	eps: f64,
) -> Vec<f32> {
	let mut out = vec![0.0; q.len()];
	for h in 0..heads {
		for t in 0..seq {
			let mut numerator = vec![0.0; d];
			let mut denominator = 0.0;
			for u in 0..seq {
				if causal && u > t {
					continue;
				}
				let weight: f64 = (0..d)
					.map(|i| phi(q[(h * seq + t) * d + i] as f64) * phi(k[(h * seq + u) * d + i] as f64))
					.sum();
				denominator += weight;
				for j in 0..d {
					numerator[j] += weight * v[(h * seq + u) * d + j] as f64;
				}
			}
			for j in 0..d {
				out[(h * seq + t) * d + j] = (numerator[j] / denominator.max(eps)) as f32;
			}
		}
	}
	out
}
fn weights(model: &impl Module) -> oa::Result<Weights> {
	model
		.all_named_parameters()?
		.into_iter()
		.map(|p| {
			Ok((
				p.path().to_owned(),
				(
					p.parameter().data().shape().to_vec(),
					p.parameter().data().read_f32()?,
				),
			))
		})
		.collect()
}
fn project(x: &[f32], rows: usize, width: usize, name: &str, w: &Weights) -> Vec<f32> {
	let weight = &w[&format!("{name}.weight")].1;
	let bias = w.get(&format!("{name}.bias"));
	(0..rows * width)
		.map(|n| {
			let (r, j) = (n / width, n % width);
			((0..width)
				.map(|i| x[r * width + i] as f64 * weight[j * width + i] as f64)
				.sum::<f64>()
				+ bias.map_or(0.0, |b| b.1[j] as f64)) as f32
		})
		.collect()
}
fn model_oracle(
	x: &[f32],
	batch: usize,
	seq: usize,
	width: usize,
	heads: usize,
	causal: bool,
	w: &Weights,
) -> Vec<f32> {
	let d = width / heads;
	let split = |x: Vec<f32>| {
		(0..x.len())
			.map(|n| {
				let i = n % d;
				let t = n / d % seq;
				let h = n / d / seq % heads;
				let b = n / d / seq / heads;
				x[(b * seq + t) * width + h * d + i]
			})
			.collect::<Vec<_>>()
	};
	let q = split(project(x, batch * seq, width, "q_proj", w));
	let k = split(project(x, batch * seq, width, "k_proj", w));
	let v = split(project(x, batch * seq, width, "v_proj", w));
	let context = oracle(&q, &k, &v, batch * heads, seq, d, causal, 1.0e-6);
	let merged = (0..x.len())
		.map(|n| {
			let i = n % width % d;
			let h = n % width / d;
			let t = n / width % seq;
			let b = n / width / seq;
			context[((b * heads + h) * seq + t) * d + i]
		})
		.collect::<Vec<_>>();
	project(&merged, batch * seq, width, "out_proj", w)
}
fn model_forward_gate(engine: &oa::Engine, causal: bool) -> oa::Result<()> {
	use oa::ml::nn::{CausalLinearAttention, LinearAttentionMode};
	for (batch, seq, width, heads) in [(2, 5, 6, 2), (1, 1, 4, 2), (1, 3, 65, 1)] {
		let vals = (0..batch * seq * width)
			.map(|i| ((i * 13 % 23) as f32 - 11.0) / 9.0)
			.collect::<Vec<_>>();
		let model = CausalLinearAttention::new_with_seed(engine, width, heads, seq, 947)?;
		model.set_mode(if causal {
			LinearAttentionMode::Causal
		} else {
			LinearAttentionMode::Bidirectional
		});
		let input = oa::Matrix::new_from_f32(engine, [batch * seq, width], &vals)?;
		let expected = model_oracle(&vals, batch, seq, width, heads, causal, &weights(&model)?);
		assert_close(&model.forward(&input)?.read_f32()?, &expected, 2.0e-4);
	}
	Ok(())
}
fn model_gradient_gate(engine: &oa::Engine, causal: bool) -> oa::Result<()> {
	use oa::ml::nn::{CausalLinearAttention, LinearAttentionMode};
	let (seq, width, heads) = (3, 4, 2);
	let vals = (0..seq * width)
		.map(|i| ((i * 7 % 13) as f32 - 6.0) / 5.0)
		.collect::<Vec<_>>();
	let model = CausalLinearAttention::new_with_seed(engine, width, heads, seq, 4532)?;
	model.set_mode(if causal {
		LinearAttentionMode::Causal
	} else {
		LinearAttentionMode::Bidirectional
	});
	let input = oa::Matrix::new_from_f32(engine, [seq, width], &vals)?;
	let target = oa::Matrix::new_from_f32(engine, [seq, width], &vec![0.0; seq * width])?;
	let tape = oa::ml::GradientTape::new();
	let out = model.forward(&input)?;
	let loss = oa::ml::loss::mse(&out, &target)?;
	tape.backward(&loss)?;
	let mut w = weights(&model)?;
	for p in model.all_named_parameters()? {
		let name = p.path();
		let gradient = p
			.parameter()
			.gradient()
			.expect("missing gradient")
			.read_f32()?;
		let mut expected = Vec::new();
		for i in 0..gradient.len() {
			let original = w[name].1[i];
			let eps = 0.001_f32;
			let mut losses = [0.0; 2];
			for (j, delta) in [eps, -eps].into_iter().enumerate() {
				w.get_mut(name).unwrap().1[i] = original + delta;
				let output = model_oracle(&vals, 1, seq, width, heads, causal, &w);
				losses[j] = output.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / output.len() as f64;
			}
			w.get_mut(name).unwrap().1[i] = original;
			expected.push(((losses[0] - losses[1]) / (2.0 * f64::from(eps))) as f32);
		}
		assert_close(&gradient, &expected, 4.0e-4);
	}
	Ok(())
}

// ── geometry validation ───────────────────────────────────────────────────────

test_vk!(linear_attention_rejects_invalid_construction, engine, {
	use oa::ml::nn::CausalLinearAttention;

	// zero model_width
	assert_eq!(
		CausalLinearAttention::new_with_seed(&engine, 0, 2, 4, 0)
			.err()
			.expect("zero model_width was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	// zero num_heads
	assert_eq!(
		CausalLinearAttention::new_with_seed(&engine, 8, 0, 4, 0)
			.err()
			.expect("zero num_heads was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	// zero sequence_length
	assert_eq!(
		CausalLinearAttention::new_with_seed(&engine, 8, 2, 0, 0)
			.err()
			.expect("zero sequence_length was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	// model_width not divisible by num_heads
	assert_eq!(
		CausalLinearAttention::new_with_seed(&engine, 6, 4, 4, 0)
			.err()
			.expect("indivisible width/heads was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});

test_vk!(linear_attention_rejects_wrong_input_shape, engine, {
	use oa::ml::nn::CausalLinearAttention;

	let attn = CausalLinearAttention::new_with_seed(&engine, 8, 2, 4, 0x4c41)?;
	// Wrong width.
	let bad_width = oa::Matrix::new_from_f32(&engine, [4, 6], &[0.0; 24])?;
	assert_eq!(
		attn
			.forward(&bad_width)
			.err()
			.expect("wrong width was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	// Row count not a multiple of sequence_length.
	let bad_rows = oa::Matrix::new_from_f32(&engine, [3, 8], &[0.0; 24])?;
	assert_eq!(
		attn
			.forward(&bad_rows)
			.err()
			.expect("non-multiple row count was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});

// ── bidirectional forward matches host oracle ─────────────────────────────────

test_vk!(
	bidirectional_linear_attention_matches_host_oracle,
	engine,
	{ model_forward_gate(&engine, false) }
);

// ── causal forward is strictly causal (position i ignores i+1..S-1) ──────────

test_vk!(causal_linear_attention_is_strictly_causal, engine, {
	use oa::ml::nn::CausalLinearAttention;

	// S=4, D=4, H=1, d_head=4, B=1
	let seq = 4_usize;
	let width = 4_usize;
	let count = seq * width;

	let base_vals: Vec<f32> = (0..count)
		.map(|i| ((i * 11 % 23) as f32 - 11.0) / 10.0)
		.collect();

	let attn = CausalLinearAttention::new_with_seed(&engine, width, 1, seq, 0x4341_5553)?;

	// Replace key position 3 (the last) with different values and check that
	// positions 0..2 produce identical output (they cannot see position 3).
	let mut perturbed = base_vals.clone();
	for item in &mut perturbed[3 * width..4 * width] {
		*item = -*item + 0.5;
	}

	let input_base = oa::Matrix::new_from_f32(&engine, [seq, width], &base_vals)?;
	let input_perturbed = oa::Matrix::new_from_f32(&engine, [seq, width], &perturbed)?;

	let out_base = attn.forward(&input_base)?.read_f32()?;
	let out_perturbed = attn.forward(&input_perturbed)?.read_f32()?;

	// Positions 0..2 (rows 0..3*width) must be identical to within float noise.
	assert_close(
		&out_base[0..3 * width],
		&out_perturbed[0..3 * width],
		1.0e-4,
	);
	// Position 3 is allowed to differ.
	assert!(
		out_base[3 * width..]
			.iter()
			.zip(&out_perturbed[3 * width..])
			.any(|(a, b)| (a - b).abs() > 1.0e-4),
		"causal output at final position was identical despite perturbed input"
	);
	Ok(())
});

// ── host oracle agreement (no-proj, direct feature-map inputs) ────────────────

test_vk!(
	linear_attention_forward_matches_host_oracle_on_known_projections,
	engine,
	{ model_forward_gate(&engine, true) }
);

// ── autograd: gradients match finite differences ──────────────────────────────

test_vk!(
	linear_attention_causal_reverse_matches_finite_differences,
	engine,
	{ model_gradient_gate(&engine, true) }
);

test_vk!(
	linear_attention_bidirectional_reverse_matches_finite_differences,
	engine,
	{ model_gradient_gate(&engine, false) }
);

// ── parameter layout ──────────────────────────────────────────────────────────

test_vk!(
	linear_attention_parameter_layout_matches_multi_head_attention,
	engine,
	{
		use oa::ml::{Module, nn::CausalLinearAttention};

		// Biased: 4 weights + 4 biases = 8 named parameters.
		let biased = CausalLinearAttention::new_with_seed(&engine, 8, 2, 4, 0x4c41_4f55)?;
		assert!(biased.has_bias());
		let named = biased.all_named_parameters()?;
		assert_eq!(named.len(), 8);
		assert_eq!(named[0].path(), "q_proj.weight");
		assert_eq!(named[1].path(), "q_proj.bias");
		assert_eq!(named[2].path(), "k_proj.weight");
		assert_eq!(named[3].path(), "k_proj.bias");
		assert_eq!(named[4].path(), "v_proj.weight");
		assert_eq!(named[5].path(), "v_proj.bias");
		assert_eq!(named[6].path(), "out_proj.weight");
		assert_eq!(named[7].path(), "out_proj.bias");

		// Bias-free: 4 weights only.
		let no_bias =
			CausalLinearAttention::new_with_seed_and_bias(&engine, 8, 2, 4, false, 0x4e4f_4249)?;
		assert!(!no_bias.has_bias());
		let named = no_bias.all_named_parameters()?;
		assert_eq!(named.len(), 4);
		assert_eq!(named[0].path(), "q_proj.weight");
		assert_eq!(named[1].path(), "k_proj.weight");
		assert_eq!(named[2].path(), "v_proj.weight");
		assert_eq!(named[3].path(), "out_proj.weight");
		Ok(())
	}
);

// ── API surface ───────────────────────────────────────────────────────────────

test_vk!(linear_attention_accessors_and_mode_switch, engine, {
	use oa::ml::nn::{CausalLinearAttention, LinearAttentionMode};

	let attn = CausalLinearAttention::new_with_seed(&engine, 8, 4, 6, 0x4150_4953)?;
	assert_eq!(attn.model_width(), 8);
	assert_eq!(attn.num_heads(), 4);
	assert_eq!(attn.sequence_length(), 6);
	assert_eq!(attn.mode(), LinearAttentionMode::Causal);

	attn.set_mode(LinearAttentionMode::Bidirectional);
	assert_eq!(attn.mode(), LinearAttentionMode::Bidirectional);

	attn.set_sequence_length(3)?;
	assert_eq!(attn.sequence_length(), 3);

	assert_eq!(
		attn
			.set_sequence_length(0)
			.expect_err("zero sequence_length was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	assert_eq!(attn.sequence_length(), 3);
	Ok(())
});

// ── AttentionBackend::Linear guard on MultiHeadAttention ─────────────────────

test_vk!(multi_head_attention_rejects_linear_backend, engine, {
	use oa::ml::nn::{AttentionBackend, MultiHeadAttention};

	let attn = MultiHeadAttention::new_with_seed(&engine, 8, 2, 4, 0x4c49_4e45)?;
	attn.set_backend(AttentionBackend::Linear);
	let input = oa::Matrix::new_from_f32(&engine, [4, 8], &[0.1_f32; 32])?;
	assert_eq!(
		attn
			.forward(&input)
			.err()
			.expect("Linear backend on MultiHeadAttention was accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});

test_vk!(fused_linear_attention_extremes_tails_and_capture, engine, {
	use oa::ml::matrix::causal_linear_attention;
	for (h, s, d) in [(2, 5, 3), (1, 1, 1), (1, 3, 24), (1, 2, 64)] {
		for extreme in [false, true] {
			let n = h * s * d;
			let q = (0..n)
				.map(|i| {
					if extreme {
						-20.0 - (i % 3) as f32 * 30.0
					} else {
						((i * 7 % 19) as f32 - 9.0) / 8.0
					}
				})
				.collect::<Vec<_>>();
			let k = (0..n)
				.map(|i| {
					if extreme {
						-20.0
					} else {
						((i * 13 % 23) as f32 - 11.0) / 9.0
					}
				})
				.collect::<Vec<_>>();
			let v = (0..n)
				.map(|i| ((i * 11 % 29) as f32 - 14.0) / 7.0)
				.collect::<Vec<_>>();
			let q = oa::Matrix::new_from_f32(&engine, [h, s, d], &q)?;
			let k = oa::Matrix::new_from_f32(&engine, [h, s, d], &k)?;
			let v = oa::Matrix::new_from_f32(&engine, [h, s, d], &v)?;
			let expected = oracle(
				&q.read_f32()?,
				&k.read_f32()?,
				&v.read_f32()?,
				h,
				s,
				d,
				true,
				1e-6,
			);
			let (plan, out) = engine.capture(|| causal_linear_attention(&q, &k, &v, 1e-6))?;
			for _ in 0..2 {
				engine.submit(&plan)?.wait()?;
				let actual = out.read_f32()?;
				assert_close(&actual, &expected, 4e-5);
				if extreme {
					assert!(
						actual.iter().any(|x| *x != 0.0),
						"negative feature branch cancelled to zero"
					);
					for (a, b) in actual.iter().zip(&expected) {
						assert!(
							(a - b).abs() <= 1e-4 * b.abs() + 1e-16,
							"extreme feature value: {a} != {b}"
						);
					}
				}
			}
		}
	}
	let q = oa::Matrix::new_from_f32(&engine, [1, 2, 3], &[0.0; 6])?;
	for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
		assert!(causal_linear_attention(&q, &q, &q, eps).is_err());
	}
	let too_wide = oa::Matrix::new_from_f32(&engine, [1, 1, 65], &[0.0; 65])?;
	assert!(causal_linear_attention(&too_wide, &too_wide, &too_wide, 1e-6).is_err());
	Ok(())
});

test_vk!(
	fused_linear_attention_qkv_gradients_floor_and_alias,
	engine,
	{
		use oa::ml::nn::Embedding;
		for (extreme, alias) in [(false, false), (true, false), (false, true)] {
			let (h, s, d) = (2, 3, 3);
			let n = h * s * d;
			let q = (0..n)
				.map(|i| {
					if extreme {
						-8.0
					} else {
						((i * 7 % 19) as f32 - 9.0) / 8.0
					}
				})
				.collect::<Vec<_>>();
			let k = (0..n)
				.map(|i| {
					if extreme {
						-8.0
					} else {
						((i * 13 % 23) as f32 - 11.0) / 9.0
					}
				})
				.collect::<Vec<_>>();
			let v = (0..n)
				.map(|i| ((i * 11 % 29) as f32 - 14.0) / 7.0)
				.collect::<Vec<_>>();
			let target = (0..n)
				.map(|i| i as f32 / n as f32 - 0.4)
				.collect::<Vec<_>>();
			let tables = [q.clone(), k.clone(), v.clone()]
				.into_iter()
				.map(|x| Embedding::new_from_matrix(oa::Matrix::new_from_f32(&engine, [h * s, d], &x)?))
				.collect::<oa::Result<Vec<_>>>()?;
			let indices =
				oa::Matrix::new_from_slice(&engine, [h * s], &(0..(h * s) as u32).collect::<Vec<_>>())?;
			let target_device = oa::Matrix::new_from_f32(&engine, [h, s, d], &target)?;
			engine.checkpoint()?.wait()?;
			let (plan, loss) = engine.capture(|| {
				let tape = oa::ml::GradientTape::new();
				let q = tables[0].forward(&indices)?.reshape([h, s, d])?;
				let k = if alias {
					q.clone()
				} else {
					tables[1].forward(&indices)?.reshape([h, s, d])?
				};
				let v = if alias {
					q.clone()
				} else {
					tables[2].forward(&indices)?.reshape([h, s, d])?
				};
				let out = oa::ml::matrix::causal_linear_attention(&q, &k, &v, 1e-6)?;
				let loss = oa::ml::loss::mse(&out, &target_device)?;
				tape.backward(&loss)?;
				Ok(loss)
			})?;
			assert!(
				plan
					.semantic_graph()
					.operations()
					.iter()
					.any(|op| op.name() == "oa::ml::matrix::causal_linear_attention_backward")
			);
			for _ in 0..2 {
				engine.submit(&plan)?.wait()?;
				assert!(loss.read_f32()?[0].is_finite());
			}
			let mut raw = [q, k, v];
			for index in 0..if alias { 1 } else { 3 } {
				let actual = tables[index]
					.weight()
					.gradient()
					.expect("missing QKV adjoint")
					.read_f32()?;
				let mut expected = Vec::new();
				for i in 0..n {
					let original = raw[index][i];
					let eps = 0.002_f32;
					let mut losses = [0.0; 2];
					for (side, delta) in [eps, -eps].into_iter().enumerate() {
						raw[index][i] = original + delta;
						let (k, v) = if alias {
							(&raw[0], &raw[0])
						} else {
							(&raw[1], &raw[2])
						};
						let out = oracle(&raw[0], k, v, h, s, d, true, 1e-6);
						losses[side] = out
							.iter()
							.zip(&target)
							.map(|(a, b)| f64::from(a - b).powi(2))
							.sum::<f64>()
							/ n as f64;
					}
					raw[index][i] = original;
					expected.push(((losses[0] - losses[1]) / (2.0 * f64::from(eps))) as f32);
				}
				assert_close(&actual, &expected, 8e-5);
			}
		}
		Ok(())
	}
);
