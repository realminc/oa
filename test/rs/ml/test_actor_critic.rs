use std::collections::BTreeMap;

use oa::ml::{CategoricalActorCritic, CategoricalActorCriticConfig, Module, NamedParameter};

type ParameterValues = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

fn parameters(module: &dyn Module) -> oa::Result<ParameterValues> {
	module
		.all_named_parameters()?
		.into_iter()
		.map(|named: NamedParameter| {
			let matrix = named.parameter().data();
			Ok((
				named.path().to_owned(),
				(matrix.shape().to_vec(), matrix.read_f32()?),
			))
		})
		.collect()
}

fn linear(
	input: &[f32],
	batch: usize,
	input_width: usize,
	output_width: usize,
	weight: &[f32],
	bias: &[f32],
	relu: bool,
) -> Vec<f32> {
	let mut output = vec![0.0; batch * output_width];
	for row in 0..batch {
		for out in 0..output_width {
			let mut value = bias[out];
			for inner in 0..input_width {
				value += input[row * input_width + inner] * weight[out * input_width + inner];
			}
			output[row * output_width + out] = if relu { value.max(0.0) } else { value };
		}
	}
	output
}

fn parameter<'a>(parameters: &'a ParameterValues, path: &str) -> &'a [f32] {
	&parameters.get(path).expect("registered parameter").1
}

test_vk!(
	categorical_actor_critic_matches_independent_cpu_towers,
	engine,
	{
		let config = CategoricalActorCriticConfig {
			observation_size: 3,
			action_count: 2,
			hidden_size: 4,
			seed: 0x1234_5678,
		};
		let model = CategoricalActorCritic::new(&engine, config)?;
		let input_values = [1.0, -2.0, 0.5, -0.25, 0.75, 2.0];
		let input = oa::Matrix::from_f32(&engine, [2, 3], &input_values)?;
		let output = model.evaluate(&input)?;
		assert_eq!(output.logits.shape(), [2, 2]);
		assert_eq!(output.value.shape(), [2]);

		let parameters = parameters(&model)?;
		let expected_paths = [
			"policy.bias",
			"policy.weight",
			"policy_0.bias",
			"policy_0.weight",
			"policy_1.bias",
			"policy_1.weight",
			"value.bias",
			"value.weight",
			"value_0.bias",
			"value_0.weight",
			"value_1.bias",
			"value_1.weight",
		];
		assert_eq!(
			parameters.keys().map(String::as_str).collect::<Vec<_>>(),
			expected_paths
		);

		let policy_0 = linear(
			&input_values,
			2,
			3,
			4,
			parameter(&parameters, "policy_0.weight"),
			parameter(&parameters, "policy_0.bias"),
			true,
		);
		let policy_1 = linear(
			&policy_0,
			2,
			4,
			4,
			parameter(&parameters, "policy_1.weight"),
			parameter(&parameters, "policy_1.bias"),
			true,
		);
		let expected_logits = linear(
			&policy_1,
			2,
			4,
			2,
			parameter(&parameters, "policy.weight"),
			parameter(&parameters, "policy.bias"),
			false,
		);
		let value_0 = linear(
			&input_values,
			2,
			3,
			4,
			parameter(&parameters, "value_0.weight"),
			parameter(&parameters, "value_0.bias"),
			true,
		);
		let value_1 = linear(
			&value_0,
			2,
			4,
			4,
			parameter(&parameters, "value_1.weight"),
			parameter(&parameters, "value_1.bias"),
			true,
		);
		let expected_value = linear(
			&value_1,
			2,
			4,
			1,
			parameter(&parameters, "value.weight"),
			parameter(&parameters, "value.bias"),
			false,
		);
		for (actual, expected) in output.logits.read_f32()?.into_iter().zip(expected_logits) {
			assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
		}
		for (actual, expected) in output.value.read_f32()?.into_iter().zip(expected_value) {
			assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
		}
		Ok(())
	}
);

test_vk!(
	categorical_actor_critic_rejects_invalid_observations,
	engine,
	{
		assert!(
			CategoricalActorCritic::new(
				&engine,
				CategoricalActorCriticConfig {
					observation_size: 3,
					action_count: 1,
					hidden_size: 4,
					seed: 1,
				},
			)
			.is_err()
		);
		let model = CategoricalActorCritic::new(
			&engine,
			CategoricalActorCriticConfig {
				observation_size: 3,
				action_count: 2,
				hidden_size: 4,
				seed: 1,
			},
		)?;
		assert!(
			model
				.evaluate(&oa::Matrix::from_f32(&engine, [2, 2], &[0.0; 4])?)
				.is_err()
		);
		assert!(
			model
				.evaluate(&oa::Matrix::from_slice(&engine, [2, 3], &[0_i32; 6])?)
				.is_err()
		);
		Ok(())
	}
);
