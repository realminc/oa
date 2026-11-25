use oa::{FusionStats, Matrix, matrix};

fn node_outcome_sum(stats: FusionStats) -> u64 {
	stats.disabled_node_count()
		+ stats.not_elementwise_node_count()
		+ stats.policy_excluded_node_count()
		+ stats.hazard_node_count()
		+ stats.isolated_node_count()
		+ stats.extent_boundary_node_count()
		+ stats.cross_invocation_node_count()
		+ stats.program_rejected_node_count()
		+ stats.foreign_device_node_count()
		+ stats.tape_bound_node_count()
		+ stats.tape_capacity_node_count()
		+ stats.resource_node_count()
		+ stats.specialization_declined_node_count()
		+ stats.tape_fused_node_count()
		+ stats.specialized_fused_node_count()
}

fn values(count: usize) -> Vec<f32> {
	(0..count)
		.map(|index| ((index % 29) as f32 - 14.0) / 16.0)
		.collect()
}

#[test]
fn fusion_stats_default_is_zero_and_formats_every_counter() {
	let stats = FusionStats::default();
	assert_eq!(stats.examined_node_count(), 0);
	assert_eq!(node_outcome_sum(stats), 0);
	let text = stats.to_string();
	assert!(text.starts_with("examined=0 "));
	assert!(text.ends_with(" dnn_fallback=0"));
	assert!(text.split(' ').all(|field| field.ends_with("=0")));
}

test_vk!(
	fusion_stats_account_for_every_examined_eager_node,
	runtime,
	{
		let before = runtime.fusion_stats();
		assert_eq!(
			before,
			runtime.fusion_stats(),
			"an idle query must not fuse"
		);
		let elements = 1024;
		let input = Matrix::new_from_f32([1, elements], &values(elements))?;
		let chain = matrix::exp(&matrix::scale(&matrix::add(&input, &input)?, 0.25)?)?;
		let expected = values(elements)
			.iter()
			.map(|value| (value * 2.0 * 0.25).exp())
			.collect::<Vec<_>>();
		let actual = chain.read_f32()?;
		for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
			assert!(
				(actual - expected).abs() <= 1e-5 * expected.abs().max(1.0),
				"element {index}: {actual} != {expected}"
			);
		}
		let after = runtime.fusion_stats();
		assert!(after.pass_count() > before.pass_count());
		let examined = after.examined_node_count() - before.examined_node_count();
		assert!(examined >= 3, "three elementwise nodes were offered");
		assert_eq!(
			node_outcome_sum(after) - node_outcome_sum(before),
			examined,
			"every examined node has exactly one outcome"
		);
		let fused = (after.tape_fused_node_count() + after.specialized_fused_node_count())
			- (before.tape_fused_node_count() + before.specialized_fused_node_count());
		assert!(fused >= 3, "the add/scale/exp chain fuses: {after}");
		if !oa::EnvFlag::is_set("OA_JIT_FUSION") {
			assert!(after.jit_disabled_count() > before.jit_disabled_count());
			assert_eq!(after.specialized_fused_node_count(), 0);
		}
		Ok(())
	}
);

test_vk!(fusion_stats_attribute_a_lone_elementwise_node, runtime, {
	let elements = 512;
	let input = Matrix::new_from_f32([1, elements], &values(elements))?;
	input.read_f32()?;
	let before = runtime.fusion_stats();
	let output = matrix::exp(&input)?;
	output.read_f32()?;
	let after = runtime.fusion_stats();
	let lone = (after.isolated_node_count() + after.extent_boundary_node_count())
		- (before.isolated_node_count() + before.extent_boundary_node_count());
	assert!(lone >= 1, "a single exp has no fusible neighbour: {after}");
	assert_eq!(
		node_outcome_sum(after) - node_outcome_sum(before),
		after.examined_node_count() - before.examined_node_count()
	);
	Ok(())
});

test_vk!(fusion_stats_do_not_change_on_plan_replay, runtime, {
	let elements = 1024;
	let input = Matrix::new_from_f32([1, elements], &values(elements))?;
	input.read_f32()?;
	let (plan, output) =
		runtime.capture(|| matrix::exp(&matrix::scale(&matrix::add(&input, &input)?, 0.5)?))?;
	let captured = runtime.fusion_stats();
	for _ in 0..3 {
		runtime.submit(&plan)?.wait()?;
	}
	output.read_f32()?;
	assert_eq!(runtime.fusion_stats(), captured);
	Ok(())
});
