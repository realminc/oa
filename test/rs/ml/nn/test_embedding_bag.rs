//! EmbeddingBag module contracts: bag reduction, offsets validation, registry.

use oa::ml::nn::{EmbeddingBag, EmbeddingBagMode};
use oa::ml::{GradientTape, Module};

fn table() -> oa::Result<oa::Matrix> {
	// Distinct rows make mean/sum results exact on the host oracle.
	oa::Matrix::new_from_f32(
		[5, 3],
		&[
			1.0, 2.0, 3.0, // row 0
			4.0, 5.0, 6.0, // row 1
			7.0, 8.0, 9.0, // row 2
			1.0, 0.0, -1.0, // row 3
			0.5, 0.5, 0.5, // row 4
		],
	)
}

test_vk!(embedding_bag_mean_matches_host_oracle, runtime, {
	let bag = EmbeddingBag::new_from_matrix(table()?, EmbeddingBagMode::Mean)?;
	// bags: [rows 0,1] and [rows 2,3,4]
	let indices = oa::Matrix::new_from_slice([5], &[0_u32, 1, 2, 3, 4])?;
	let output = bag.forward(&indices, &[0, 2, 5])?;
	assert_eq!(output.shape(), &[2, 3]);
	let values = output.read_f32()?;
	let expected = [
		(1.0 + 4.0) / 2.0,
		(2.0 + 5.0) / 2.0,
		(3.0 + 6.0) / 2.0,
		(7.0 + 1.0 + 0.5) / 3.0,
		(8.0 + 0.0 + 0.5) / 3.0,
		(9.0 - 1.0 + 0.5) / 3.0,
	];
	for (i, (a, e)) in values.iter().zip(expected.iter()).enumerate() {
		assert!(
			(a - e).abs() <= 1.0e-5,
			"bag mean deviates at {i}: expected {e}, found {a}"
		);
	}
	Ok(())
});

test_vk!(
	embedding_bag_sum_mode_and_gradients_reach_weight,
	runtime,
	{
		let bag = EmbeddingBag::new_from_matrix(table()?, EmbeddingBagMode::Sum)?;
		assert_eq!(bag.mode(), EmbeddingBagMode::Sum);
		let indices = oa::Matrix::new_from_slice([3], &[1_u32, 2, 4])?;
		let output = bag.forward(&indices, &[0, 1, 3])?;
		let values = output.read_f32()?;
		let expected = [4.0, 5.0, 6.0, 7.5, 8.5, 9.5];
		for (a, e) in values.iter().zip(expected.iter()) {
			assert!((a - e).abs() <= 1.0e-5);
		}
		// Registry exposes the shared lookup table under the child path.
		let names: Vec<String> = bag
			.all_named_parameters()?
			.iter()
			.map(|named| named.path().to_string())
			.collect();
		assert_eq!(names, ["embedding.weight"]);

		let tape = GradientTape::new();
		let output = bag.forward(&indices, &[0, 1, 3])?;
		let loss = oa::ml::loss::mse(&output, &oa::Matrix::new_from_f32([2, 3], &[0.0; 6])?)?;
		tape.backward(&loss)?;
		let gradient = bag.embedding().weight().gradient();
		assert!(gradient.is_some());
		assert_eq!(gradient.expect("weight gradient").shape(), &[5, 3]);
		Ok(())
	}
);

test_vk!(
	embedding_bag_rejects_malformed_offsets_and_empty_bags,
	runtime,
	{
		let bag = EmbeddingBag::new_from_matrix(table()?, EmbeddingBagMode::Mean)?;
		let indices = oa::Matrix::new_from_slice([4], &[0_u32, 1, 2, 3])?;
		assert!(bag.forward(&indices, &[0]).is_err()); // < 2 boundaries
		assert!(bag.forward(&indices, &[0, 3]).is_err()); // not ending at count
		assert!(bag.forward(&indices, &[0, 2, 2, 4]).is_err()); // empty bag
		assert!(bag.forward(&indices, &[0, 3, 2, 4]).is_err()); // decreasing
		assert!(bag.forward(&indices, &[1, 4]).is_err()); // not starting at 0
		assert!(bag.forward(&indices, &[]).is_err());
		assert!(
			Module::forward(&bag, &indices).is_err(),
			"single-input Module::forward must reject"
		);
		Ok(())
	}
);

test_vk!(
	embedding_bag_normal_constructor_is_deterministic,
	runtime,
	{
		let first =
			EmbeddingBag::new_with_seed_normal_on(&runtime, 8, 4, 0.5, EmbeddingBagMode::Mean, 0xE8A6)?;
		let second =
			EmbeddingBag::new_with_seed_normal_on(&runtime, 8, 4, 0.5, EmbeddingBagMode::Mean, 0xE8A6)?;
		let a = first.embedding().weight().data().read_f32()?;
		let b = second.embedding().weight().data().read_f32()?;
		assert_eq!(a, b, "same-seed normal init must be deterministic");
		// Sample variance should sit near stddev^2 = 0.25 for 32 draws.
		let mean = a.iter().sum::<f32>() / a.len() as f32;
		let variance = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / a.len() as f32;
		assert!(
			variance > 0.05 && variance < 1.0,
			"normal init variance {variance} strays far from 0.25"
		);
		Ok(())
	}
);
