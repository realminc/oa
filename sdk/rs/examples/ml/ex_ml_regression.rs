// OA_DOC_BEGIN: ml-regression
use oa::{Engine, Matrix, ml};

fn main() -> oa::Result<()> {
	let engine = Engine::new()?;

	let x = Matrix::from_f32(&engine, [5, 1], &[-2.0, -1.0, 0.0, 1.0, 2.0])?;
	let target = Matrix::from_f32(&engine, [5, 1], &[-3.0, -1.0, 1.0, 3.0, 5.0])?;

	let model = ml::nn::Linear::with_seed(&engine, 1, 1, 0)?;
	let mut optimizer = ml::Sgd::new(model.parameters(), 0.05, 0.0, 0.0)?;

	let mut initial_loss = 0.0_f32;
	let mut loss_matrix = Matrix::from_f32(&engine, [1], &[0.0])?;

	for step in 0..80 {
		optimizer.zero_grad();
		let tape = ml::GradientTape::new();
		let prediction = model.forward(&x)?;
		loss_matrix = ml::loss::mse(&prediction, &target)?;
		tape.backward(&loss_matrix)?;
		drop(tape);
		optimizer.step()?;

		if step == 0 {
			initial_loss = loss_matrix.read_f32()?[0];
		}
	}

	let final_loss = loss_matrix.read_f32()?[0];
	let values = model.forward(&x)?.read_f32()?;

	assert!(final_loss < initial_loss * 0.01);
	assert!(
		values
			.iter()
			.zip([-3.0_f32, -1.0, 1.0, 3.0, 5.0])
			.all(|(&a, b)| (a - b).abs() < 0.1)
	);

	println!("loss: {initial_loss:.6} -> {final_loss:.6}");
	println!("{values:?}");
	Ok(())
}
// OA_DOC_END: ml-regression
