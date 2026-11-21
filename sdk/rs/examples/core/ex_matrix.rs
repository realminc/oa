// OA_DOC_BEGIN: core-matrix-add
use oa::matrix;

fn main() -> oa::Result<()> {
	let one = matrix::ones([2, 3])?;
	let two = matrix::full([2, 3], 2.0)?;
	let result = matrix::add(&one, &two)?;

	let values = result.read_f32()?;
	assert_eq!(values.len(), 6);
	assert!(values.iter().all(|&v| (v - 3.0).abs() <= 1e-6));

	println!("Matrix addition verified: every value is 3");
	Ok(())
}
// OA_DOC_END: core-matrix-add
