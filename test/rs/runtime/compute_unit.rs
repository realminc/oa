use super::*;

/// Compile the embedded `matrix.add.f32` artifact as an extension kernel
/// and prove an end-to-end dispatch against real matrices. The artifact is
/// a known-good SPIR-V stream, so failures isolate the extension path
/// rather than shader authoring.
#[test]
#[ignore = "requires a Vulkan device"]
fn extension_add_computes() -> Result<()> {
	let engine = Engine::new()?;
	let spirv = include_bytes!(concat!(env!("OA_SHADER_DIR"), "/matrix_add_f32.spv"));
	let kernel = Kernel::compile(&engine, "extension.test.matrix_add", spirv, 3, [256, 1, 1])?;
	assert_eq!(kernel.binding_count(), 3);
	assert_eq!(kernel.push_words(), 1);

	let a = Matrix::new_from_f32([2, 2], &[1.0, 2.0, 3.0, 4.0])?;
	let b = Matrix::new_from_f32([2, 2], &[10.0, 20.0, 30.0, 40.0])?;
	let out = Matrix::new_from_f32([2, 2], &[0.0; 4])?;
	engine.dispatch_kernel(
		&kernel,
		&[
			KernelBinding::Read(&a),
			KernelBinding::Read(&b),
			KernelBinding::Write(&out),
		],
		&[KernelPush::U32(4)],
		[1, 1, 1],
	)?;
	engine.checkpoint()?.wait()?;
	assert_eq!(out.read_f32()?, vec![11.0, 22.0, 33.0, 44.0]);
	Ok(())
}
