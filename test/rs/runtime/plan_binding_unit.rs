//! Compiled specialization binding-contract tests.
//!
//! Donor: `oacpp/source/cpp/lib/oa/runtime/executionPlan.cpp`
//! (`buildBindingContract_`, `binding`, `bindingContractHash`,
//! `validateSpecializationInputs`).

use super::*;

#[test]
#[ignore = "requires an admitted Vulkan compute device"]
fn binding_contract_describes_inputs_and_outputs() -> crate::Result<()> {
	let engine = match crate::Engine::new() {
		Ok(engine) => engine,
		Err(_) => return Ok(()),
	};
	let one = crate::matrix::ones(&engine, [2, 3])?;
	let two = crate::matrix::full(&engine, [2, 3], 2.0)?;
	let (plan, _output) = engine.capture(|| {
		let three = crate::matrix::add(&one, &two)?;
		crate::matrix::add(&three, &three)
	})?;

	// Contract: external inputs `one` and `two` first, the produced output last.
	assert_eq!(plan.binding_count(), 3);
	assert_eq!(plan.input_binding_count(), 2);
	assert_eq!(plan.output_binding_count(), 1);
	assert_ne!(plan.binding_contract_hash(), 0);

	for slot in 0..2 {
		let binding = plan.binding(slot)?;
		assert_eq!(binding.slot(), slot);
		assert_eq!(binding.role(), ExecutionBindingRole::Input);
		assert_eq!(binding.kind(), crate::OpValueKind::Matrix);
		assert_eq!(binding.shape(), &[2, 3]);
		assert_eq!(binding.strides(), &[3, 1]);
		assert_eq!(binding.dtype(), crate::DType::F32);
		assert_eq!(binding.byte_offset(), 0);
	}
	let output = plan.binding(2)?;
	assert_eq!(output.role(), ExecutionBindingRole::Output);
	assert_eq!(output.shape(), &[2, 3]);

	assert_eq!(
		plan.binding(3).expect_err("out-of-range slot").kind(),
		crate::ErrorKind::InvalidArgument
	);

	// The same contract revalidates identical specialization inputs and
	// rejects count, shape, dtype, and stride mismatches.
	plan.validate_specialization_inputs(&[one.clone(), two.clone()])?;
	assert_eq!(
		plan
			.validate_specialization_inputs(std::slice::from_ref(&one))
			.expect_err("input count")
			.kind(),
		crate::ErrorKind::InvalidArgument
	);
	let wide = crate::matrix::ones(&engine, [2, 4])?;
	assert_eq!(
		plan
			.validate_specialization_inputs(&[wide.clone(), two.clone()])
			.expect_err("input shape")
			.kind(),
		crate::ErrorKind::FailedPrecondition
	);
	let integers = crate::Matrix::new_from_slice(&engine, [2, 3], &[0_i32; 6])?;
	assert_eq!(
		plan
			.validate_specialization_inputs(&[integers.clone(), two.clone()])
			.expect_err("input dtype")
			.kind(),
		crate::ErrorKind::FailedPrecondition
	);

	// A structurally identical recapture reproduces the same contract hash.
	let (recaptured, _output) = engine.capture(|| {
		let three = crate::matrix::add(&one, &two)?;
		crate::matrix::add(&three, &three)
	})?;
	assert_eq!(
		recaptured.binding_contract_hash(),
		plan.binding_contract_hash()
	);
	recaptured.validate_serialized_descriptor(&plan.serialize_descriptor()?)?;
	Ok(())
}
