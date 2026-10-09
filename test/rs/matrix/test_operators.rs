//! Operators share fallible semantic operations; no panic or implicit readback.

#[test]
fn operator_outputs_preserve_result_contract() {
	fn binary<T: std::ops::Add<Output = oa::Result<oa::Matrix>>>() {}
	fn unary<T: std::ops::Neg<Output = oa::Result<oa::Matrix>>>() {}
	binary::<&oa::Matrix>();
	unary::<&oa::Matrix>();
	unary::<oa::Matrix>();
}

test_vk!(
	operators_match_named_operations_and_propagate_errors,
	engine,
	{
		let a = oa::Matrix::from_f32(&engine, [2], &[2.0, 4.0])?;
		let b = oa::Matrix::from_f32(&engine, [2], &[1.0, 2.0])?;
		for (actual, expected) in [
			((&a + &b)?, vec![3.0, 6.0]),
			((&a - &b)?, vec![1.0, 2.0]),
			((&a * &b)?, vec![2.0, 8.0]),
			(a.mul(&b)?, vec![2.0, 8.0]),
			(oa::Matrix::mul(&a, &b)?, vec![2.0, 8.0]),
			(oa::matrix::mul(&a, &b)?, vec![2.0, 8.0]),
			(a.add(&b)?, vec![3.0, 6.0]),
			(a.sub(&b)?, vec![1.0, 2.0]),
			(a.div(&b)?, vec![2.0, 2.0]),
			((&a / &b)?, vec![2.0, 2.0]),
			((-&a)?, vec![-2.0, -4.0]),
			((-a.clone())?, vec![-2.0, -4.0]),
			((&a + 1.0)?, vec![3.0, 5.0]),
			((&a - 1.0)?, vec![1.0, 3.0]),
			((&a * 2.0)?, vec![4.0, 8.0]),
			((&a / 2.0)?, vec![1.0, 2.0]),
			((2.0 * &a)?, vec![4.0, 8.0]),
			((2.0 + &a)?, vec![4.0, 6.0]),
		] {
			assert_eq!(actual.read_f32()?, expected);
		}
		let bad = oa::Matrix::from_f32(&engine, [3], &[1.0; 3])?;
		let named_error = oa::matrix::add(&a, &bad).err().expect("bad shape");
		let operator_error = (&a + &bad).err().expect("fallible operator");
		assert_eq!(named_error.kind(), operator_error.kind());
		assert_eq!(named_error.message(), operator_error.message());
		let method_error = a.add(&bad).err().expect("fallible method");
		assert_eq!(named_error.kind(), method_error.kind());
		assert_eq!(named_error.message(), method_error.message());
		Ok(())
	}
);
