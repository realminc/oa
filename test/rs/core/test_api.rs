use std::any::TypeId;

#[test]
fn root_exports_are_identities_of_core_contracts() {
	assert_eq!(TypeId::of::<oa::Matrix>(), TypeId::of::<oa::core::Matrix>());
	assert_eq!(TypeId::of::<oa::Image>(), TypeId::of::<oa::core::Image>());
	assert_eq!(TypeId::of::<oa::DType>(), TypeId::of::<oa::core::DType>());
	assert_eq!(TypeId::of::<oa::Error>(), TypeId::of::<oa::core::Error>());

	let result: oa::core::Result<()> = Ok(());
	let _: oa::Result<()> = result;
}

#[test]
fn admitted_rust_elements_map_to_exact_dense_dtypes() {
	fn dtype_of<T: oa::core::Element>() -> oa::DType {
		T::DTYPE
	}

	assert_eq!(dtype_of::<f32>(), oa::DType::F32);
	assert_eq!(dtype_of::<i32>(), oa::DType::I32);
	assert_eq!(dtype_of::<u32>(), oa::DType::U32);
	assert_eq!(oa::DType::F32.token(), "f32");
	assert_eq!(oa::DType::I32.token(), "i32");
	assert_eq!(oa::DType::U32.token(), "u32");
	assert_eq!(oa::DType::F32.size_bytes(), size_of::<f32>());
	assert_eq!(oa::DType::I32.size_bytes(), size_of::<i32>());
	assert_eq!(oa::DType::U32.size_bytes(), size_of::<u32>());
}
