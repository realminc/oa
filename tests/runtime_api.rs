use std::any::TypeId;

#[test]
fn root_exports_are_identities_of_runtime_contracts() {
	assert_eq!(
		TypeId::of::<oa::EngineBuilder>(),
		TypeId::of::<oa::runtime::EngineBuilder>()
	);
	assert_eq!(
		TypeId::of::<oa::DeviceSelection>(),
		TypeId::of::<oa::runtime::DeviceSelection>()
	);
	assert_eq!(
		TypeId::of::<oa::Event>(),
		TypeId::of::<oa::runtime::Event>()
	);

	let _: oa::EngineBuilder = oa::Engine::builder().devices(oa::DeviceSelection::Automatic);
}
