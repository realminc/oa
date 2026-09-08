use std::any::TypeId;

#[test]
fn root_exports_are_identities_of_runtime_contracts() -> oa::Result<()> {
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
	assert_eq!(
		TypeId::of::<oa::ExecutionPlan>(),
		TypeId::of::<oa::runtime::ExecutionPlan>()
	);
	assert_eq!(
		TypeId::of::<oa::ExecutionPlanDiagnostics>(),
		TypeId::of::<oa::runtime::ExecutionPlanDiagnostics>()
	);
	assert_eq!(
		TypeId::of::<oa::LogOptions>(),
		TypeId::of::<oa::runtime::LogOptions>()
	);
	assert_eq!(
		TypeId::of::<oa::LogLevel>(),
		TypeId::of::<oa::runtime::LogLevel>()
	);
	assert_eq!(
		TypeId::of::<oa::LogComponent>(),
		TypeId::of::<oa::runtime::LogComponent>()
	);

	let logging = oa::LogOptions::new()
		.minimum_level(oa::LogLevel::Warn)
		.console_output(false);
	assert_eq!(logging.level(), oa::LogLevel::Warn);
	let _: oa::EngineBuilder = oa::Engine::builder()
		.devices(oa::DeviceSelection::Automatic)
		.logging(logging);
	assert_eq!(oa::LogComponent::new("DNA")?.tag(), "DNA");
	if false {
		oa::log_info!(oa::LogComponent::APP, "external macro surface {}", 1);
	}
	Ok(())
}
