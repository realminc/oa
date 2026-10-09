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
		TypeId::of::<oa::DeviceSummary>(),
		TypeId::of::<oa::runtime::DeviceSummary>()
	);
	assert_eq!(
		TypeId::of::<oa::DeviceInfo>(),
		TypeId::of::<oa::runtime::DeviceInfo>()
	);
	assert_eq!(
		TypeId::of::<oa::DeviceKind>(),
		TypeId::of::<oa::runtime::DeviceKind>()
	);
	assert_eq!(
		TypeId::of::<oa::EngineRequirements>(),
		TypeId::of::<oa::runtime::EngineRequirements>()
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
		TypeId::of::<oa::CapturedResourceDesc>(),
		TypeId::of::<oa::runtime::CapturedResourceDesc>()
	);
	assert_eq!(
		TypeId::of::<oa::SemanticStorageBinding>(),
		TypeId::of::<oa::runtime::SemanticStorageBinding>()
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
		.requirements(oa::EngineRequirements::default().graphics())
		.logging(logging);
	assert_eq!(oa::LogComponent::new("DNA")?.tag(), "DNA");
	if false {
		oa::log_info!(oa::LogComponent::APP, "external macro surface {}", 1);
	}
	Ok(())
}

#[test]
fn engine_requirements_are_composable_and_deduplicate_exact_profiles() {
	let decode = oa::video::VideoDecodeProfile::h264_420_8bit(oa::video::H264Profile::High);
	let encode = oa::video::VideoEncodeProfile::h265_main_420_8bit();
	let once = oa::EngineRequirements::default()
		.presentation()
		.video_decode(decode)
		.video_encode(encode);
	let repeated = oa::EngineRequirements::default()
		.presentation()
		.video_decode(decode)
		.video_decode(decode)
		.video_encode(encode)
		.video_encode(encode);
	assert_eq!(once, repeated);
}
