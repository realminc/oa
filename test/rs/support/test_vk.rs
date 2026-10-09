/// Define an explicitly requested Vulkan test with a fresh thread-affine engine.
///
/// Rust's parallel test harness cannot share the current `!Send + !Sync`
/// engine. Keeping construction here removes call-site boilerplate while
/// preserving per-test isolation and an explicit ignored-test requirement.
macro_rules! test_vk {
	($name:ident, $engine:ident, $body:block) => {
		test_vk!(
			$name,
			$engine,
			"requires an admitted Vulkan compute device",
			$body
		);
	};
	($name:ident, $engine:ident, $requirement:literal, $body:block) => {
		#[test]
		#[ignore = $requirement]
		fn $name() -> oa::Result<()> {
			let selection = match std::env::var("OA_TEST_DEVICE_INDEX") {
				Ok(index) => oa::DeviceSelection::Index(
					index
						.parse()
						.expect("OA_TEST_DEVICE_INDEX must be a nonnegative integer"),
				),
				Err(std::env::VarError::NotPresent) => oa::DeviceSelection::Automatic,
				Err(_) => panic!("OA_TEST_DEVICE_INDEX must be Unicode"),
			};
			let $engine = oa::Engine::builder().devices(selection).build()?;
			$body
		}
	};
}
