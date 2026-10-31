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

/// Define a heavyweight Vulkan test that additionally requires `OA_TEST_SLOW`.
///
/// Corpus and multi-parameter PQC proofs compile several ~600 KiB SPIR-V
/// modules per fresh engine; on weak drivers that means minutes of shader
/// compilation per test. These tests stay `#[ignore]`d like [`test_vk!`] and
/// early-return unless `OA_TEST_SLOW` names a truthy value, so a plain
/// `--ignored` run stays fast and `OA_TEST_SLOW=1` opts into the full battery.
#[allow(unused_macros)]
macro_rules! test_vk_slow {
	($name:ident, $engine:ident, $body:block) => {
		test_vk_slow!(
			$name,
			$engine,
			"requires OA_TEST_SLOW=1 and an admitted Vulkan compute device",
			$body
		);
	};
	($name:ident, $engine:ident, $requirement:literal, $body:block) => {
		#[test]
		#[ignore = $requirement]
		fn $name() -> oa::Result<()> {
			if !oa::EnvFlag::is_set("OA_TEST_SLOW") {
				eprintln!(concat!(
					stringify!($name),
					": skipped (set OA_TEST_SLOW=1 to run)"
				));
				return Ok(());
			}
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
