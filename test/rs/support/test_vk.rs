/// Define a hardware Vulkan test with a fresh thread-affine engine.
///
/// Rust's parallel test harness cannot share the current `!Send + !Sync`
/// engine. Keeping construction here removes call-site boilerplate while
/// preserving per-test isolation and an explicit ignored-test requirement.
macro_rules! test_vk {
	($name:ident, $engine:ident, $body:block) => {
		test_vk!(
			$name,
			$engine,
			"requires a hardware Vulkan 1.3 compute device",
			$body
		);
	};
	($name:ident, $engine:ident, $requirement:literal, $body:block) => {
		#[test]
		#[ignore = $requirement]
		fn $name() -> oa::Result<()> {
			let $engine = oa::Engine::new()?;
			$body
		}
	};
}
