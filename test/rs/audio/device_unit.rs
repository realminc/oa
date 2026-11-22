use super::*;

#[test]
fn usable_default_does_not_probe_bridges() {
	let bridges = std::iter::from_fn(|| panic!("default must take precedence"));
	assert_eq!(
		select_candidate(Some("default"), bridges, |_| Ok(48_000)).unwrap(),
		("default", 48_000)
	);
}

#[test]
fn failed_default_tries_server_bridges_in_order() {
	let mut attempts = Vec::new();
	let selected = select_candidate(Some("default"), ["pipewire", "pulse"], |device| {
		attempts.push(*device);
		if *device == "pulse" {
			Ok(2)
		} else {
			Err(Error::missing_capability("unavailable"))
		}
	})
	.unwrap();
	assert_eq!(selected, ("pulse", 2));
	assert_eq!(attempts, ["default", "pipewire", "pulse"]);
}

#[test]
fn all_failures_preserve_the_original_error() {
	let result = select_candidate(Some("default"), ["pipewire"], |device| {
		Err::<(), _>(Error::missing_capability(*device))
	});
	assert_eq!(result.unwrap_err().message(), "default");
	assert!(select_candidate::<&str, ()>(None, [], |_| Ok(())).is_err());
}
