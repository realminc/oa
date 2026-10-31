//! Private host-device selection shared by playback and capture.
//!
//! Donor: `source/cpp/lib/oa/audio/audioPlayer.cpp` and `audioCapture.cpp`.
//! Adapter-policy adaptation: miniaudio negotiated desktop-server backends;
//! CPAL's Linux ALSA host needs an explicit bridge when `default` still names
//! physical hardware occupied by PipeWire/PulseAudio. Preserve a usable default
//! first, then try only installed server-default bridges, never arbitrary cards.

use cpal::{
	Device, Host, HostId, SupportedStreamConfig,
	traits::{DeviceTrait, HostTrait},
};

use super::player::{cpal_error, select_config};
use crate::{Engine, Error, LogComponent, LogLevel, Result};

#[derive(Clone, Copy)]
pub(super) enum Direction {
	Input,
	Output,
}

/// Choose an exact PCM configuration without opening a callback stream.
pub(super) fn select_device(
	engine: &Engine,
	host: &Host,
	direction: Direction,
	sample_rate: u32,
	channels: u32,
) -> Result<(Device, SupportedStreamConfig)> {
	let default = match direction {
		Direction::Input => host.default_input_device(),
		Direction::Output => host.default_output_device(),
	};
	let bridges = bridge_names(host.id())
		.iter()
		.filter_map(|name| host.device_by_id(&cpal::DeviceId::new(host.id(), name)));
	let selected = select_candidate(default, bridges, |device| match direction {
		Direction::Input => select_config(
			device
				.supported_input_configs()
				.map_err(|error| cpal_error("input configuration query", error))?,
			sample_rate,
			channels,
			"input",
		),
		Direction::Output => select_config(
			device
				.supported_output_configs()
				.map_err(|error| cpal_error("output configuration query", error))?,
			sample_rate,
			channels,
			"output",
		),
	})?;
	if let Ok(id) = selected.0.id()
		&& bridge_names(host.id()).contains(&id.id())
	{
		engine.log(
			LogLevel::Warn,
			LogComponent::AUDIO,
			format!(
				"Configured audio default unavailable; using {} desktop-server bridge",
				id.id()
			),
		)?;
	}
	Ok(selected)
}

fn bridge_names(host: HostId) -> &'static [&'static str] {
	#[cfg(any(
		target_os = "linux",
		target_os = "freebsd",
		target_os = "dragonfly",
		target_os = "netbsd"
	))]
	if host == HostId::Alsa {
		return &["pipewire", "pulse"];
	}
	let _ = host;
	&[]
}

fn select_candidate<D, C>(
	default: Option<D>,
	bridges: impl IntoIterator<Item = D>,
	mut configure: impl FnMut(&D) -> Result<C>,
) -> Result<(D, C)> {
	let mut first_error = None;
	for device in default.into_iter().chain(bridges) {
		match configure(&device) {
			Ok(config) => return Ok((device, config)),
			Err(error) => {
				first_error.get_or_insert(error);
			}
		}
	}
	Err(first_error.unwrap_or_else(|| {
		Error::no_suitable_device("audio found no default device or usable desktop-server bridge")
	}))
}

#[cfg(test)]
#[path = "../../../test/rs/audio/device_unit.rs"]
mod tests;
