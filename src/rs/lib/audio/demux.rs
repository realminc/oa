//! Host probing for audio-only files and embedded audio tracks.

use std::{fs::File, path::Path};

use symphonia::core::{
	formats::{FormatOptions, FormatReader, probe::Hint},
	io::{MediaSource, MediaSourceStream},
	meta::MetadataOptions,
};

use crate::{Error, Result};

pub(super) fn probe_file(
	path: &Path,
	open_operation: &'static str,
	probe_operation: &'static str,
) -> Result<Box<dyn FormatReader>> {
	let file = File::open(path).map_err(|source| Error::io(open_operation, source))?;
	let mut hint = Hint::new();
	if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
		hint.with_extension(extension);
	}
	probe(Box::new(file), &hint, probe_operation)
}

pub(super) fn probe(
	source: Box<dyn MediaSource>,
	hint: &Hint,
	operation: &'static str,
) -> Result<Box<dyn FormatReader>> {
	symphonia::default::get_probe()
		.probe(
			hint,
			MediaSourceStream::new(source, Default::default()),
			FormatOptions::default(),
			MetadataOptions::default(),
		)
		.map_err(|source| Error::backend_failure("Symphonia", operation, source))
}
