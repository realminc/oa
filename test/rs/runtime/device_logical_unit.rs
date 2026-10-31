use super::Device;

impl Device {
	pub(in crate::runtime) fn initialized_pipeline_count(&self) -> usize {
		self
			.logical
			.pipelines
			.iter()
			.flatten()
			.filter(|slot| slot.get().is_some())
			.count()
	}
}
