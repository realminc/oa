//! Backend-neutral device identity, queried facts, and allocated capacities.

/// Broad Vulkan physical-device kind reported for diagnostics.
///
/// Election never assigns a performance rank to this value. Integrated,
/// discrete, virtual, and other hardware all compete using queried features
/// and resource capacities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
	Integrated,
	Discrete,
	Virtual,
	/// Software Vulkan executing on the CPU. Selected explicitly by index.
	Cpu,
	Other,
}

/// Hardware identity and resource facts captured at engine construction.
///
/// Vulkan limits and exposed memory capacity can change with driver policy.
/// Budget and usage are snapshots, not immutable hardware properties.
/// Descriptor capacities describe the allocated heap after negotiation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceHardwareInfo {
	pub(crate) index: usize,
	pub(crate) name: String,
	pub(crate) kind: DeviceKind,
	pub(crate) vendor_id: u32,
	pub(crate) device_id: u32,
	pub(crate) local_memory_bytes: u64,
	pub(crate) local_memory_budget_bytes: Option<u64>,
	pub(crate) local_memory_usage_bytes: Option<u64>,
	pub(crate) max_compute_shared_memory_bytes: u32,
	pub(crate) max_compute_workgroup_invocations: u32,
	pub(crate) subgroup_size: u32,
	pub(crate) storage_buffer_descriptors: u32,
	pub(crate) storage_image_descriptors: u32,
	pub(crate) graphics: bool,
	pub(crate) swapchain: bool,
}

/// Immutable software facts for the device owned by one engine.
///
/// These reflect the installed driver: API version, driver version, driver
/// identity, and queried feature support. They may change across driver
/// updates without any hardware change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceSoftwareInfo {
	pub(crate) api_version: String,
	pub(crate) driver_name: String,
	pub(crate) driver_info: String,
	pub(crate) driver_version: u32,
	pub(crate) capability_identity: u64,
	pub(crate) shader_float16: bool,
	pub(crate) shader_float64: bool,
	pub(crate) shader_integer_dot_product: bool,
	pub(crate) cooperative_matrix_tuple_count: usize,
}

/// Immutable device facts captured for the device owned by one engine.
///
/// Composes [`DeviceHardwareInfo`] (PCI, local memory, limits, queue presence) and
/// [`DeviceSoftwareInfo`] (driver/API version, feature booleans).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
	pub hardware: DeviceHardwareInfo,
	pub software: DeviceSoftwareInfo,
}

impl DeviceInfo {
	pub const fn index(&self) -> usize {
		self.hardware.index
	}

	pub fn name(&self) -> &str {
		&self.hardware.name
	}

	pub const fn kind(&self) -> DeviceKind {
		self.hardware.kind
	}

	pub fn api_version(&self) -> &str {
		&self.software.api_version
	}

	pub const fn vendor_id(&self) -> u32 {
		self.hardware.vendor_id
	}

	pub const fn device_id(&self) -> u32 {
		self.hardware.device_id
	}

	pub fn driver_name(&self) -> &str {
		&self.software.driver_name
	}

	pub fn driver_info(&self) -> &str {
		&self.software.driver_info
	}

	pub const fn driver_version(&self) -> u32 {
		self.software.driver_version
	}

	/// Stable hash of device and driver UUIDs, limits, and queried capabilities.
	///
	/// Live memory budget and usage are deliberately excluded.
	pub const fn capability_identity(&self) -> u64 {
		self.software.capability_identity
	}

	pub const fn local_memory_bytes(&self) -> u64 {
		self.hardware.local_memory_bytes
	}

	/// Local memory budget observed when the engine was created, if advertised.
	pub const fn local_memory_budget_bytes(&self) -> Option<u64> {
		self.hardware.local_memory_budget_bytes
	}

	/// Local memory usage observed when the engine was created, if advertised.
	pub const fn local_memory_usage_bytes(&self) -> Option<u64> {
		self.hardware.local_memory_usage_bytes
	}

	pub const fn max_compute_shared_memory_bytes(&self) -> u32 {
		self.hardware.max_compute_shared_memory_bytes
	}

	pub const fn max_compute_workgroup_invocations(&self) -> u32 {
		self.hardware.max_compute_workgroup_invocations
	}

	pub const fn subgroup_size(&self) -> u32 {
		self.hardware.subgroup_size
	}

	/// Allocated storage-buffer descriptor capacity after heap negotiation.
	pub const fn storage_buffer_descriptors(&self) -> u32 {
		self.hardware.storage_buffer_descriptors
	}

	/// Allocated storage-image descriptor capacity after heap negotiation.
	pub const fn storage_image_descriptors(&self) -> u32 {
		self.hardware.storage_image_descriptors
	}

	pub const fn supports_shader_float16(&self) -> bool {
		self.software.shader_float16
	}

	pub const fn supports_shader_float64(&self) -> bool {
		self.software.shader_float64
	}

	pub const fn supports_shader_integer_dot_product(&self) -> bool {
		self.software.shader_integer_dot_product
	}

	pub const fn cooperative_matrix_tuple_count(&self) -> usize {
		self.software.cooperative_matrix_tuple_count
	}

	pub const fn supports_graphics(&self) -> bool {
		self.hardware.graphics
	}

	pub const fn supports_swapchain(&self) -> bool {
		self.hardware.swapchain
	}
}

/// Discovery-only identity for an enumerated Vulkan device.
///
/// Includes software and unsupported devices. Discovery creates no logical
/// device or pipelines and does not imply OA admission. Indices belong to the
/// current loader/ICD enumeration; they are not persistent device identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceSummary {
	pub(crate) index: usize,
	pub(crate) name: String,
	pub(crate) kind: DeviceKind,
	pub(crate) api_version: String,
}

impl DeviceSummary {
	/// Physical-device ordinal accepted by [`crate::runtime::DeviceSelection::Index`].
	pub const fn index(&self) -> usize {
		self.index
	}
	/// Driver-reported device name.
	pub fn name(&self) -> &str {
		&self.name
	}
	/// Driver-reported device kind, without a performance ranking.
	pub const fn kind(&self) -> DeviceKind {
		self.kind
	}
	/// Reported Vulkan API version; admission checks capabilities separately.
	pub fn api_version(&self) -> &str {
		&self.api_version
	}
}

// Private Vulkan identity and structured device diagnostics.

/// Raw Vulkan-side identity facts for one elected physical device.
///
/// These complement the public [`DeviceInfo`](crate::DeviceInfo): they carry
/// Vulkan-specific fields (driver ID enum, conformance version, device-type
/// label string) that are not part of the backend-neutral public API.
#[derive(Clone)]
pub(in crate::runtime) struct DevicePhysicalInfo {
	pub(in crate::runtime) index: usize,
	pub(in crate::runtime) name: String,
	pub(in crate::runtime) kind: DeviceKind,
	pub(in crate::runtime) device_type: &'static str,
	pub(in crate::runtime) api_version: String,
	pub(in crate::runtime) vendor_id: u32,
	pub(in crate::runtime) device_id: u32,
	pub(in crate::runtime) driver_name: String,
	pub(in crate::runtime) driver_info: String,
	pub(in crate::runtime) driver_id: ash::vk::DriverId,
	pub(in crate::runtime) driver_version: u32,
	pub(in crate::runtime) conformance_version: ash::vk::ConformanceVersion,
	pub(in crate::runtime) capability_identity: u64,
	pub(in crate::runtime) local_memory_bytes: u64,
	pub(in crate::runtime) local_memory_budget_bytes: Option<u64>,
	pub(in crate::runtime) local_memory_usage_bytes: Option<u64>,
}

/// Format a byte count as GiB or MiB.
pub(super) fn format_capacity_value(bytes: u64) -> String {
	const GIB: u64 = 1024 * 1024 * 1024;
	const MIB: u64 = 1024 * 1024;
	if bytes >= GIB {
		format!("{:.2} GiB", bytes as f64 / GIB as f64)
	} else {
		format!("{:.2} MiB", bytes as f64 / MIB as f64)
	}
}

/// One candidate entry for [`log_device_election`].
pub(super) struct ElectionCandidate<'a> {
	pub(super) index: usize,
	pub(super) name: &'a str,
	pub(super) local_heap_bytes: u64,
	pub(super) cooperative_matrix_compute: bool,
	pub(super) shader_float16: bool,
	pub(super) shader_integer_dot_product: bool,
	pub(super) on_frontier: bool,
}

/// Log the Vulkan device election summary.
///
/// Called by [`super::physical::DevicePhysical::select`] after the Pareto
/// frontier is computed, before the winner is chosen.  All presentation data
/// is passed as plain types so no `ash` or selection-policy types leak into
/// this module.
pub(super) fn log_device_election(
	enumerated_count: usize,
	candidates: &[ElectionCandidate<'_>],
	rejections: &[String],
) {
	let frontier_count = candidates.iter().filter(|c| c.on_frontier).count();
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"Vulkan device election · {} enumerated · {} admitted · {} Pareto-front candidates {{",
		enumerated_count,
		candidates.len(),
		frontier_count,
	);
	// Frontier entries first, then dominated; stable within each group.
	let mut order: Vec<usize> = (0..candidates.len()).collect();
	order.sort_by_key(|&i| !candidates[i].on_frontier);
	for i in order {
		let c = &candidates[i];
		let tier = match (
			c.cooperative_matrix_compute,
			c.shader_float16 || c.shader_integer_dot_product,
		) {
			(true, _) => "hpc",
			(false, true) => "mid",
			_ => "edge",
		};
		let status = if c.on_frontier {
			"frontier"
		} else {
			"dominated"
		};
		crate::log_info!(
			crate::LogComponent::RUNTIME,
			"    [{}] {} · local-memory {} · {} · {}",
			c.index,
			c.name,
			format_capacity_value(c.local_heap_bytes),
			tier,
			status,
		);
	}
	for rejection in rejections {
		crate::log_info!(crate::LogComponent::RUNTIME, "    rejected {rejection}");
	}
	crate::log_info!(crate::LogComponent::RUNTIME, "}}");
}

/// Log one `oa::ComputeDevice` block.
///
/// `p` is the line prefix (indentation).
/// `is_last` controls whether the closing brace is `}` (last/only device) or
/// `},` (more follow).
pub(super) fn log_compute_device(device: &super::Device, p: &str, is_last: bool) {
	let pp = format!("{}  ", p);
	let ppp = format!("{}    ", p);
	let physical = device.physical();
	let configuration = device.configuration();
	let info = &physical.info;
	let conformance = info.conformance_version;

	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}oa::ComputeDevice [{}] {{",
		p,
		info.index
	);
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}name:   {} · {}",
		pp,
		info.name,
		info.device_type,
	);
	match (
		info.local_memory_budget_bytes,
		info.local_memory_usage_bytes,
	) {
		(Some(budget), Some(usage)) => crate::log_info!(
			crate::LogComponent::RUNTIME,
			"{}local-memory:   {} / {}",
			pp,
			format_capacity_value(usage),
			format_capacity_value(budget),
		),
		_ => crate::log_info!(
			crate::LogComponent::RUNTIME,
			"{}local-memory:   {}",
			pp,
			format_capacity_value(info.local_memory_bytes),
		),
	}
	let tier = match (
		physical
			.cooperative_matrix
			.supported_stages
			.contains(ash::vk::ShaderStageFlags::COMPUTE)
			&& !physical.cooperative_matrix.tuples.is_empty(),
		physical.features.shader_float16 || physical.features.shader_integer_dot_product,
	) {
		(true, _) => "hpc accelerator",
		(false, true) => "mid-end",
		_ => "edge",
	};
	crate::log_info!(crate::LogComponent::RUNTIME, "{}kind:   {}", pp, tier);
	crate::log_info!(crate::LogComponent::RUNTIME, "{}driver {{", pp);
	if info.driver_info.is_empty() {
		crate::log_info!(
			crate::LogComponent::RUNTIME,
			"{}name: {}",
			ppp,
			info.driver_name
		);
	} else {
		crate::log_info!(
			crate::LogComponent::RUNTIME,
			"{}name: {}",
			ppp,
			info.driver_info
		);
	}
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}id:   {:?} · version 0x{:08x} · conformance {}.{}.{}.{}",
		ppp,
		info.driver_id,
		info.driver_version,
		conformance.major,
		conformance.minor,
		conformance.subminor,
		conformance.patch
	);
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}api:  Vulkan {}",
		ppp,
		info.api_version
	);
	crate::log_info!(crate::LogComponent::RUNTIME, "{}}}", pp);
	crate::log_info!(crate::LogComponent::RUNTIME, "{}hardware {{", pp);
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}pci:                 {:04x}:{:04x}",
		ppp,
		info.vendor_id,
		info.device_id,
	);
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}capability_identity: oa.device.v1.{:016x}",
		ppp,
		info.capability_identity
	);
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}profile:             {}",
		ppp,
		device.execution_profile_name()
	);
	crate::log_info!(crate::LogComponent::RUNTIME, "{}}}", pp);
	crate::log_info!(crate::LogComponent::RUNTIME, "{}queues {{", pp);
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}compute:      family {} · available {} · created {}",
		ppp,
		physical.compute_queue_family,
		physical.compute_queue_count,
		configuration.queue_count(physical.compute_queue_family),
	);
	match physical.graphics_queue_family {
		Some(gfx) => crate::log_info!(
			crate::LogComponent::RUNTIME,
			"{}graphics:     family {} · available {} · created {}",
			ppp,
			gfx,
			physical.graphics_queue_count,
			configuration.queue_count(gfx),
		),
		None => crate::log_info!(
			crate::LogComponent::RUNTIME,
			"{}graphics:     not supported",
			ppp
		),
	}
	match physical.transfer_queue_family {
		Some(xfr) => crate::log_info!(
			crate::LogComponent::RUNTIME,
			"{}transfer:     family {} · available {} · created {}",
			ppp,
			xfr,
			physical.transfer_queue_count,
			configuration.queue_count(xfr),
		),
		None => crate::log_info!(
			crate::LogComponent::RUNTIME,
			"{}transfer:     not supported",
			ppp
		),
	}
	{
		let v = &physical.video;
		match v.decode_queue_family {
			Some(fam) => {
				let mut codecs = Vec::new();
				if v.h264_decode {
					codecs.push("H.264");
				}
				if v.h265_decode {
					codecs.push("H.265");
				}
				if v.av1_decode {
					codecs.push("AV1");
				}
				if v.vp9_decode {
					codecs.push("VP9");
				}
				let codec_str = if codecs.is_empty() {
					String::new()
				} else {
					format!(" · {}", codecs.join(" "))
				};
				crate::log_info!(
					crate::LogComponent::RUNTIME,
					"{}video-decode: family {}{}",
					ppp,
					fam,
					codec_str
				);
			}
			None => crate::log_info!(
				crate::LogComponent::RUNTIME,
				"{}video-decode: not supported",
				ppp
			),
		}
		match v.encode_queue_family {
			Some(fam) => {
				let mut codecs = Vec::new();
				if v.h264_encode {
					codecs.push("H.264");
				}
				if v.h265_encode {
					codecs.push("H.265");
				}
				if v.av1_encode {
					codecs.push("AV1");
				}
				let codec_str = if codecs.is_empty() {
					String::new()
				} else {
					format!(" · {}", codecs.join(" "))
				};
				crate::log_info!(
					crate::LogComponent::RUNTIME,
					"{}video-encode: family {}{}",
					ppp,
					fam,
					codec_str
				);
			}
			None => crate::log_info!(
				crate::LogComponent::RUNTIME,
				"{}video-encode: not supported",
				ppp
			),
		}
	}
	if device.supports_swapchain() {
		crate::log_info!(crate::LogComponent::RUNTIME, "{}swapchain:    enabled", ppp);
	}
	crate::log_info!(crate::LogComponent::RUNTIME, "{}}}", pp);
	let lim = &physical.limits;
	crate::log_info!(crate::LogComponent::RUNTIME, "{}compute {{", pp);
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}shared:       {} KiB/workgroup · {} invocations · subgroup {}",
		ppp,
		lim.max_compute_shared_memory_size / 1024,
		lim.max_compute_work_group_invocations,
		lim.subgroup_size
	);
	let (buffer_capacity, image_capacity) = device.descriptor_capacities();
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}descriptors:  {} storage-buffer · {} storage-image (allocated)",
		ppp,
		buffer_capacity,
		image_capacity
	);
	crate::log_info!(crate::LogComponent::RUNTIME, "{}}}", pp);
	let feat = &physical.features;
	let enabled = configuration.features();
	let coop_compute = physical
		.cooperative_matrix
		.supported_stages
		.contains(ash::vk::ShaderStageFlags::COMPUTE)
		&& !physical.cooperative_matrix.tuples.is_empty();
	let coop_tuples = physical.cooperative_matrix.tuples.len();
	crate::log_info!(crate::LogComponent::RUNTIME, "{}precision {{", pp);
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}f64:  {}",
		ppp,
		feature_state(feat.shader_float64, enabled.shader_float64)
	);
	crate::log_info!(crate::LogComponent::RUNTIME, "{}f32:  enabled", ppp);
	crate::log_info!(crate::LogComponent::RUNTIME, "{}bf16: not queried", ppp);
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}f16:  {}",
		ppp,
		feature_state(feat.shader_float16, enabled.shader_float16)
	);
	let i8_reported = feat.shader_int8 && feat.storage_buffer_8_bit_access;
	let i8_enabled = enabled.shader_int8 && enabled.storage_buffer_8_bit_access;
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}i8:   {}",
		ppp,
		feature_state(i8_reported, i8_enabled)
	);
	crate::log_info!(
		crate::LogComponent::RUNTIME,
		"{}int-dot: {}",
		ppp,
		feature_state(
			feat.shader_integer_dot_product,
			enabled.shader_integer_dot_product
		)
	);
	if coop_compute {
		crate::log_info!(
			crate::LogComponent::RUNTIME,
			"{}cooperative-matrix: {} ({} tuples)",
			ppp,
			feature_state(coop_compute, enabled.cooperative_matrix),
			coop_tuples
		);
	} else {
		crate::log_info!(
			crate::LogComponent::RUNTIME,
			"{}cooperative-matrix: not supported",
			ppp
		);
	}
	crate::log_info!(crate::LogComponent::RUNTIME, "{}}}", pp);
	if is_last {
		crate::log_info!(crate::LogComponent::RUNTIME, "{}}}", p);
	} else {
		crate::log_info!(crate::LogComponent::RUNTIME, "{}}},", p);
	}
}

fn feature_state(reported: bool, enabled: bool) -> &'static str {
	if enabled {
		"enabled"
	} else if reported {
		"supported, disabled"
	} else {
		"not supported"
	}
}
