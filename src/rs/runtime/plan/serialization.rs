//! Persisted execution-plan descriptor serialization and validation.
//!
//! Donor: `oacpp/source/cpp/lib/oa/runtime/executionPlanSerialization.cpp`.
//! The envelope contract is donor-identical: an `"OAEXPLAN"` magic, a format
//! version, a payload byte count, and an FNV-1a 64-bit payload checksum. The
//! payload is a Rust-native deterministic record of the compiled plan — DNN
//! planner output, captured resource and binding contracts, the queried device
//! contract, and the executable graph — with no Vulkan handles, device
//! addresses, or pointer identities. Validation reserializes the live plan and
//! requires byte-for-byte equality, matching the donor's cache-revalidation
//! semantic.

use crate::{Error, Result};

use super::super::{Buffer, BufferAccess, PushConstant, executable_graph::ResourceLifetime};
use super::ExecutionPlan;

/// `"OAEXPLAN"` — identical to the donor descriptor magic.
const DESCRIPTOR_MAGIC: u64 = 0x4E41_4C50_5854_414F;
/// Rust descriptor payload format version. The donor's version `4` describes
/// the C++ payload; the Rust payload is its own format lineage and is not
/// byte-compatible with the donor.
const DESCRIPTOR_VERSION: u32 = 1;
const DESCRIPTOR_HEADER_BYTES: usize = 28;

/// Little-endian descriptor writer mirroring the donor's `DescriptorWriter`.
/// `count` and `string` reject values above the u32 format limit by marking the
/// writer invalid; callers surface that as `ResourceExhausted`.
pub(in crate::runtime) struct DescriptorWriter {
	bytes: Vec<u8>,
	valid: bool,
}

impl DescriptorWriter {
	pub(in crate::runtime) fn new() -> Self {
		Self {
			bytes: Vec::new(),
			valid: true,
		}
	}

	pub(in crate::runtime) fn u8(&mut self, value: u8) {
		self.bytes.push(value);
	}

	pub(in crate::runtime) fn bool(&mut self, value: bool) {
		self.u8(u8::from(value));
	}

	pub(in crate::runtime) fn u32(&mut self, value: u32) {
		self.bytes.extend_from_slice(&value.to_le_bytes());
	}

	pub(in crate::runtime) fn u64(&mut self, value: u64) {
		self.bytes.extend_from_slice(&value.to_le_bytes());
	}

	pub(in crate::runtime) fn i64(&mut self, value: i64) {
		self.u64(value.cast_unsigned());
	}

	pub(in crate::runtime) fn usize(&mut self, value: usize) {
		self.i64(value as i64);
	}

	pub(in crate::runtime) fn isize(&mut self, value: isize) {
		self.i64(value as i64);
	}

	pub(in crate::runtime) fn count(&mut self, value: usize) {
		if let Ok(value) = u32::try_from(value) {
			self.u32(value);
		} else {
			self.valid = false;
			self.u32(0);
		}
	}

	pub(in crate::runtime) fn string(&mut self, value: &str) {
		self.count(value.len());
		if self.valid && !value.is_empty() {
			self.bytes.extend_from_slice(value.as_bytes());
		}
	}

	pub(in crate::runtime) fn bytes(&mut self, value: &[u8]) {
		self.bytes.extend_from_slice(value);
	}

	pub(in crate::runtime) const fn valid(&self) -> bool {
		self.valid
	}

	pub(in crate::runtime) fn take(self) -> Vec<u8> {
		self.bytes
	}
}

/// FNV-1a 64-bit checksum, identical to the donor's `descriptorChecksum`.
fn descriptor_checksum(bytes: &[u8]) -> u64 {
	let mut hash = 0xcbf2_9ce4_8422_2325_u64;
	for byte in bytes {
		hash ^= u64::from(*byte);
		hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
	}
	hash
}

fn resource_id(lifetimes: &[ResourceLifetime], buffer: &Buffer) -> Result<u32> {
	let resource = lifetimes
		.iter()
		.position(|lifetime| lifetime.buffer.same_as(buffer))
		.ok_or_else(|| {
			Error::internal(
				"compiled execution plan references a buffer outside the captured resource inventory",
			)
		})?;
	u32::try_from(resource)
		.map_err(|_| Error::resource_exhausted("captured resource identity exceeds u32"))
}

fn write_resource_contract(plan: &ExecutionPlan, out: &mut DescriptorWriter) -> Result<()> {
	let lifetimes = plan.graph.resource_lifetimes();
	out.count(plan.captured_resources.len());
	for resource in &plan.captured_resources {
		out.u32(resource.resource());
		out.u64(resource.byte_len() as u64);
		out.count(resource.first_access());
		out.count(resource.last_access());
		out.u8(
			u8::from(resource.stable_replay_input())
				| u8::from(resource.stable_transient()) << 1
				| u8::from(resource.semantic_external()) << 2
				| u8::from(resource.observed_output()) << 3,
		);
	}
	out.count(plan.input_bindings.len());
	for binding in &plan.input_bindings {
		out.u32(resource_id(&lifetimes, &binding.original)?);
	}
	out.count(plan.semantic_bindings.len());
	for binding in plan.semantic_bindings() {
		out.u32(binding.value().index());
		out.u32(binding.resource());
		out.u8(
			u8::from(binding.semantic_external())
				| u8::from(binding.stable_replay_input()) << 1
				| u8::from(binding.observed_output()) << 2,
		);
	}
	Ok(())
}

fn write_device_contract(plan: &ExecutionPlan, out: &mut DescriptorWriter) {
	let info = plan.engine.device_info();
	out.u32(info.hardware.vendor_id);
	out.u32(info.hardware.device_id);
	out.u8(info.hardware.kind as u8);
	out.string(&info.hardware.name);
	out.u64(info.hardware.local_memory_bytes);
	out.u32(info.hardware.max_compute_shared_memory_bytes);
	out.u32(info.hardware.max_compute_workgroup_invocations);
	out.u32(info.hardware.subgroup_size);
	out.u32(info.hardware.storage_buffer_descriptors);
	out.u32(info.hardware.storage_image_descriptors);
	out.bool(info.hardware.graphics);
	out.bool(info.hardware.swapchain);
	out.string(&info.software.api_version);
	out.string(&info.software.driver_name);
	out.string(&info.software.driver_info);
	out.u32(info.software.driver_version);
	out.u64(info.software.capability_identity);
	out.bool(info.software.shader_float16);
	out.bool(info.software.shader_float64);
	out.bool(info.software.shader_integer_dot_product);
	out.count(info.software.cooperative_matrix_tuple_count);
}

fn write_executable_graph(plan: &ExecutionPlan, out: &mut DescriptorWriter) -> Result<()> {
	let lifetimes = plan.graph.resource_lifetimes();
	let nodes = plan.graph.nodes();
	out.count(nodes.len());
	for (index, node) in nodes.iter().enumerate() {
		out.string(node.operation);
		out.count(plan.graph.barriers_before(index).len());
		out.count(node.semantic_ops.len());
		for operation in &node.semantic_ops {
			out.u32(operation.index());
		}
		out.u64(node.op_contract_hash);
		out.u32(u32::from(node.kernel as u16));
		let artifact = node.kernel.artifact();
		out.u64(artifact.content_id());
		match artifact.physical_write {
			None => out.bool(false),
			Some(contract) => {
				out.bool(true);
				out.count(contract.writes.len());
				for write in contract.writes {
					out.u8(write.binding);
					out.u8(write.domain as u8);
					out.u8(write.partition as u8);
					out.u8(write.extent as u8);
					out.u8(write.collision as u8);
					out.u8(write.tail as u8);
				}
				out.u8(contract.workspace as u8);
			}
		}
		out.count(node.buffers.len());
		for buffer_use in &node.buffers {
			out.u32(resource_id(&lifetimes, &buffer_use.buffer)?);
			out.u64(buffer_use.buffer.byte_len() as u64);
			out.u8(match buffer_use.access {
				BufferAccess::Read => 0,
				BufferAccess::Write => 1,
				BufferAccess::ReadWrite => 2,
			});
		}
		out.count(node.push_constants.len());
		for constant in &node.push_constants {
			match *constant {
				PushConstant::U32(value) => {
					out.u8(0);
					out.u32(value);
				}
				PushConstant::F32(value) => {
					out.u8(1);
					out.u32(value.to_bits());
				}
			}
		}
		for dimension in node.workgroups {
			out.u32(dimension);
		}
		match &node.indirect {
			None => out.bool(false),
			Some(indirect) => {
				out.bool(true);
				out.u32(resource_id(&lifetimes, &indirect.buffer)?);
				out.u64(indirect.offset);
			}
		}
	}
	Ok(())
}

impl ExecutionPlan {
	/// Serialize this compiled plan into a deterministic, handle-free
	/// descriptor suitable for persisting alongside cache metadata.
	///
	/// The `"OAEXPLAN"` envelope and FNV-1a checksum match the donor's
	/// `ExecutionPlan::serializeDescriptor`; the payload is a Rust-native format
	/// at its own version and is not byte-compatible with the C++ descriptor.
	///
	/// # Errors
	///
	/// Returns an error when a captured resource identity or serialized count
	/// exceeds the format limits, or when the compiled plan is internally
	/// inconsistent.
	pub fn serialize_descriptor(&self) -> Result<Vec<u8>> {
		let mut payload = DescriptorWriter::new();
		payload.u64(self.graph_id);
		payload.u64(self.device_capability_identity);
		self.dnn.write_descriptor(&mut payload)?;
		write_resource_contract(self, &mut payload)?;
		write_device_contract(self, &mut payload);
		write_executable_graph(self, &mut payload)?;
		if !payload.valid() {
			return Err(Error::resource_exhausted(
				"execution plan descriptor exceeds the format limits",
			));
		}
		let payload = payload.take();
		let mut descriptor = DescriptorWriter::new();
		descriptor.u64(DESCRIPTOR_MAGIC);
		descriptor.u32(DESCRIPTOR_VERSION);
		descriptor.u64(payload.len() as u64);
		descriptor.u64(descriptor_checksum(&payload));
		descriptor.bytes(&payload);
		Ok(descriptor.take())
	}

	/// Validate a persisted descriptor against this compiled plan.
	///
	/// The envelope must be well formed and the payload must equal a fresh
	/// serialization of this plan byte for byte — the donor's cache
	/// revalidation semantic.
	///
	/// # Errors
	///
	/// - `InvalidArgument`: truncated header or invalid magic.
	/// - `FailedPrecondition`: unsupported version or descriptor mismatch.
	/// - `DataLoss`: invalid payload length or checksum mismatch.
	pub fn validate_serialized_descriptor(&self, descriptor: &[u8]) -> Result<()> {
		if descriptor.len() < DESCRIPTOR_HEADER_BYTES {
			return Err(Error::invalid_argument(
				"serialized execution plan descriptor is truncated",
			));
		}
		let magic = u64::from_le_bytes(descriptor[0..8].try_into().expect("slice length checked"));
		let version = u32::from_le_bytes(descriptor[8..12].try_into().expect("slice length checked"));
		let payload_bytes =
			u64::from_le_bytes(descriptor[12..20].try_into().expect("slice length checked"));
		let checksum = u64::from_le_bytes(descriptor[20..28].try_into().expect("slice length checked"));
		if magic != DESCRIPTOR_MAGIC {
			return Err(Error::invalid_argument(
				"serialized execution plan descriptor has an invalid magic value",
			));
		}
		if version != DESCRIPTOR_VERSION {
			return Err(Error::failed_precondition(
				"serialized execution plan descriptor version is unsupported",
			));
		}
		if payload_bytes != (descriptor.len() - DESCRIPTOR_HEADER_BYTES) as u64 {
			return Err(Error::data_loss(
				"serialized execution plan descriptor has an invalid payload length",
			));
		}
		if descriptor_checksum(&descriptor[DESCRIPTOR_HEADER_BYTES..]) != checksum {
			return Err(Error::data_loss(
				"serialized execution plan descriptor checksum does not match",
			));
		}
		let current = self.serialize_descriptor()?;
		if current != descriptor {
			return Err(Error::failed_precondition(
				"serialized execution plan descriptor does not match this compiled plan",
			));
		}
		Ok(())
	}
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/plan_serialization_unit.rs"]
mod tests;
