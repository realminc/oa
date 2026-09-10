//! Private target-independent shader artifacts and metadata.

use std::sync::OnceLock;

use crate::{Error, Result};

const SPIRV_MAGIC: u32 = 0x0723_0203;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LogicalWriteDomain {
	AxisSlices,
	OutputElements,
	Rows,
	Scalar,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WritePartition {
	ExclusivePerInvocation,
	ExclusivePerWorkgroup,
	SharedAtomicContributors,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WriteExtent {
	AxisSlice,
	OneElement,
	OneScalar,
	RowWidth,
	Tile16x16,
	UpToTwoElements,
	UpToFourElements,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CollisionPolicy {
	Exclusive,
	AtomicU32,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TailPolicy {
	BoundsChecked,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkspacePartition {
	None,
	ExclusivePerWorkgroup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PhysicalWrite {
	pub(crate) binding: u8,
	pub(crate) domain: LogicalWriteDomain,
	pub(crate) partition: WritePartition,
	pub(crate) extent: WriteExtent,
	pub(crate) collision: CollisionPolicy,
	pub(crate) tail: TailPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PhysicalWriteContract {
	pub(crate) writes: &'static [PhysicalWrite],
	pub(crate) workspace: WorkspacePartition,
}

impl LogicalWriteDomain {
	pub(crate) const fn token(self) -> &'static str {
		match self {
			Self::AxisSlices => "axis_slices",
			Self::OutputElements => "output_elements",
			Self::Rows => "rows",
			Self::Scalar => "scalar",
		}
	}
}

impl WritePartition {
	pub(crate) const fn token(self) -> &'static str {
		match self {
			Self::ExclusivePerInvocation => "exclusive_per_invocation",
			Self::ExclusivePerWorkgroup => "exclusive_per_workgroup",
			Self::SharedAtomicContributors => "shared_atomic_contributors",
		}
	}
}

impl WriteExtent {
	pub(crate) const fn token(self) -> &'static str {
		match self {
			Self::AxisSlice => "axis_slice",
			Self::OneElement => "one_element",
			Self::OneScalar => "one_scalar",
			Self::RowWidth => "row_width",
			Self::Tile16x16 => "tile_16x16",
			Self::UpToTwoElements => "up_to_two_elements",
			Self::UpToFourElements => "up_to_four_elements",
		}
	}
}

impl CollisionPolicy {
	pub(crate) const fn token(self) -> &'static str {
		match self {
			Self::Exclusive => "exclusive",
			Self::AtomicU32 => "atomic_u32",
		}
	}
}

impl TailPolicy {
	pub(crate) const fn token(self) -> &'static str {
		match self {
			Self::BoundsChecked => "bounds_checked",
		}
	}
}

impl WorkspacePartition {
	pub(crate) const fn token(self) -> &'static str {
		match self {
			Self::None => "none",
			Self::ExclusivePerWorkgroup => "exclusive_per_workgroup",
		}
	}
}

/// A build-validated shader artifact embedded into the OA library.
pub struct ShaderArtifact {
	bytes: &'static [u8],
	pub workgroup_size: [u32; 3],
	pub dispatch_tile_size: [u32; 3],
	pub(crate) physical_write: Option<PhysicalWriteContract>,
	content_id: OnceLock<u64>,
}

impl ShaderArtifact {
	/// Decode the embedded little-endian SPIR-V bytes for Vulkan module creation.
	pub fn spirv_words(&self) -> Result<Vec<u32>> {
		decode_spirv(self.bytes)
	}

	/// Reflect the exact push-constant block size from the embedded SPIR-V.
	pub fn push_constant_size(&self) -> Result<u32> {
		reflect_push_constant_size(&self.spirv_words()?)
	}

	pub(crate) fn content_id(&self) -> u64 {
		*self.content_id.get_or_init(|| {
			self.bytes
				.iter()
				.fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
					(hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
				})
		})
	}
}

#[path = "shader/registry.gen.rs"]
mod registry;

pub(crate) use registry::{KernelId, TrainingReplayRole};

fn decode_spirv(bytes: &[u8]) -> Result<Vec<u32>> {
	if bytes.len() < size_of::<u32>() || !bytes.len().is_multiple_of(size_of::<u32>()) {
		return Err(invalid_artifact(
			"SPIR-V byte length is not a non-empty word sequence",
		));
	}
	let (word_bytes, remainder) = bytes.as_chunks::<4>();
	debug_assert!(remainder.is_empty());
	let words = word_bytes
		.iter()
		.map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
		.collect::<Vec<_>>();
	if words[0] != SPIRV_MAGIC {
		return Err(invalid_artifact("SPIR-V magic number is invalid"));
	}
	Ok(words)
}

#[derive(Clone, Default)]
enum SpirvType {
	#[default]
	Unknown,
	Scalar(u32),
	Vector {
		component: u32,
		count: u32,
	},
	Struct(Vec<u32>),
}

fn reflect_push_constant_size(words: &[u32]) -> Result<u32> {
	const OP_TYPE_INT: u16 = 21;
	const OP_TYPE_FLOAT: u16 = 22;
	const OP_TYPE_VECTOR: u16 = 23;
	const OP_TYPE_STRUCT: u16 = 30;
	const OP_TYPE_POINTER: u16 = 32;
	const OP_MEMBER_DECORATE: u16 = 72;
	const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
	const DECORATION_OFFSET: u32 = 35;
	const MAX_ID_BOUND: usize = 1 << 22;

	if words.len() < 5 || words[0] != SPIRV_MAGIC {
		return Err(invalid_artifact("SPIR-V header is invalid"));
	}
	let bound =
		usize::try_from(words[3]).map_err(|_| invalid_artifact("SPIR-V ID bound exceeds usize"))?;
	if bound == 0 || bound > MAX_ID_BOUND {
		return Err(invalid_artifact("SPIR-V ID bound is invalid"));
	}

	let mut types = vec![SpirvType::Unknown; bound];
	let mut member_offsets = vec![Vec::<Option<u32>>::new(); bound];
	let mut push_pointees = Vec::new();
	let mut cursor = 5_usize;
	while cursor < words.len() {
		let instruction = words[cursor];
		let word_count = usize::from((instruction >> 16) as u16);
		let opcode = (instruction & 0xffff) as u16;
		if word_count == 0 {
			return Err(invalid_artifact("SPIR-V instruction has zero words"));
		}
		let end = cursor
			.checked_add(word_count)
			.ok_or_else(|| invalid_artifact("SPIR-V instruction range overflows"))?;
		if end > words.len() {
			return Err(invalid_artifact("SPIR-V instruction exceeds module length"));
		}
		let operands = &words[cursor + 1..end];
		match opcode {
			OP_TYPE_INT | OP_TYPE_FLOAT if operands.len() >= 2 => {
				let id = checked_spirv_id(operands[0], bound)?;
				let width = operands[1];
				if width == 0 || !width.is_multiple_of(8) {
					return Err(invalid_artifact("SPIR-V scalar width is not byte-sized"));
				}
				types[id] = SpirvType::Scalar(width / 8);
			}
			OP_TYPE_VECTOR if operands.len() >= 3 => {
				let id = checked_spirv_id(operands[0], bound)?;
				checked_spirv_id(operands[1], bound)?;
				types[id] = SpirvType::Vector {
					component: operands[1],
					count: operands[2],
				};
			}
			OP_TYPE_STRUCT if !operands.is_empty() => {
				let id = checked_spirv_id(operands[0], bound)?;
				for member in &operands[1..] {
					checked_spirv_id(*member, bound)?;
				}
				types[id] = SpirvType::Struct(operands[1..].to_vec());
			}
			OP_TYPE_POINTER if operands.len() >= 3 => {
				checked_spirv_id(operands[0], bound)?;
				let pointee = checked_spirv_id(operands[2], bound)?;
				if operands[1] == STORAGE_CLASS_PUSH_CONSTANT {
					push_pointees.push(pointee);
				}
			}
			OP_MEMBER_DECORATE if operands.len() >= 4 && operands[2] == DECORATION_OFFSET => {
				let structure = checked_spirv_id(operands[0], bound)?;
				let member = usize::try_from(operands[1])
					.map_err(|_| invalid_artifact("SPIR-V member index exceeds usize"))?;
				let required = member
					.checked_add(1)
					.ok_or_else(|| invalid_artifact("SPIR-V member index overflows"))?;
				member_offsets[structure].resize(required, None);
				member_offsets[structure][member] = Some(operands[3]);
			}
			_ => {}
		}
		cursor = end;
	}

	let push_struct = push_pointees
		.into_iter()
		.find(|id| matches!(types[*id], SpirvType::Struct(_)))
		.ok_or_else(|| invalid_artifact("SPIR-V has no directly reflected push-constant block"))?;
	let SpirvType::Struct(members) = &types[push_struct] else {
		return Err(invalid_artifact(
			"SPIR-V push-constant type is not a struct",
		));
	};
	let offsets = &member_offsets[push_struct];
	if members.is_empty() || offsets.len() != members.len() {
		return Err(invalid_artifact(
			"SPIR-V push-constant member offsets are incomplete",
		));
	}
	let mut size = 0_u32;
	for (member, offset) in members.iter().zip(offsets) {
		let member_size = spirv_type_size(&types, *member)?;
		let offset = offset.ok_or_else(|| {
			invalid_artifact("SPIR-V push-constant member is missing its byte offset")
		})?;
		let end = offset
			.checked_add(member_size)
			.ok_or_else(|| invalid_artifact("SPIR-V push-constant size overflows u32"))?;
		size = size.max(end);
	}
	if size == 0 || !size.is_multiple_of(4) {
		return Err(invalid_artifact(
			"SPIR-V push-constant block size is not a nonzero four-byte multiple",
		));
	}
	Ok(size)
}

fn checked_spirv_id(id: u32, bound: usize) -> Result<usize> {
	let id = usize::try_from(id).map_err(|_| invalid_artifact("SPIR-V ID exceeds usize"))?;
	if id == 0 || id >= bound {
		return Err(invalid_artifact("SPIR-V ID exceeds the module bound"));
	}
	Ok(id)
}

fn spirv_type_size(types: &[SpirvType], id: u32) -> Result<u32> {
	let index = checked_spirv_id(id, types.len())?;
	match types[index] {
		SpirvType::Scalar(size) => Ok(size),
		SpirvType::Vector { component, count } => spirv_type_size(types, component)?
			.checked_mul(count)
			.ok_or_else(|| invalid_artifact("SPIR-V vector byte size overflows u32")),
		SpirvType::Unknown | SpirvType::Struct(_) => Err(invalid_artifact(
			"SPIR-V push-constant member type cannot be sized exactly",
		)),
	}
}

fn invalid_artifact(message: impl Into<String>) -> Error {
	Error::backend_failure(
		"Slang",
		"shader-artifact validation",
		std::io::Error::other(message.into()),
	)
}

#[cfg(test)]
mod tests {
	use super::{KernelId, SPIRV_MAGIC, decode_spirv, reflect_push_constant_size};

	#[test]
	fn embedded_artifacts_match_their_validated_abi() -> crate::Result<()> {
		for kernel in KernelId::ALL {
			let artifact = kernel.artifact();
			assert!(!artifact.spirv_words()?.is_empty());
			assert_ne!(artifact.content_id(), 0);
			assert_eq!(artifact.content_id(), artifact.content_id());
			let push_constant_size = artifact.push_constant_size()?;
			assert!(push_constant_size.is_multiple_of(4));
			assert!(push_constant_size <= 128);
			assert!(artifact.workgroup_size.into_iter().all(|extent| extent > 0));
			assert!(artifact.workgroup_size.into_iter().product::<u32>() <= 1024);
			assert!(
				artifact
					.dispatch_tile_size
					.into_iter()
					.all(|extent| extent > 0)
			);
		}
		Ok(())
	}

	#[test]
	fn generated_kernel_registry_distinguishes_semantic_and_lowering_only_kernels() {
		for kernel in KernelId::ALL {
			if let Some(contract) = kernel.semantic_contract() {
				assert!(contract.name().starts_with("oa::"));
			}
		}
		for kernel in [
			KernelId::MlCrossEntropySumF32,
			KernelId::MlQkvProjectionBiasF32,
			KernelId::MlGateUpSwigluBiasF32,
			KernelId::MlAdamWMany4F32,
			KernelId::MlAdamWMany4GraphF32,
			KernelId::MlMuonNormalizeF32,
			KernelId::MlMuonMatMulAxpbyF32,
			KernelId::MlMuonLinearCombinationF32,
			KernelId::MlMuonTransposeF32,
			KernelId::MlMuonApplyF32,
		] {
			assert!(kernel.semantic_contract().is_none());
		}
		assert!(KernelId::MlCrossEntropyF32.semantic_contract().is_some());
		assert_eq!(
			KernelId::MlMuonVectorF32
				.semantic_contract()
				.map(|contract| contract.name()),
			Some(crate::core::operation::ml::MUON.name())
		);
	}

	#[test]
	fn generated_registry_classifies_only_the_connected_write_ownership_slice() {
		for kernel in [
			KernelId::MatrixSoftmaxF32,
			KernelId::MatrixSoftmaxBackwardF32,
			KernelId::MatrixSumF32,
			KernelId::MatrixSumAxisF32,
			KernelId::MatrixSumBackwardF32,
			KernelId::MlLayerNormF32,
			KernelId::MlLayerNormBackwardF32,
			KernelId::MlRmsNormF32,
			KernelId::MlRmsNormBackwardF32,
		] {
			let contract = kernel
				.artifact()
				.physical_write
				.expect("connected candidate must have physical-write metadata");
			assert!(!contract.writes.is_empty());
		}
		assert!(KernelId::MatrixAddF32.artifact().physical_write.is_none());
	}

	#[test]
	fn malformed_spirv_is_rejected() {
		assert!(decode_spirv(&[]).is_err());
		assert!(decode_spirv(&[0, 0, 0, 0]).is_err());
		assert!(decode_spirv(&[3, 2, 35]).is_err());
		assert!(reflect_push_constant_size(&[SPIRV_MAGIC, 0, 0, 1, 0]).is_err());
		assert!(reflect_push_constant_size(&[SPIRV_MAGIC, 0, 0, 2, 0, 0]).is_err());
	}
}
