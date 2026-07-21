#pragma once

#include <Oa/Core/Operation.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/SemanticGraphFwd.h>

class OaComputeGraph;

enum class OaSemanticAccessMode : OaU8 {
	Read,
	Write,
	ReadWrite,
};

// Handle-free description of one logical value. Storage ownership and Vulkan
// resources belong to the executable graph; this graph preserves meaning.
struct OaSemanticValueDesc {
	OaSemanticValueId Id = OaInvalidSemanticValueId;
	OaString Name;
	OaOperationValueKind Kind = OaOperationValueKind::Matrix;
	OaMatrixShape Shape{};
	OaScalarType Dtype = OaScalarType::Float32;
	OaBool External = false;
	OaBool Virtual = false;
	OaSemanticOperationId Producer = OaInvalidSemanticOperationId;
};

struct OaSemanticValueAccess {
	OaSemanticValueId Value = OaInvalidSemanticValueId;
	OaSemanticAccessMode Mode = OaSemanticAccessMode::Read;
};

struct OaSemanticAliasDesc {
	OaSemanticValueId Output = OaInvalidSemanticValueId;
	OaSemanticValueId Input = OaInvalidSemanticValueId;
};

// Provenance link between one reverse-differentiable semantic output and the
// concrete autograd tape node attached while authoring the graph. The tape
// remains the current backward executor; this record makes its origin explicit
// without embedding ML objects or host pointers in the semantic IR.
struct OaSemanticAutogradDesc {
	OaSemanticOperationId ForwardOperation = OaInvalidSemanticOperationId;
	OaSemanticValueId Output = OaInvalidSemanticValueId;
	OaU32 OutputIndex = 0;
	OaU64 Sequence = 0;
	OaSemanticOperationId BackwardFirstOperation =
		OaInvalidSemanticOperationId;
	OaU32 BackwardOperationCount = 0;
	OaBool BackwardExpanded = false;
};

// One mathematical/domain operation before kernel, queue, launch geometry,
// transfer, barrier, or device selection. One semantic operation may lower to
// zero, one, or many executable nodes.
struct OaSemanticOperationDesc {
	OaSemanticOperationId Id = OaInvalidSemanticOperationId;
	OaString Name;
	OaU64 ContractHash = 0;
	OaOperationDifferentiation Differentiation =
		OaOperationDifferentiation::None;
	OaOperationLowering Lowering = OaOperationLowering::Dispatch;
	OaOperationControlFlow ControlFlow = OaOperationControlFlow::StraightLine;
	OaVec<OaSemanticValueId> Inputs;
	OaVec<OaSemanticValueId> Outputs;
	OaVec<OaOperationAttribute> Attributes;
	OaVec<OaSemanticValueAccess> Accesses;
	OaVec<OaSemanticValueId> MutatedInputs;
	OaVec<OaSemanticAliasDesc> Aliases;
	OaVec<OaSemanticOperationId> ControlDependencies;
	OaSemanticOperationId BackwardOf = OaInvalidSemanticOperationId;
	OaU64 BackwardSequence = 0;
};

// Canonical semantic graph. It is deliberately independent of Vulkan and of
// executable dispatch descriptions so graph compilation can decompose, fuse,
// partition, place, and schedule without losing public operation semantics.
class OaSemanticGraph {
public:
	[[nodiscard]] OaResult<OaSemanticValueId> AddValue(
		const OaSemanticValueDesc& InValue);
	[[nodiscard]] OaResult<OaSemanticOperationId> AddOperation(
		const OaOperationContract& InContract,
		OaSpan<const OaSemanticValueId> InInputs,
		OaSpan<const OaSemanticValueId> InOutputs,
		OaSpan<const OaSemanticOperationId> InControlDependencies = {},
		OaSpan<const OaOperationAttribute> InAttributes = {});
	[[nodiscard]] OaStatus AttachAutograd(
		OaSemanticOperationId InForwardOperation,
		OaU32 InOutputIndex,
		OaU64 InSequence);
	[[nodiscard]] OaStatus CompleteAutograd(
		OaSemanticOperationId InForwardOperation,
		OaU64 InSequence,
		OaSemanticOperationId InBackwardFirstOperation,
		OaU32 InBackwardOperationCount);
	[[nodiscard]] OaStatus Validate() const;
	[[nodiscard]] OaStatus CopyFrom(const OaSemanticGraph& InSource);

	[[nodiscard]] const OaSemanticValueDesc* FindValue(
		OaSemanticValueId InId) const noexcept;
	[[nodiscard]] OaSpan<const OaSemanticValueDesc> Values() const noexcept {
		return {Values_.Data(), Values_.Size()};
	}
	[[nodiscard]] OaSpan<const OaSemanticOperationDesc> Operations() const noexcept {
		return {Operations_.Data(), Operations_.Size()};
	}
	[[nodiscard]] OaSpan<const OaSemanticAutogradDesc> Autograd() const noexcept {
		return {Autograd_.Data(), Autograd_.Size()};
	}
	[[nodiscard]] OaU32 ValueCount() const noexcept {
		return static_cast<OaU32>(Values_.Size());
	}
	[[nodiscard]] OaU32 OperationCount() const noexcept {
		return static_cast<OaU32>(Operations_.Size());
	}

	// Deterministic and handle-free evidence for graph/compiler tests.
	[[nodiscard]] OaString DebugReportJson(OaStringView InName = "") const;
	void Reset() noexcept;

private:
	OaVec<OaSemanticValueDesc> Values_;
	OaVec<OaSemanticOperationDesc> Operations_;
	OaVec<OaSemanticAutogradDesc> Autograd_;
};

// Deterministic evidence for the provenance relation between semantic
// operations and executable nodes. One semantic operation with multiple nodes
// is a decomposition; executable nodes without semantic ownership remain
// visible compatibility debt while schema migration is incomplete.
class OaSemanticLoweringAnalysis {
public:
	[[nodiscard]] OaU32 OperationCount() const noexcept {
		return static_cast<OaU32>(ExecutableNodeCounts_.Size());
	}
	[[nodiscard]] OaU32 ExecutableNodeCount(
		OaSemanticOperationId InOperation) const noexcept
	{
		return InOperation < ExecutableNodeCounts_.Size()
			? ExecutableNodeCounts_[InOperation] : 0U;
	}
	[[nodiscard]] OaU32 SchemaOwnedNodeCount() const noexcept {
		return SchemaOwnedNodeCount_;
	}
	[[nodiscard]] OaU32 CompatibilityNodeCount() const noexcept {
		return CompatibilityNodeCount_;
	}
	[[nodiscard]] OaU32 DirectOperationCount() const noexcept {
		return DirectOperationCount_;
	}
	[[nodiscard]] OaU32 DecomposedOperationCount() const noexcept {
		return DecomposedOperationCount_;
	}
	[[nodiscard]] OaU32 FusedOperationCount() const noexcept {
		return FusedOperationCount_;
	}
	[[nodiscard]] OaU32 FusedNodeCount() const noexcept {
		return FusedNodeCount_;
	}
	[[nodiscard]] OaU32 MaximumNodesPerOperation() const noexcept {
		return MaximumNodesPerOperation_;
	}
	[[nodiscard]] OaU32 MaximumOperationsPerNode() const noexcept {
		return MaximumOperationsPerNode_;
	}

private:
	friend OaResult<OaSemanticLoweringAnalysis> OaAnalyzeSemanticLowering(
		const OaSemanticGraph& InSemantic,
		const OaComputeGraph& InExecutable);

	OaVec<OaU32> ExecutableNodeCounts_;
	OaU32 SchemaOwnedNodeCount_ = 0;
	OaU32 CompatibilityNodeCount_ = 0;
	OaU32 DirectOperationCount_ = 0;
	OaU32 DecomposedOperationCount_ = 0;
	OaU32 FusedOperationCount_ = 0;
	OaU32 FusedNodeCount_ = 0;
	OaU32 MaximumNodesPerOperation_ = 0;
	OaU32 MaximumOperationsPerNode_ = 0;
};

// Validate semantic identity and return the exact many-to-many lowering shape.
// One operation may decompose to several nodes and one fused node may retain
// provenance for several semantic operations.
[[nodiscard]] OaResult<OaSemanticLoweringAnalysis> OaAnalyzeSemanticLowering(
	const OaSemanticGraph& InSemantic,
	const OaComputeGraph& InExecutable);

// Validate the provenance edge between semantic operations and their concrete
// executable lowerings. Compatibility executable nodes may remain unowned
// while schema migration is in progress, but every semantic operation must
// lower to at least one node and every owned node must match its contract.
[[nodiscard]] OaStatus OaValidateSemanticLowering(
	const OaSemanticGraph& InSemantic,
	const OaComputeGraph& InExecutable);
