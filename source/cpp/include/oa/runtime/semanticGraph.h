#pragma once

#include <oa/core/op.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/runtime/semanticGraphFwd.h>

namespace oa {

class ExecutableGraph;

enum class SemanticAccessMode : oa::U8 {
	Read,
	Write,
	ReadWrite,
};

// handle-free description of one logical value. storage ownership and vulkan
// resources belong to the executable graph; this graph preserves meaning.
struct SemanticValueDesc {
	oa::U32 id = oa::invalidSemanticValueId;
	oa::String name;
	oa::OpValueKind kind = oa::OpValueKind::Matrix;
	oa::MatrixShape shape{};
	oa::Array<oa::I64, OA_MAX_TENSOR_DIMS> strides{};
	oa::ScalarType dtype = oa::ScalarType::Float32;
	oa::Bool external = false;
	oa::Bool isVirtual = false;
	oa::U32 producer = oa::invalidSemanticOpId;
	// Metadata-only values are represented as values derived from an earlier
	// value, not as fake operations with fake executable lowerings.
	oa::U32 viewSource = oa::invalidSemanticValueId;
	oa::I64 viewByteOffset = 0;
};

struct SemanticValueAccess {
	oa::U32 value = oa::invalidSemanticValueId;
	oa::SemanticAccessMode mode = oa::SemanticAccessMode::Read;
};

struct SemanticAliasDesc {
	oa::U32 output = oa::invalidSemanticValueId;
	oa::U32 input = oa::invalidSemanticValueId;
};

// Provenance link between one reverse-differentiable semantic output and the
// concrete autograd tape node attached while authoring the graph. The tape
// remains the current backward executor; this record makes its origin explicit
// without embedding ML objects or host pointers in the semantic IR.
struct SemanticAutogradDesc {
	oa::U32 forwardOp = oa::invalidSemanticOpId;
	oa::U32 output = oa::invalidSemanticValueId;
	oa::U32 outputIndex = 0;
	oa::U64 sequence = 0;
	oa::U32 backwardFirstOp =
		oa::invalidSemanticOpId;
	oa::U32 backwardOpCount = 0;
	oa::Bool backwardExpanded = false;
};

// One mathematical/domain operation before kernel, queue, launch geometry,
// transfer, barrier, or device selection. One semantic operation may lower to
// zero, one, or many executable nodes.
struct SemanticOpDesc {
	oa::U32 id = oa::invalidSemanticOpId;
	oa::String name;
	oa::U64 contractHash = 0;
	oa::OpDifferentiation differentiation =
		oa::OpDifferentiation::None;
	oa::OpLowering lowering = oa::OpLowering::Dispatch;
	oa::OpControlFlow controlFlow = oa::OpControlFlow::StraightLine;
	oa::U8 optionalInputMask = 0U;
	oa::Vec<oa::U32> inputs;
	oa::Vec<oa::U32> outputs;
	oa::Vec<oa::OpAttribute> attributes;
	oa::Vec<oa::SemanticValueAccess> accesses;
	oa::Vec<oa::U32> mutatedInputs;
	oa::Vec<oa::SemanticAliasDesc> aliases;
	oa::Vec<oa::U32> controlDependencies;
	oa::U32 backwardOf = oa::invalidSemanticOpId;
	oa::U64 backwardSequence = 0;
};

// Canonical semantic graph. It is deliberately independent of vulkan and of
// executable dispatch descriptions so graph compilation can decompose, fuse,
// partition, place, and schedule without losing public operation semantics.
class SemanticGraph {
public:
	[[nodiscard]] oa::Result<oa::U32> addValue(
		const oa::SemanticValueDesc& inValue);
	[[nodiscard]] oa::Result<oa::U32> addOp(
		const oa::OpContract& inContract,
		oa::Span<const oa::U32> inInputs,
		oa::Span<const oa::U32> inOutputs,
		oa::Span<const oa::U32> inControlDependencies = {},
		oa::Span<const oa::OpAttribute> inAttributes = {});
	[[nodiscard]] oa::Status addView(
		oa::U32 inSource,
		oa::U32 inView,
		oa::I64 inByteOffset);
	[[nodiscard]] oa::Status attachAutograd(
		oa::U32 inForwardOp,
		oa::U32 inOutputIndex,
		oa::U64 inSequence);
	[[nodiscard]] oa::Status completeAutograd(
		oa::U32 inForwardOp,
		oa::U64 inSequence,
		oa::U32 inBackwardFirstOp,
		oa::U32 inBackwardOpCount);
	[[nodiscard]] oa::Status validate() const;
	[[nodiscard]] oa::Status copyFrom(const oa::SemanticGraph& inSource);

	[[nodiscard]] const oa::SemanticValueDesc* findValue(
		oa::U32 inId) const noexcept;
	[[nodiscard]] oa::Span<const oa::SemanticValueDesc> values() const noexcept {
		return {values_.data(), values_.size()};
	}
	[[nodiscard]] oa::Span<const oa::SemanticOpDesc> operations() const noexcept {
		return {operations_.data(), operations_.size()};
	}
	[[nodiscard]] oa::Span<const oa::SemanticAutogradDesc> autograd() const noexcept {
		return {autograd_.data(), autograd_.size()};
	}
	[[nodiscard]] oa::U32 valueCount() const noexcept {
		return static_cast<oa::U32>(values_.size());
	}
	[[nodiscard]] oa::U32 operationCount() const noexcept {
		return static_cast<oa::U32>(operations_.size());
	}
	[[nodiscard]] oa::U64 generation() const noexcept { return generation_; }
	[[nodiscard]] oa::U32 viewCount() const noexcept;

	// Deterministic and handle-free evidence for graph/compiler tests.
	[[nodiscard]] oa::String debugReportJson(oa::StringView inName = "") const;
	void reset() noexcept;

private:
	oa::Vec<oa::SemanticValueDesc> values_;
	oa::Vec<oa::SemanticOpDesc> operations_;
	oa::Vec<oa::SemanticAutogradDesc> autograd_;
	oa::U64 generation_ = 1;
};

// Deterministic evidence for the provenance relation between semantic
// operations and executable nodes. One semantic operation with multiple nodes
// is a decomposition; executable nodes without semantic ownership remain
// visible compatibility debt while schema migration is incomplete.
class SemanticLoweringAnalysis {
public:
	[[nodiscard]] oa::U32 operationCount() const noexcept {
		return static_cast<oa::U32>(executableNodeCounts_.size());
	}
	[[nodiscard]] oa::U32 executableNodeCount(
		oa::U32 inOperation) const noexcept
	{
		return inOperation < executableNodeCounts_.size()
			? executableNodeCounts_[inOperation] : 0U;
	}
	[[nodiscard]] oa::U32 schemaOwnedNodeCount() const noexcept {
		return schemaOwnedNodeCount_;
	}
	[[nodiscard]] oa::U32 compatibilityNodeCount() const noexcept {
		return compatibilityNodeCount_;
	}
	[[nodiscard]] oa::U32 directOpCount() const noexcept {
		return directOpCount_;
	}
	[[nodiscard]] oa::U32 decomposedOpCount() const noexcept {
		return decomposedOpCount_;
	}
	[[nodiscard]] oa::U32 fusedOpCount() const noexcept {
		return fusedOpCount_;
	}
	[[nodiscard]] oa::U32 fusedNodeCount() const noexcept {
		return fusedNodeCount_;
	}
	[[nodiscard]] oa::U32 maximumNodesPerOp() const noexcept {
		return maximumNodesPerOp_;
	}
	[[nodiscard]] oa::U32 maximumOpsPerNode() const noexcept {
		return maximumOpsPerNode_;
	}

private:
	friend oa::Result<oa::SemanticLoweringAnalysis> analyzeSemanticLowering(
		const oa::SemanticGraph& inSemantic,
		const oa::ExecutableGraph& inExecutable);

	oa::Vec<oa::U32> executableNodeCounts_;
	oa::U32 schemaOwnedNodeCount_ = 0;
	oa::U32 compatibilityNodeCount_ = 0;
	oa::U32 directOpCount_ = 0;
	oa::U32 decomposedOpCount_ = 0;
	oa::U32 fusedOpCount_ = 0;
	oa::U32 fusedNodeCount_ = 0;
	oa::U32 maximumNodesPerOp_ = 0;
	oa::U32 maximumOpsPerNode_ = 0;
};

// validate semantic identity and return the exact many-to-many lowering shape.
// One operation may decompose to several nodes and one fused node may retain
// provenance for several semantic operations.
[[nodiscard]] oa::Result<oa::SemanticLoweringAnalysis> analyzeSemanticLowering(
	const oa::SemanticGraph& inSemantic,
	const oa::ExecutableGraph& inExecutable);

// validate the provenance edge between semantic operations and their concrete
// executable lowerings. Compatibility executable nodes may remain unowned
// while schema migration is in progress, but every semantic operation must
// lower to at least one node and every owned node must match its contract.
[[nodiscard]] oa::Status validateSemanticLowering(
	const oa::SemanticGraph& inSemantic,
	const oa::ExecutableGraph& inExecutable);

} // namespace oa
