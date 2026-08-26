#include <oa/runtime/semanticGraph.h>
#include <oa/core/jsonWriter.h>

#include <oa/runtime/executableGraph.h>

namespace {

oa::StringView kindName(oa::OpValueKind inKind) {
	switch (inKind) {
		case oa::OpValueKind::None: return "none";
		case oa::OpValueKind::Matrix: return "matrix";
		case oa::OpValueKind::Image: return "image";
		case oa::OpValueKind::Audio: return "audio";
		case oa::OpValueKind::VideoFrame: return "video_frame";
		case oa::OpValueKind::QuantMatrix: return "quant_matrix";
	}
	return "unknown";
}

oa::StringView accessName(oa::SemanticAccessMode inMode) {
	switch (inMode) {
		case oa::SemanticAccessMode::Read: return "read";
		case oa::SemanticAccessMode::Write: return "write";
		case oa::SemanticAccessMode::ReadWrite: return "read_write";
	}
	return "unknown";
}

oa::StringView loweringName(oa::OpLowering inLowering) {
	switch (inLowering) {
		case oa::OpLowering::Dispatch: return "dispatch";
		case oa::OpLowering::Gemm: return "gemm";
	}
	return "unknown";
}

oa::StringView differentiationName(oa::OpDifferentiation inDifferentiation) {
	switch (inDifferentiation) {
		case oa::OpDifferentiation::None: return "none";
		case oa::OpDifferentiation::Reverse: return "reverse";
	}
	return "unknown";
}

oa::StringView controlFlowName(oa::OpControlFlow inControlFlow) {
	switch (inControlFlow) {
		case oa::OpControlFlow::StraightLine: return "straight_line";
		case oa::OpControlFlow::Conditional: return "conditional";
		case oa::OpControlFlow::Loop: return "loop";
	}
	return "unknown";
}

oa::StringView attributeKindName(oa::OpAttributeKind inKind) {
	switch (inKind) {
		case oa::OpAttributeKind::Boolean: return "boolean";
		case oa::OpAttributeKind::SignedInteger: return "signed_integer";
		case oa::OpAttributeKind::UnsignedInteger: return "unsigned_integer";
		case oa::OpAttributeKind::Float: return "float";
		case oa::OpAttributeKind::String: return "string";
		case oa::OpAttributeKind::Shape: return "shape";
		case oa::OpAttributeKind::Enum: return "enum";
	}
	return "unknown";
}

void writeJsonString(oa::internal::JsonWriter& out, oa::StringView inValue) {
	out.writeString(inValue);
}

oa::SemanticValueAccess* findAccess(
	oa::Vec<oa::SemanticValueAccess>& inAccesses,
	oa::U32 inValue)
{
	for (auto& access : inAccesses) {
		if (access.value == inValue) return &access;
	}
	return nullptr;
}

void mergeAccess(
	oa::Vec<oa::SemanticValueAccess>& inAccesses,
	oa::U32 inValue,
	oa::SemanticAccessMode inMode)
{
	auto* existing = findAccess(inAccesses, inValue);
	if (not existing) {
		inAccesses.pushBack({.value = inValue, .mode = inMode});
		return;
	}
	if (existing->mode != inMode) existing->mode = oa::SemanticAccessMode::ReadWrite;
}

} // namespace

oa::Result<oa::U32> oa::SemanticGraph::addValue(
	const oa::SemanticValueDesc& inValue)
{
	if (inValue.id != oa::invalidSemanticValueId) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic graph value id is assigned by the graph");
	}
	if (inValue.producer != oa::invalidSemanticOpId) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic graph value producer is assigned by the graph");
	}
	if (inValue.viewSource != oa::invalidSemanticValueId
		or inValue.viewByteOffset != 0)
	{
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic graph value view provenance is assigned by the graph");
	}
	if (inValue.shape.rank < 0 or inValue.shape.rank > OA_MAX_TENSOR_DIMS) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic graph value rank is outside the supported range");
	}
	for (oa::I32 dimension = 0; dimension < inValue.shape.rank; ++dimension) {
		if (inValue.shape[dimension] < 0) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"semantic graph value has a negative dimension");
		}
	}

	oa::SemanticValueDesc value = inValue;
	value.id = static_cast<oa::U32>(values_.size());
	values_.pushBack(oa::move(value));
	return values_.back().id;
}

oa::Status oa::SemanticGraph::addView(
	oa::U32 inSource,
	oa::U32 inView,
	oa::I64 inByteOffset)
{
	if (inSource >= values_.size() or inView >= values_.size()) {
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"semantic view references an unknown value");
	}
	if (inSource >= inView) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic view source must precede its derived value");
	}
	auto& view = values_[inView];
	const auto& source = values_[inSource];
	if (view.producer != oa::invalidSemanticOpId) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic view cannot also have an operation producer");
	}
	if (view.viewSource != oa::invalidSemanticValueId) {
		return oa::Status::error(oa::StatusCode::AlreadyExists,
			"semantic value already has a view source");
	}
	if (view.kind != source.kind or view.dtype != source.dtype) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic view must preserve value kind and dtype");
	}
	if (view.external) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic view cannot be an external value");
	}
	view.viewSource = inSource;
	view.viewByteOffset = inByteOffset;
	return oa::Status::ok();
}

oa::U32 oa::SemanticGraph::viewCount() const noexcept {
	oa::U32 count = 0;
	for (const auto& value : values_) {
		if (value.viewSource != oa::invalidSemanticValueId) ++count;
	}
	return count;
}

oa::Result<oa::U32> oa::SemanticGraph::addOp(
	const oa::OpContract& inContract,
	oa::Span<const oa::U32> inInputs,
	oa::Span<const oa::U32> inOutputs,
	oa::Span<const oa::U32> inControlDependencies,
	oa::Span<const oa::OpAttribute> inAttributes)
{
	constexpr oa::U8 MaxPackedValueKinds =
		static_cast<oa::U8>(sizeof(oa::U32) * 2U);
	if (inContract.name.empty() or inContract.hash == 0U) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic graph operation requires a named, hashed contract");
	}
	if (not inContract.acceptsInputCount(inInputs.size())
		or not inContract.acceptsOutputCount(inOutputs.size()))
	{
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic graph operation arity does not match its contract");
	}
	if (inContract.inputCount > MaxPackedValueKinds
		or inContract.outputCount > MaxPackedValueKinds)
	{
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"semantic graph contract exceeds packed value-kind capacity");
	}
	if (
		(inContract.hasVariadicInputs()
			!= (inContract.minimumVariadicInputCount > 0U))
		or (inContract.hasVariadicOutputs()
			!= (inContract.minimumVariadicOutputCount > 0U))
	)
	{
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic graph contract has an invalid variadic descriptor");
	}
	const oa::U8 validInputMask = inContract.inputCount == 0U
		? 0U
		: static_cast<oa::U8>((1U << inContract.inputCount) - 1U);
	if ((inContract.optionalInputMask & static_cast<oa::U8>(~validInputMask)) != 0U
		or (inContract.optionalInputMask & inContract.mutatedInputMask) != 0U)
	{
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic graph contract has an invalid optional-input mask");
	}
	OA_RETURN_IF_ERROR(oa::validateOpAttributes(inContract, inAttributes));

	for (oa::U32 index = 0; index < inInputs.size(); ++index) {
		if (inInputs[index] == oa::invalidSemanticValueId) {
			if (inContract.isInputOptional(index)) continue;
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"semantic graph operation omits a required input");
		}
		const auto* value = findValue(inInputs[index]);
		if (not value) {
			return oa::Status::error(oa::StatusCode::OutOfRange,
				"semantic graph operation references an unknown input");
		}
		if (value->kind != inContract.inputKindAt(index)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"semantic graph input kind does not match its contract");
		}
	}
	for (oa::U32 index = 0; index < inOutputs.size(); ++index) {
		auto* value = inOutputs[index] < values_.size()
			? &values_[inOutputs[index]] : nullptr;
		if (not value) {
			return oa::Status::error(oa::StatusCode::OutOfRange,
				"semantic graph operation references an unknown output");
		}
		if (value->kind != inContract.outputKindAt(index)) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"semantic graph output kind does not match its contract");
		}
		if (value->producer != oa::invalidSemanticOpId) {
			return oa::Status::error(oa::StatusCode::AlreadyExists,
				"semantic graph value already has a producer");
		}
		const oa::U8 aliasInput = inContract.aliasInputForOutput(index);
		if (aliasInput != oa::OpContract::NoAliasInput
			and aliasInput >= inInputs.size())
		{
			return oa::Status::error(oa::StatusCode::OutOfRange,
				"semantic graph output alias references an unknown input");
		}
		if (aliasInput != oa::OpContract::NoAliasInput
			and inContract.isInputOptional(aliasInput))
		{
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"semantic graph output cannot alias an optional input");
		}
	}
	for (oa::U32 index = 0; index < inContract.inputCount; ++index) {
		if (inContract.mutatesInput(index) and index >= inInputs.size()) {
			return oa::Status::error(oa::StatusCode::OutOfRange,
				"semantic graph mutation references an unknown input");
		}
	}
	for (oa::U32 index = 0; index < inControlDependencies.size(); ++index) {
		const auto dependency = inControlDependencies[index];
		if (dependency >= operations_.size()) {
			return oa::Status::error(oa::StatusCode::OutOfRange,
				"semantic graph control dependency must reference an earlier operation");
		}
		for (oa::U32 previous = 0; previous < index; ++previous) {
			if (inControlDependencies[previous] == dependency) {
				return oa::Status::error(oa::StatusCode::AlreadyExists,
					"semantic graph control dependency is duplicated");
			}
		}
	}

	oa::SemanticOpDesc operation;
	operation.id = static_cast<oa::U32>(operations_.size());
	operation.name = oa::String(inContract.name);
	operation.contractHash = inContract.hash;
	operation.differentiation = inContract.differentiation;
	operation.lowering = inContract.lowering;
	operation.controlFlow = inContract.controlFlow;
	operation.optionalInputMask = inContract.optionalInputMask;
	operation.inputs = oa::Vec<oa::U32>(inInputs.begin(), inInputs.end());
	operation.outputs = oa::Vec<oa::U32>(inOutputs.begin(), inOutputs.end());
	operation.attributes = oa::Vec<oa::OpAttribute>(
		inAttributes.begin(), inAttributes.end());
	operation.controlDependencies = oa::Vec<oa::U32>(
		inControlDependencies.begin(), inControlDependencies.end());
	if ((static_cast<oa::U8>(inContract.effects)
		& static_cast<oa::U8>(oa::OpEffect::ReadInputs)) != 0U)
	{
		for (oa::U32 index = 0; index < inInputs.size(); ++index) {
			if (inInputs[index] == oa::invalidSemanticValueId) continue;
			mergeAccess(operation.accesses, inInputs[index],
				inContract.mutatesInput(index)
					? oa::SemanticAccessMode::ReadWrite
					: oa::SemanticAccessMode::Read);
		}
	}
	if ((static_cast<oa::U8>(inContract.effects)
		& static_cast<oa::U8>(oa::OpEffect::WriteOutputs)) != 0U)
	{
		for (const auto output : inOutputs) {
			mergeAccess(operation.accesses, output, oa::SemanticAccessMode::Write);
		}
	}
	for (oa::U32 index = 0; index < inInputs.size(); ++index) {
		if (not inContract.mutatesInput(index)) continue;
		if (inInputs[index] == oa::invalidSemanticValueId) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"semantic graph operation cannot mutate an absent input");
		}
		operation.mutatedInputs.pushBack(inInputs[index]);
		if ((static_cast<oa::U8>(inContract.effects)
			& static_cast<oa::U8>(oa::OpEffect::ReadInputs)) == 0U)
		{
			mergeAccess(operation.accesses, inInputs[index],
				oa::SemanticAccessMode::Write);
		}
	}
	for (oa::U32 index = 0; index < inOutputs.size(); ++index) {
		const oa::U8 aliasInput = inContract.aliasInputForOutput(index);
		if (aliasInput == oa::OpContract::NoAliasInput) continue;
		if (inInputs[aliasInput] == oa::invalidSemanticValueId) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"semantic graph output cannot alias an absent input");
		}
		operation.aliases.pushBack({
			.output = inOutputs[index],
			.input = inInputs[aliasInput],
		});
	}

	for (const auto output : inOutputs) values_[output].producer = operation.id;
	operations_.pushBack(oa::move(operation));
	return operations_.back().id;
}

oa::Status oa::SemanticGraph::attachAutograd(
	oa::U32 inForwardOp,
	oa::U32 inOutputIndex,
	oa::U64 inSequence)
{
	if (inForwardOp >= operations_.size()) {
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"semantic autograd attachment references an unknown operation");
	}
	const auto& operation = operations_[inForwardOp];
	if (operation.differentiation != oa::OpDifferentiation::Reverse) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic autograd attachment requires a reverse-mode contract");
	}
	if (inOutputIndex >= operation.outputs.size()) {
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"semantic autograd attachment references an unknown output");
	}
	if (inSequence == 0U) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"semantic autograd attachment requires a nonzero tape sequence");
	}
	const auto output = operation.outputs[inOutputIndex];
	for (const auto& attachment : autograd_) {
		if (attachment.output == output) {
			return oa::Status::error(oa::StatusCode::AlreadyExists,
				"semantic output already has an autograd attachment");
		}
		if (attachment.sequence == inSequence) {
			return oa::Status::error(oa::StatusCode::AlreadyExists,
				"semantic autograd sequence is already attached");
		}
	}
	autograd_.pushBack({
		.forwardOp = inForwardOp,
		.output = output,
		.outputIndex = inOutputIndex,
		.sequence = inSequence,
	});
	return oa::Status::ok();
}

oa::Status oa::SemanticGraph::completeAutograd(
	oa::U32 inForwardOp,
	oa::U64 inSequence,
	oa::U32 inBackwardFirstOp,
	oa::U32 inBackwardOpCount)
{
	oa::SemanticAutogradDesc* attachment = nullptr;
	for (auto& candidate : autograd_) {
		if (candidate.forwardOp == inForwardOp
			and candidate.sequence == inSequence)
		{
			attachment = &candidate;
			break;
		}
	}
	if (not attachment) {
		return oa::Status::error(oa::StatusCode::NotFound,
			"semantic backward expansion has no matching tape attachment");
	}
	if (attachment->backwardExpanded) {
		return oa::Status::error(oa::StatusCode::AlreadyExists,
			"semantic backward expansion is already complete");
	}
	if (inBackwardFirstOp > operations_.size()
		or inBackwardOpCount
			> operations_.size() - inBackwardFirstOp)
	{
		return oa::Status::error(oa::StatusCode::OutOfRange,
			"semantic backward expansion range is outside the graph");
	}
	for (oa::U32 offset = 0; offset < inBackwardOpCount; ++offset) {
		const auto operationId = inBackwardFirstOp + offset;
		const auto& operation = operations_[operationId];
		if (operationId <= inForwardOp) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"semantic backward expansion must follow its forward operation");
		}
		if (operation.backwardOf != oa::invalidSemanticOpId) {
			return oa::Status::error(oa::StatusCode::AlreadyExists,
				"semantic operation already belongs to a backward expansion");
		}
	}

	attachment->backwardFirstOp = inBackwardFirstOp;
	attachment->backwardOpCount = inBackwardOpCount;
	attachment->backwardExpanded = true;
	for (oa::U32 offset = 0; offset < inBackwardOpCount; ++offset) {
		auto& operation = operations_[inBackwardFirstOp + offset];
		operation.backwardOf = inForwardOp;
		operation.backwardSequence = inSequence;
	}
	return oa::Status::ok();
}

oa::Status oa::SemanticGraph::validate() const {
	for (oa::U32 index = 0; index < values_.size(); ++index) {
		const auto& value = values_[index];
		if (value.id != index) {
			return oa::Status::error(oa::StatusCode::Internal,
				"semantic graph value ids are not canonical");
		}
		if (value.producer != oa::invalidSemanticOpId
			and value.producer >= operations_.size())
		{
			return oa::Status::error(oa::StatusCode::Internal,
				"semantic graph value has an invalid producer");
		}
		if (value.viewSource != oa::invalidSemanticValueId) {
			if (value.viewSource >= index
				or value.producer != oa::invalidSemanticOpId
				or value.external
				or values_[value.viewSource].kind != value.kind
				or values_[value.viewSource].dtype != value.dtype)
			{
				return oa::Status::error(oa::StatusCode::Internal,
					"semantic graph value has an invalid view source");
			}
		}
	}
	for (oa::U32 index = 0; index < operations_.size(); ++index) {
		const auto& operation = operations_[index];
		if (operation.id != index or operation.name.empty()
			or operation.contractHash == 0U)
		{
			return oa::Status::error(oa::StatusCode::Internal,
				"semantic graph operation identity is invalid");
		}
		for (oa::U32 inputIndex = 0;
			inputIndex < operation.inputs.size(); ++inputIndex)
		{
			const auto input = operation.inputs[inputIndex];
			if (input == oa::invalidSemanticValueId
				and inputIndex < oa::OpContract::MaxValues
				and (operation.optionalInputMask
					& static_cast<oa::U8>(1U << inputIndex)) != 0U)
			{
				continue;
			}
			if (input >= values_.size()) {
				return oa::Status::error(oa::StatusCode::Internal,
					"semantic graph operation has an invalid input");
			}
		}
		for (const auto output : operation.outputs) {
			if (output >= values_.size() or values_[output].producer != index) {
				return oa::Status::error(oa::StatusCode::Internal,
					"semantic graph producer/output relationship is invalid");
			}
		}
		for (oa::U32 attribute = 0;
			attribute < operation.attributes.size(); ++attribute)
		{
			if (not operation.attributes[attribute].validate().isOk()) {
				return oa::Status::error(oa::StatusCode::Internal,
					"semantic graph operation has an invalid attribute");
			}
			for (oa::U32 previous = 0; previous < attribute; ++previous) {
				if (operation.attributes[previous].name
					== operation.attributes[attribute].name)
				{
					return oa::Status::error(oa::StatusCode::Internal,
						"semantic graph operation has duplicate attribute names");
				}
			}
		}
		for (const auto& access : operation.accesses) {
			if (access.value >= values_.size()) {
				return oa::Status::error(oa::StatusCode::Internal,
					"semantic graph operation has an invalid access");
			}
		}
		for (const auto mutated : operation.mutatedInputs) {
			if (mutated >= values_.size()) {
				return oa::Status::error(oa::StatusCode::Internal,
					"semantic graph operation has an invalid mutation");
			}
		}
		for (const auto& alias : operation.aliases) {
			if (alias.output >= values_.size() or alias.input >= values_.size()) {
				return oa::Status::error(oa::StatusCode::Internal,
					"semantic graph operation has an invalid alias");
			}
		}
		for (const auto dependency : operation.controlDependencies) {
			if (dependency >= index) {
				return oa::Status::error(oa::StatusCode::Internal,
					"semantic graph operation has an invalid control dependency");
			}
		}
		if (operation.backwardOf != oa::invalidSemanticOpId) {
			if (operation.backwardOf >= index
				or operation.backwardSequence == 0U)
			{
				return oa::Status::error(oa::StatusCode::Internal,
					"semantic backward operation has invalid provenance");
			}
		}
	}
	for (oa::U32 index = 0; index < autograd_.size(); ++index) {
		const auto& attachment = autograd_[index];
		if (attachment.forwardOp >= operations_.size()) {
			return oa::Status::error(oa::StatusCode::Internal,
				"semantic autograd attachment has an invalid operation");
		}
		const auto& operation = operations_[attachment.forwardOp];
		if (operation.differentiation != oa::OpDifferentiation::Reverse
			or attachment.outputIndex >= operation.outputs.size()
			or operation.outputs[attachment.outputIndex] != attachment.output
			or attachment.sequence == 0U)
		{
			return oa::Status::error(oa::StatusCode::Internal,
				"semantic autograd attachment contract is invalid");
		}
		for (oa::U32 previous = 0; previous < index; ++previous) {
			if (autograd_[previous].output == attachment.output
				or autograd_[previous].sequence == attachment.sequence)
			{
				return oa::Status::error(oa::StatusCode::Internal,
					"semantic autograd attachment identity is duplicated");
			}
		}
		if (attachment.backwardExpanded) {
			if (attachment.backwardFirstOp > operations_.size()
				or attachment.backwardOpCount
					> operations_.size() - attachment.backwardFirstOp)
			{
				return oa::Status::error(oa::StatusCode::Internal,
					"semantic backward expansion range is invalid");
			}
			for (oa::U32 offset = 0;
				offset < attachment.backwardOpCount; ++offset)
			{
				const auto& backward = operations_[
					attachment.backwardFirstOp + offset];
				if (backward.backwardOf != attachment.forwardOp
					or backward.backwardSequence != attachment.sequence)
				{
					return oa::Status::error(oa::StatusCode::Internal,
						"semantic backward operation disagrees with its expansion");
				}
			}
		} else if (attachment.backwardFirstOp
				!= oa::invalidSemanticOpId
			or attachment.backwardOpCount != 0U)
		{
			return oa::Status::error(oa::StatusCode::Internal,
				"incomplete semantic backward expansion owns a range");
		}
	}
	return oa::Status::ok();
}

oa::Status oa::SemanticGraph::copyFrom(const oa::SemanticGraph& inSource) {
	if (this == &inSource) {
		return oa::Status::invalidArgument(
			"oa::SemanticGraph::copyFrom cannot copy a graph onto itself");
	}
	if (not values_.empty() or not operations_.empty() or not autograd_.empty()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::SemanticGraph::copyFrom requires an empty destination graph");
	}
	OA_RETURN_IF_ERROR(inSource.validate());
	values_ = inSource.values_;
	operations_ = inSource.operations_;
	autograd_ = inSource.autograd_;
	return oa::Status::ok();
}

oa::Result<oa::SemanticLoweringAnalysis> oa::analyzeSemanticLowering(
	const oa::SemanticGraph& inSemantic,
	const oa::ExecutableGraph& inExecutable)
{
	OA_RETURN_IF_ERROR(inSemantic.validate());

	const auto operations = inSemantic.operations();
	oa::SemanticLoweringAnalysis analysis;
	analysis.executableNodeCounts_.resize(operations.size(), 0U);
	oa::Vec<oa::U8> fusionMembership(operations.size(), 0U);
	for (const auto& node : inExecutable.nodes()) {
		const oa::U32 ownerCount =
			static_cast<oa::U32>(node.semanticOps.size());
		analysis.maximumOpsPerNode_ = oa::max(
			analysis.maximumOpsPerNode_, ownerCount);
		if (ownerCount == 0U) {
			++analysis.compatibilityNodeCount_;
			continue;
		}
		if (ownerCount > 1U) {
			if (node.operation.empty() or node.opContractHash != 0U) {
				return oa::Status::error(oa::StatusCode::FailedPrecondition,
					"fused executable node requires a distinct implementation identity");
			}
			++analysis.fusedNodeCount_;
		}
		for (oa::U32 owner = 0; owner < ownerCount; ++owner) {
			const auto semanticOp = node.semanticOps[owner];
			if (semanticOp >= operations.size()) {
				return oa::Status::error(oa::StatusCode::OutOfRange,
					"executable node references an unknown semantic operation");
			}
			for (oa::U32 previous = 0; previous < owner; ++previous) {
				if (node.semanticOps[previous] == semanticOp) {
					return oa::Status::error(oa::StatusCode::AlreadyExists,
						"executable node repeats a semantic operation owner");
				}
			}
			const auto& operation = operations[semanticOp];
			if (ownerCount == 1U
				and (node.opContractHash != operation.contractHash
					or node.operation != operation.name))
			{
				return oa::Status::error(oa::StatusCode::FailedPrecondition,
					"executable node semantic identity does not match its operation");
			}
			++analysis.executableNodeCounts_[semanticOp];
			if (ownerCount > 1U) fusionMembership[semanticOp] = 1U;
		}
		++analysis.schemaOwnedNodeCount_;
	}
	for (oa::U32 operation = 0;
		operation < analysis.executableNodeCounts_.size(); ++operation)
	{
		const oa::U32 nodeCount = analysis.executableNodeCounts_[operation];
		if (nodeCount == 0U) {
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"semantic operation has no executable lowering");
		}
		if (nodeCount == 1U and fusionMembership[operation] == 0U) {
			++analysis.directOpCount_;
		}
		if (nodeCount > 1U) ++analysis.decomposedOpCount_;
		if (fusionMembership[operation] != 0U) {
			++analysis.fusedOpCount_;
		}
		analysis.maximumNodesPerOp_ = oa::max(
			analysis.maximumNodesPerOp_, nodeCount);
	}
	return analysis;
}

oa::Status oa::validateSemanticLowering(
	const oa::SemanticGraph& inSemantic,
	const oa::ExecutableGraph& inExecutable)
{
	const auto analysis = oa::analyzeSemanticLowering(inSemantic, inExecutable);
	if (not analysis.isOk()) return analysis.getStatus();
	return oa::Status::ok();
}

const oa::SemanticValueDesc* oa::SemanticGraph::findValue(
	oa::U32 inId) const noexcept
{
	return inId < values_.size() ? &values_[inId] : nullptr;
}

oa::String oa::SemanticGraph::debugReportJson(oa::StringView inName) const {
	oa::internal::JsonWriter out;
	out << "{\n  \"schema\": \"oa.semantic_graph.v2\",\n  \"name\": ";
	writeJsonString(out, inName);
	out << ",\n  \"values\": [";
	for (oa::U32 index = 0; index < values_.size(); ++index) {
		const auto& value = values_[index];
		out << (index == 0 ? "\n" : ",\n") << "    {\"id\": " << value.id
			<< ", \"name\": ";
		writeJsonString(out, value.name);
		out << ", \"kind\": ";
		writeJsonString(out, kindName(value.kind));
		out << ", \"dtype\": ";
		writeJsonString(out, oa::scalarTypeName(value.dtype));
		out << ", \"shape\": [";
		for (oa::I32 dimension = 0; dimension < value.shape.rank; ++dimension) {
			if (dimension != 0) out << ", ";
			out << value.shape[dimension];
		}
		out << "], \"external\": " << (value.external ? "true" : "false")
			<< ", \"virtual\": " << (value.isVirtual ? "true" : "false")
			<< ", \"strides\": [";
		for (oa::I32 dimension = 0; dimension < value.shape.rank; ++dimension) {
			if (dimension != 0) out << ", ";
			out << value.strides[static_cast<oa::Usize>(dimension)];
		}
		out << "], \"view_source\": ";
		if (value.viewSource == oa::invalidSemanticValueId) out << "null";
		else out << value.viewSource;
		out << ", \"view_byte_offset\": " << value.viewByteOffset
			<< ", \"producer\": ";
		if (value.producer == oa::invalidSemanticOpId) out << "null";
		else out << value.producer;
		out << '}';
	}
	if (not values_.empty()) out << '\n';
	out << "  ],\n  \"operations\": [";
	for (oa::U32 index = 0; index < operations_.size(); ++index) {
		const auto& operation = operations_[index];
		out << (index == 0 ? "\n" : ",\n") << "    {\"id\": " << operation.id
			<< ", \"name\": ";
		writeJsonString(out, operation.name);
		out << ", \"contract_hash\": ";
		out.writeHexId(operation.contractHash);
		out << ", \"lowering\": ";
		writeJsonString(out, loweringName(operation.lowering));
		out << ", \"differentiation\": ";
		writeJsonString(out, differentiationName(operation.differentiation));
		out << ", \"control_flow\": ";
		writeJsonString(out, controlFlowName(operation.controlFlow));
		out << ", \"inputs\": [";
		for (oa::U32 value = 0; value < operation.inputs.size(); ++value) {
			if (value != 0) out << ", ";
			if (operation.inputs[value] == oa::invalidSemanticValueId) out << "null";
			else out << operation.inputs[value];
		}
		out << "], \"outputs\": [";
		for (oa::U32 value = 0; value < operation.outputs.size(); ++value) {
			if (value != 0) out << ", ";
			out << operation.outputs[value];
		}
		out << "], \"attributes\": [";
		for (oa::U32 attribute = 0;
			attribute < operation.attributes.size(); ++attribute)
		{
			if (attribute != 0) out << ", ";
			const auto& value = operation.attributes[attribute];
			out << "{\"name\": ";
			writeJsonString(out, value.name);
			out << ", \"kind\": ";
			writeJsonString(out, attributeKindName(value.kind));
			out << ", \"value\": ";
			switch (value.kind) {
				case oa::OpAttributeKind::Boolean:
					out << (value.boolean ? "true" : "false");
					break;
				case oa::OpAttributeKind::SignedInteger:
					out << value.signedInteger;
					break;
				case oa::OpAttributeKind::UnsignedInteger:
					out << value.unsignedInteger;
					break;
				case oa::OpAttributeKind::Float:
					out.writeFloat(value.floatVal);
					break;
				case oa::OpAttributeKind::String:
				case oa::OpAttributeKind::Enum:
					writeJsonString(out, value.text);
					break;
				case oa::OpAttributeKind::Shape:
					out << '[';
					for (oa::I32 dimension = 0;
						dimension < value.shape.rank; ++dimension)
					{
						if (dimension != 0) out << ", ";
						out << value.shape[dimension];
					}
					out << ']';
					break;
			}
			out << '}';
		}
		out << "], \"accesses\": [";
		for (oa::U32 access = 0; access < operation.accesses.size(); ++access) {
			if (access != 0) out << ", ";
			out << "{\"value\": " << operation.accesses[access].value
				<< ", \"mode\": ";
			writeJsonString(out, accessName(operation.accesses[access].mode));
			out << '}';
		}
		out << "], \"mutated_inputs\": [";
		for (oa::U32 value = 0; value < operation.mutatedInputs.size(); ++value) {
			if (value != 0) out << ", ";
			out << operation.mutatedInputs[value];
		}
		out << "], \"aliases\": [";
		for (oa::U32 alias = 0; alias < operation.aliases.size(); ++alias) {
			if (alias != 0) out << ", ";
			out << "{\"output\": " << operation.aliases[alias].output
				<< ", \"input\": " << operation.aliases[alias].input << '}';
		}
		out << "], \"control_dependencies\": [";
		for (oa::U32 dependency = 0;
			dependency < operation.controlDependencies.size(); ++dependency)
		{
			if (dependency != 0) out << ", ";
			out << operation.controlDependencies[dependency];
		}
		out << "], \"backward_of\": ";
		if (operation.backwardOf == oa::invalidSemanticOpId) out << "null";
		else out << operation.backwardOf;
		out << ", \"backward_sequence\": " << operation.backwardSequence
			<< '}';
	}
	if (not operations_.empty()) out << '\n';
	out << "  ],\n  \"autograd\": [";
	for (oa::U32 index = 0; index < autograd_.size(); ++index) {
		const auto& attachment = autograd_[index];
		out << (index == 0 ? "\n" : ",\n")
			<< "    {\"forward_operation\": " << attachment.forwardOp
			<< ", \"output\": " << attachment.output
			<< ", \"output_index\": " << attachment.outputIndex
			<< ", \"sequence\": " << attachment.sequence
			<< ", \"backward_expanded\": "
			<< (attachment.backwardExpanded ? "true" : "false")
			<< ", \"backward_first_operation\": ";
		if (attachment.backwardFirstOp
			== oa::invalidSemanticOpId)
		{
			out << "null";
		} else {
			out << attachment.backwardFirstOp;
		}
		out << ", \"backward_operation_count\": "
			<< attachment.backwardOpCount << '}';
	}
	if (not autograd_.empty()) out << '\n';
	out << "  ]\n}\n";
	return out.take();
}

void oa::SemanticGraph::reset() noexcept {
	values_.clear();
	operations_.clear();
	autograd_.clear();
	generation_ = generation_ == UINT64_MAX ? 1U : generation_ + 1U;
}
