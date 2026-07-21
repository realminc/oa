#include <Oa/Runtime/SemanticGraph.h>

#include <Oa/Runtime/ComputeGraph.h>

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace {

OaOperationValueKind ContractKind(OaU32 InPackedKinds, OaU32 InIndex) {
	return static_cast<OaOperationValueKind>(
		(InPackedKinds >> (InIndex * 4U)) & 0x0fU);
}

OaStringView KindName(OaOperationValueKind InKind) {
	switch (InKind) {
		case OaOperationValueKind::Matrix: return "matrix";
		case OaOperationValueKind::Image: return "image";
		case OaOperationValueKind::AudioBuffer: return "audio_buffer";
		case OaOperationValueKind::VideoFrame: return "video_frame";
	}
	return "unknown";
}

OaStringView AccessName(OaSemanticAccessMode InMode) {
	switch (InMode) {
		case OaSemanticAccessMode::Read: return "read";
		case OaSemanticAccessMode::Write: return "write";
		case OaSemanticAccessMode::ReadWrite: return "read_write";
	}
	return "unknown";
}

OaStringView LoweringName(OaOperationLowering InLowering) {
	switch (InLowering) {
		case OaOperationLowering::Dispatch: return "dispatch";
		case OaOperationLowering::Gemm: return "gemm";
	}
	return "unknown";
}

OaStringView DifferentiationName(OaOperationDifferentiation InDifferentiation) {
	switch (InDifferentiation) {
		case OaOperationDifferentiation::None: return "none";
		case OaOperationDifferentiation::Reverse: return "reverse";
	}
	return "unknown";
}

OaStringView ControlFlowName(OaOperationControlFlow InControlFlow) {
	switch (InControlFlow) {
		case OaOperationControlFlow::StraightLine: return "straight_line";
		case OaOperationControlFlow::Conditional: return "conditional";
		case OaOperationControlFlow::Loop: return "loop";
	}
	return "unknown";
}

OaStringView AttributeKindName(OaOperationAttributeKind InKind) {
	switch (InKind) {
		case OaOperationAttributeKind::Boolean: return "boolean";
		case OaOperationAttributeKind::SignedInteger: return "signed_integer";
		case OaOperationAttributeKind::UnsignedInteger: return "unsigned_integer";
		case OaOperationAttributeKind::Float: return "float";
		case OaOperationAttributeKind::String: return "string";
		case OaOperationAttributeKind::Shape: return "shape";
		case OaOperationAttributeKind::Enum: return "enum";
	}
	return "unknown";
}

void WriteJsonString(std::ostringstream& Out, OaStringView InValue) {
	Out << '"';
	for (const char value : InValue) {
		switch (value) {
			case '"': Out << "\\\""; break;
			case '\\': Out << "\\\\"; break;
			case '\b': Out << "\\b"; break;
			case '\f': Out << "\\f"; break;
			case '\n': Out << "\\n"; break;
			case '\r': Out << "\\r"; break;
			case '\t': Out << "\\t"; break;
			default:
				if (static_cast<unsigned char>(value) < 0x20U) {
					Out << "\\u" << std::hex << std::setw(4)
						<< std::setfill('0') << static_cast<unsigned>(value)
						<< std::dec << std::setfill(' ');
				} else {
					Out << value;
				}
		}
	}
	Out << '"';
}

OaSemanticValueAccess* FindAccess(
	OaVec<OaSemanticValueAccess>& InAccesses,
	OaSemanticValueId InValue)
{
	for (auto& access : InAccesses) {
		if (access.Value == InValue) return &access;
	}
	return nullptr;
}

void MergeAccess(
	OaVec<OaSemanticValueAccess>& InAccesses,
	OaSemanticValueId InValue,
	OaSemanticAccessMode InMode)
{
	auto* existing = FindAccess(InAccesses, InValue);
	if (not existing) {
		InAccesses.PushBack({.Value = InValue, .Mode = InMode});
		return;
	}
	if (existing->Mode != InMode) existing->Mode = OaSemanticAccessMode::ReadWrite;
}

} // namespace

OaResult<OaSemanticValueId> OaSemanticGraph::AddValue(
	const OaSemanticValueDesc& InValue)
{
	if (InValue.Id != OaInvalidSemanticValueId) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"semantic graph value id is assigned by the graph");
	}
	if (InValue.Producer != OaInvalidSemanticOperationId) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"semantic graph value producer is assigned by the graph");
	}
	if (InValue.Shape.Rank < 0 or InValue.Shape.Rank > OA_MAX_TENSOR_DIMS) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"semantic graph value rank is outside the supported range");
	}
	for (OaI32 dimension = 0; dimension < InValue.Shape.Rank; ++dimension) {
		if (InValue.Shape[dimension] < 0) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"semantic graph value has a negative dimension");
		}
	}

	OaSemanticValueDesc value = InValue;
	value.Id = static_cast<OaSemanticValueId>(Values_.Size());
	Values_.PushBack(OaStdMove(value));
	return Values_.Back().Id;
}

OaResult<OaSemanticOperationId> OaSemanticGraph::AddOperation(
	const OaOperationContract& InContract,
	OaSpan<const OaSemanticValueId> InInputs,
	OaSpan<const OaSemanticValueId> InOutputs,
	OaSpan<const OaSemanticOperationId> InControlDependencies,
	OaSpan<const OaOperationAttribute> InAttributes)
{
	constexpr OaU8 MaxPackedValueKinds =
		static_cast<OaU8>(sizeof(OaU32) * 2U);
	if (InContract.Name.Empty() or InContract.Hash == 0U) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"semantic graph operation requires a named, hashed contract");
	}
	if (InInputs.Size() != InContract.InputCount
		or InOutputs.Size() != InContract.OutputCount)
	{
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"semantic graph operation arity does not match its contract");
	}
	if (InContract.InputCount > MaxPackedValueKinds
		or InContract.OutputCount > MaxPackedValueKinds)
	{
		return OaStatus::Error(OaStatusCode::OutOfRange,
			"semantic graph contract exceeds packed value-kind capacity");
	}
	OA_RETURN_IF_ERROR(OaValidateOperationAttributes(InContract, InAttributes));

	for (OaU32 index = 0; index < InInputs.Size(); ++index) {
		const auto* value = FindValue(InInputs[index]);
		if (not value) {
			return OaStatus::Error(OaStatusCode::OutOfRange,
				"semantic graph operation references an unknown input");
		}
		if (value->Kind != ContractKind(InContract.InputKinds, index)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"semantic graph input kind does not match its contract");
		}
	}
	for (OaU32 index = 0; index < InOutputs.Size(); ++index) {
		auto* value = InOutputs[index] < Values_.Size()
			? &Values_[InOutputs[index]] : nullptr;
		if (not value) {
			return OaStatus::Error(OaStatusCode::OutOfRange,
				"semantic graph operation references an unknown output");
		}
		if (value->Kind != ContractKind(InContract.OutputKinds, index)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"semantic graph output kind does not match its contract");
		}
		if (value->Producer != OaInvalidSemanticOperationId) {
			return OaStatus::Error(OaStatusCode::AlreadyExists,
				"semantic graph value already has a producer");
		}
		const OaU8 aliasInput = InContract.AliasInputForOutput(index);
		if (aliasInput != OaOperationContract::NoAliasInput
			and aliasInput >= InInputs.Size())
		{
			return OaStatus::Error(OaStatusCode::OutOfRange,
				"semantic graph output alias references an unknown input");
		}
	}
	for (OaU32 index = 0; index < InContract.InputCount; ++index) {
		if (InContract.MutatesInput(index) and index >= InInputs.Size()) {
			return OaStatus::Error(OaStatusCode::OutOfRange,
				"semantic graph mutation references an unknown input");
		}
	}
	for (OaU32 index = 0; index < InControlDependencies.Size(); ++index) {
		const auto dependency = InControlDependencies[index];
		if (dependency >= Operations_.Size()) {
			return OaStatus::Error(OaStatusCode::OutOfRange,
				"semantic graph control dependency must reference an earlier operation");
		}
		for (OaU32 previous = 0; previous < index; ++previous) {
			if (InControlDependencies[previous] == dependency) {
				return OaStatus::Error(OaStatusCode::AlreadyExists,
					"semantic graph control dependency is duplicated");
			}
		}
	}

	OaSemanticOperationDesc operation;
	operation.Id = static_cast<OaSemanticOperationId>(Operations_.Size());
	operation.Name = OaString(InContract.Name);
	operation.ContractHash = InContract.Hash;
	operation.Differentiation = InContract.Differentiation;
	operation.Lowering = InContract.Lowering;
	operation.ControlFlow = InContract.ControlFlow;
	operation.Inputs = OaVec<OaSemanticValueId>(InInputs.Begin(), InInputs.End());
	operation.Outputs = OaVec<OaSemanticValueId>(InOutputs.Begin(), InOutputs.End());
	operation.Attributes = OaVec<OaOperationAttribute>(
		InAttributes.Begin(), InAttributes.End());
	operation.ControlDependencies = OaVec<OaSemanticOperationId>(
		InControlDependencies.Begin(), InControlDependencies.End());
	if ((static_cast<OaU8>(InContract.Effects)
		& static_cast<OaU8>(OaOperationEffect::ReadInputs)) != 0U)
	{
		for (OaU32 index = 0; index < InInputs.Size(); ++index) {
			MergeAccess(operation.Accesses, InInputs[index],
				InContract.MutatesInput(index)
					? OaSemanticAccessMode::ReadWrite
					: OaSemanticAccessMode::Read);
		}
	}
	if ((static_cast<OaU8>(InContract.Effects)
		& static_cast<OaU8>(OaOperationEffect::WriteOutputs)) != 0U)
	{
		for (const auto output : InOutputs) {
			MergeAccess(operation.Accesses, output, OaSemanticAccessMode::Write);
		}
	}
	for (OaU32 index = 0; index < InInputs.Size(); ++index) {
		if (not InContract.MutatesInput(index)) continue;
		operation.MutatedInputs.PushBack(InInputs[index]);
		if ((static_cast<OaU8>(InContract.Effects)
			& static_cast<OaU8>(OaOperationEffect::ReadInputs)) == 0U)
		{
			MergeAccess(operation.Accesses, InInputs[index],
				OaSemanticAccessMode::Write);
		}
	}
	for (OaU32 index = 0; index < InOutputs.Size(); ++index) {
		const OaU8 aliasInput = InContract.AliasInputForOutput(index);
		if (aliasInput == OaOperationContract::NoAliasInput) continue;
		operation.Aliases.PushBack({
			.Output = InOutputs[index],
			.Input = InInputs[aliasInput],
		});
	}

	for (const auto output : InOutputs) Values_[output].Producer = operation.Id;
	Operations_.PushBack(OaStdMove(operation));
	return Operations_.Back().Id;
}

OaStatus OaSemanticGraph::AttachAutograd(
	OaSemanticOperationId InForwardOperation,
	OaU32 InOutputIndex,
	OaU64 InSequence)
{
	if (InForwardOperation >= Operations_.Size()) {
		return OaStatus::Error(OaStatusCode::OutOfRange,
			"semantic autograd attachment references an unknown operation");
	}
	const auto& operation = Operations_[InForwardOperation];
	if (operation.Differentiation != OaOperationDifferentiation::Reverse) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"semantic autograd attachment requires a reverse-mode contract");
	}
	if (InOutputIndex >= operation.Outputs.Size()) {
		return OaStatus::Error(OaStatusCode::OutOfRange,
			"semantic autograd attachment references an unknown output");
	}
	if (InSequence == 0U) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"semantic autograd attachment requires a nonzero tape sequence");
	}
	const auto output = operation.Outputs[InOutputIndex];
	for (const auto& attachment : Autograd_) {
		if (attachment.Output == output) {
			return OaStatus::Error(OaStatusCode::AlreadyExists,
				"semantic output already has an autograd attachment");
		}
		if (attachment.Sequence == InSequence) {
			return OaStatus::Error(OaStatusCode::AlreadyExists,
				"semantic autograd sequence is already attached");
		}
	}
	Autograd_.PushBack({
		.ForwardOperation = InForwardOperation,
		.Output = output,
		.OutputIndex = InOutputIndex,
		.Sequence = InSequence,
	});
	return OaStatus::Ok();
}

OaStatus OaSemanticGraph::CompleteAutograd(
	OaSemanticOperationId InForwardOperation,
	OaU64 InSequence,
	OaSemanticOperationId InBackwardFirstOperation,
	OaU32 InBackwardOperationCount)
{
	OaSemanticAutogradDesc* attachment = nullptr;
	for (auto& candidate : Autograd_) {
		if (candidate.ForwardOperation == InForwardOperation
			and candidate.Sequence == InSequence)
		{
			attachment = &candidate;
			break;
		}
	}
	if (not attachment) {
		return OaStatus::Error(OaStatusCode::NotFound,
			"semantic backward expansion has no matching tape attachment");
	}
	if (attachment->BackwardExpanded) {
		return OaStatus::Error(OaStatusCode::AlreadyExists,
			"semantic backward expansion is already complete");
	}
	if (InBackwardFirstOperation > Operations_.Size()
		or InBackwardOperationCount
			> Operations_.Size() - InBackwardFirstOperation)
	{
		return OaStatus::Error(OaStatusCode::OutOfRange,
			"semantic backward expansion range is outside the graph");
	}
	for (OaU32 offset = 0; offset < InBackwardOperationCount; ++offset) {
		const auto operationId = InBackwardFirstOperation + offset;
		const auto& operation = Operations_[operationId];
		if (operationId <= InForwardOperation) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"semantic backward expansion must follow its forward operation");
		}
		if (operation.BackwardOf != OaInvalidSemanticOperationId) {
			return OaStatus::Error(OaStatusCode::AlreadyExists,
				"semantic operation already belongs to a backward expansion");
		}
	}

	attachment->BackwardFirstOperation = InBackwardFirstOperation;
	attachment->BackwardOperationCount = InBackwardOperationCount;
	attachment->BackwardExpanded = true;
	for (OaU32 offset = 0; offset < InBackwardOperationCount; ++offset) {
		auto& operation = Operations_[InBackwardFirstOperation + offset];
		operation.BackwardOf = InForwardOperation;
		operation.BackwardSequence = InSequence;
	}
	return OaStatus::Ok();
}

OaStatus OaSemanticGraph::Validate() const {
	for (OaU32 index = 0; index < Values_.Size(); ++index) {
		const auto& value = Values_[index];
		if (value.Id != index) {
			return OaStatus::Error(OaStatusCode::Internal,
				"semantic graph value ids are not canonical");
		}
		if (value.Producer != OaInvalidSemanticOperationId
			and value.Producer >= Operations_.Size())
		{
			return OaStatus::Error(OaStatusCode::Internal,
				"semantic graph value has an invalid producer");
		}
	}
	for (OaU32 index = 0; index < Operations_.Size(); ++index) {
		const auto& operation = Operations_[index];
		if (operation.Id != index or operation.Name.Empty()
			or operation.ContractHash == 0U)
		{
			return OaStatus::Error(OaStatusCode::Internal,
				"semantic graph operation identity is invalid");
		}
		for (const auto input : operation.Inputs) {
			if (input >= Values_.Size()) {
				return OaStatus::Error(OaStatusCode::Internal,
					"semantic graph operation has an invalid input");
			}
		}
		for (const auto output : operation.Outputs) {
			if (output >= Values_.Size() or Values_[output].Producer != index) {
				return OaStatus::Error(OaStatusCode::Internal,
					"semantic graph producer/output relationship is invalid");
			}
		}
		for (OaU32 attribute = 0;
			attribute < operation.Attributes.Size(); ++attribute)
		{
			if (not operation.Attributes[attribute].Validate().IsOk()) {
				return OaStatus::Error(OaStatusCode::Internal,
					"semantic graph operation has an invalid attribute");
			}
			for (OaU32 previous = 0; previous < attribute; ++previous) {
				if (operation.Attributes[previous].Name
					== operation.Attributes[attribute].Name)
				{
					return OaStatus::Error(OaStatusCode::Internal,
						"semantic graph operation has duplicate attribute names");
				}
			}
		}
		for (const auto& access : operation.Accesses) {
			if (access.Value >= Values_.Size()) {
				return OaStatus::Error(OaStatusCode::Internal,
					"semantic graph operation has an invalid access");
			}
		}
		for (const auto mutated : operation.MutatedInputs) {
			if (mutated >= Values_.Size()) {
				return OaStatus::Error(OaStatusCode::Internal,
					"semantic graph operation has an invalid mutation");
			}
		}
		for (const auto& alias : operation.Aliases) {
			if (alias.Output >= Values_.Size() or alias.Input >= Values_.Size()) {
				return OaStatus::Error(OaStatusCode::Internal,
					"semantic graph operation has an invalid alias");
			}
		}
		for (const auto dependency : operation.ControlDependencies) {
			if (dependency >= index) {
				return OaStatus::Error(OaStatusCode::Internal,
					"semantic graph operation has an invalid control dependency");
			}
		}
		if (operation.BackwardOf != OaInvalidSemanticOperationId) {
			if (operation.BackwardOf >= index
				or operation.BackwardSequence == 0U)
			{
				return OaStatus::Error(OaStatusCode::Internal,
					"semantic backward operation has invalid provenance");
			}
		}
	}
	for (OaU32 index = 0; index < Autograd_.Size(); ++index) {
		const auto& attachment = Autograd_[index];
		if (attachment.ForwardOperation >= Operations_.Size()) {
			return OaStatus::Error(OaStatusCode::Internal,
				"semantic autograd attachment has an invalid operation");
		}
		const auto& operation = Operations_[attachment.ForwardOperation];
		if (operation.Differentiation != OaOperationDifferentiation::Reverse
			or attachment.OutputIndex >= operation.Outputs.Size()
			or operation.Outputs[attachment.OutputIndex] != attachment.Output
			or attachment.Sequence == 0U)
		{
			return OaStatus::Error(OaStatusCode::Internal,
				"semantic autograd attachment contract is invalid");
		}
		for (OaU32 previous = 0; previous < index; ++previous) {
			if (Autograd_[previous].Output == attachment.Output
				or Autograd_[previous].Sequence == attachment.Sequence)
			{
				return OaStatus::Error(OaStatusCode::Internal,
					"semantic autograd attachment identity is duplicated");
			}
		}
		if (attachment.BackwardExpanded) {
			if (attachment.BackwardFirstOperation > Operations_.Size()
				or attachment.BackwardOperationCount
					> Operations_.Size() - attachment.BackwardFirstOperation)
			{
				return OaStatus::Error(OaStatusCode::Internal,
					"semantic backward expansion range is invalid");
			}
			for (OaU32 offset = 0;
				offset < attachment.BackwardOperationCount; ++offset)
			{
				const auto& backward = Operations_[
					attachment.BackwardFirstOperation + offset];
				if (backward.BackwardOf != attachment.ForwardOperation
					or backward.BackwardSequence != attachment.Sequence)
				{
					return OaStatus::Error(OaStatusCode::Internal,
						"semantic backward operation disagrees with its expansion");
				}
			}
		} else if (attachment.BackwardFirstOperation
				!= OaInvalidSemanticOperationId
			or attachment.BackwardOperationCount != 0U)
		{
			return OaStatus::Error(OaStatusCode::Internal,
				"incomplete semantic backward expansion owns a range");
		}
	}
	return OaStatus::Ok();
}

OaStatus OaSemanticGraph::CopyFrom(const OaSemanticGraph& InSource) {
	if (this == &InSource) {
		return OaStatus::InvalidArgument(
			"OaSemanticGraph::CopyFrom cannot copy a graph onto itself");
	}
	if (not Values_.Empty() or not Operations_.Empty() or not Autograd_.Empty()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaSemanticGraph::CopyFrom requires an empty destination graph");
	}
	OA_RETURN_IF_ERROR(InSource.Validate());
	Values_ = InSource.Values_;
	Operations_ = InSource.Operations_;
	Autograd_ = InSource.Autograd_;
	return OaStatus::Ok();
}

OaResult<OaSemanticLoweringAnalysis> OaAnalyzeSemanticLowering(
	const OaSemanticGraph& InSemantic,
	const OaComputeGraph& InExecutable)
{
	OA_RETURN_IF_ERROR(InSemantic.Validate());

	const auto operations = InSemantic.Operations();
	OaSemanticLoweringAnalysis analysis;
	analysis.ExecutableNodeCounts_.Resize(operations.Size(), 0U);
	OaVec<OaU8> fusionMembership(operations.Size(), 0U);
	for (const auto& node : InExecutable.Nodes()) {
		const OaU32 ownerCount =
			static_cast<OaU32>(node.SemanticOperations.Size());
		analysis.MaximumOperationsPerNode_ = std::max(
			analysis.MaximumOperationsPerNode_, ownerCount);
		if (ownerCount == 0U) {
			++analysis.CompatibilityNodeCount_;
			continue;
		}
		if (ownerCount > 1U) {
			if (node.Operation.Empty() or node.OperationContractHash != 0U) {
				return OaStatus::Error(OaStatusCode::FailedPrecondition,
					"fused executable node requires a distinct implementation identity");
			}
			++analysis.FusedNodeCount_;
		}
		for (OaU32 owner = 0; owner < ownerCount; ++owner) {
			const auto semanticOperation = node.SemanticOperations[owner];
			if (semanticOperation >= operations.Size()) {
				return OaStatus::Error(OaStatusCode::OutOfRange,
					"executable node references an unknown semantic operation");
			}
			for (OaU32 previous = 0; previous < owner; ++previous) {
				if (node.SemanticOperations[previous] == semanticOperation) {
					return OaStatus::Error(OaStatusCode::AlreadyExists,
						"executable node repeats a semantic operation owner");
				}
			}
			const auto& operation = operations[semanticOperation];
			if (ownerCount == 1U
				and (node.OperationContractHash != operation.ContractHash
					or node.Operation != operation.Name))
			{
				return OaStatus::Error(OaStatusCode::FailedPrecondition,
					"executable node semantic identity does not match its operation");
			}
			++analysis.ExecutableNodeCounts_[semanticOperation];
			if (ownerCount > 1U) fusionMembership[semanticOperation] = 1U;
		}
		++analysis.SchemaOwnedNodeCount_;
	}
	for (OaU32 operation = 0;
		operation < analysis.ExecutableNodeCounts_.Size(); ++operation)
	{
		const OaU32 nodeCount = analysis.ExecutableNodeCounts_[operation];
		if (nodeCount == 0U) {
			return OaStatus::Error(OaStatusCode::FailedPrecondition,
				"semantic operation has no executable lowering");
		}
		if (nodeCount == 1U and fusionMembership[operation] == 0U) {
			++analysis.DirectOperationCount_;
		}
		if (nodeCount > 1U) ++analysis.DecomposedOperationCount_;
		if (fusionMembership[operation] != 0U) {
			++analysis.FusedOperationCount_;
		}
		analysis.MaximumNodesPerOperation_ = std::max(
			analysis.MaximumNodesPerOperation_, nodeCount);
	}
	return analysis;
}

OaStatus OaValidateSemanticLowering(
	const OaSemanticGraph& InSemantic,
	const OaComputeGraph& InExecutable)
{
	const auto analysis = OaAnalyzeSemanticLowering(InSemantic, InExecutable);
	if (not analysis.IsOk()) return analysis.GetStatus();
	return OaStatus::Ok();
}

const OaSemanticValueDesc* OaSemanticGraph::FindValue(
	OaSemanticValueId InId) const noexcept
{
	return InId < Values_.Size() ? &Values_[InId] : nullptr;
}

OaString OaSemanticGraph::DebugReportJson(OaStringView InName) const {
	std::ostringstream out;
	out << "{\n  \"schema\": \"oa.semantic_graph.v2\",\n  \"name\": ";
	WriteJsonString(out, InName);
	out << ",\n  \"values\": [";
	for (OaU32 index = 0; index < Values_.Size(); ++index) {
		const auto& value = Values_[index];
		out << (index == 0 ? "\n" : ",\n") << "    {\"id\": " << value.Id
			<< ", \"name\": ";
		WriteJsonString(out, value.Name);
		out << ", \"kind\": ";
		WriteJsonString(out, KindName(value.Kind));
		out << ", \"dtype\": ";
		WriteJsonString(out, OaScalarTypeName(value.Dtype));
		out << ", \"shape\": [";
		for (OaI32 dimension = 0; dimension < value.Shape.Rank; ++dimension) {
			if (dimension != 0) out << ", ";
			out << value.Shape[dimension];
		}
		out << "], \"external\": " << (value.External ? "true" : "false")
			<< ", \"virtual\": " << (value.Virtual ? "true" : "false")
			<< ", \"producer\": ";
		if (value.Producer == OaInvalidSemanticOperationId) out << "null";
		else out << value.Producer;
		out << '}';
	}
	if (not Values_.Empty()) out << '\n';
	out << "  ],\n  \"operations\": [";
	for (OaU32 index = 0; index < Operations_.Size(); ++index) {
		const auto& operation = Operations_[index];
		out << (index == 0 ? "\n" : ",\n") << "    {\"id\": " << operation.Id
			<< ", \"name\": ";
		WriteJsonString(out, operation.Name);
		out << ", \"contract_hash\": \"0x" << std::hex << std::setw(16)
			<< std::setfill('0') << operation.ContractHash << std::dec
			<< std::setfill(' ') << "\", \"lowering\": ";
		WriteJsonString(out, LoweringName(operation.Lowering));
		out << ", \"differentiation\": ";
		WriteJsonString(out, DifferentiationName(operation.Differentiation));
		out << ", \"control_flow\": ";
		WriteJsonString(out, ControlFlowName(operation.ControlFlow));
		out << ", \"inputs\": [";
		for (OaU32 value = 0; value < operation.Inputs.Size(); ++value) {
			if (value != 0) out << ", ";
			out << operation.Inputs[value];
		}
		out << "], \"outputs\": [";
		for (OaU32 value = 0; value < operation.Outputs.Size(); ++value) {
			if (value != 0) out << ", ";
			out << operation.Outputs[value];
		}
		out << "], \"attributes\": [";
		for (OaU32 attribute = 0;
			attribute < operation.Attributes.Size(); ++attribute)
		{
			if (attribute != 0) out << ", ";
			const auto& value = operation.Attributes[attribute];
			out << "{\"name\": ";
			WriteJsonString(out, value.Name);
			out << ", \"kind\": ";
			WriteJsonString(out, AttributeKindName(value.Kind));
			out << ", \"value\": ";
			switch (value.Kind) {
				case OaOperationAttributeKind::Boolean:
					out << (value.Boolean ? "true" : "false");
					break;
				case OaOperationAttributeKind::SignedInteger:
					out << value.SignedInteger;
					break;
				case OaOperationAttributeKind::UnsignedInteger:
					out << value.UnsignedInteger;
					break;
				case OaOperationAttributeKind::Float:
					out << std::setprecision(17) << value.Float;
					break;
				case OaOperationAttributeKind::String:
				case OaOperationAttributeKind::Enum:
					WriteJsonString(out, value.Text);
					break;
				case OaOperationAttributeKind::Shape:
					out << '[';
					for (OaI32 dimension = 0;
						dimension < value.Shape.Rank; ++dimension)
					{
						if (dimension != 0) out << ", ";
						out << value.Shape[dimension];
					}
					out << ']';
					break;
			}
			out << '}';
		}
		out << "], \"accesses\": [";
		for (OaU32 access = 0; access < operation.Accesses.Size(); ++access) {
			if (access != 0) out << ", ";
			out << "{\"value\": " << operation.Accesses[access].Value
				<< ", \"mode\": ";
			WriteJsonString(out, AccessName(operation.Accesses[access].Mode));
			out << '}';
		}
		out << "], \"mutated_inputs\": [";
		for (OaU32 value = 0; value < operation.MutatedInputs.Size(); ++value) {
			if (value != 0) out << ", ";
			out << operation.MutatedInputs[value];
		}
		out << "], \"aliases\": [";
		for (OaU32 alias = 0; alias < operation.Aliases.Size(); ++alias) {
			if (alias != 0) out << ", ";
			out << "{\"output\": " << operation.Aliases[alias].Output
				<< ", \"input\": " << operation.Aliases[alias].Input << '}';
		}
		out << "], \"control_dependencies\": [";
		for (OaU32 dependency = 0;
			dependency < operation.ControlDependencies.Size(); ++dependency)
		{
			if (dependency != 0) out << ", ";
			out << operation.ControlDependencies[dependency];
		}
		out << "], \"backward_of\": ";
		if (operation.BackwardOf == OaInvalidSemanticOperationId) out << "null";
		else out << operation.BackwardOf;
		out << ", \"backward_sequence\": " << operation.BackwardSequence
			<< '}';
	}
	if (not Operations_.Empty()) out << '\n';
	out << "  ],\n  \"autograd\": [";
	for (OaU32 index = 0; index < Autograd_.Size(); ++index) {
		const auto& attachment = Autograd_[index];
		out << (index == 0 ? "\n" : ",\n")
			<< "    {\"forward_operation\": " << attachment.ForwardOperation
			<< ", \"output\": " << attachment.Output
			<< ", \"output_index\": " << attachment.OutputIndex
			<< ", \"sequence\": " << attachment.Sequence
			<< ", \"backward_expanded\": "
			<< (attachment.BackwardExpanded ? "true" : "false")
			<< ", \"backward_first_operation\": ";
		if (attachment.BackwardFirstOperation
			== OaInvalidSemanticOperationId)
		{
			out << "null";
		} else {
			out << attachment.BackwardFirstOperation;
		}
		out << ", \"backward_operation_count\": "
			<< attachment.BackwardOperationCount << '}';
	}
	if (not Autograd_.Empty()) out << '\n';
	out << "  ]\n}\n";
	return OaString(out.str());
}

void OaSemanticGraph::Reset() noexcept {
	Values_.Clear();
	Operations_.Clear();
	Autograd_.Clear();
}
