#include <oa/runtime/dnn.h>
#include <oa/runtime/semanticGraph.h>

#include <algorithm>
#include <cstring>

namespace {

void hashMix(oa::U64& inOut, oa::U64 inValue) {
	inOut ^= inValue;
	inOut *= 0x100000001b3ULL;
}

void hashString(oa::U64& inOut, oa::StringView inValue) {
	hashMix(inOut, inValue.size());
	for (const char value : inValue) {
		hashMix(inOut, static_cast<unsigned char>(value));
	}
}

void hashShape(oa::U64& inOut, const oa::MatrixShape& inShape) {
	hashMix(inOut, static_cast<oa::U64>(inShape.rank));
	for (oa::I32 dimension = 0; dimension < inShape.rank; ++dimension) {
		hashMix(inOut, static_cast<oa::U64>(inShape[dimension]));
	}
}

void hashAttribute(oa::U64& inOut, const oa::OpAttribute& inAttribute) {
	hashString(inOut, inAttribute.name);
	hashMix(inOut, static_cast<oa::U8>(inAttribute.kind));
	switch (inAttribute.kind) {
		case oa::OpAttributeKind::Boolean:
			hashMix(inOut, inAttribute.boolean); break;
		case oa::OpAttributeKind::SignedInteger:
			hashMix(inOut, static_cast<oa::U64>(inAttribute.signedInteger)); break;
		case oa::OpAttributeKind::UnsignedInteger:
			hashMix(inOut, inAttribute.unsignedInteger); break;
		case oa::OpAttributeKind::Float: {
			oa::U64 bits = 0;
			static_assert(sizeof(bits) == sizeof(inAttribute.floatVal));
			std::memcpy(&bits, &inAttribute.floatVal, sizeof(bits));
			hashMix(inOut, bits); break;
		}
		case oa::OpAttributeKind::String:
			hashString(inOut, inAttribute.text); break;
		case oa::OpAttributeKind::Shape:
			hashShape(inOut, inAttribute.shape); break;
		case oa::OpAttributeKind::Enum:
			hashString(inOut, inAttribute.text); break;
	}
}

struct DnnOpRole {
	oa::StringView name;
	oa::U64 contractHash = 0U;
	oa::DnnOpType type = oa::DnnOpType::Portable;
	oa::GemmEpilogue epilogue = oa::GemmEpilogue::None;
	oa::U8 epilogueRequiredInput = UINT8_MAX;
};

#include "dnn/opRoles.gen.inc"

const DnnOpRole* findOpRole(
	const oa::SemanticOpDesc& inOperation) {
	for (const auto& role : kDnnOpRoles) {
		if (role.contractHash == inOperation.contractHash
			and role.name == inOperation.name) {
			return &role;
		}
	}
	return nullptr;
}

oa::GemmEpilogue capturedEpilogue(
	const oa::SemanticOpDesc& inOperation,
	const DnnOpRole* inRole) {
	if (inRole == nullptr) return oa::GemmEpilogue::None;
	if (inRole->epilogueRequiredInput != UINT8_MAX) {
		const auto input = inRole->epilogueRequiredInput;
		if (input >= inOperation.inputs.size()
			or inOperation.inputs[input] == oa::invalidSemanticValueId) {
			return oa::GemmEpilogue::None;
		}
	}
	return inRole->epilogue;
}

bool consumes(const oa::DnnOpDesc& inConsumer, oa::U32 inValue) {
	return std::find(inConsumer.inputs.begin(), inConsumer.inputs.end(), inValue)	!= inConsumer.inputs.end();
}

bool singleEdge(const oa::DnnOpDesc& inA, const oa::DnnOpDesc& inB) {
	return inA.outputs.size() == 1U and consumes(inB, inA.outputs[0]);
}

oa::U32 sourceOp(
	const oa::DnnOpDesc& inOp, oa::U32 inFallback) {
	return inOp.sourceOp != oa::invalidDnnOpId
		? inOp.sourceOp : inFallback;
}

oa::DnnPartition portable(const oa::DnnOpDesc& inOp, oa::U32 inFallback) {
	oa::DnnPartition p;
	p.ops.pushBack(sourceOp(inOp, inFallback));
	return p;
}

oa::Bool isRecognized(oa::DnnEngineType inEngine) {
	return inEngine != oa::DnnEngineType::Portable;
}

oa::U64 semanticGraphHash(
	const oa::SemanticGraph& inGraph, const oa::DnnPolicy& inPolicy) {
	oa::U64 hash = 0xcbf29ce484222325ULL;
	hashMix(hash, 2U);
	hashMix(hash, inPolicy.maxWorkspaceBytes);
	hashMix(hash, inPolicy.requireDeterministic);
	hashMix(hash, inPolicy.allowRecompute);
	hashMix(hash, inGraph.valueCount());
	hashMix(hash, inGraph.operationCount());
	hashMix(hash, inGraph.autograd().size());
	for (const auto& value : inGraph.values()) {
		hashMix(hash, value.id);
		hashString(hash, value.name);
		hashMix(hash, static_cast<oa::U8>(value.kind));
		hashShape(hash, value.shape);
		for (oa::I32 dimension = 0; dimension < value.shape.rank; ++dimension) {
			hashMix(hash, static_cast<oa::U64>(
				value.strides[static_cast<oa::Usize>(dimension)]));
		}
		hashMix(hash, static_cast<oa::U8>(value.dtype));
		hashMix(hash, value.external);
		hashMix(hash, value.isVirtual);
		hashMix(hash, value.producer);
		hashMix(hash, value.viewSource);
		hashMix(hash, static_cast<oa::U64>(value.viewByteOffset));
	}
	for (const auto& op : inGraph.operations()) {
		hashMix(hash, op.id);
		hashString(hash, op.name);
		hashMix(hash, op.contractHash);
		hashMix(hash, static_cast<oa::U8>(op.differentiation));
		hashMix(hash, static_cast<oa::U8>(op.lowering));
		hashMix(hash, static_cast<oa::U8>(op.controlFlow));
		hashMix(hash, op.optionalInputMask);
		hashMix(hash, op.inputs.size());
		for (const auto input : op.inputs) hashMix(hash, input);
		hashMix(hash, op.outputs.size());
		for (const auto output : op.outputs) hashMix(hash, output);
		hashMix(hash, op.attributes.size());
		for (const auto& attribute : op.attributes) hashAttribute(hash, attribute);
		hashMix(hash, op.accesses.size());
		for (const auto& access : op.accesses) {
			hashMix(hash, access.value);
			hashMix(hash, static_cast<oa::U8>(access.mode));
		}
		hashMix(hash, op.mutatedInputs.size());
		for (const auto input : op.mutatedInputs) hashMix(hash, input);
		hashMix(hash, op.aliases.size());
		for (const auto& alias : op.aliases) {
			hashMix(hash, alias.output); hashMix(hash, alias.input);
		}
		hashMix(hash, op.controlDependencies.size());
		for (const auto dependency : op.controlDependencies) {
			hashMix(hash, dependency);
		}
		hashMix(hash, op.backwardOf);
		hashMix(hash, op.backwardSequence);
	}
	for (const auto& autograd : inGraph.autograd()) {
		hashMix(hash, autograd.forwardOp);
		hashMix(hash, autograd.output);
		hashMix(hash, autograd.outputIndex);
		hashMix(hash, autograd.sequence);
		hashMix(hash, autograd.backwardFirstOp);
		hashMix(hash, autograd.backwardOpCount);
		hashMix(hash, autograd.backwardExpanded);
	}
	return hash;
}

} // Namespace

oa::Status oa::DnnGraph::addMatrix(const oa::DnnMatrixDesc& inMatrix) {
	if (inMatrix.id == oa::invalidDnnMatrixId or inMatrix.shape.rank <= 0
		or inMatrix.shape.rank > 4 or inMatrix.shape.numElements() <= 0
	) {
		return oa::Status::invalidArgument("DNN matrix has an invalid id or shape");
	}
	for (oa::I32 dimension = 0; dimension < inMatrix.shape.rank; ++dimension) {
		if (inMatrix.shape[dimension] <= 0) {
			return oa::Status::invalidArgument(
				"DNN matrix dimensions must be positive");
		}
	}
	if (findMatrix(inMatrix.id) != nullptr) {
		return oa::Status::invalidArgument("DNN matrix ids must be unique");
	}
	matrices_.pushBack(inMatrix);
	return oa::Status::ok();
}

oa::Status oa::DnnGraph::addOp(const oa::DnnOpDesc& inOp) {
	if (inOp.inputs.empty() or inOp.outputs.empty()) {
		return oa::Status::invalidArgument("DNN operation needs inputs and outputs");
	}
	oa::DnnOpDesc op = inOp;
	if (op.sourceOp == oa::invalidDnnOpId) {
		op.sourceOp = static_cast<oa::U32>(ops_.size());
	}
	for (const auto& existing : ops_) {
		if (existing.sourceOp == op.sourceOp) {
			return oa::Status::invalidArgument(
				"DNN source operation ids must be unique");
		}
	}
	ops_.pushBack(oa::move(op));
	return oa::Status::ok();
}

const oa::DnnMatrixDesc* oa::DnnGraph::findMatrix(oa::U32 inId) const {
	for (const auto& matrix : matrices_) {
		if (matrix.id == inId) return &matrix;
	}
	return nullptr;
}

oa::Status oa::DnnGraph::validate() const {
	if (matrices_.empty() or ops_.empty()) {
		return oa::Status::invalidArgument("DNN graph cannot be empty");
	}
	oa::Vec<oa::U32> produced;
	for (oa::U32 opIdx = 0; opIdx < ops_.size(); ++opIdx) {
		const auto& op = ops_[opIdx];
		for (auto id : op.inputs) {
			const auto* matrix = findMatrix(id);
			if (matrix == nullptr) return oa::Status::invalidArgument("DNN op has a dangling input");
			const bool hasProducer = std::find(produced.begin(), produced.end(), id) != produced.end();
			if (not matrix->external and not hasProducer) {
				return oa::Status::invalidArgument("DNN op consumes a value before its producer");
			}
		}
		for (auto id : op.outputs) {
			const auto* matrix = findMatrix(id);
			if (matrix == nullptr) return oa::Status::invalidArgument("DNN op has a dangling output");
			if (std::find(produced.begin(), produced.end(), id) != produced.end()) {
				return oa::Status::invalidArgument("DNN graph violates single-assignment output semantics");
			}
			produced.pushBack(id);
		}
	}
	return oa::Status::ok();
}

oa::Result<oa::DnnPlan> oa::DnnPlanner::plan(const oa::DnnGraph& inGraph, const oa::DnnPolicy& inPolicy) {
	const auto status = inGraph.validate();
	if (not status.isOk()) {
		return status;
	}

	oa::DnnPlan plan;
	oa::U64 hash = 0xcbf29ce484222325ULL;
	hashMix(hash, plan.plannerAbi);
	hashMix(hash, inPolicy.maxWorkspaceBytes);
	hashMix(hash, inPolicy.requireDeterministic);
	hashMix(hash, inPolicy.allowRecompute);
	hashMix(hash, inGraph.matrices().size());
	hashMix(hash, inGraph.ops().size());
	for (const auto& matrix : inGraph.matrices()) {
		hashMix(hash, matrix.id); hashMix(hash, static_cast<oa::U8>(matrix.dtype));
		hashShape(hash, matrix.shape);
		hashMix(hash, matrix.external); hashMix(hash, matrix.isVirtual);
	}
	for (const auto& op : inGraph.ops()) {
		hashMix(hash, op.sourceOp);
		hashMix(hash, static_cast<oa::U8>(op.type));
		hashMix(hash, static_cast<oa::U8>(op.epilogue));
		hashMix(hash, op.training);
		hashMix(hash, op.inputs.size());
		for (const auto input : op.inputs) hashMix(hash, input);
		hashMix(hash, op.outputs.size());
		for (const auto output : op.outputs) hashMix(hash, output);
	}

	const auto ops = inGraph.ops();
	plan.sourceOpCount = static_cast<oa::U32>(ops.size());
	plan.capturedOpCount = static_cast<oa::U32>(ops.size());
	for (oa::U32 i = 0; i < ops.size();) {
		const auto& op = ops[i];

		// Three projections sharing one activation are one semantic QKV region.
		// The partition name does not promise a packed execution strategy.
		if (i + 2U < ops.size() and op.type == oa::DnnOpType::Matmul
			and ops[i + 1U].type == oa::DnnOpType::Matmul
			and ops[i + 2U].type == oa::DnnOpType::Matmul
			and not op.inputs.empty() and not ops[i + 1U].inputs.empty()
			and not ops[i + 2U].inputs.empty()
			and op.inputs[0] == ops[i + 1U].inputs[0]
			and op.inputs[0] == ops[i + 2U].inputs[0]) {
			oa::DnnPartition p; p.engine = oa::DnnEngineType::QkvProjectionGroup;
			p.ops = {
				sourceOp(op, i),
				sourceOp(ops[i + 1U], i + 1U),
				sourceOp(ops[i + 2U], i + 2U)};
			if (op.training) p.savedForBackward.pushBack(op.inputs[0]);
			plan.partitions.pushBack(std::move(p)); i += 3U; continue;
		}

		// gate matmul, up matmul, siLU(gate), multiply(silu, up)
		if (i + 3U < ops.size() and op.type == oa::DnnOpType::Matmul
			and ops[i + 1U].type == oa::DnnOpType::Matmul
			and ops[i + 2U].type == oa::DnnOpType::Silu
			and ops[i + 3U].type == oa::DnnOpType::Multiply
			and not op.inputs.empty() and not ops[i + 1U].inputs.empty()
			and op.inputs[0] == ops[i + 1U].inputs[0]
			and singleEdge(op, ops[i + 2U])
			and singleEdge(ops[i + 2U], ops[i + 3U])
			and singleEdge(ops[i + 1U], ops[i + 3U])) {
			oa::DnnPartition p; p.engine = oa::DnnEngineType::GatedFfn;
			p.ops = {
				sourceOp(op, i),
				sourceOp(ops[i + 1U], i + 1U),
				sourceOp(ops[i + 2U], i + 2U),
				sourceOp(ops[i + 3U], i + 3U)};
			if (op.training) p.savedForBackward.pushBack(op.inputs[0]);
			plan.partitions.pushBack(std::move(p)); i += 4U; continue;
		}
		// Shipping two-input SwiGLU is already one semantic gated-multiply op.
		if (i + 2U < ops.size() and op.type == oa::DnnOpType::Matmul
			and ops[i + 1U].type == oa::DnnOpType::Matmul
			and ops[i + 2U].type == oa::DnnOpType::GatedMultiply
			and not op.inputs.empty() and not ops[i + 1U].inputs.empty()
			and op.inputs[0] == ops[i + 1U].inputs[0]
			and singleEdge(op, ops[i + 2U])
			and singleEdge(ops[i + 1U], ops[i + 2U])) {
			oa::DnnPartition p; p.engine = oa::DnnEngineType::GatedFfn;
			p.ops = {
				sourceOp(op, i),
				sourceOp(ops[i + 1U], i + 1U),
				sourceOp(ops[i + 2U], i + 2U)};
			if (op.training) p.savedForBackward.pushBack(op.inputs[0]);
			plan.partitions.pushBack(std::move(p)); i += 3U; continue;
		}

		// generated Linear* contracts already carry the exact epilogue and do
		// not need to be decomposed into invented Matmul/Bias/Activation nodes.
		if (op.type == oa::DnnOpType::Matmul
			and op.epilogue != oa::GemmEpilogue::None) {
			auto p = portable(op, i);
			p.engine = oa::DnnEngineType::BlasLtEpilogue;
			if (op.training and not inPolicy.allowRecompute
				and not op.outputs.empty()) {
				p.savedForBackward.pushBack(op.outputs[0]);
			}
			plan.partitions.pushBack(oa::move(p)); ++i; continue;
		}

		if (op.type == oa::DnnOpType::Matmul and i + 1U < ops.size()
			and ops[i + 1U].type == oa::DnnOpType::BiasAdd and singleEdge(op, ops[i + 1U])) {
			oa::DnnPartition p; p.engine = oa::DnnEngineType::BlasLtEpilogue;
			p.ops = {
				sourceOp(op, i),
				sourceOp(ops[i + 1U], i + 1U)};
			if (i + 2U < ops.size()
				and (ops[i + 2U].type == oa::DnnOpType::Relu
					or ops[i + 2U].type == oa::DnnOpType::Gelu
					or ops[i + 2U].type == oa::DnnOpType::Silu)
				and singleEdge(ops[i + 1U], ops[i + 2U])) {
				p.ops.pushBack(sourceOp(ops[i + 2U], i + 2U));
				if (op.training and not inPolicy.allowRecompute) {
					p.savedForBackward.pushBack(ops[i + 1U].outputs[0]);
				}
			}
			const oa::U32 consumed = static_cast<oa::U32>(p.ops.size());
			plan.partitions.pushBack(std::move(p)); i += consumed; continue;
		}

		if (op.type == oa::DnnOpType::FlashAttentionCausal) {
			auto p = portable(op, i); p.engine = oa::DnnEngineType::FlashAttention;
			if (op.training) {
				for (auto id : op.outputs) p.savedForBackward.pushBack(id);
			}
			plan.partitions.pushBack(std::move(p)); ++i; continue;
		}
		if (op.type == oa::DnnOpType::GroupedGemm) {
			auto p = portable(op, i); p.engine = oa::DnnEngineType::GroupedMoe;
			if (op.training and not op.inputs.empty()) p.savedForBackward.pushBack(op.inputs[0]);
			plan.partitions.pushBack(std::move(p)); ++i; continue;
		}
		if (op.type == oa::DnnOpType::Add and i + 1U < ops.size()
			and ops[i + 1U].type == oa::DnnOpType::RmsNorm and singleEdge(op, ops[i + 1U])) {
			oa::DnnPartition p; p.engine = oa::DnnEngineType::ResidualNorm;
			p.ops = {
				sourceOp(op, i),
				sourceOp(ops[i + 1U], i + 1U)};
			plan.partitions.pushBack(std::move(p)); i += 2U; continue;
		}
		if (op.type == oa::DnnOpType::ResidualRmsNorm) {
			auto p = portable(op, i); p.engine = oa::DnnEngineType::ResidualNorm;
			plan.partitions.pushBack(oa::move(p)); ++i; continue;
		}

		plan.partitions.pushBack(portable(op, i)); ++i;
	}
	plan.graphHash = hash;
	for (const auto& partition : plan.partitions) {
		if (isRecognized(partition.engine)) ++plan.recognizedPartitionCount;
	}
	return plan;
}

oa::Result<oa::DnnPlan> oa::DnnPlanner::plan(
	const oa::SemanticGraph& inGraph, const oa::DnnPolicy& inPolicy) {
	const auto validation = inGraph.validate();
	if (not validation.isOk()) return validation;

	oa::DnnGraph captured;
	oa::Vec<oa::U8> trainingOperations;
	trainingOperations.resize(inGraph.operationCount(), 0U);
	for (const auto& autograd : inGraph.autograd()) {
		trainingOperations[autograd.forwardOp] = 1U;
	}
	for (const auto& value : inGraph.values()) {
		if (value.kind != oa::OpValueKind::Matrix
			or value.shape.rank <= 0 or value.shape.numElements() <= 0) {
			continue;
		}
		const auto status = captured.addMatrix({
			.id = value.id,
			.shape = value.shape,
			.dtype = value.dtype,
			.external = value.external
				or value.producer == oa::invalidSemanticOpId,
			.isVirtual = value.isVirtual,
		});
		if (not status.isOk()) return status;
	}

	for (const auto& operation : inGraph.operations()) {
		const auto* role = findOpRole(operation);
		oa::DnnOpDesc op;
		op.sourceOp = operation.id;
		op.type = role != nullptr ? role->type : oa::DnnOpType::Portable;
		op.epilogue = capturedEpilogue(operation, role);
		op.training = trainingOperations[operation.id] != 0U;

		oa::Bool matrixOnly = true;
		for (const auto input : operation.inputs) {
			if (input == oa::invalidSemanticValueId) continue;
			if (captured.findMatrix(input) == nullptr) {
				matrixOnly = false; break;
			}
			op.inputs.pushBack(input);
		}
		if (not matrixOnly) continue;
		for (const auto output : operation.outputs) {
			if (captured.findMatrix(output) == nullptr) {
				matrixOnly = false; break;
			}
			op.outputs.pushBack(output);
		}
		if (not matrixOnly or op.inputs.empty() or op.outputs.empty()) continue;
		const auto status = captured.addOp(op);
		if (not status.isOk()) return status;
	}

	const oa::U64 graphHash = semanticGraphHash(inGraph, inPolicy);
	if (captured.ops().empty()) {
		oa::DnnPlan plan;
		plan.graphHash = graphHash;
		plan.sourceOpCount = inGraph.operationCount();
		return plan;
	}

	auto result = plan(captured, inPolicy);
	if (not result.isOk()) return result.getStatus();
	auto plan = oa::move(result).getValue();
	plan.graphHash = graphHash;
	plan.sourceOpCount = inGraph.operationCount();
	plan.capturedOpCount = static_cast<oa::U32>(captured.ops().size());
	return plan;
}
