#include <oa/ml/transferWeights.h>

#include "safeTensorsWeightSource.h"
#include "shardedWeightSource.h"

#include <oa/core/filesystem.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/limits.h>
#include <oa/core/std/sync.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>

namespace {

struct TranslatorRegistry {
	oa::Mutex mutex;
	oa::Vector<oa::UniquePtr<oa::ModelTranslator>> translators;
};

TranslatorRegistry& getTranslatorRegistry() {
	static TranslatorRegistry registry;
	return registry;
}

oa::Result<oa::U64> elementCount(oa::Span<const oa::I64> inShape) {
	oa::U64 count = 1;
	for (const oa::I64 dim : inShape) {
		if (dim < 0) return oa::Status::invalidArgument("weight shape contains a negative dimension");
		if (dim == 0) return oa::U64{0};
		const auto value = static_cast<oa::U64>(dim);
		if (count > oa::Limits<oa::U64>::max() / value) {
			return oa::Status::error(oa::StatusCode::OutOfRange, "weight element count overflow");
		}
		count *= value;
	}
	return count;
}

bool sameShape(oa::Span<const oa::I64> inA, oa::Span<const oa::I64> inB) {
	if (inA.size() != inB.size()) return false;
	for (oa::Usize i = 0; i < inA.size(); ++i) if (inA[i] != inB[i]) return false;
	return true;
}

oa::Result<oa::Vector<oa::U8>> readConverted(
	const oa::WeightSource& inSource, const oa::WeightInfo& inInfo, oa::ScalarType inDtype) {
	const oa::U64 scalarBytes = oa::scalarSize(inDtype);
	if (scalarBytes == 0 || inInfo.elementCount > oa::Limits<oa::U64>::max() / scalarBytes) {
		return oa::Status::error(oa::StatusCode::DtypeMismatch, "Invalid transfer target dtype");
	}
	oa::Vector<oa::U8> result(inInfo.elementCount * scalarBytes);
	OA_RETURN_IF_ERROR(inSource.read(
		inInfo.name, oa::Span<oa::U8>(result.data(), result.size()), inDtype));
	return result;
}

oa::Result<oa::Vector<oa::U8>> executeMapping(
	const oa::WeightSource& inSource, const oa::WeightMapping& inMapping) {
	if (inMapping.sources.empty()) {
		return oa::Status::invalidArgument(oa::String("weight mapping has no source: ") + inMapping.target);
	}
	if (inMapping.targetShape.size() > oa::kModelFileMaxRank) {
		return oa::Status::invalidArgument(oa::String("target rank exceeds OAM limit: ") + inMapping.target);
	}
	auto targetCountResult = elementCount(oa::Span<const oa::I64>(
		inMapping.targetShape.data(), inMapping.targetShape.size()));
	if (targetCountResult.isError()) return targetCountResult.getStatus();
	const oa::U64 targetCount = targetCountResult.getValue();
	const oa::U64 scalarBytes = oa::scalarSize(inMapping.targetDtype);
	if (scalarBytes == 0 || targetCount > oa::Limits<oa::U64>::max() / scalarBytes) {
		return oa::Status::error(oa::StatusCode::DtypeMismatch,
			oa::String("Invalid target dtype for mapping: ") + inMapping.target);
	}
	oa::Vector<oa::U8> output(targetCount * scalarBytes);

	oa::Vector<const oa::WeightInfo*> sources;
	sources.reserve(inMapping.sources.size());
	for (const auto& name : inMapping.sources) {
		const auto* info = inSource.find(name);
		if (!info) return oa::Status::notFound(oa::String("source weight not found: ") + name);
		sources.pushBack(info);
	}

	if (inMapping.transform == oa::WeightTransform::Identity) {
		if (sources.size() != 1 || !sameShape(
			oa::Span<const oa::I64>(sources[0]->shape.data(), sources[0]->shape.size()),
			oa::Span<const oa::I64>(inMapping.targetShape.data(), inMapping.targetShape.size()))) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				oa::String("Identity mapping shape mismatch: ") + inMapping.target);
		}
		OA_RETURN_IF_ERROR(inSource.read(inMapping.sources[0],
			oa::Span<oa::U8>(output.data(), output.size()), inMapping.targetDtype));
		return output;
	}

	if (inMapping.transform == oa::WeightTransform::Transpose2D) {
		if (sources.size() != 1 || sources[0]->shape.size() != 2 ||
			inMapping.targetShape.size() != 2 ||
			inMapping.targetShape[0] != sources[0]->shape[1] ||
			inMapping.targetShape[1] != sources[0]->shape[0]) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				oa::String("Transpose2D mapping shape mismatch: ") + inMapping.target);
		}
		auto inputResult = readConverted(inSource, *sources[0], inMapping.targetDtype);
		if (inputResult.isError()) return inputResult.getStatus();
		const auto& input = inputResult.getValue();
		const oa::U64 rows = static_cast<oa::U64>(sources[0]->shape[0]);
		const oa::U64 cols = static_cast<oa::U64>(sources[0]->shape[1]);
		for (oa::U64 row = 0; row < rows; ++row) for (oa::U64 col = 0; col < cols; ++col) {
			oa::memcpy(output.data() + (col * rows + row) * scalarBytes,
				input.data() + (row * cols + col) * scalarBytes, scalarBytes);
		}
		return output;
	}

	const oa::Usize rank = inMapping.targetShape.size();
	const oa::I32 axis = inMapping.transform == oa::WeightTransform::Slice
		? inMapping.slice.axis : inMapping.concatAxis;
	if (axis < 0 || static_cast<oa::Usize>(axis) >= rank) {
		return oa::Status::invalidArgument(oa::String("weight mapping axis is out of range: ") + inMapping.target);
	}
	const oa::Usize axisIndex = static_cast<oa::Usize>(axis);
	oa::U64 outer = 1;
	oa::U64 inner = 1;
	for (oa::Usize d = 0; d < axisIndex; ++d) outer *= static_cast<oa::U64>(inMapping.targetShape[d]);
	for (oa::Usize d = axisIndex + 1; d < rank; ++d)
		inner *= static_cast<oa::U64>(inMapping.targetShape[d]);

	if (inMapping.transform == oa::WeightTransform::Concat) {
		oa::I64 concatenated = 0;
		for (const auto* info : sources) {
			if (info->shape.size() != rank) return oa::Status::error(oa::StatusCode::ShapeMismatch, "Concat rank mismatch");
			for (oa::Usize d = 0; d < rank; ++d) {
				if (static_cast<oa::I32>(d) != axis && info->shape[d] != inMapping.targetShape[d])
					return oa::Status::error(oa::StatusCode::ShapeMismatch, "Concat non-axis dimension mismatch");
			}
			concatenated += info->shape[axisIndex];
		}
		if (concatenated != inMapping.targetShape[axisIndex])
			return oa::Status::error(oa::StatusCode::ShapeMismatch, "Concat target axis mismatch");

		oa::Vector<oa::Vector<oa::U8>> inputBuffers;
		inputBuffers.reserve(sources.size());
		for (const auto* info : sources) {
			auto inputResult = readConverted(inSource, *info, inMapping.targetDtype);
			if (inputResult.isError()) return inputResult.getStatus();
			inputBuffers.pushBack(oa::move(inputResult.getValue()));
		}
		for (oa::U64 out = 0; out < outer; ++out) {
			oa::U64 targetAxisOffset = 0;
			for (oa::Usize sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
				const auto* info = sources[sourceIndex];
				const auto& input = inputBuffers[sourceIndex];
				const oa::U64 axisLength = static_cast<oa::U64>(info->shape[axisIndex]);
				const oa::U64 blockBytes = axisLength * inner * scalarBytes;
				oa::memcpy(output.data() +
					(out * static_cast<oa::U64>(inMapping.targetShape[axisIndex]) * inner + targetAxisOffset * inner) * scalarBytes,
					input.data() + out * axisLength * inner * scalarBytes, blockBytes);
				targetAxisOffset += axisLength;
			}
		}
		return output;
	}

	if (inMapping.transform == oa::WeightTransform::Slice) {
		if (sources.size() != 1 || sources[0]->shape.size() != rank ||
			inMapping.slice.begin < 0 || inMapping.slice.length < 0 ||
			inMapping.slice.begin + inMapping.slice.length > sources[0]->shape[axisIndex]) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch, "Slice bounds or rank mismatch");
		}
		for (oa::Usize d = 0; d < rank; ++d) {
			const oa::I64 expected = d == axisIndex
				? inMapping.slice.length : sources[0]->shape[d];
			if (inMapping.targetShape[d] != expected)
				return oa::Status::error(oa::StatusCode::ShapeMismatch, "Slice target shape mismatch");
		}
		auto inputResult = readConverted(inSource, *sources[0], inMapping.targetDtype);
		if (inputResult.isError()) return inputResult.getStatus();
		const auto& input = inputResult.getValue();
		const oa::U64 sourceAxis = static_cast<oa::U64>(sources[0]->shape[axisIndex]);
		const oa::U64 sliceAxis = static_cast<oa::U64>(inMapping.slice.length);
		const oa::U64 begin = static_cast<oa::U64>(inMapping.slice.begin);
		for (oa::U64 out = 0; out < outer; ++out) {
			oa::memcpy(output.data() + out * sliceAxis * inner * scalarBytes,
				input.data() + (out * sourceAxis * inner + begin * inner) * scalarBytes,
				sliceAxis * inner * scalarBytes);
		}
		return output;
	}

	return oa::Status::error(oa::StatusCode::Unimplemented, "Unknown weight mapping transform");
}

} // namespace

oa::Status oa::registerModelTranslator(oa::UniquePtr<oa::ModelTranslator> inTranslator) {
	if (!inTranslator || inTranslator->name().empty()) {
		return oa::Status::invalidArgument("Cannot register an empty model translator");
	}
	auto& registry = getTranslatorRegistry();
	oa::ScopedLock lock(registry.mutex);
	for (const auto& translator : registry.translators) {
		if (translator->name() == inTranslator->name()) {
			return oa::Status::error(oa::StatusCode::AlreadyExists,
				oa::String("Model translator is already registered: ") + inTranslator->name());
		}
	}
	registry.translators.pushBack(oa::move(inTranslator));
	return oa::Status::ok();
}

const oa::ModelTranslator* oa::findModelTranslator(oa::StringView inName) {
	auto& registry = getTranslatorRegistry();
	oa::ScopedLock lock(registry.mutex);
	for (const auto& translator : registry.translators) {
		if (translator->name() == inName) return translator.get();
	}
	return nullptr;
}

oa::Vector<oa::String> oa::listModelTranslators() {
	auto& registry = getTranslatorRegistry();
	oa::ScopedLock lock(registry.mutex);
	oa::Vector<oa::String> result;
	result.reserve(registry.translators.size());
	for (const auto& translator : registry.translators) result.pushBack(oa::String(translator->name()));
	return result;
}

oa::Status oa::WeightSource::readMatrix(
	oa::Engine& inEngine, oa::StringView inName, oa::Matrix& outMatrix,
	oa::ScalarType inTargetDtype) const
{
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	const auto* info = find(inName);
	if (!info) return oa::Status::notFound(oa::String("weight not found: ") + inName);
	if (info->shape.empty() || info->shape.size() > 4) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch, "oa::Matrix weight rank must be 1-4");
	}
	oa::MatrixShape shape;
	shape.rank = static_cast<oa::I32>(info->shape.size());
	for (oa::I32 i = 0; i < shape.rank; ++i) {
		const auto index = static_cast<oa::Usize>(i);
		shape.dims[index] = info->shape[index];
	}
	outMatrix = oa::FnMatrix::empty(shape, inTargetDtype);
	const oa::U64 byteSize = info->elementCount * oa::scalarSize(inTargetDtype);
	oa::Vector<oa::U8> bytes(byteSize);
	OA_RETURN_IF_ERROR(read(inName, oa::Span<oa::U8>(bytes.data(), bytes.size()), inTargetDtype));
	OA_RETURN_IF_ERROR(oa::EngineResourceAccess::uploadBuffer(inEngine,
		oa::MatrixAccess::descriptor(outMatrix), outMatrix.byteOffset(),
		bytes.data(), bytes.size()));
	return oa::Status::ok();
}

oa::Result<oa::UniquePtr<oa::WeightSource>> oa::openWeightSource(
	const oa::Path& inPath, oa::WeightFormat inFormat) {
	oa::Path path = inPath;
	if (oa::Filesystem::isDirectory(path)) path /= "model.safetensors.index.json";
	oa::WeightFormat format = inFormat;
	if (format == oa::WeightFormat::Auto) {
		const oa::String extension = path.extension().string();
		if (extension == ".safetensors") format = oa::WeightFormat::SafeTensors;
		else if (extension == ".json") format = oa::WeightFormat::SafeTensors;
		else if (extension == ".oam") format = oa::WeightFormat::ModelFile;
		else if (extension == ".gguf") format = oa::WeightFormat::Gguf;
		else if (extension == ".onnx") format = oa::WeightFormat::Onnx;
		else return oa::Status::invalidArgument(oa::String("Cannot infer weight source format: ") + path.string());
	}
	if (format == oa::WeightFormat::SafeTensors) {
		if (path.extension().string() == ".json") {
			auto package = oa::makeUnique<oa::ShardedWeightSource>();
			OA_RETURN_IF_ERROR(package->open(path));
			oa::UniquePtr<oa::WeightSource> source(oa::move(package));
			return oa::Result<oa::UniquePtr<oa::WeightSource>>(oa::move(source));
		}
		auto backend = oa::makeUnique<oa::SafeTensorsWeightSource>();
		OA_RETURN_IF_ERROR(backend->open(path));
		oa::UniquePtr<oa::WeightSource> source(oa::move(backend));
		return oa::Result<oa::UniquePtr<oa::WeightSource>>(oa::move(source));
	}
	return oa::Status::error(oa::StatusCode::Unimplemented, "weight source backend is not implemented");
}

oa::Result<oa::WeightMap> oa::makeRawWeightMap(const oa::WeightSource& inSource) {
	oa::WeightMap map;
	map.architecture = "raw";
	map.requireAllSourceWeights = true;
	for (const auto& info : inSource.list()) {
		oa::WeightMapping mapping;
		mapping.sources.pushBack(info.name);
		mapping.target = info.name;
		mapping.targetShape = info.shape;
		mapping.targetDtype = info.dtype;
		mapping.transform = oa::WeightTransform::Identity;
		map.mappings.pushBack(oa::move(mapping));
	}
	return map;
}

oa::Result<oa::WeightTransferReport> oa::transferWeights(
	const oa::WeightSource& inSource, const oa::WeightMap& inMap, oa::ModelFile& outModel) {
	if (inMap.architecture.empty() || inMap.architecture.size() >= sizeof(outModel.config.architecture)) {
		return oa::Status::invalidArgument("weight map architecture is empty or too long");
	}
	oa::HashSet<oa::String> targets;
	oa::HashSet<oa::String> usedSources;
	oa::ModelFile model;
	if (inMap.archConfig.size() > oa::Limits<oa::U32>::max()) {
		return oa::Status::invalidArgument("architecture config exceeds OAM limit");
	}
	model.config = inMap.config;
	oa::memset(model.config.architecture, 0, sizeof(model.config.architecture));
	oa::memcpy(model.config.architecture, inMap.architecture.data(), inMap.architecture.size());
	model.config.architecture[inMap.architecture.size()] = '\0';
	model.config.configVersion = inMap.configVersion;
	model.config.archConfigSize = static_cast<oa::U32>(inMap.archConfig.size());
	model.archConfig = inMap.archConfig;

	oa::WeightTransferReport report;
	report.sourceWeights = inSource.list().size();
	for (const auto& mapping : inMap.mappings) {
		if (mapping.target.empty() || mapping.target.size() >= oa::kModelFileMaxName) {
			return oa::Status::invalidArgument(oa::String("Invalid OAM target weight name: ") + mapping.target);
		}
		if (!targets.insert(mapping.target).second) {
			return oa::Status::invalidArgument(oa::String("Duplicate target weight mapping: ") + mapping.target);
		}
		for (const auto& source : mapping.sources) usedSources.insert(source);
		auto outputResult = executeMapping(inSource, mapping);
		if (outputResult.isError()) return outputResult.getStatus();
		auto& output = outputResult.getValue();
		oa::Vector<oa::U64> shape;
		shape.reserve(mapping.targetShape.size());
		for (const oa::I64 dim : mapping.targetShape) shape.pushBack(static_cast<oa::U64>(dim));
		model.addWeight(mapping.target.cStr(), mapping.targetDtype,
			oa::Span<const oa::U64>(shape.data(), shape.size()), output.data(), output.size());
		report.outputBytes += output.size();
	}

	for (const auto& info : inSource.list()) {
		if (usedSources.contains(info.name)) ++report.usedSourceWeights;
		else report.unusedSources.pushBack(info.name);
	}
	report.outputWeights = model.weightIndex.size();
	if (inMap.requireAllSourceWeights && !report.unusedSources.empty()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			oa::String("weight map left source tensors unused; first: ") + report.unusedSources[0]);
	}
	outModel = oa::move(model);
	return report;
}
