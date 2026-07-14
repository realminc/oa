#include <Oa/Ml/TransferWeights.h>

#include "SafeTensorsWeightSource.h"
#include "ShardedWeightSource.h"

#include <Oa/Core/FileIo.h>
#include <Oa/Core/FnMatrix.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>

namespace {

struct AdapterRegistry {
	std::mutex Mutex;
	OaVec<OaUniquePtr<OaModelWeightAdapter>> Adapters;
};

AdapterRegistry& GetAdapterRegistry() {
	static AdapterRegistry registry;
	return registry;
}

OaResult<OaU64> ElementCount(OaSpan<const OaI64> InShape) {
	OaU64 count = 1;
	for (const OaI64 dim : InShape) {
		if (dim < 0) return OaStatus::InvalidArgument("Weight shape contains a negative dimension");
		if (dim == 0) return OaU64{0};
		const auto value = static_cast<OaU64>(dim);
		if (count > std::numeric_limits<OaU64>::max() / value) {
			return OaStatus::Error(OaStatusCode::OutOfRange, "Weight element count overflow");
		}
		count *= value;
	}
	return count;
}

bool SameShape(OaSpan<const OaI64> InA, OaSpan<const OaI64> InB) {
	if (InA.Size() != InB.Size()) return false;
	for (OaUsize i = 0; i < InA.Size(); ++i) if (InA[i] != InB[i]) return false;
	return true;
}

OaResult<OaVec<OaU8>> ReadConverted(
	const OaWeightSource& InSource, const OaWeightInfo& InInfo, OaScalarType InDtype) {
	const OaU64 scalarBytes = OaScalarSize(InDtype);
	if (scalarBytes == 0 || InInfo.ElementCount > std::numeric_limits<OaU64>::max() / scalarBytes) {
		return OaStatus::Error(OaStatusCode::DtypeMismatch, "Invalid transfer target dtype");
	}
	OaVec<OaU8> result(InInfo.ElementCount * scalarBytes);
	OA_RETURN_IF_ERROR(InSource.Read(
		InInfo.Name, OaSpan<OaU8>(result.Data(), result.Size()), InDtype));
	return result;
}

OaResult<OaVec<OaU8>> ExecuteMapping(
	const OaWeightSource& InSource, const OaWeightMapping& InMapping) {
	if (InMapping.Sources.Empty()) {
		return OaStatus::InvalidArgument(OaString("Weight mapping has no source: ") + InMapping.Target);
	}
	if (InMapping.TargetShape.Size() > OAM_MAX_RANK) {
		return OaStatus::InvalidArgument(OaString("Target rank exceeds OAM limit: ") + InMapping.Target);
	}
	auto targetCountResult = ElementCount(OaSpan<const OaI64>(
		InMapping.TargetShape.Data(), InMapping.TargetShape.Size()));
	if (targetCountResult.IsError()) return targetCountResult.GetStatus();
	const OaU64 targetCount = targetCountResult.GetValue();
	const OaU64 scalarBytes = OaScalarSize(InMapping.TargetDtype);
	if (scalarBytes == 0 || targetCount > std::numeric_limits<OaU64>::max() / scalarBytes) {
		return OaStatus::Error(OaStatusCode::DtypeMismatch,
			OaString("Invalid target dtype for mapping: ") + InMapping.Target);
	}
	OaVec<OaU8> output(targetCount * scalarBytes);

	OaVec<const OaWeightInfo*> sources;
	sources.Reserve(InMapping.Sources.Size());
	for (const auto& name : InMapping.Sources) {
		const auto* info = InSource.Find(name);
		if (!info) return OaStatus::NotFound(OaString("Source weight not found: ") + name);
		sources.PushBack(info);
	}

	if (InMapping.Transform == OaWeightTransform::Identity) {
		if (sources.Size() != 1 || !SameShape(
			OaSpan<const OaI64>(sources[0]->Shape.Data(), sources[0]->Shape.Size()),
			OaSpan<const OaI64>(InMapping.TargetShape.Data(), InMapping.TargetShape.Size()))) {
			return OaStatus::Error(OaStatusCode::ShapeMismatch,
				OaString("Identity mapping shape mismatch: ") + InMapping.Target);
		}
		OA_RETURN_IF_ERROR(InSource.Read(InMapping.Sources[0],
			OaSpan<OaU8>(output.Data(), output.Size()), InMapping.TargetDtype));
		return output;
	}

	if (InMapping.Transform == OaWeightTransform::Transpose2D) {
		if (sources.Size() != 1 || sources[0]->Shape.Size() != 2 ||
			InMapping.TargetShape.Size() != 2 ||
			InMapping.TargetShape[0] != sources[0]->Shape[1] ||
			InMapping.TargetShape[1] != sources[0]->Shape[0]) {
			return OaStatus::Error(OaStatusCode::ShapeMismatch,
				OaString("Transpose2D mapping shape mismatch: ") + InMapping.Target);
		}
		auto inputResult = ReadConverted(InSource, *sources[0], InMapping.TargetDtype);
		if (inputResult.IsError()) return inputResult.GetStatus();
		const auto& input = inputResult.GetValue();
		const OaU64 rows = static_cast<OaU64>(sources[0]->Shape[0]);
		const OaU64 cols = static_cast<OaU64>(sources[0]->Shape[1]);
		for (OaU64 row = 0; row < rows; ++row) for (OaU64 col = 0; col < cols; ++col) {
			OaMemcpy(output.Data() + (col * rows + row) * scalarBytes,
				input.Data() + (row * cols + col) * scalarBytes, scalarBytes);
		}
		return output;
	}

	const OaUsize rank = InMapping.TargetShape.Size();
	const OaI32 axis = InMapping.Transform == OaWeightTransform::Slice
		? InMapping.Slice.Axis : InMapping.ConcatAxis;
	if (axis < 0 || static_cast<OaUsize>(axis) >= rank) {
		return OaStatus::InvalidArgument(OaString("Weight mapping axis is out of range: ") + InMapping.Target);
	}
	const OaUsize axisIndex = static_cast<OaUsize>(axis);
	OaU64 outer = 1;
	OaU64 inner = 1;
	for (OaUsize d = 0; d < axisIndex; ++d) outer *= static_cast<OaU64>(InMapping.TargetShape[d]);
	for (OaUsize d = axisIndex + 1; d < rank; ++d)
		inner *= static_cast<OaU64>(InMapping.TargetShape[d]);

	if (InMapping.Transform == OaWeightTransform::Concat) {
		OaI64 concatenated = 0;
		for (const auto* info : sources) {
			if (info->Shape.Size() != rank) return OaStatus::Error(OaStatusCode::ShapeMismatch, "Concat rank mismatch");
			for (OaUsize d = 0; d < rank; ++d) {
				if (static_cast<OaI32>(d) != axis && info->Shape[d] != InMapping.TargetShape[d])
					return OaStatus::Error(OaStatusCode::ShapeMismatch, "Concat non-axis dimension mismatch");
			}
			concatenated += info->Shape[axisIndex];
		}
		if (concatenated != InMapping.TargetShape[axisIndex])
			return OaStatus::Error(OaStatusCode::ShapeMismatch, "Concat target axis mismatch");

		OaVec<OaVec<OaU8>> inputBuffers;
		inputBuffers.Reserve(sources.Size());
		for (const auto* info : sources) {
			auto inputResult = ReadConverted(InSource, *info, InMapping.TargetDtype);
			if (inputResult.IsError()) return inputResult.GetStatus();
			inputBuffers.PushBack(OaStdMove(inputResult.GetValue()));
		}
		for (OaU64 out = 0; out < outer; ++out) {
			OaU64 targetAxisOffset = 0;
			for (OaUsize sourceIndex = 0; sourceIndex < sources.Size(); ++sourceIndex) {
				const auto* info = sources[sourceIndex];
				const auto& input = inputBuffers[sourceIndex];
				const OaU64 axisLength = static_cast<OaU64>(info->Shape[axisIndex]);
				const OaU64 blockBytes = axisLength * inner * scalarBytes;
				OaMemcpy(output.Data() +
					(out * static_cast<OaU64>(InMapping.TargetShape[axisIndex]) * inner + targetAxisOffset * inner) * scalarBytes,
					input.Data() + out * axisLength * inner * scalarBytes, blockBytes);
				targetAxisOffset += axisLength;
			}
		}
		return output;
	}

	if (InMapping.Transform == OaWeightTransform::Slice) {
		if (sources.Size() != 1 || sources[0]->Shape.Size() != rank ||
			InMapping.Slice.Begin < 0 || InMapping.Slice.Length < 0 ||
			InMapping.Slice.Begin + InMapping.Slice.Length > sources[0]->Shape[axisIndex]) {
			return OaStatus::Error(OaStatusCode::ShapeMismatch, "Slice bounds or rank mismatch");
		}
		for (OaUsize d = 0; d < rank; ++d) {
			const OaI64 expected = d == axisIndex
				? InMapping.Slice.Length : sources[0]->Shape[d];
			if (InMapping.TargetShape[d] != expected)
				return OaStatus::Error(OaStatusCode::ShapeMismatch, "Slice target shape mismatch");
		}
		auto inputResult = ReadConverted(InSource, *sources[0], InMapping.TargetDtype);
		if (inputResult.IsError()) return inputResult.GetStatus();
		const auto& input = inputResult.GetValue();
		const OaU64 sourceAxis = static_cast<OaU64>(sources[0]->Shape[axisIndex]);
		const OaU64 sliceAxis = static_cast<OaU64>(InMapping.Slice.Length);
		const OaU64 begin = static_cast<OaU64>(InMapping.Slice.Begin);
		for (OaU64 out = 0; out < outer; ++out) {
			OaMemcpy(output.Data() + out * sliceAxis * inner * scalarBytes,
				input.Data() + (out * sourceAxis * inner + begin * inner) * scalarBytes,
				sliceAxis * inner * scalarBytes);
		}
		return output;
	}

	return OaStatus::Error(OaStatusCode::Unimplemented, "Unknown weight mapping transform");
}

} // namespace

OaStatus OaRegisterModelWeightAdapter(OaUniquePtr<OaModelWeightAdapter> InAdapter) {
	if (!InAdapter || InAdapter->Name().empty()) {
		return OaStatus::InvalidArgument("Cannot register an empty model weight adapter");
	}
	auto& registry = GetAdapterRegistry();
	std::lock_guard lock(registry.Mutex);
	for (const auto& adapter : registry.Adapters) {
		if (adapter->Name() == InAdapter->Name()) {
			return OaStatus::Error(OaStatusCode::AlreadyExists,
				OaString("Model weight adapter is already registered: ") + InAdapter->Name());
		}
	}
	registry.Adapters.PushBack(OaStdMove(InAdapter));
	return OaStatus::Ok();
}

const OaModelWeightAdapter* OaFindModelWeightAdapter(OaStringView InName) {
	auto& registry = GetAdapterRegistry();
	std::lock_guard lock(registry.Mutex);
	for (const auto& adapter : registry.Adapters) {
		if (adapter->Name() == InName) return adapter.Get();
	}
	return nullptr;
}

OaVec<OaString> OaListModelWeightAdapters() {
	auto& registry = GetAdapterRegistry();
	std::lock_guard lock(registry.Mutex);
	OaVec<OaString> result;
	result.Reserve(registry.Adapters.Size());
	for (const auto& adapter : registry.Adapters) result.PushBack(OaString(adapter->Name()));
	return result;
}

OaStatus OaWeightSource::ReadMatrix(
	OaStringView InName, OaMatrix& OutMatrix, OaScalarType InTargetDtype) const {
	const auto* info = Find(InName);
	if (!info) return OaStatus::NotFound(OaString("Weight not found: ") + InName);
	if (info->Shape.Empty() || info->Shape.Size() > 4) {
		return OaStatus::Error(OaStatusCode::ShapeMismatch, "OaMatrix weight rank must be 1-4");
	}
	OaMatrixShape shape;
	shape.Rank = static_cast<OaI32>(info->Shape.Size());
	for (OaI32 i = 0; i < shape.Rank; ++i) {
		const auto index = static_cast<OaUsize>(i);
		shape.Dims[index] = info->Shape[index];
	}
	OutMatrix = OaFnMatrix::Empty(shape, InTargetDtype);
	const OaU64 byteSize = info->ElementCount * OaScalarSize(InTargetDtype);
	OaVec<OaU8> bytes(byteSize);
	OA_RETURN_IF_ERROR(Read(InName, OaSpan<OaU8>(bytes.Data(), bytes.Size()), InTargetDtype));
	OaMemcpy(OutMatrix.Data(), bytes.Data(), bytes.Size());
	return OaStatus::Ok();
}

OaResult<OaUniquePtr<OaWeightSource>> OaOpenWeightSource(
	const OaPath& InPath, OaWeightFormat InFormat) {
	OaPath path = InPath;
	if (OaFileIo::IsDirectory(path)) path /= "model.safetensors.index.json";
	OaWeightFormat format = InFormat;
	if (format == OaWeightFormat::Auto) {
		const OaString extension = OaFileIo::GetExtension(path);
		if (extension == ".safetensors") format = OaWeightFormat::SafeTensors;
		else if (extension == ".json") format = OaWeightFormat::SafeTensors;
		else if (extension == ".oam") format = OaWeightFormat::Oam;
		else if (extension == ".gguf") format = OaWeightFormat::Gguf;
		else if (extension == ".onnx") format = OaWeightFormat::Onnx;
		else return OaStatus::InvalidArgument(OaString("Cannot infer weight source format: ") + path.String());
	}
	if (format == OaWeightFormat::SafeTensors) {
		if (OaFileIo::GetExtension(path) == ".json") {
			auto package = OaMakeUniquePtr<OaShardedWeightSource>();
			OA_RETURN_IF_ERROR(package->Open(path));
			OaUniquePtr<OaWeightSource> source(OaStdMove(package));
			return OaResult<OaUniquePtr<OaWeightSource>>(OaStdMove(source));
		}
		auto backend = OaMakeUniquePtr<OaSafeTensorsWeightSource>();
		OA_RETURN_IF_ERROR(backend->Open(path));
		OaUniquePtr<OaWeightSource> source(OaStdMove(backend));
		return OaResult<OaUniquePtr<OaWeightSource>>(OaStdMove(source));
	}
	return OaStatus::Error(OaStatusCode::Unimplemented, "Weight source backend is not implemented");
}

OaResult<OaWeightMap> OaMakeRawWeightMap(const OaWeightSource& InSource) {
	OaWeightMap map;
	map.Architecture = "raw";
	map.RequireAllSourceWeights = true;
	for (const auto& info : InSource.List()) {
		OaWeightMapping mapping;
		mapping.Sources.PushBack(info.Name);
		mapping.Target = info.Name;
		mapping.TargetShape = info.Shape;
		mapping.TargetDtype = info.Dtype;
		mapping.Transform = OaWeightTransform::Identity;
		map.Mappings.PushBack(OaStdMove(mapping));
	}
	return map;
}

OaResult<OaWeightTransferReport> OaTransferWeights(
	const OaWeightSource& InSource, const OaWeightMap& InMap, OamModel& OutModel) {
	if (InMap.Architecture.empty() || InMap.Architecture.size() >= sizeof(OutModel.Config.Architecture)) {
		return OaStatus::InvalidArgument("Weight map architecture is empty or too long");
	}
	OaHashSet<OaString> targets;
	OaHashSet<OaString> usedSources;
	OamModel model;
	if (InMap.ArchConfig.Size() > std::numeric_limits<OaU32>::max()) {
		return OaStatus::InvalidArgument("Architecture config exceeds OAM limit");
	}
	model.Config = InMap.Config;
	OaMemset(model.Config.Architecture, 0, sizeof(model.Config.Architecture));
	OaMemcpy(model.Config.Architecture, InMap.Architecture.Data(), InMap.Architecture.Size());
	model.Config.Architecture[InMap.Architecture.size()] = '\0';
	model.Config.ConfigVersion = InMap.ConfigVersion;
	model.Config.ArchConfigSize = static_cast<OaU32>(InMap.ArchConfig.Size());
	model.ArchConfig = InMap.ArchConfig;

	OaWeightTransferReport report;
	report.SourceWeights = InSource.List().Size();
	for (const auto& mapping : InMap.Mappings) {
		if (mapping.Target.empty() || mapping.Target.size() >= OAM_MAX_NAME) {
			return OaStatus::InvalidArgument(OaString("Invalid OAM target weight name: ") + mapping.Target);
		}
		if (!targets.Insert(mapping.Target).second) {
			return OaStatus::InvalidArgument(OaString("Duplicate target weight mapping: ") + mapping.Target);
		}
		for (const auto& source : mapping.Sources) usedSources.Insert(source);
		auto outputResult = ExecuteMapping(InSource, mapping);
		if (outputResult.IsError()) return outputResult.GetStatus();
		auto& output = outputResult.GetValue();
		OaVec<OaU64> shape;
		shape.Reserve(mapping.TargetShape.Size());
		for (const OaI64 dim : mapping.TargetShape) shape.PushBack(static_cast<OaU64>(dim));
		model.AddWeight(mapping.Target.c_str(), mapping.TargetDtype,
			OaSpan<const OaU64>(shape.Data(), shape.Size()), output.Data(), output.Size());
		report.OutputBytes += output.Size();
	}

	for (const auto& info : InSource.List()) {
		if (usedSources.Contains(info.Name)) ++report.UsedSourceWeights;
		else report.UnusedSources.PushBack(info.Name);
	}
	report.OutputWeights = model.WeightIndex.Size();
	if (InMap.RequireAllSourceWeights && !report.UnusedSources.Empty()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			OaString("Weight map left source tensors unused; first: ") + report.UnusedSources[0]);
	}
	OutModel = OaStdMove(model);
	return report;
}
