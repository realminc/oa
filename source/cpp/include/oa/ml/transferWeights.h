#pragma once

// TransferWeights — format-neutral external model import.
//
// File containers are private backends. Model translators describe complete,
// checked source-to-OA mappings using this API; callers never depend on a vendor
// container class. The only durable output artifact is .oam.

#include <oa/core/matrix.h>
#include <oa/core/status.h>
#include <oa/ml/modelFile.h>

namespace oa {

class Engine;

enum class WeightFormat : oa::U8 {
	Auto,
	SafeTensors,
	ModelFile,
	Gguf,
	Onnx,
};

struct WeightInfo {
	oa::String name;
	oa::Vector<oa::I64> shape;
	oa::ScalarType dtype = oa::ScalarType::Float32;
	oa::U64 byteSize = 0;
	oa::U64 elementCount = 0;
};

// Immutable named-weight source. Implementations may mmap one file, aggregate a
// sharded manifest, or expose another model container. Returned byte spans remain
// valid while the source remains alive.
class WeightSource {
public:
	virtual ~WeightSource() = default;

	[[nodiscard]] virtual WeightFormat format() const noexcept = 0;
	[[nodiscard]] virtual const oa::Path& path() const noexcept = 0;
	[[nodiscard]] virtual oa::Vector<WeightInfo> list() const = 0;
	[[nodiscard]] virtual const WeightInfo* find(oa::StringView inName) const = 0;
	[[nodiscard]] virtual oa::Result<oa::Span<const oa::U8>> bytes(oa::StringView inName) const = 0;
	[[nodiscard]] virtual oa::HashMap<oa::String, oa::String> metadata() const = 0;
	[[nodiscard]] virtual oa::U64 sourceBytes() const noexcept = 0;

	// Checked copy with optional scalar conversion. Shape transforms belong to a
	// weight mapping, never to the source backend.
	virtual oa::Status read(oa::StringView inName, oa::Span<oa::U8> outData,
		oa::ScalarType inTargetDtype) const = 0;

	[[nodiscard]] oa::Status readMatrix(
		oa::Engine& inEngine, oa::StringView inName, oa::Matrix& outMatrix,
		oa::ScalarType inTargetDtype) const;
};

[[nodiscard]] oa::Result<oa::UniquePtr<WeightSource>> openWeightSource(
	const oa::Path& inPath, WeightFormat inFormat = WeightFormat::Auto);

enum class WeightTransform : oa::U8 {
	Identity,
	Transpose2D,
	Concat,
	Slice,
};

struct WeightSlice {
	oa::I32 axis = 0;
	oa::I64 begin = 0;
	oa::I64 length = 0;
};

struct WeightMapping {
	// Concat accepts multiple sources in order. Other transforms require one.
	oa::Vector<oa::String> sources;
	oa::String target;
	oa::Vector<oa::I64> targetShape;
	oa::ScalarType targetDtype = oa::ScalarType::Float32;
	WeightTransform transform = WeightTransform::Identity;
	oa::I32 concatAxis = 0;
	WeightSlice slice;
};

struct WeightMap {
	oa::String architecture;
	oa::U32 configVersion = 1;
	ModelFileConfig config;
	oa::Vector<oa::U8> archConfig;
	oa::Vector<WeightMapping> mappings;
	bool requireAllSourceWeights = true;
};

struct WeightTransferReport {
	oa::U64 sourceWeights = 0;
	oa::U64 usedSourceWeights = 0;
	oa::U64 outputWeights = 0;
	oa::U64 outputBytes = 0;
	oa::Vector<oa::String> unusedSources;
};

// Model translators are small declarative mapping layers. They inspect the
// opened source/config assets and emit a complete map; transfer mechanics remain
// shared and independently tested.
class ModelTranslator {
public:
	virtual ~ModelTranslator() = default;
	[[nodiscard]] virtual oa::StringView name() const noexcept = 0;
	[[nodiscard]] virtual oa::Result<WeightMap> buildMap(
		const WeightSource& inSource) const = 0;
};

// Process-wide translator registry used by modelctl and applications. Extension
// libraries register model translators without adding model-specific logic
// to the transfer engine.
oa::Status registerModelTranslator(oa::UniquePtr<ModelTranslator> inTranslator);
[[nodiscard]] const ModelTranslator* findModelTranslator(oa::StringView inName);
[[nodiscard]] oa::Vector<oa::String> listModelTranslators();

[[nodiscard]] oa::Result<WeightTransferReport> transferWeights(
	const WeightSource& inSource, const WeightMap& inMap,
	ModelFile& outModel);

// Exact name-preserving file plan, useful for container round-trip tests and
// inspection. It is not a model translator and does not prove model support.
[[nodiscard]] oa::Result<WeightMap> makeRawWeightMap(
	const WeightSource& inSource);

} // namespace oa
