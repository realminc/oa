#pragma once

// TransferWeights — format-neutral external model weight import.
//
// File containers are private backends. Architecture adapters describe complete,
// checked source-to-OA mappings using this API; callers never depend on a vendor
// container class. The only durable output artifact is .oam.

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Ml/Oam.h>

enum class OaWeightFormat : OaU8 {
	Auto,
	SafeTensors,
	Oam,
	Gguf,
	Onnx,
};

struct OaWeightInfo {
	OaString Name;
	OaVec<OaI64> Shape;
	OaScalarType Dtype = OaScalarType::Float32;
	OaU64 ByteSize = 0;
	OaU64 ElementCount = 0;
};

// Immutable named-weight source. Implementations may mmap one file, aggregate a
// sharded manifest, or expose another model container. Returned byte spans remain
// valid while the source remains alive.
class OaWeightSource {
public:
	virtual ~OaWeightSource() = default;

	[[nodiscard]] virtual OaWeightFormat Format() const noexcept = 0;
	[[nodiscard]] virtual const OaPath& Path() const noexcept = 0;
	[[nodiscard]] virtual OaVec<OaWeightInfo> List() const = 0;
	[[nodiscard]] virtual const OaWeightInfo* Find(OaStringView InName) const = 0;
	[[nodiscard]] virtual OaResult<OaSpan<const OaU8>> Bytes(OaStringView InName) const = 0;
	[[nodiscard]] virtual OaHashMap<OaString, OaString> Metadata() const = 0;
	[[nodiscard]] virtual OaU64 SourceBytes() const noexcept = 0;

	// Checked copy with optional scalar conversion. Shape transforms belong to a
	// weight mapping, never to the source backend.
	virtual OaStatus Read(OaStringView InName, OaSpan<OaU8> OutData,
		OaScalarType InTargetDtype) const = 0;

	[[nodiscard]] OaStatus ReadMatrix(OaStringView InName, OaMatrix& OutMatrix,
		OaScalarType InTargetDtype) const;
};

[[nodiscard]] OaResult<OaUniquePtr<OaWeightSource>> OaOpenWeightSource(
	const OaPath& InPath, OaWeightFormat InFormat = OaWeightFormat::Auto);

enum class OaWeightTransform : OaU8 {
	Identity,
	Transpose2D,
	Concat,
	Slice,
};

struct OaWeightSlice {
	OaI32 Axis = 0;
	OaI64 Begin = 0;
	OaI64 Length = 0;
};

struct OaWeightMapping {
	// Concat accepts multiple sources in order. Other transforms require one.
	OaVec<OaString> Sources;
	OaString Target;
	OaVec<OaI64> TargetShape;
	OaScalarType TargetDtype = OaScalarType::Float32;
	OaWeightTransform Transform = OaWeightTransform::Identity;
	OaI32 ConcatAxis = 0;
	OaWeightSlice Slice;
};

struct OaWeightMap {
	OaString Architecture;
	OaU32 ConfigVersion = 1;
	OamConfig Config;
	OaVec<OaU8> ArchConfig;
	OaVec<OaWeightMapping> Mappings;
	bool RequireAllSourceWeights = true;
};

struct OaWeightTransferReport {
	OaU64 SourceWeights = 0;
	OaU64 UsedSourceWeights = 0;
	OaU64 OutputWeights = 0;
	OaU64 OutputBytes = 0;
	OaVec<OaString> UnusedSources;
};

// Architecture adapters are small declarative mapping layers. They inspect the
// opened source/config assets and emit a complete map; transfer mechanics remain
// shared and independently tested.
class OaModelWeightAdapter {
public:
	virtual ~OaModelWeightAdapter() = default;
	[[nodiscard]] virtual OaStringView Name() const noexcept = 0;
	[[nodiscard]] virtual OaResult<OaWeightMap> BuildMap(
		const OaWeightSource& InSource) const = 0;
};

// Process-wide adapter registry used by modelctl and applications. Extension
// libraries register architecture adapters without adding model-specific logic
// to the transfer engine.
OaStatus OaRegisterModelWeightAdapter(OaUniquePtr<OaModelWeightAdapter> InAdapter);
[[nodiscard]] const OaModelWeightAdapter* OaFindModelWeightAdapter(OaStringView InName);
[[nodiscard]] OaVec<OaString> OaListModelWeightAdapters();

[[nodiscard]] OaResult<OaWeightTransferReport> OaTransferWeights(
	const OaWeightSource& InSource, const OaWeightMap& InMap,
	OamModel& OutModel);

// Exact name-preserving archive plan, useful for container round-trip tests and
// inspection. It is not an architecture adapter and does not prove model support.
[[nodiscard]] OaResult<OaWeightMap> OaMakeRawWeightMap(
	const OaWeightSource& InSource);
