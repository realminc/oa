#pragma once

// Native model file (.oam). The extension is a wire-format identity; the
// C++ API describes the semantic artifact rather than repeating that acronym.
//
// One file. One model. Everything in it.
// Generic across architectures — REALM-P, transformer, SSM, trading RL.
//
// file layout:
//   [ModelFileHeader        ]  64 bytes
//   [ModelFileSectionHeader x N ]  N x 64 bytes
//   [padding to 4096      ]  page-align first section
//   [section data ...     ]  each section at absolute offset

#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/ml/quantMatrix.h>

#include <cstring>

namespace oa {

class Engine;

// CONSTANTS

constexpr oa::U32 kModelFileMagic = 0x004D414F; // "OAM\0"
// v2 integrity-checks the complete file/section metadata through
// FileHeader.checksum in addition to the per-section payload hashes. These
// non-cryptographic hashes detect accidental corruption; they do not establish
// artifact authenticity. v3 gives every tensor an explicit dense/Q4/Q8
// encoding without changing ModelTensorEntry's binary size. The loader remains
// compatible with v1/v2; their zeroed encoding bytes mean Dense.
constexpr oa::U32 kModelFileVersion = 3;
constexpr oa::U32 kModelFileMinVersion = 1;
constexpr oa::Usize kModelFilePageSize = 4096;
constexpr oa::Usize kModelFileMaxRank = 8;
constexpr oa::Usize kModelFileMaxName = 128;
constexpr oa::Usize kModelFileHeaderSize = 64;
constexpr oa::Usize kModelFileSectionHeaderSize = 64;

// ENUMS

enum class ModelFileCompression : oa::U8 {
	None = 0,
};

enum class ModelFileSection : oa::U32 {
	Config = 1,
	Weights = 2,
	State = 3,
	Optimizer = 4,
	Progress = 5,
	// Reserved for reading files written before kernel authority moved into
	// the engine-owned manifest. New files never write this section.
	LegacyKernelCache = 6,
};

enum class ModelTensorEncoding : oa::U8 {
	Dense = 0,
	Q4 = 1,
	Q8 = 2,
};

// PACKED BINARY STRUCTS

#pragma pack(push, 1)

class ModelFileHeader {
public:
	oa::U32 magic = kModelFileMagic;
	oa::U32 version = kModelFileVersion;
	oa::U32 numSections = 0;
	oa::U32 flags = 0;
	oa::U64 totalSize = 0;
	oa::U64 checksum = 0;
	oa::U8 reserved[32] = {};
};
static_assert(sizeof(ModelFileHeader) == kModelFileHeaderSize);

class ModelFileSectionHeader {
public:
	oa::U32 type = 0;
	ModelFileCompression compression = ModelFileCompression::None;
	oa::U8 reserved0[3] = {};
	oa::U32 flags = 0;
	oa::U64 offset = 0;
	oa::U64 size = 0;
	oa::U64 compressedSize = 0;
	oa::U64 checksum = 0;
	oa::U8 reserved1[20] = {};
};
static_assert(sizeof(ModelFileSectionHeader) == kModelFileSectionHeaderSize);

// Generic config — universal fields that every architecture has.
// architecture-specific config follows immediately after (archConfigSize
// bytes).
class ModelFileConfig {
public:
	char architecture[32] = {};
	oa::U32 configVersion = 1;
	oa::U32 flags = 0;
	oa::U32 dModel = 0;
	oa::U32 nLayers = 0;
	oa::U32 dVocab = 256;
	oa::U32 archConfigSize = 0;
	oa::U8 weightDtype = static_cast<oa::U8>(oa::ScalarType::Float32);
	oa::U8 stateDtype = static_cast<oa::U8>(oa::ScalarType::Float32);
	oa::U8 computeDtype = static_cast<oa::U8>(oa::ScalarType::Float32);
	oa::U8 reserved0 = 0;
	oa::U8 reserved1[48] = {};
};

class ModelTensorEntry {
public:
	char name[kModelFileMaxName] = {};
	oa::U64 blobOffset = 0;
	oa::U64 numBytes = 0;
	oa::ScalarType dtype = oa::ScalarType::Float32;
	oa::U8 rank = 0;
	ModelTensorEncoding encoding = ModelTensorEncoding::Dense;
	oa::U8 blockSize = 0;
	oa::U8 reserved[4] = {};
	oa::U64 shape[kModelFileMaxRank] = {};
};
static_assert(sizeof(ModelTensorEntry) == 216);

class ModelOptimizerState {
public:
	char type[16] = "AdamW";
	oa::F32 lr = 3e-4f;
	oa::F32 beta1 = 0.9f;
	oa::F32 beta2 = 0.999f;
	oa::F32 eps = 1e-8f;
	oa::F32 weightDecay = 0.1f;
	oa::I64 step = 0;
	oa::U64 numParams = 0;
	oa::U8 reserved[16] = {};
};

class ModelTrainingProgress {
public:
	oa::U8 phase = 0;
	oa::U8 reserved0[3] = {};
	oa::I64 step = 0;
	oa::U64 bytesSeen = 0;
	oa::U64 envSteps = 0;
	oa::F32 lr = 3e-4f;
	oa::F32 bestMetric = 0.0f;
	oa::U8 lowerIsBetter = 1;
	oa::U8 reserved1[3] = {};
	char metricName[32] = "loss";
	oa::U8 reserved2[32] = {};
};

#pragma pack(pop)

/// A complete OA model file loaded in memory.
///
/// `ModelFile` owns model configuration, named weights, persistent state,
/// optimizer state, and training progress. `load()` and `save()` implement the
/// native `.oam` wire format. External formats are imported by
/// `oa::ModelTranslator`; they are not alternative representations of this
/// type.

class ModelFile {
public:
	oa::U32 formatVersion = kModelFileVersion;
	ModelFileConfig config;
	oa::Vec<oa::U8> archConfig;

	oa::Vec<ModelTensorEntry> weightIndex;
	oa::Vec<oa::U8> weightBlob;

	oa::Vec<ModelTensorEntry> stateIndex;
	oa::Vec<oa::U8> stateBlob;

	ModelOptimizerState optimizer = {};
	oa::Vec<oa::F32> adamM;
	oa::Vec<oa::F32> adamV;
	oa::Bool optimizerPresent = false;

	ModelTrainingProgress progress = {};

	[[nodiscard]] bool hasWeights() const { return !weightBlob.empty(); }
	[[nodiscard]] bool hasState() const { return !stateBlob.empty(); }
	[[nodiscard]] bool hasOptimizer() const {
		return optimizerPresent || !adamM.empty() || !adamV.empty();
	}
	[[nodiscard]] const ModelTensorEntry* findWeight(const char* inName) const;
	[[nodiscard]] const void* weightPtr(const char* inName) const;
	[[nodiscard]] const ModelTensorEntry* findState(const char* inName) const;
	[[nodiscard]] const void* statePtr(const char* inName) const;
	[[nodiscard]] void* statePtr(const char* inName);

	void addWeight(const char* inName, oa::ScalarType inDtype, oa::Span<const oa::U64> inShape,
				   const void* inData, oa::U64 inBytes);
	void addState(const char* inName, oa::ScalarType inDtype, oa::Span<const oa::U64> inShape,
				  const void* inData, oa::U64 inBytes);
	// Produce a weight-only inference artifact. Dense Float32 tensors with rank
	// >= 2 become native OA Q4/Q8 blocks; scalar/vector weights and all state
	// remain dense. Optimizer state is deliberately removed.
	[[nodiscard]] oa::Result<ModelFile> quantizeWeights(oa::Quantization inQuantization) const;

	// upload one encoded weight as the same semantic value consumed by the
	// fused vkDNN MatMulNt path. Dense entries fail closed.
	[[nodiscard]] oa::Result<oa::QuantMatrix> loadQuantMatrix(oa::Engine& inEngine,
															  const char* inName) const;

	[[nodiscard]] oa::Status save(const oa::String& inPath) const;
	[[nodiscard]] static oa::Result<ModelFile> load(const oa::String& inPath);
};

[[nodiscard]] inline bool modelFileOptimizerTypeIs(const ModelOptimizerState& inHeader,
												   const char* inType) {
	return std::strncmp(inHeader.type, inType, sizeof(inHeader.type)) == 0;
}

[[nodiscard]] inline bool modelFileHasKnownOptimizer(const ModelOptimizerState& inHeader) {
	return modelFileOptimizerTypeIs(inHeader, "SGD") or
		   modelFileOptimizerTypeIs(inHeader, "Adam") or
		   modelFileOptimizerTypeIs(inHeader, "AdamW") or
		   modelFileOptimizerTypeIs(inHeader, "Muon");
}

[[nodiscard]] inline bool modelFileIsMuonOnly(const ModelOptimizerState& inHeader) {
	return modelFileOptimizerTypeIs(inHeader, "Muon");
}

// UTILITIES

[[nodiscard]] constexpr oa::Usize modelFilePageAlign(oa::Usize inSize) {
	return (inSize + kModelFilePageSize - 1) & ~(kModelFilePageSize - 1);
}

[[nodiscard]] const char* modelFileSectionName(ModelFileSection inType);
[[nodiscard]] const char* modelFileTensorEncodingName(ModelTensorEncoding inEncoding);

// FNV-1a hash used for .oam checksums
[[nodiscard]] oa::U64 modelFileHash(const oa::U8* inData, oa::Usize inSize);

void dumpModelFile(const oa::String& inPath);

} // namespace oa
