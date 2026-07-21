#pragma once

// OAM — Native model checkpoint (.oam). Types: OamModel, section layout, on-disk structs.
// Dataset archives (.oad, mmap train/val/test) live in oad.h — keep the two extensions separate.
//
// One file. One model. Everything in it.
// Generic across architectures — REALM-P, transformer, SSM, trading RL.
//
// File layout:
//   [OamFileHeader        ]  64 bytes
//   [OamSectionHeader x N ]  N x 64 bytes
//   [padding to 4096      ]  page-align first section
//   [section data ...     ]  each section at absolute offset

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>

#include <cstring>

// CONSTANTS

constexpr OaU32  OAM_MAGIC         = 0x004D414F;  // "OAM\0"
// v2 integrity-checks the complete file/section metadata through
// FileHeader.Checksum in addition to the per-section payload hashes. These
// non-cryptographic hashes detect accidental corruption; they do not establish
// artifact authenticity. The loader remains compatible with v1 and verifies
// its legacy XOR-of-section-hashes file checksum.
constexpr OaU32  OAM_VERSION       = 2;
constexpr OaU32  OAM_MIN_VERSION   = 1;
constexpr OaUsize OAM_PAGE_SIZE    = 4096;
constexpr OaUsize OAM_MAX_RANK     = 8;
constexpr OaUsize OAM_MAX_NAME     = 128;
constexpr OaUsize OAM_FILE_HDR_SIZE = 64;
constexpr OaUsize OAM_SECT_HDR_SIZE = 64;

// ENUMS

enum class OamCompression : OaU8 {
	None  = 0,
};

enum class OamSectionType : OaU32 {
	Config    = 1,
	Weights   = 2,
	State     = 3,
	Optimizer = 4,
	Progress  = 5,
	Spirv     = 6,
};

// PACKED BINARY STRUCTS

#pragma pack(push, 1)

class OamFileHeader {
public:
	OaU32 Magic        = OAM_MAGIC;
	OaU32 Version      = OAM_VERSION;
	OaU32 NumSections  = 0;
	OaU32 Flags        = 0;
	OaU64 TotalSize    = 0;
	OaU64 Checksum     = 0;
	OaU8  Reserved[32] = {};
};
static_assert(sizeof(OamFileHeader) == OAM_FILE_HDR_SIZE);

class OamSectionHeader {
public:
	OaU32          Type           = 0;
	OamCompression Compression    = OamCompression::None;
	OaU8           Reserved0[3]   = {};
	OaU32          Flags          = 0;
	OaU64          Offset         = 0;
	OaU64          Size           = 0;
	OaU64          CompressedSize = 0;
	OaU64          Checksum       = 0;
	OaU8           Reserved1[20]  = {};
};
static_assert(sizeof(OamSectionHeader) == OAM_SECT_HDR_SIZE);

// Generic config — universal fields that every architecture has.
// Architecture-specific config follows immediately after (ArchConfigSize bytes).
class OamConfig {
public:
	char  Architecture[32] = {};
	OaU32 ConfigVersion    = 1;
	OaU32 Flags            = 0;
	OaU32 DModel           = 0;
	OaU32 NLayers          = 0;
	OaU32 DVocab           = 256;
	OaU32 ArchConfigSize   = 0;
	OaU8  WeightDtype      = static_cast<OaU8>(OaScalarType::Float32);
	OaU8  StateDtype       = static_cast<OaU8>(OaScalarType::Float32);
	OaU8  ComputeDtype     = static_cast<OaU8>(OaScalarType::Float32);
	OaU8  Reserved0        = 0;
	OaU8  Reserved1[48]    = {};
};

class OamTensorEntry {
public:
	char     Name[OAM_MAX_NAME]  = {};
	OaU64    BlobOffset          = 0;
	OaU64    NumBytes            = 0;
	OaScalarType Dtype           = OaScalarType::Float32;
	OaU8     Rank                = 0;
	OaU8     Reserved[6]         = {};
	OaU64    Shape[OAM_MAX_RANK] = {};
};

class OamOptimizerHeader {
public:
	char  Type[16]     = "AdamW";
	OaF32 Lr           = 3e-4f;
	OaF32 Beta1        = 0.9f;
	OaF32 Beta2        = 0.999f;
	OaF32 Eps          = 1e-8f;
	OaF32 WeightDecay  = 0.1f;
	OaI64 Step         = 0;
	OaU64 NumParams    = 0;
	OaU8  Reserved[16] = {};
};

class OamProgress {
public:
	OaU8  Phase         = 0;
	OaU8  Reserved0[3]  = {};
	OaI64 Step          = 0;
	OaU64 BytesSeen     = 0;
	OaU64 EnvSteps      = 0;
	OaF32 Lr            = 3e-4f;
	OaF32 BestMetric    = 0.0f;
	OaU8  LowerIsBetter = 1;
	OaU8  Reserved1[3]  = {};
	char  MetricName[32] = "loss";
	OaU8  Reserved2[32]  = {};
};

class OamSpirvEntry {
public:
	char  Name[64]     = {};
	OaU64 SourceHash   = 0;
	OaU64 BlobOffset   = 0;
	OaU64 NumBytes     = 0;
	OaU8  Reserved[16] = {};
};

#pragma pack(pop)

// OamModel — in-memory representation of a loaded .oam file

class OamModel {
public:
	OaU32                  FormatVersion = OAM_VERSION;
	OamConfig               Config;
	OaVec<OaU8>             ArchConfig;

	OaVec<OamTensorEntry>   WeightIndex;
	OaVec<OaU8>             WeightBlob;

	OaVec<OamTensorEntry>   StateIndex;
	OaVec<OaU8>             StateBlob;

	OamOptimizerHeader      Optimizer = {};
	OaVec<OaF32>            AdamM;
	OaVec<OaF32>            AdamV;
	OaVec<OaF32>            MuonM;  // Muon momentum (Muon-only or MuonAdamW hybrid)
	OaBool                   OptimizerPresent = false;

	OamProgress             Progress = {};

	OaVec<OamSpirvEntry>    SpirvIndex;
	OaVec<OaU8>             SpirvBlob;

	[[nodiscard]] bool HasWeights()    const { return !WeightBlob.Empty(); }
	[[nodiscard]] bool HasState()      const { return !StateBlob.Empty(); }
	[[nodiscard]] bool HasOptimizer()  const {
		return OptimizerPresent || !AdamM.Empty() || !AdamV.Empty() || !MuonM.Empty();
	}
	[[nodiscard]] bool HasSpirvCache() const { return !SpirvBlob.Empty(); }

	[[nodiscard]] const OamTensorEntry* FindWeight(const char* InName) const;
	[[nodiscard]] const void* WeightPtr(const char* InName) const;
	[[nodiscard]] const OamTensorEntry* FindState(const char* InName) const;
	[[nodiscard]] const void* StatePtr(const char* InName) const;
	[[nodiscard]] void* StatePtr(const char* InName);

	[[nodiscard]] const OamSpirvEntry* FindSpirv(const char* InName) const;
	[[nodiscard]] OaSpan<const OaU8> SpirvData(const OamSpirvEntry& InEntry) const;

	void AddWeight(const char* InName, OaScalarType InDtype, OaSpan<const OaU64> InShape, const void* InData, OaU64 InBytes);
	void AddState(const char* InName, OaScalarType InDtype, OaSpan<const OaU64> InShape, const void* InData, OaU64 InBytes);
	void AddSpirv(const char* InName, const OaU8* InData, OaU64 InBytes);

	[[nodiscard]] OaStatus Save(const OaString& InPath) const;
	[[nodiscard]] static OaResult<OamModel> Load(const OaString& InPath);
};

// Optimizer section helpers (Reserved[0..7] = Muon flat element count for MuonAdamW).
[[nodiscard]] inline OaU64 OamGetMuonNumParams(const OamOptimizerHeader& InHdr) {
	OaU64 n = 0;
	std::memcpy(&n, InHdr.Reserved, sizeof(n));
	return n;
}

inline void OamSetMuonNumParams(OamOptimizerHeader& InOutHdr, OaU64 InNum) {
	std::memcpy(InOutHdr.Reserved, &InNum, sizeof(InNum));
}

[[nodiscard]] inline bool OamOptimizerTypeIs(
	const OamOptimizerHeader& InHdr, const char* InType)
{
	return std::strncmp(InHdr.Type, InType, sizeof(InHdr.Type)) == 0;
}

[[nodiscard]] inline bool OamIsKnownOptimizerType(
	const OamOptimizerHeader& InHdr)
{
	return OamOptimizerTypeIs(InHdr, "SGD")
		or OamOptimizerTypeIs(InHdr, "Adam")
		or OamOptimizerTypeIs(InHdr, "AdamW")
		or OamOptimizerTypeIs(InHdr, "Muon")
		or OamOptimizerTypeIs(InHdr, "MuonAdamW");
}

[[nodiscard]] inline bool OamIsMuonAdamWType(const OamOptimizerHeader& InHdr) {
	return OamOptimizerTypeIs(InHdr, "MuonAdamW");
}

[[nodiscard]] inline bool OamIsMuonOnlyType(const OamOptimizerHeader& InHdr) {
	return OamOptimizerTypeIs(InHdr, "Muon");
}

// UTILITIES

[[nodiscard]] constexpr OaUsize OamPageAlign(OaUsize InSize) {
	return (InSize + OAM_PAGE_SIZE - 1) & ~(OAM_PAGE_SIZE - 1);
}

[[nodiscard]] const char* OamSectionTypeName(OamSectionType InType);

// FNV-1a hash used for .oam checksums
[[nodiscard]] OaU64 OamHash(const OaU8* InData, OaUsize InSize);

void OamDump(const OaString& InPath);
