#include <Oa/Ml/Oam.h>
#include <Oa/Core/Log.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

// FNV-1a hash (no external deps — replaces xxhash)
OaU64 OamHash(const OaU8* InData, OaUsize InSize) {
	OaU64 hash = 0xcbf29ce484222325ULL;
	for (OaUsize i = 0; i < InSize; ++i) {
		hash ^= InData[i];
		hash *= 0x100000001b3ULL;
	}
	return hash;
}

namespace {

constexpr OaU32 kOamMaxSections = 32;
constexpr OaUsize kHashChunkBytes = 1024 * 1024;

OaStatus OamCorrupt(OaString InMessage) {
	return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
		"corrupt .oam: " + InMessage);
}

void OamHashUpdate(OaU64& InOutHash, const OaU8* InData, OaUsize InSize) {
	for (OaUsize i = 0; i < InSize; ++i) {
		InOutHash ^= InData[i];
		InOutHash *= 0x100000001b3ULL;
	}
}

OaU64 OamManifestHash(
	const OamFileHeader& InHeader,
	const OaVec<OamSectionHeader>& InSections)
{
	OamFileHeader normalized = InHeader;
	normalized.Checksum = 0;
	OaU64 hash = 0xcbf29ce484222325ULL;
	OamHashUpdate(hash, reinterpret_cast<const OaU8*>(&normalized),
		sizeof(normalized));
	if (not InSections.Empty()) {
		OamHashUpdate(hash,
			reinterpret_cast<const OaU8*>(InSections.Data()),
			InSections.Size() * sizeof(OamSectionHeader));
	}
	return hash;
}

bool OamCheckedAdd(OaU64 InA, OaU64 InB, OaU64& Out) {
	if (InB > std::numeric_limits<OaU64>::max() - InA) return false;
	Out = InA + InB;
	return true;
}

bool OamCheckedMul(OaU64 InA, OaU64 InB, OaU64& Out) {
	if (InA != 0 and InB > std::numeric_limits<OaU64>::max() / InA) return false;
	Out = InA * InB;
	return true;
}

bool OamReadExact(
	std::ifstream& InFile,
	OaU64 InOffset,
	void* OutData,
	OaU64 InBytes)
{
	if (InOffset > static_cast<OaU64>(std::numeric_limits<std::streamoff>::max())
		or InBytes > static_cast<OaU64>(std::numeric_limits<std::streamsize>::max()))
	{
		return false;
	}
	InFile.clear();
	InFile.seekg(static_cast<std::streamoff>(InOffset), std::ios::beg);
	if (not InFile) return false;
	if (InBytes == 0) return true;
	InFile.read(static_cast<char*>(OutData), static_cast<std::streamsize>(InBytes));
	return InFile.good()
		or (InFile.eof() and static_cast<OaU64>(InFile.gcount()) == InBytes);
}

OaResult<OaU64> OamHashRange(
	std::ifstream& InFile,
	OaU64 InOffset,
	OaU64 InBytes)
{
	OaVec<OaU8> chunk(std::min<OaU64>(InBytes, kHashChunkBytes));
	OaU64 hash = 0xcbf29ce484222325ULL;
	OaU64 consumed = 0;
	while (consumed < InBytes) {
		const OaU64 bytes = std::min<OaU64>(chunk.Size(), InBytes - consumed);
		if (not OamReadExact(InFile, InOffset + consumed, chunk.Data(), bytes)) {
			return OamCorrupt("truncated section payload");
		}
		OamHashUpdate(hash, chunk.Data(), static_cast<OaUsize>(bytes));
		consumed += bytes;
	}
	return hash;
}

OaString OamTemporaryPath(const OaString& InFinalPath) {
	static std::atomic<OaU64> sequence{0};
	const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
	return InFinalPath + ".tmp." +
		std::to_string(static_cast<unsigned long long>(ticks)) + "." +
		std::to_string(static_cast<unsigned long long>(++sequence));
}

OaStatus OamAtomicReplace(
	const OaString& InTemporaryPath,
	const OaString& InFinalPath)
{
#ifdef _WIN32
	const std::filesystem::path temporary(InTemporaryPath.StdStr());
	const std::filesystem::path final(InFinalPath.StdStr());
	HANDLE handle = CreateFileW(temporary.c_str(), GENERIC_READ,
		FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (handle == INVALID_HANDLE_VALUE) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"cannot open temporary .oam for durable commit");
	}
	const BOOL flushed = FlushFileBuffers(handle);
	CloseHandle(handle);
	if (not flushed or not MoveFileExW(temporary.c_str(), final.c_str(),
		MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
	{
		return OaStatus::Error(OaStatusCode::Unavailable,
			"atomic .oam replacement failed");
	}
#else
	const int fileFd = ::open(InTemporaryPath.CStr(), O_RDONLY | O_CLOEXEC);
	if (fileFd < 0) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"cannot open temporary .oam for durable commit");
	}
	const int syncResult = ::fsync(fileFd);
	::close(fileFd);
	if (syncResult != 0 or ::rename(InTemporaryPath.CStr(), InFinalPath.CStr()) != 0) {
		return OaStatus::Error(OaStatusCode::Unavailable,
			"atomic .oam replacement failed");
	}
	const auto parent = std::filesystem::path(InFinalPath.StdStr()).parent_path();
	const OaString parentString(parent.empty() ? "." : parent.string());
	const int dirFd = ::open(parentString.CStr(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
	if (dirFd >= 0) {
		(void)::fsync(dirFd);
		::close(dirFd);
	}
#endif
	return OaStatus::Ok();
}

bool OamHasTerminator(const char* InData, OaUsize InSize) {
	return std::memchr(InData, '\0', InSize) != nullptr;
}

bool OamValidScalarType(OaScalarType InType) {
	return static_cast<OaU8>(InType)
		<= static_cast<OaU8>(OaScalarType::Q8_K);
}

} // namespace

// Name lookups
const char* OamSectionTypeName(OamSectionType InType) {
	switch (InType) {
		case OamSectionType::Config:    return "Config";
		case OamSectionType::Weights:   return "Weights";
		case OamSectionType::State:     return "State";
		case OamSectionType::Optimizer: return "Optimizer";
		case OamSectionType::Progress:  return "Progress";
		case OamSectionType::Spirv:     return "Kernels";
		default:                        return "Unknown";
	}
}


// OamModel methods
const OamTensorEntry* OamModel::FindWeight(const char* InName) const {
	for (const auto& e : WeightIndex) {
		if (std::strncmp(e.Name, InName, OAM_MAX_NAME) == 0) {
			return &e;
		}
	}
	return nullptr;
}

const void* OamModel::WeightPtr(const char* InName) const {
	const auto* entry = FindWeight(InName);
	if (!entry || WeightBlob.Empty()) {
		return nullptr;
	}
	return WeightBlob.Data() + entry->BlobOffset;
}

const OamTensorEntry* OamModel::FindState(const char* InName) const {
	for (const auto& e : StateIndex) {
		if (std::strncmp(e.Name, InName, OAM_MAX_NAME) == 0) {
			return &e;
		}
	}
	return nullptr;
}

const void* OamModel::StatePtr(const char* InName) const {
	const auto* entry = FindState(InName);
	if (!entry || StateBlob.Empty()) {
		return nullptr;
	}
	return StateBlob.Data() + entry->BlobOffset;
}

void* OamModel::StatePtr(const char* InName) {
	return const_cast<void*>(static_cast<const OamModel&>(*this).StatePtr(InName));
}

static void AddTensor(
	OaVec<OamTensorEntry>& OutIndex,
	OaVec<OaU8>& OutBlob,
	const char* InName,
	OaScalarType InDtype,
	OaSpan<const OaU64> InShape,
	const void* InData,
	OaU64 InBytes
) {
	assert(InShape.Size() <= OAM_MAX_RANK);
	assert(InData != nullptr || InBytes == 0);
	OamTensorEntry entry;
	std::strncpy(entry.Name, InName, OAM_MAX_NAME - 1);
	entry.BlobOffset = OutBlob.Size();
	entry.NumBytes = InBytes;
	entry.Dtype = InDtype;
	entry.Rank = static_cast<OaU8>(InShape.size());
	for (OaUsize i = 0; i < InShape.size() && i < OAM_MAX_RANK; ++i)
		entry.Shape[i] = InShape[i];
	OutIndex.PushBack(entry);

	OaUsize oldSize = OutBlob.Size();
	OutBlob.Resize(oldSize + InBytes);
	if (InBytes > 0) {
		std::memcpy(OutBlob.Data() + oldSize, InData, InBytes);
	}
}

void OamModel::AddWeight(
	const char* InName, OaScalarType InDtype,
	OaSpan<const OaU64> InShape, const void* InData, OaU64 InBytes
) {
	AddTensor(WeightIndex, WeightBlob, InName, InDtype, InShape, InData, InBytes);
}

void OamModel::AddState(
	const char* InName, OaScalarType InDtype,
	OaSpan<const OaU64> InShape, const void* InData, OaU64 InBytes
) {
	AddTensor(StateIndex, StateBlob, InName, InDtype, InShape, InData, InBytes);
}

const OamSpirvEntry* OamModel::FindSpirv(const char* InName) const {
	for (const auto& e : SpirvIndex)
		if (std::strncmp(e.Name, InName, sizeof(e.Name)) == 0) return &e;
	return nullptr;
}

OaSpan<const OaU8> OamModel::SpirvData(const OamSpirvEntry& InEntry) const {
	if (InEntry.BlobOffset + InEntry.NumBytes > SpirvBlob.Size()) return {};
	return {SpirvBlob.Data() + InEntry.BlobOffset, static_cast<OaUsize>(InEntry.NumBytes)};
}

void OamModel::AddSpirv(const char* InName, const OaU8* InData, OaU64 InBytes) {
	OamSpirvEntry entry;
	std::strncpy(entry.Name, InName, sizeof(entry.Name) - 1);
	entry.BlobOffset = SpirvBlob.Size();
	entry.NumBytes = InBytes;
	entry.SourceHash = OamHash(InData, static_cast<OaUsize>(InBytes));
	SpirvIndex.PushBack(entry);

	OaUsize oldSize = SpirvBlob.Size();
	SpirvBlob.Resize(oldSize + InBytes);
	std::memcpy(SpirvBlob.Data() + oldSize, InData, InBytes);
}

// Tensor index serialization

static OaVec<OaU8> SerializeTensorIndex(
	const OaVec<OamTensorEntry>& InIndex, const OaVec<OaU8>& InBlob
) {
	OaU32 count = static_cast<OaU32>(InIndex.Size());
	OaU32 reserved = 0;
	OaUsize indexBytes = sizeof(OaU32) * 2 + sizeof(OamTensorEntry) * count;

	OaVec<OaU8> raw(indexBytes + InBlob.Size());
	OaU8* p = raw.Data();
	std::memcpy(p, &count, sizeof(OaU32)); p += sizeof(OaU32);
	std::memcpy(p, &reserved, sizeof(OaU32)); p += sizeof(OaU32);
	for (const auto& e : InIndex) {
		std::memcpy(p, &e, sizeof(OamTensorEntry));
		p += sizeof(OamTensorEntry);
	}
	std::memcpy(p, InBlob.Data(), InBlob.Size());
	return raw;
}

// Save

OaStatus OamModel::Save(const OaString& InPath) const {
	{
		auto parent = std::filesystem::path(InPath.StdStr()).parent_path();
		if (!parent.empty()) {
			std::error_code ec;
			std::filesystem::create_directories(parent, ec);
		}
	}

	struct Payload {
		OamSectionType Type;
		OaVec<OaU8>   Data;
	};
	OaVec<Payload> payloads;

	// Config + optional ArchConfig
	{
		OaVec<OaU8> raw(sizeof(OamConfig) + ArchConfig.Size());
		std::memcpy(raw.Data(), &Config, sizeof(OamConfig));
		if (!ArchConfig.Empty())
			std::memcpy(raw.Data() + sizeof(OamConfig), ArchConfig.Data(), ArchConfig.Size());
		payloads.PushBack({OamSectionType::Config, std::move(raw)});
	}

	if (HasWeights()) {
		payloads.PushBack({OamSectionType::Weights,
			SerializeTensorIndex(WeightIndex, WeightBlob)});
	}

	if (HasState()) {
		payloads.PushBack({OamSectionType::State,
			SerializeTensorIndex(StateIndex, StateBlob)});
	}

	if (HasOptimizer()) {
		if (not OamIsKnownOptimizerType(Optimizer)) {
			return OaStatus::Error(OaStatusCode::InvalidArgument,
				"cannot save .oam with an unknown optimizer type");
		}
		OamOptimizerHeader hdr = Optimizer;
		hdr.NumParams = AdamM.Size();
		if (OamIsMuonAdamWType(hdr)) {
			OamSetMuonNumParams(hdr, MuonM.Size());
		}
		const OaUsize adamMBytes = AdamM.Size() * sizeof(OaF32);
		const OaUsize adamVBytes = AdamV.Size() * sizeof(OaF32);
		const OaUsize muonMBytes = MuonM.Size() * sizeof(OaF32);
		OaVec<OaU8> raw(sizeof(OamOptimizerHeader) + adamMBytes + adamVBytes + muonMBytes);
		OaU8* p = raw.Data();
		std::memcpy(p, &hdr, sizeof(OamOptimizerHeader)); p += sizeof(OamOptimizerHeader);
		if (adamMBytes > 0) {
			std::memcpy(p, AdamM.Data(), adamMBytes); p += adamMBytes;
		}
		if (adamVBytes > 0) {
			std::memcpy(p, AdamV.Data(), adamVBytes); p += adamVBytes;
		}
		if (muonMBytes > 0) {
			std::memcpy(p, MuonM.Data(), muonMBytes);
		}
		payloads.PushBack({OamSectionType::Optimizer, std::move(raw)});
	}

	{
		OaVec<OaU8> raw(sizeof(OamProgress));
		std::memcpy(raw.Data(), &Progress, sizeof(OamProgress));
		payloads.PushBack({OamSectionType::Progress, std::move(raw)});
	}

	if (HasSpirvCache()) {
		OaU32 count = static_cast<OaU32>(SpirvIndex.Size());
		OaU32 reserved = 0;
		OaUsize indexBytes = sizeof(OaU32) * 2 + sizeof(OamSpirvEntry) * count;
		OaVec<OaU8> raw(indexBytes + SpirvBlob.Size());
		OaU8* p = raw.Data();
		std::memcpy(p, &count, sizeof(OaU32)); p += sizeof(OaU32);
		std::memcpy(p, &reserved, sizeof(OaU32)); p += sizeof(OaU32);
		for (const auto& e : SpirvIndex) {
			std::memcpy(p, &e, sizeof(OamSpirvEntry));
			p += sizeof(OamSpirvEntry);
		}
		std::memcpy(p, SpirvBlob.Data(), SpirvBlob.Size());
		payloads.PushBack({OamSectionType::Spirv, std::move(raw)});
	}

	OaU32 numSections = static_cast<OaU32>(payloads.Size());
	OaUsize headerBytes = OAM_FILE_HDR_SIZE + numSections * OAM_SECT_HDR_SIZE;
	OaUsize dataStart = OamPageAlign(headerBytes);

	OaVec<OamSectionHeader> sectionHeaders(numSections);
	OaU64 offset = dataStart;

	for (OaU32 i = 0; i < numSections; ++i) {
		auto& sh = sectionHeaders[i];
		auto& sp = payloads[i];
		sh.Type = static_cast<OaU32>(sp.Type);
		sh.Compression = OamCompression::None;
		sh.Offset = offset;
		sh.Size = sp.Data.Size();
		sh.CompressedSize = 0;
		sh.Checksum = OamHash(sp.Data.Data(), sp.Data.Size());

		bool needsAlign = (sp.Type == OamSectionType::Weights || sp.Type == OamSectionType::State);
		offset += needsAlign ? OamPageAlign(sp.Data.Size()) : sp.Data.Size();
	}

	OamFileHeader fileHeader;
	fileHeader.NumSections = numSections;
	fileHeader.TotalSize = offset;
	fileHeader.Checksum = OamManifestHash(fileHeader, sectionHeaders);

	const OaString temporaryPath = OamTemporaryPath(InPath);
	std::ofstream file(temporaryPath.c_str(), std::ios::binary | std::ios::trunc);
	if (!file) return OaStatus::Error("Failed to open for write: " + temporaryPath);

	file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
	for (const auto& sh : sectionHeaders)
		file.write(reinterpret_cast<const char*>(&sh), sizeof(sh));

	{
		OaUsize written = OAM_FILE_HDR_SIZE + numSections * OAM_SECT_HDR_SIZE;
		OaVec<OaU8> zeros(dataStart - written, 0);
		file.write(reinterpret_cast<const char*>(zeros.Data()), zeros.Size());
	}

	for (OaU32 i = 0; i < numSections; ++i) {
		const auto& sp = payloads[i];
		file.write(reinterpret_cast<const char*>(sp.Data.Data()), sp.Data.Size());
		bool needsAlign = (sp.Type == OamSectionType::Weights || sp.Type == OamSectionType::State);
		if (needsAlign) {
			OaUsize pad = OamPageAlign(sp.Data.Size()) - sp.Data.Size();
			if (pad > 0) {
				OaVec<OaU8> zeros(pad, 0);
				file.write(reinterpret_cast<const char*>(zeros.Data()), pad);
			}
		}
	}
	file.flush();
	if (not file.good()) {
		file.close();
		std::error_code ignored;
		std::filesystem::remove(temporaryPath.StdStr(), ignored);
		return OaStatus::Error(OaStatusCode::DiskFull,
			"failed to write complete .oam: " + InPath);
	}
	file.close();
	if (file.fail()) {
		std::error_code ignored;
		std::filesystem::remove(temporaryPath.StdStr(), ignored);
		return OaStatus::Error(OaStatusCode::Unavailable,
			"failed to close .oam temporary file: " + InPath);
	}
	const auto commitStatus = OamAtomicReplace(temporaryPath, InPath);
	if (not commitStatus.IsOk()) {
		std::error_code ignored;
		std::filesystem::remove(temporaryPath.StdStr(), ignored);
		return commitStatus;
	}

	OA_LOG_INFO(OaLogComponent::ML, "[oam] Saved %s | %zu weights | %zu kernels | %.1f MB",
		InPath.c_str(), WeightIndex.Size(), SpirvIndex.Size(), fileHeader.TotalSize / 1e6);
	return OaStatus::Ok();
}

// Load

OaResult<OamModel> OamModel::Load(const OaString& InPath) {
	std::ifstream file(InPath.c_str(), std::ios::binary);
	if (!file) return OaStatus::Error("Cannot open: " + InPath);

	file.seekg(0, std::ios::end);
	const auto end = file.tellg();
	if (end < 0) return OamCorrupt("cannot determine file size");
	const OaU64 fileSize = static_cast<OaU64>(end);
	if (fileSize < sizeof(OamFileHeader)) return OamCorrupt("truncated file header");

	OamFileHeader fh;
	if (not OamReadExact(file, 0, &fh, sizeof(fh))) {
		return OamCorrupt("failed to read file header");
	}
	if (fh.Magic != OAM_MAGIC) return OamCorrupt("invalid magic");
	if (fh.Version < OAM_MIN_VERSION or fh.Version > OAM_VERSION) {
		return OamCorrupt("unsupported format version");
	}
	if (fh.NumSections == 0 or fh.NumSections > kOamMaxSections) {
		return OamCorrupt("invalid section count");
	}
	OaU64 sectionTableBytes = 0;
	if (not OamCheckedMul(fh.NumSections, sizeof(OamSectionHeader), sectionTableBytes)) {
		return OamCorrupt("section table size overflow");
	}
	OaU64 headerBytes = 0;
	if (not OamCheckedAdd(sizeof(OamFileHeader), sectionTableBytes, headerBytes)
		or headerBytes > fileSize)
	{
		return OamCorrupt("truncated section table");
	}
	if (fh.TotalSize != fileSize) return OamCorrupt("file size does not match header");

	OaVec<OamSectionHeader> sections(fh.NumSections);
	if (not OamReadExact(file, sizeof(OamFileHeader), sections.Data(), sectionTableBytes)) {
		return OamCorrupt("failed to read section table");
	}
	if (fh.Version >= 2 and fh.Checksum != OamManifestHash(fh, sections)) {
		return OamCorrupt("file metadata checksum mismatch");
	}

	const OaU64 dataStart = OamPageAlign(static_cast<OaUsize>(headerBytes));
	std::array<bool, static_cast<OaUsize>(OamSectionType::Spirv) + 1> seen{};
	std::vector<std::pair<OaU64, OaU64>> ranges;
	OaU64 legacyFileChecksum = 0;
	for (const auto& sh : sections) {
		if (sh.Type < static_cast<OaU32>(OamSectionType::Config)
			or sh.Type > static_cast<OaU32>(OamSectionType::Spirv))
		{
			return OamCorrupt("unknown section type");
		}
		if (seen[sh.Type]) return OamCorrupt("duplicate section type");
		seen[sh.Type] = true;
		if (sh.Compression != OamCompression::None or sh.CompressedSize != 0) {
			return OamCorrupt("unsupported section compression");
		}
		OaU64 sectionEnd = 0;
		if (sh.Offset < dataStart or not OamCheckedAdd(sh.Offset, sh.Size, sectionEnd)
			or sectionEnd > fileSize)
		{
			return OamCorrupt("section range is outside the file");
		}
		// OamSectionHeader is packed on disk.  Do not forward references to its
		// 64-bit members into standard-library constructors: those references
		// would retain the packed, potentially misaligned address.
		const OaU64 sectionOffset = sh.Offset;
		ranges.emplace_back(sectionOffset, sectionEnd);
		auto hash = OamHashRange(file, sh.Offset, sh.Size);
		if (not hash.IsOk()) return hash.GetStatus();
		if (hash.GetValue() != sh.Checksum) {
			return OamCorrupt("section payload checksum mismatch");
		}
		legacyFileChecksum ^= hash.GetValue();
	}
	std::sort(ranges.begin(), ranges.end());
	for (OaUsize i = 1; i < ranges.size(); ++i) {
		if (ranges[i].first < ranges[i - 1].second) {
			return OamCorrupt("overlapping section ranges");
		}
	}
	if (fh.Version == 1 and legacyFileChecksum != fh.Checksum) {
		return OamCorrupt("legacy file checksum mismatch");
	}
	if (not seen[static_cast<OaUsize>(OamSectionType::Config)]
		or not seen[static_cast<OaUsize>(OamSectionType::Progress)])
	{
		return OamCorrupt("required section is missing");
	}

	auto findSection = [&](OamSectionType InType) -> const OamSectionHeader* {
		for (const auto& sh : sections) {
			if (sh.Type == static_cast<OaU32>(InType)) return &sh;
		}
		return nullptr;
	};

	OamModel model;
	model.FormatVersion = fh.Version;
	const auto* configSection = findSection(OamSectionType::Config);
	if (configSection->Size < sizeof(OamConfig)
		or not OamReadExact(file, configSection->Offset,
			&model.Config, sizeof(OamConfig)))
	{
		return OamCorrupt("invalid config section");
	}
	OaU64 expectedConfigSize = 0;
	if (not OamCheckedAdd(sizeof(OamConfig), model.Config.ArchConfigSize,
			expectedConfigSize)
		or expectedConfigSize != configSection->Size)
	{
		return OamCorrupt("architecture config size mismatch");
	}
	if (not OamHasTerminator(model.Config.Architecture,
			sizeof(model.Config.Architecture))
		or not OamValidScalarType(static_cast<OaScalarType>(model.Config.WeightDtype))
		or not OamValidScalarType(static_cast<OaScalarType>(model.Config.StateDtype))
		or not OamValidScalarType(static_cast<OaScalarType>(model.Config.ComputeDtype)))
	{
		return OamCorrupt("invalid config metadata");
	}
	if (model.Config.ArchConfigSize > 0) {
		model.ArchConfig.Resize(model.Config.ArchConfigSize);
		if (not OamReadExact(file, configSection->Offset + sizeof(OamConfig),
				model.ArchConfig.Data(), model.ArchConfig.Size()))
		{
			return OamCorrupt("truncated architecture config");
		}
	}

	auto parseTensorSection = [&](OamSectionType InType,
		OaVec<OamTensorEntry>& OutIndex, OaVec<OaU8>& OutBlob) -> OaStatus
	{
		const auto* sh = findSection(InType);
		if (sh == nullptr) return OaStatus::Ok();
		if (sh->Size < sizeof(OaU32) * 2) return OamCorrupt("truncated tensor index");
		OaU32 countAndReserved[2]{};
		if (not OamReadExact(file, sh->Offset, countAndReserved, sizeof(countAndReserved))) {
			return OamCorrupt("cannot read tensor index header");
		}
		if (countAndReserved[1] != 0) return OamCorrupt("tensor index reserved field is nonzero");
		OaU64 entriesBytes = 0;
		OaU64 indexBytes = 0;
		if (not OamCheckedMul(countAndReserved[0], sizeof(OamTensorEntry), entriesBytes)
			or not OamCheckedAdd(sizeof(countAndReserved), entriesBytes, indexBytes)
			or indexBytes > sh->Size)
		{
			return OamCorrupt("tensor index size overflow");
		}
		OutIndex.Resize(countAndReserved[0]);
		if (entriesBytes > 0 and not OamReadExact(file,
				sh->Offset + sizeof(countAndReserved), OutIndex.Data(), entriesBytes))
		{
			return OamCorrupt("truncated tensor index entries");
		}
		const OaU64 blobSize = sh->Size - indexBytes;
		OaHashSet<OaString> names;
		for (const auto& entry : OutIndex) {
			if (not OamHasTerminator(entry.Name, sizeof(entry.Name))
				or entry.Name[0] == '\0' or entry.Rank > OAM_MAX_RANK
				or not OamValidScalarType(entry.Dtype))
			{
				return OamCorrupt("invalid tensor metadata");
			}
			OaU64 entryEnd = 0;
			if (not OamCheckedAdd(entry.BlobOffset, entry.NumBytes, entryEnd)
				or entryEnd > blobSize)
			{
				return OamCorrupt("tensor payload range is outside its section");
			}
			if (not names.Insert(OaString(entry.Name)).second) {
				return OamCorrupt("duplicate tensor name");
			}
			const OaUsize scalarBytes = OaScalarSize(entry.Dtype);
			if (scalarBytes != 0) {
				OaU64 elements = 1;
				for (OaU8 dim = 0; dim < entry.Rank; ++dim) {
					if (not OamCheckedMul(elements, entry.Shape[dim], elements)) {
						return OamCorrupt("tensor element count overflow");
					}
				}
				OaU64 expectedBytes = 0;
				if (not OamCheckedMul(elements, scalarBytes, expectedBytes)
					or expectedBytes != entry.NumBytes)
				{
					return OamCorrupt("tensor byte count does not match shape/dtype");
				}
			}
		}
		OutBlob.Resize(static_cast<OaUsize>(blobSize));
		if (blobSize > 0 and not OamReadExact(file, sh->Offset + indexBytes,
				OutBlob.Data(), blobSize))
		{
			return OamCorrupt("truncated tensor payload");
		}
		return OaStatus::Ok();
	};
	OA_RETURN_IF_ERROR(parseTensorSection(
		OamSectionType::Weights, model.WeightIndex, model.WeightBlob));
	OA_RETURN_IF_ERROR(parseTensorSection(
		OamSectionType::State, model.StateIndex, model.StateBlob));

	if (const auto* sh = findSection(OamSectionType::Optimizer)) {
		model.OptimizerPresent = true;
		if (sh->Size < sizeof(OamOptimizerHeader)
			or not OamReadExact(file, sh->Offset, &model.Optimizer,
				sizeof(OamOptimizerHeader))
			or not OamHasTerminator(model.Optimizer.Type,
				sizeof(model.Optimizer.Type)))
		{
			return OamCorrupt("invalid optimizer header");
		}
		if (not OamIsKnownOptimizerType(model.Optimizer)) {
			return OamCorrupt("unknown optimizer type");
		}
		const OaU64 adamNum = model.Optimizer.NumParams;
		const OaU64 muonNum = OamIsMuonAdamWType(model.Optimizer)
			? OamGetMuonNumParams(model.Optimizer) : 0;
		OaU64 adamBytes = 0;
		OaU64 expectedSize = sizeof(OamOptimizerHeader);
		if (not OamCheckedMul(adamNum, sizeof(OaF32), adamBytes)
			or not OamCheckedAdd(expectedSize, adamBytes, expectedSize))
		{
			return OamCorrupt("optimizer size overflow");
		}
		if (not OamIsMuonOnlyType(model.Optimizer)
			and not OamCheckedAdd(expectedSize, adamBytes, expectedSize))
		{
			return OamCorrupt("optimizer size overflow");
		}
		OaU64 muonBytes = 0;
		if (not OamCheckedMul(muonNum, sizeof(OaF32), muonBytes)
			or not OamCheckedAdd(expectedSize, muonBytes, expectedSize)
			or expectedSize != sh->Size)
		{
			return OamCorrupt("optimizer payload size mismatch");
		}
		OaU64 offset = sizeof(OamOptimizerHeader);
		model.AdamM.Resize(static_cast<OaUsize>(adamNum));
		if (adamBytes > 0 and not OamReadExact(file, sh->Offset + offset,
				model.AdamM.Data(), adamBytes))
		{
			return OamCorrupt("truncated optimizer first moment");
		}
		offset += adamBytes;
		if (not OamIsMuonOnlyType(model.Optimizer)) {
			model.AdamV.Resize(static_cast<OaUsize>(adamNum));
			if (adamBytes > 0 and not OamReadExact(file, sh->Offset + offset,
					model.AdamV.Data(), adamBytes))
			{
				return OamCorrupt("truncated optimizer second moment");
			}
			offset += adamBytes;
		}
		model.MuonM.Resize(static_cast<OaUsize>(muonNum));
		if (muonBytes > 0 and not OamReadExact(file, sh->Offset + offset,
				model.MuonM.Data(), muonBytes))
		{
			return OamCorrupt("truncated Muon momentum");
		}
	}

	const auto* progressSection = findSection(OamSectionType::Progress);
	if (progressSection->Size != sizeof(OamProgress)
		or not OamReadExact(file, progressSection->Offset,
			&model.Progress, sizeof(OamProgress))
		or not OamHasTerminator(model.Progress.MetricName,
			sizeof(model.Progress.MetricName)))
	{
		return OamCorrupt("invalid progress section");
	}

	if (const auto* sh = findSection(OamSectionType::Spirv)) {
		if (sh->Size < sizeof(OaU32) * 2) return OamCorrupt("truncated SPIR-V index");
		OaU32 countAndReserved[2]{};
		if (not OamReadExact(file, sh->Offset, countAndReserved, sizeof(countAndReserved))
			or countAndReserved[1] != 0)
		{
			return OamCorrupt("invalid SPIR-V index header");
		}
		OaU64 entriesBytes = 0;
		OaU64 indexBytes = 0;
		if (not OamCheckedMul(countAndReserved[0], sizeof(OamSpirvEntry), entriesBytes)
			or not OamCheckedAdd(sizeof(countAndReserved), entriesBytes, indexBytes)
			or indexBytes > sh->Size)
		{
			return OamCorrupt("SPIR-V index size overflow");
		}
		model.SpirvIndex.Resize(countAndReserved[0]);
		if (entriesBytes > 0 and not OamReadExact(file,
				sh->Offset + sizeof(countAndReserved), model.SpirvIndex.Data(), entriesBytes))
		{
			return OamCorrupt("truncated SPIR-V index entries");
		}
		const OaU64 blobSize = sh->Size - indexBytes;
		OaHashSet<OaString> names;
		for (const auto& entry : model.SpirvIndex) {
			OaU64 entryEnd = 0;
			if (not OamHasTerminator(entry.Name, sizeof(entry.Name))
				or entry.Name[0] == '\0'
				or not OamCheckedAdd(entry.BlobOffset, entry.NumBytes, entryEnd)
				or entryEnd > blobSize
				or not names.Insert(OaString(entry.Name)).second)
			{
				return OamCorrupt("invalid SPIR-V entry metadata");
			}
		}
		model.SpirvBlob.Resize(static_cast<OaUsize>(blobSize));
		if (blobSize > 0 and not OamReadExact(file, sh->Offset + indexBytes,
				model.SpirvBlob.Data(), blobSize))
		{
			return OamCorrupt("truncated SPIR-V payload");
		}
		for (const auto& entry : model.SpirvIndex) {
			if (OamHash(model.SpirvBlob.Data() + entry.BlobOffset,
					static_cast<OaUsize>(entry.NumBytes)) != entry.SourceHash)
			{
				return OamCorrupt("SPIR-V source hash mismatch");
			}
		}
	}

	OA_LOG_INFO(OaLogComponent::ML, "[oam] Loaded %s | arch=%s | %zu weights | optimizer=%s | kernels=%zu",
		InPath.c_str(), model.Config.Architecture,
		model.WeightIndex.Size(),
		model.HasOptimizer() ? "yes" : "no",
		model.SpirvIndex.Size());
	return model;
}

// Dump

void OamDump(const OaString& InPath) {
	{
		auto verified = OamModel::Load(InPath);
		if (not verified.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::ML, "[oam] Refusing to inspect %s: %s",
				InPath.c_str(), verified.GetStatus().ToString().c_str());
			return;
		}
	}
	std::ifstream file(InPath.c_str(), std::ios::binary);
	if (!file) { OA_LOG_ERROR(OaLogComponent::ML, "[oam] Cannot open: %s", InPath.c_str()); return; }

	OamFileHeader fh;
	file.read(reinterpret_cast<char*>(&fh), sizeof(fh));
	if (fh.Magic != OAM_MAGIC) { OA_LOG_ERROR(OaLogComponent::ML, "[oam] Not an .oam file"); return; }

	OA_LOG_INFO(OaLogComponent::ML, "=== %s ===", InPath.c_str());
	OA_LOG_INFO(OaLogComponent::ML, "  Sections: %u  Size: %.1f MB", fh.NumSections, fh.TotalSize / 1e6);

	OaVec<OamSectionHeader> sections(fh.NumSections);
	for (OaU32 i = 0; i < fh.NumSections; ++i)
		file.read(reinterpret_cast<char*>(&sections[i]), sizeof(OamSectionHeader));

	for (OaU32 i = 0; i < fh.NumSections; ++i) {
		const auto& sh = sections[i];
		auto type = static_cast<OamSectionType>(sh.Type);
		OA_LOG_INFO(OaLogComponent::ML, "  [%u] %s  offset=%lu  %.2f MB",
			i, OamSectionTypeName(type), sh.Offset, sh.Size / 1e6);

		if (type == OamSectionType::Config) {
			OamConfig cfg;
			file.seekg(sh.Offset);
			file.read(reinterpret_cast<char*>(&cfg), std::min(sh.Size, static_cast<OaU64>(sizeof(OamConfig))));
			OA_LOG_INFO(OaLogComponent::ML, "       arch=%s DModel=%u NLayers=%u DVocab=%u",
				cfg.Architecture, cfg.DModel, cfg.NLayers, cfg.DVocab
			);
		}

		if (type == OamSectionType::Weights || type == OamSectionType::State) {
			OaU32 count = 0, reserved = 0;
			file.seekg(sh.Offset);
			file.read(reinterpret_cast<char*>(&count), sizeof(OaU32));
			file.read(reinterpret_cast<char*>(&reserved), sizeof(OaU32));
			for (OaU32 j = 0; j < count; ++j) {
				OamTensorEntry e;
				file.read(reinterpret_cast<char*>(&e), sizeof(e));
				OA_LOG_INFO(OaLogComponent::ML, "       %s  %.*s  %.3f MB",
					e.Name, static_cast<int>(OaScalarTypeName(e.Dtype).size()),
					OaScalarTypeName(e.Dtype).data(), e.NumBytes / 1e6);
			}
		}

		if (type == OamSectionType::Spirv) {
			OaU32 count = 0, reserved = 0;
			file.seekg(sh.Offset);
			file.read(reinterpret_cast<char*>(&count), sizeof(OaU32));
			file.read(reinterpret_cast<char*>(&reserved), sizeof(OaU32));
			for (OaU32 j = 0; j < count; ++j) {
				OamSpirvEntry e;
				file.read(reinterpret_cast<char*>(&e), sizeof(e));
				OA_LOG_INFO(OaLogComponent::ML, "       %s  %.1f KB  hash=%016lx",
					e.Name, e.NumBytes / 1e3, e.SourceHash);
			}
		}
	}
}
