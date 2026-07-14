#include <Oa/Ml/Oam.h>
#include <Oa/Core/Log.h>

#include <cstring>
#include <fstream>
#include <filesystem>

// FNV-1a hash (no external deps — replaces xxhash)
OaU64 OamHash(const OaU8* InData, OaUsize InSize) {
	OaU64 hash = 0xcbf29ce484222325ULL;
	for (OaUsize i = 0; i < InSize; ++i) {
		hash ^= InData[i];
		hash *= 0x100000001b3ULL;
	}
	return hash;
}

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
	{
		OaU64 checksum = 0;
		for (const auto& sp : payloads) checksum ^= OamHash(sp.Data.Data(), sp.Data.Size());
		fileHeader.Checksum = checksum;
	}

	std::ofstream file(InPath.c_str(), std::ios::binary);
	if (!file) return OaStatus::Error("Failed to open for write: " + InPath);

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

	OA_LOG_INFO(OaLogComponent::ML, "[oam] Saved %s | %zu weights | %zu kernels | %.1f MB",
		InPath.c_str(), WeightIndex.Size(), SpirvIndex.Size(), fileHeader.TotalSize / 1e6);
	return OaStatus::Ok();
}

// Load

OaResult<OamModel> OamModel::Load(const OaString& InPath) {
	std::ifstream file(InPath.c_str(), std::ios::binary);
	if (!file) return OaStatus::Error("Cannot open: " + InPath);

	OamFileHeader fh;
	file.read(reinterpret_cast<char*>(&fh), sizeof(fh));
	if (!file) return OaStatus::Error("Failed to read file header");
	if (fh.Magic != OAM_MAGIC) return OaStatus::Error("Not an .oam file");
	if (fh.Version > OAM_VERSION) return OaStatus::Error("Unsupported .oam version");

	OaVec<OamSectionHeader> sections(fh.NumSections);
	for (OaU32 i = 0; i < fh.NumSections; ++i)
		file.read(reinterpret_cast<char*>(&sections[i]), sizeof(OamSectionHeader));

	OamModel model;

	for (const auto& sh : sections) {
		auto type = static_cast<OamSectionType>(sh.Type);

		switch (type) {
			case OamSectionType::Config: {
				OaVec<OaU8> raw(sh.Size);
				file.seekg(sh.Offset);
				file.read(reinterpret_cast<char*>(raw.Data()), sh.Size);
				if (raw.Size() >= sizeof(OamConfig))
					std::memcpy(&model.Config, raw.Data(), sizeof(OamConfig));
				if (model.Config.ArchConfigSize > 0 && raw.Size() >= sizeof(OamConfig) + model.Config.ArchConfigSize) {
					model.ArchConfig.Resize(model.Config.ArchConfigSize);
					std::memcpy(model.ArchConfig.Data(), raw.Data() + sizeof(OamConfig), model.Config.ArchConfigSize);
				}
				break;
			}

			case OamSectionType::Weights:
			case OamSectionType::State: {
				bool isWeights = (type == OamSectionType::Weights);
				OaU32 count = 0, reserved = 0;
				file.seekg(sh.Offset);
				file.read(reinterpret_cast<char*>(&count), sizeof(OaU32));
				file.read(reinterpret_cast<char*>(&reserved), sizeof(OaU32));

				OaUsize indexBytes = sizeof(OaU32) * 2 + sizeof(OamTensorEntry) * count;
				OaVec<OamTensorEntry> index(count);
				file.read(reinterpret_cast<char*>(index.Data()), sizeof(OamTensorEntry) * count);

				OaUsize blobSize = sh.Size - indexBytes;
				OaVec<OaU8> blob(blobSize);
				file.read(reinterpret_cast<char*>(blob.Data()), blobSize);

				if (isWeights) {
					model.WeightIndex = std::move(index);
					model.WeightBlob = std::move(blob);
				} else {
					model.StateIndex = std::move(index);
					model.StateBlob = std::move(blob);
				}
				break;
			}

			case OamSectionType::Optimizer: {
				OaVec<OaU8> raw(sh.Size);
				file.seekg(sh.Offset);
				file.read(reinterpret_cast<char*>(raw.Data()), sh.Size);
				if (raw.Size() < sizeof(OamOptimizerHeader)) break;
				std::memcpy(&model.Optimizer, raw.Data(), sizeof(OamOptimizerHeader));
				const OaU64 adamNum = model.Optimizer.NumParams;
				OaUsize offset = sizeof(OamOptimizerHeader);
				const OaUsize adamMBytes = adamNum * sizeof(OaF32);

				if (adamNum > 0 && offset + adamMBytes <= raw.Size()) {
					model.AdamM.Resize(adamNum);
					std::memcpy(model.AdamM.Data(), raw.Data() + offset, adamMBytes);
					offset += adamMBytes;
				}

				if (OamIsMuonAdamWType(model.Optimizer)) {
					const OaU64 muonNum = OamGetMuonNumParams(model.Optimizer);
					const OaUsize adamVBytes = adamNum * sizeof(OaF32);
					const OaUsize muonMBytes = muonNum * sizeof(OaF32);
					if (offset + adamVBytes + muonMBytes <= raw.Size()) {
						if (adamNum > 0) {
							model.AdamV.Resize(adamNum);
							std::memcpy(model.AdamV.Data(), raw.Data() + offset, adamVBytes);
							offset += adamVBytes;
						}
						if (muonNum > 0) {
							model.MuonM.Resize(muonNum);
							std::memcpy(model.MuonM.Data(), raw.Data() + offset, muonMBytes);
						}
					}
				} else if (OamIsMuonOnlyType(model.Optimizer)) {
					model.AdamV.Clear();
				} else {
					const OaUsize adamVBytes = adamNum * sizeof(OaF32);
					if (adamNum > 0 && offset + adamVBytes <= raw.Size()) {
						model.AdamV.Resize(adamNum);
						std::memcpy(model.AdamV.Data(), raw.Data() + offset, adamVBytes);
					}
				}
				break;
			}

			case OamSectionType::Progress: {
				file.seekg(sh.Offset);
				file.read(reinterpret_cast<char*>(&model.Progress),
					std::min(sh.Size, static_cast<OaU64>(sizeof(OamProgress))));
				break;
			}

			case OamSectionType::Spirv: {
				OaU32 count = 0, reserved = 0;
				file.seekg(sh.Offset);
				file.read(reinterpret_cast<char*>(&count), sizeof(OaU32));
				file.read(reinterpret_cast<char*>(&reserved), sizeof(OaU32));
				OaUsize indexBytes = sizeof(OaU32) * 2 + sizeof(OamSpirvEntry) * count;
				model.SpirvIndex.Resize(count);
				file.read(reinterpret_cast<char*>(model.SpirvIndex.Data()), sizeof(OamSpirvEntry) * count);
				OaUsize blobSize = sh.Size - indexBytes;
				model.SpirvBlob.Resize(blobSize);
				file.read(reinterpret_cast<char*>(model.SpirvBlob.Data()), blobSize);
				break;
			}

			default: break;
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
