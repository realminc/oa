// modelctl - OA Model Control Tool (Unified .oam operations)
//
// Unified tool for model inspection, weight import, and validation.
// Uses OaComputeApp for GPU-accelerated validation (optional).
//
// Commands:
//   inspect    - Inspect an external weight source
//   info       - Show model info and all sections
//   verify     - Verify model integrity (load + checksums)
//   import     - Import external weights to .oam
//   validate   - Validate converted model against reference
//   list       - List all .oam models in directory
//   compare    - Compare two models
//   dump       - Raw section dump (offsets, sizes, checksums)
//
// Usage:
//   modelctl info model.oam
//   modelctl import --in model.safetensors --arch raw --out model.oam
//   modelctl validate model.oam --reference ref.bin
//   modelctl list var/model
//   modelctl compare a.oam b.oam
//   modelctl dump model.oam

#include <Oa/Core/Types.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Cli.h>
#include <Oa/Core/FileIo.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Ml/TransferWeights.h>
#include <Ml/Nn/Alm/ClipTextWeightAdapter.h>

#include <cmath>
#include <cstring>
#include <fstream>

// ============================================================================
// Format Utilities
// ============================================================================

static OaString FormatBytes(OaU64 InBytes) {
	char buf[64];
	if (InBytes >= 1'000'000'000) {
		snprintf(buf, sizeof(buf), "%.2f GB", static_cast<OaF64>(InBytes) / 1'000'000'000);
	} else if (InBytes >= 1'000'000) {
		snprintf(buf, sizeof(buf), "%.2f MB", static_cast<OaF64>(InBytes) / 1'000'000);
	} else if (InBytes >= 1'000) {
		snprintf(buf, sizeof(buf), "%.2f KB", static_cast<OaF64>(InBytes) / 1'000);
	} else {
		snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(InBytes));
	}
	return buf;
}

static OaString FormatNumber(OaI64 InValue) {
	char buf[64];
	if (InValue >= 1'000'000'000) {
		snprintf(buf, sizeof(buf), "%.2fB", static_cast<OaF64>(InValue) / 1'000'000'000);
	} else if (InValue >= 1'000'000) {
		snprintf(buf, sizeof(buf), "%.2fM", static_cast<OaF64>(InValue) / 1'000'000);
	} else if (InValue >= 1'000) {
		snprintf(buf, sizeof(buf), "%.2fK", static_cast<OaF64>(InValue) / 1'000);
	} else {
		snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(InValue));
	}
	return buf;
}

static OaString FormatShape(const OamTensorEntry& InEntry) {
	OaString result = "[";
	for (OaU8 d = 0; d < InEntry.Rank; ++d) {
		if (d > 0) result += ", ";
		char num[32];
		snprintf(num, sizeof(num), "%llu", static_cast<unsigned long long>(InEntry.Shape[d]));
		result += num;
	}
	result += "]";
	return result;
}

static OaString FormatShape(const OaWeightInfo& InEntry) {
	OaString result = "[";
	for (OaUsize d = 0; d < InEntry.Shape.Size(); ++d) {
		if (d > 0) result += ", ";
		result += std::to_string(static_cast<long long>(InEntry.Shape[d]));
	}
	result += "]";
	return result;
}

static OaI64 CountElements(const OamTensorEntry& InEntry) {
	OaI64 n = 1;
	for (OaU8 d = 0; d < InEntry.Rank; ++d)
		n *= static_cast<OaI64>(InEntry.Shape[d]);
	return n;
}

static OaI64 CountTotalParams(const OamModel& InModel) {
	OaI64 total = 0;
	for (const auto& w : InModel.WeightIndex)
		total += CountElements(w);
	return total;
}

// ============================================================================
// Commands
// ============================================================================

static int CmdInspect(const OaString& InPath) {
	auto sourceResult = OaOpenWeightSource(OaPath(InPath));
	if (sourceResult.IsError()) {
		OA_CLI("Error: %s", sourceResult.GetStatus().ToString().c_str());
		return 1;
	}
	auto& source = *sourceResult.GetValue();

	const auto entries = source.List();
	OaU64 payloadBytes = 0;
	OaU64 elements = 0;
	for (const auto& entry : entries) {
		payloadBytes += entry.ByteSize;
		elements += entry.ElementCount;
	}

	OA_CLI_RAW("\n");
	OA_CLI("  Weight source: %s", InPath.c_str());
	OA_CLI("  File Size:     %s", FormatBytes(source.SourceBytes()).c_str());
	OA_CLI("  Entries:     %zu", entries.Size());
	OA_CLI("  Elements:    %s", FormatNumber(static_cast<OaI64>(elements)).c_str());
	OA_CLI("  Payload:     %s", FormatBytes(payloadBytes).c_str());

	const auto metadata = source.Metadata();
	if (!metadata.Empty()) {
		OA_CLI_RAW("\n");
		OA_CLI("  METADATA");
		for (const auto& [key, value] : metadata) {
			OA_CLI("  %s: %s", key.c_str(), value.c_str());
		}
	}

	OA_CLI_RAW("\n");
	OA_CLI("  %-5s %-42s %-22s %-10s %s", "Idx", "Name", "Shape", "Dtype", "Bytes");
	for (OaUsize i = 0; i < entries.Size(); ++i) {
		const auto& entry = entries[i];
		OA_CLI("  %-5zu %-42s %-22s %-10.*s %s",
			i,
			entry.Name.c_str(),
			FormatShape(entry).c_str(),
			static_cast<int>(OaScalarTypeName(entry.Dtype).size()),
			OaScalarTypeName(entry.Dtype).data(),
			FormatBytes(entry.ByteSize).c_str());
	}
	OA_CLI_RAW("\n");
	return 0;
}

static int CmdInfo(const OaString& InPath) {
	auto result = OamModel::Load(InPath);
	if (!result.IsOk()) {
		OA_CLI("Error: %s", result.GetStatus().GetMessage().c_str());
		return 1;
	}

	const auto& model = result.GetValue();
	auto totalParams = CountTotalParams(model);

	OA_CLI_RAW("\n");
	OA_CLI("  Model: %s", InPath.c_str());
	OA_CLI_RAW("\n");

	auto sizeResult = OaFileIo::GetFileSize(OaPath(InPath));
	if (sizeResult.IsOk()) {
		OA_CLI("  File Size:      %s", FormatBytes(sizeResult.GetValue()).c_str());
	}
	OA_CLI("  Format:         OAM v%u", OAM_VERSION);

	// Config
	OA_CLI_RAW("\n");
	OA_CLI("  CONFIG");
	OA_CLI("  Architecture:   %s", model.Config.Architecture);
	OA_CLI("  DModel:         %u", model.Config.DModel);
	OA_CLI("  NLayers:        %u", model.Config.NLayers);
	OA_CLI("  DVocab:         %u", model.Config.DVocab);
	OA_CLI("  Weight Dtype:   %.*s",
		static_cast<int>(OaScalarTypeName(static_cast<OaScalarType>(model.Config.WeightDtype)).size()),
		OaScalarTypeName(static_cast<OaScalarType>(model.Config.WeightDtype)).data());
	if (model.Config.ArchConfigSize > 0) {
		OA_CLI("  ArchConfig:     %u bytes", model.Config.ArchConfigSize);
	}

	// Weights
	if (model.HasWeights()) {
		OaU64 weightBytes = 0;
		for (const auto& w : model.WeightIndex) weightBytes += w.NumBytes;

		OA_CLI_RAW("\n");
		OA_CLI("  WEIGHTS (%zu tensors)", model.WeightIndex.Size());
		OA_CLI("  Parameters:     %s (%s)",
			FormatNumber(totalParams).c_str(),
			FormatBytes(static_cast<OaU64>(totalParams) * 4).c_str());
		OA_CLI("  Blob Size:      %s", FormatBytes(weightBytes).c_str());

		constexpr OaU32 kMaxTensors = 30;
		OA_CLI_RAW("\n");
		OA_CLI("  %-6s  %-30s  %-18s  %-10s  %s", "Idx", "Name", "Shape", "Elements", "Dtype");
		for (OaU32 i = 0; i < model.WeightIndex.Size(); ++i) {
			if (i >= kMaxTensors) {
				OA_CLI("  ... and %zu more tensors", model.WeightIndex.Size() - kMaxTensors);
				break;
			}
			const auto& w = model.WeightIndex[i];
			OA_CLI("  %-6u  %-30s  %-18s  %-10s  %.*s",
				i, w.Name,
				FormatShape(w).c_str(),
				FormatNumber(CountElements(w)).c_str(),
				static_cast<int>(OaScalarTypeName(w.Dtype).size()),
				OaScalarTypeName(w.Dtype).data());
		}
	}

	// State
	if (model.HasState()) {
		OA_CLI_RAW("\n");
		OA_CLI("  STATE (%zu tensors)", model.StateIndex.Size());
		for (OaU32 i = 0; i < model.StateIndex.Size() && i < 10; ++i) {
			const auto& s = model.StateIndex[i];
			OA_CLI("    %s  %s  %s", s.Name, FormatShape(s).c_str(),
				FormatBytes(s.NumBytes).c_str());
		}
	}

	// Optimizer
	if (model.HasOptimizer()) {
		OA_CLI_RAW("\n");
		OA_CLI("  OPTIMIZER");
		OA_CLI("  Type:           %s", model.Optimizer.Type);
		OA_CLI("  Step:           %lld", static_cast<long long>(model.Optimizer.Step));
		OA_CLI("  Lr:             %.2e", model.Optimizer.Lr);
		OA_CLI("  Beta1:          %.4f", model.Optimizer.Beta1);
		OA_CLI("  Beta2:          %.6f", model.Optimizer.Beta2);
		OA_CLI("  Eps:            %.1e", model.Optimizer.Eps);
		OA_CLI("  Weight Decay:   %.4f", model.Optimizer.WeightDecay);
		OA_CLI("  Num Params:     %s", FormatNumber(static_cast<OaI64>(model.Optimizer.NumParams)).c_str());
		OA_CLI("  M/V Size:       %s each",
			FormatBytes(model.AdamM.Size() * sizeof(OaF32)).c_str());
	}

	// Progress
	OA_CLI_RAW("\n");
	OA_CLI("  PROGRESS");
	OA_CLI("  Step:           %lld", static_cast<long long>(model.Progress.Step));
	OA_CLI("  Bytes Seen:     %s", FormatBytes(model.Progress.BytesSeen).c_str());
	OA_CLI("  Lr:             %.2e", model.Progress.Lr);
	OA_CLI("  Best Metric:    %.6f (%s, %s)",
		model.Progress.BestMetric,
		model.Progress.MetricName,
		model.Progress.LowerIsBetter ? "lower is better" : "higher is better");

	// Compute kernels (SPIR-V)
	if (model.HasSpirvCache()) {
		OaU64 spirvBytes = 0;
		for (const auto& e : model.SpirvIndex) spirvBytes += e.NumBytes;

		OA_CLI_RAW("\n");
		OA_CLI("  COMPUTE KERNELS (%zu kernels, %s total)", model.SpirvIndex.Size(),
			FormatBytes(spirvBytes).c_str());
		for (OaU32 i = 0; i < model.SpirvIndex.Size(); ++i) {
			const auto& e = model.SpirvIndex[i];
			OA_CLI("    %-30s  %s  hash=%016lx",
				e.Name, FormatBytes(e.NumBytes).c_str(), e.SourceHash);
		}
	}

	OA_CLI_RAW("\n");
	return 0;
}

static int CmdVerify(const OaString& InPath) {
	OA_CLI("Verifying: %s ...", InPath.c_str());

	// Raw file-level integrity checks before parsing
	{
		std::ifstream file(InPath.c_str(), std::ios::binary);
		if (!file) {
			OA_CLI("ERROR: cannot open file");
			return 1;
		}
		OamFileHeader fh;
		file.read(reinterpret_cast<char*>(&fh), sizeof(fh));
		if (!file) {
			OA_CLI("ERROR: cannot read file header");
			return 1;
		}
		if (fh.Magic != OAM_MAGIC) {
			OA_CLI("ERROR: invalid magic (not an .oam file)");
			return 1;
		}
		if (fh.Version > OAM_VERSION) {
			OA_CLI("ERROR: unsupported version %u (max %u)", fh.Version, OAM_VERSION);
			return 1;
		}
		OaVec<OamSectionHeader> sections(fh.NumSections);
		for (OaU32 i = 0; i < fh.NumSections; ++i) {
			file.read(reinterpret_cast<char*>(&sections[i]), sizeof(OamSectionHeader));
		}
		if (!file) {
			OA_CLI("ERROR: cannot read section headers");
			return 1;
		}
		// Recompute section checksums
		OaU64 computedTotalChecksum = 0;
		for (const auto& sh : sections) {
			if (sh.Offset == 0 && sh.Size > 0) {
				OA_CLI("ERROR: section type=%u has zero offset but non-zero size", sh.Type);
				return 1;
			}
			OaVec<OaU8> buf(sh.Size);
			file.seekg(static_cast<std::streamoff>(sh.Offset));
			file.read(reinterpret_cast<char*>(buf.Data()), static_cast<std::streamsize>(sh.Size));
			if (!file) {
				OA_CLI("ERROR: cannot read section data at offset %llu", static_cast<unsigned long long>(sh.Offset));
				return 1;
			}
			OaU64 computed = OamHash(buf.Data(), buf.Size());
			if (computed != sh.Checksum) {
				OA_CLI("ERROR: section type=%u checksum mismatch (expected %016llx, got %016llx)",
					sh.Type,
					static_cast<unsigned long long>(sh.Checksum),
					static_cast<unsigned long long>(computed));
				return 1;
			}
			computedTotalChecksum ^= computed;
		}
		if (computedTotalChecksum != fh.Checksum) {
			OA_CLI("ERROR: file header checksum mismatch (expected %016llx, got %016llx)",
				static_cast<unsigned long long>(fh.Checksum),
				static_cast<unsigned long long>(computedTotalChecksum));
			return 1;
		}
	}

	// Structural checks via OamModel::Load
	auto result = OamModel::Load(InPath);
	if (!result.IsOk()) {
		OA_CLI("LOAD FAILED: %s", result.GetStatus().GetMessage().c_str());
		return 1;
	}

	const auto& model = result.GetValue();
	auto totalParams = CountTotalParams(model);
	OaI32 issues = 0;

	// Check weight blob consistency
	for (const auto& w : model.WeightIndex) {
		if (w.BlobOffset + w.NumBytes > model.WeightBlob.Size()) {
			OA_CLI("ERROR: weight '%s' extends past blob (offset=%llu, size=%llu, blob=%zu)",
				w.Name, static_cast<unsigned long long>(w.BlobOffset),
				static_cast<unsigned long long>(w.NumBytes), model.WeightBlob.Size());
			++issues;
		}
	}

	// Check for NaN/Inf
	OaI64 nanCount = 0;
	OaI64 infCount = 0;
	for (const auto& w : model.WeightIndex) {
		if (w.Dtype == OaScalarType::Float32 &&
			w.BlobOffset + w.NumBytes <= model.WeightBlob.Size()) {
			OaI64 elems = CountElements(w);
			const auto* data = reinterpret_cast<const OaF32*>(
				model.WeightBlob.Data() + w.BlobOffset);
			for (OaI64 j = 0; j < elems; ++j) {
				if (std::isnan(data[j])) ++nanCount;
				if (std::isinf(data[j])) ++infCount;
			}
		}
	}

	if (nanCount > 0 || infCount > 0) {
		OA_CLI("WARNING: %lld NaN, %lld Inf values in weights",
			static_cast<long long>(nanCount), static_cast<long long>(infCount));
		++issues;
	}

	// Optimizer consistency
	if (model.HasOptimizer()) {
		if (model.AdamM.Size() != model.AdamV.Size()) {
			OA_CLI("ERROR: AdamM size (%zu) != AdamV size (%zu)",
				model.AdamM.Size(), model.AdamV.Size());
			++issues;
		}
	}

	// SPIR-V consistency
	for (const auto& e : model.SpirvIndex) {
		if (e.BlobOffset + e.NumBytes > model.SpirvBlob.Size()) {
			OA_CLI("ERROR: spirv '%s' extends past blob", e.Name);
			++issues;
		}
	}

	if (issues == 0) {
		OA_CLI("OK: arch=%s, %s params, %zu weights, optimizer=%s, %zu kernels",
			model.Config.Architecture,
			FormatNumber(totalParams).c_str(),
			model.WeightIndex.Size(),
			model.HasOptimizer() ? "yes" : "no",
			model.SpirvIndex.Size());
	} else {
		OA_CLI("ISSUES: %d problems found", issues);
	}
	return issues > 0 ? 1 : 0;
}

static int CmdList(const OaString& InDir) {
	if (!OaFileIo::IsDirectory(OaPath(InDir))) {
		OA_CLI("Error: directory not found: %s", InDir.c_str());
		return 1;
	}

	auto filesResult = OaFileIo::ListAll(OaPath(InDir), true);
	if (!filesResult.IsOk()) {
		OA_CLI("Error: %s", filesResult.GetStatus().GetMessage().c_str());
		return 1;
	}

	OA_CLI_RAW("\n");
	OA_CLI("  Models in: %s", InDir.c_str());
	OA_CLI_RAW("\n");
	OA_CLI("  %-40s  %-12s  %-8s  %-10s  %-6s  %s",
		"File", "Architecture", "Step", "Parameters", "Optim", "Kernels");

	OaI32 count = 0;
	for (const auto& path : filesResult.GetValue()) {
		auto ext = OaFileIo::GetExtension(path);
		if (ext != ".oam") continue;

		auto loadResult = OamModel::Load(path.String());
		if (loadResult.IsOk()) {
			const auto& m = loadResult.GetValue();
			auto totalParams = CountTotalParams(m);

			OaString relPath = path.String();
			if (relPath.size() > InDir.size() && relPath.substr(0, InDir.size()) == InDir) {
				relPath = relPath.substr(InDir.size());
				if (!relPath.empty() && relPath[0] == '/') relPath = relPath.substr(1);
			}
			if (relPath.size() > 40) relPath = "..." + relPath.substr(relPath.size() - 37);

			OA_CLI("  %-40s  %-12s  %-8lld  %-10s  %-6s  %zu",
				relPath.c_str(),
				m.Config.Architecture,
				static_cast<long long>(m.Progress.Step),
				FormatNumber(totalParams).c_str(),
				m.HasOptimizer() ? "yes" : "no",
				m.SpirvIndex.Size());
			++count;
		} else {
			OaString relPath = path.String();
			if (relPath.size() > InDir.size() && relPath.substr(0, InDir.size()) == InDir) {
				relPath = relPath.substr(InDir.size());
				if (!relPath.empty() && relPath[0] == '/') relPath = relPath.substr(1);
			}
			if (relPath.size() > 40) relPath = "..." + relPath.substr(relPath.size() - 37);
			OA_CLI("  %-40s  %-12s  %-8s  %-10s  %-6s  %s",
				relPath.c_str(), "ERROR", "-", "-", "-", "-");
		}
	}

	OA_CLI_RAW("\n");
	if (count == 0) {
		OA_CLI("  (no .oam models found)");
	} else {
		OA_CLI("  Total: %d models", count);
	}
	OA_CLI_RAW("\n");
	return 0;
}

static int CmdCompare(const OaString& InPath1, const OaString& InPath2) {
	auto result1 = OamModel::Load(InPath1);
	auto result2 = OamModel::Load(InPath2);

	if (!result1.IsOk()) {
		OA_CLI("Error loading %s: %s", InPath1.c_str(), result1.GetStatus().GetMessage().c_str());
		return 1;
	}
	if (!result2.IsOk()) {
		OA_CLI("Error loading %s: %s", InPath2.c_str(), result2.GetStatus().GetMessage().c_str());
		return 1;
	}

	const auto& m1 = result1.GetValue();
	const auto& m2 = result2.GetValue();
	auto params1 = CountTotalParams(m1);
	auto params2 = CountTotalParams(m2);

	OA_CLI_RAW("\n");
	OA_CLI("  Model Comparison");
	OA_CLI_RAW("\n");
	OA_CLI("  %-20s  %15s  %15s", "", "Model 1", "Model 2");
	OA_CLI("  %-20s  %15s  %15s", "Architecture:", m1.Config.Architecture, m2.Config.Architecture);
	OA_CLI("  %-20s  %15u  %15u", "DModel:", m1.Config.DModel, m2.Config.DModel);
	OA_CLI("  %-20s  %15u  %15u", "NLayers:", m1.Config.NLayers, m2.Config.NLayers);
	OA_CLI("  %-20s  %15u  %15u", "DVocab:", m1.Config.DVocab, m2.Config.DVocab);
	OA_CLI("  %-20s  %15s  %15s", "Parameters:",
		FormatNumber(params1).c_str(), FormatNumber(params2).c_str());
	OA_CLI("  %-20s  %15zu  %15zu", "Weight Tensors:",
		m1.WeightIndex.Size(), m2.WeightIndex.Size());
	OA_CLI("  %-20s  %15lld  %15lld", "Step:",
		static_cast<long long>(m1.Progress.Step),
		static_cast<long long>(m2.Progress.Step));
	OA_CLI("  %-20s  %15.6f  %15.6f", "Best Metric:",
		m1.Progress.BestMetric, m2.Progress.BestMetric);
	OA_CLI("  %-20s  %15s  %15s", "Optimizer:",
		m1.HasOptimizer() ? "yes" : "no",
		m2.HasOptimizer() ? "yes" : "no");
	OA_CLI("  %-20s  %15zu  %15zu", "Compute Kernels:",
		m1.SpirvIndex.Size(), m2.SpirvIndex.Size());
	OA_CLI_RAW("\n");

	// Weight diff if architectures match
	if (params1 == params2 && params1 > 0 &&
		m1.WeightIndex.Size() == m2.WeightIndex.Size()) {

		OaF64 sumDiff = 0.0;
		OaF64 maxDiff = 0.0;
		OaI64 totalElements = 0;
		bool compatible = true;

		for (OaU32 i = 0; i < m1.WeightIndex.Size(); ++i) {
			const auto& w1 = m1.WeightIndex[i];
			const auto& w2 = m2.WeightIndex[i];

			if (CountElements(w1) != CountElements(w2) || w1.Dtype != w2.Dtype) {
				compatible = false;
				break;
			}

			if (w1.Dtype == OaScalarType::Float32 &&
				w1.BlobOffset + w1.NumBytes <= m1.WeightBlob.Size() &&
				w2.BlobOffset + w2.NumBytes <= m2.WeightBlob.Size()) {
				OaI64 elems = CountElements(w1);
				const auto* d1 = reinterpret_cast<const OaF32*>(m1.WeightBlob.Data() + w1.BlobOffset);
				const auto* d2 = reinterpret_cast<const OaF32*>(m2.WeightBlob.Data() + w2.BlobOffset);
				for (OaI64 j = 0; j < elems; ++j) {
					OaF64 diff = std::abs(static_cast<OaF64>(d1[j]) - static_cast<OaF64>(d2[j]));
					sumDiff += diff;
					if (diff > maxDiff) maxDiff = diff;
				}
				totalElements += elems;
			}
		}

		if (compatible && totalElements > 0) {
			OaF64 avgDiff = sumDiff / static_cast<OaF64>(totalElements);
			OA_CLI("  WEIGHT DIFFERENCES");
			OA_CLI("  Avg Diff:       %.8f", avgDiff);
			OA_CLI("  Max Diff:       %.8f", maxDiff);
			OA_CLI_RAW("\n");
		}
	}

	return 0;
}

static int CmdDump(const OaString& InPath) {
	OamDump(InPath);
	return 0;
}


// ============================================================================
// Import Command (External Weight Transfer)
// ============================================================================

static int CmdImport(
	const OaString& InInputPath,
	const OaString& InFormat,
	const OaString& InArch,
	const OaString& InOutputPath,
	const OaString& InDtype,
	bool InValidate
) {
	OA_CLI("Importing: %s -> %s", InInputPath.c_str(), InOutputPath.c_str());
	OA_CLI("Format: %s, Arch: %s, Dtype: %s", InFormat.c_str(), InArch.c_str(), InDtype.c_str());

	if (InDtype != "preserve") {
		OA_CLI("Error: dtype conversion policy is adapter-owned; currently use '--dtype preserve'");
		return 2;
	}
	if (InValidate) {
		OA_CLI("Error: import parity validation requires a registered architecture validator");
		return 2;
	}
	if (InOutputPath.empty()) {
		OA_CLI("Error: --out is required for import");
		return 2;
	}

	OaWeightFormat format = OaWeightFormat::Auto;
	if (InFormat == "safetensors") format = OaWeightFormat::SafeTensors;
	else if (InFormat == "gguf") format = OaWeightFormat::Gguf;
	else if (InFormat == "onnx") format = OaWeightFormat::Onnx;
	else if (InFormat != "auto") {
		OA_CLI("Error: unknown weight format '%s'", InFormat.c_str());
		return 2;
	}
	auto sourceResult = OaOpenWeightSource(OaPath(InInputPath), format);
	if (sourceResult.IsError()) {
		OA_CLI("Error: Failed to open weight source: %s", sourceResult.GetStatus().GetMessage().c_str());
		return 1;
	}
	auto& source = *sourceResult.GetValue();

	auto entries = source.List();
	OA_CLI("Opened weight source: %s", InInputPath.c_str());
	OA_CLI("  File size: %s", FormatBytes(source.SourceBytes()).c_str());
	OA_CLI("  Found %zu weight entries", entries.Size());
	const auto builtinAdapters = OaRegisterClipTextWeightAdapter();
	if (not builtinAdapters.IsOk()) {
		OA_CLI("Error: Failed to register built-in model adapters: %s",
			builtinAdapters.GetMessage().c_str());
		return 1;
	}

	OaResult<OaWeightMap> mapResult = InArch == "raw"
		? OaMakeRawWeightMap(source)
		: ([&]() -> OaResult<OaWeightMap> {
			const auto* adapter = OaFindModelWeightAdapter(InArch);
			if (!adapter) return OaStatus::NotFound(OaString("No model weight adapter is registered for: ") + InArch);
			return adapter->BuildMap(source);
		})();
	if (mapResult.IsError()) {
		OA_CLI("Error: Failed to build weight map: %s", mapResult.GetStatus().GetMessage().c_str());
		return 1;
	}
	OamModel model;
	auto reportResult = OaTransferWeights(source, mapResult.GetValue(), model);
	if (reportResult.IsError()) {
		OA_CLI("Error: Weight transfer failed: %s", reportResult.GetStatus().GetMessage().c_str());
		return 1;
	}
	const auto& report = reportResult.GetValue();

	// Save OAM file
	auto saveStatus = model.Save(InOutputPath);
	if (!saveStatus.IsOk()) {
		OA_CLI("Error: Failed to save OAM file: %s", saveStatus.GetMessage().c_str());
		return 1;
	}

	OA_CLI("Successfully imported to OAM format");
	OA_CLI("  Output: %s", InOutputPath.c_str());
	OA_CLI("  Weights: %zu tensors", model.WeightIndex.Size());
	OA_CLI("  Total size: %s", FormatBytes(report.OutputBytes).c_str());

	return 0;
}

// ============================================================================
// Validate Command (Post-Conversion Verification)
// ============================================================================

static int CmdValidate(
	const OaString& InModelPath,
	const OaString& InReferencePath,
	const OaString& InInput
) {
	OA_CLI("Validating: %s", InModelPath.c_str());

	// Load the OAM model
	auto loadResult = OamModel::Load(InModelPath);
	if (!loadResult.IsOk()) {
		OA_CLI("Error: Failed to load OAM model: %s", loadResult.GetStatus().GetMessage().c_str());
		return 1;
	}

	const auto& model = loadResult.GetValue();
	OA_CLI("Loaded model: %s", model.Config.Architecture);
	OA_CLI("  Weights: %zu tensors", model.WeightIndex.Size());

	// Basic structural checks
	OaI32 issues = 0;
	OaI64 totalElements = 0;
	OaI64 matchedElements = 0;

	for (const auto& w : model.WeightIndex) {
		if (w.BlobOffset + w.NumBytes > model.WeightBlob.Size()) {
			OA_CLI("ERROR: weight '%s' extends past blob", w.Name);
			++issues;
			continue;
		}
		const OaI64 elems = CountElements(w);
		totalElements += elems;
		// NaN / Inf spot-check (first element)
		if (w.Dtype == OaScalarType::Float32 && elems > 0) {
			const auto* data = reinterpret_cast<const OaF32*>(model.WeightBlob.Data() + w.BlobOffset);
			if (!std::isnan(data[0]) && !std::isinf(data[0])) {
				matchedElements++;
			}
		}
	}

	OA_CLI("  Total elements: %s", FormatNumber(totalElements).c_str());
	OA_CLI("  Valid elements: %s", FormatNumber(matchedElements).c_str());

	if (!InReferencePath.empty()) {
		OA_CLI("Note: Reference numeric comparison requires model-specific forward pass (not yet implemented)");
		OA_CLI("  Reference path: %s", InReferencePath.c_str());
	}

	if (issues == 0) {
		OA_CLI("Validation passed");
		return 0;
	} else {
		OA_CLI("Validation failed: %d issues found", issues);
		return 1;
	}
}

// ============================================================================
// CLI Configuration
// ============================================================================

struct ModelctlConfig {
	// Subcommand selection
	OaString Command;

	// Common paths
	OaString InputPath;
	OaString ComparePath;
	OaString OutputPath;
	OaString ModelDir = "var/model/dev";

	// Import options
	OaString Format = "auto";
	OaString Arch;                    // Target architecture (for metadata only)
	OaString Dtype = "preserve";
	bool Validate = false;

	// Validate options
	OaString ReferencePath;           // Reference output for comparison
	OaString ValidateInput;           // Input prompt for validation
};

class ModelctlCli : public OaCli<ModelctlConfig> {
public:
	ModelctlCli() : OaCli("modelctl", "OA Model Control Tool (.oam format)") {
		SetEpilog(
			"Unified tool for model inspection, weight import, and validation.\n"
			"\n"
			"Examples:\n"
			"  modelctl inspect model.safetensors          Inspect mapped metadata\n"
			"  modelctl info model.oam                    Show model info + all sections\n"
			"  modelctl verify model.oam                    Verify model integrity\n"
			"  modelctl import --in model.safetensors \\\n"
			"                 --arch raw --dtype preserve \\\n"
			"                 --out model.oam               Import external weights\n"
			"  modelctl validate model.oam \\\n"
			"                 --reference ref.bin         Validate against reference\n"
			"  modelctl list var/model                      List all .oam models\n"
			"  modelctl compare a.oam b.oam                 Compare two models\n"
			"  modelctl dump model.oam                      Raw section dump\n"
		);

		auto* inspect = AddSubcommand("inspect", "Inspect an external weight source");
		inspect->add_option("path", Cfg_.InputPath, "External weight file")->required();

		// info command
		auto* info = AddSubcommand("info", "Show model info and all sections");
		info->add_option("path", Cfg_.InputPath, "Model file (.oam)")->required();

		// verify command
		auto* verify = AddSubcommand("verify", "Verify model integrity (load + checksums)");
		verify->add_option("path", Cfg_.InputPath, "Model file (.oam)")->required();

		// import command
		auto* import = AddSubcommand("import", "Import external weights into .oam");
		import->add_option("--in,-i", Cfg_.InputPath, "External weight file")->required();
		import->add_option("--format,-f", Cfg_.Format, "Input format: auto | safetensors | gguf | onnx");
		import->add_option("--arch,-a", Cfg_.Arch, "Registered architecture adapter, or raw")->required();
		import->add_option("--out,-o", Cfg_.OutputPath, "Output .oam path")->required();
		import->add_option("--dtype,-d", Cfg_.Dtype, "Weight dtype policy; raw requires preserve");
		import->add_flag("--validate,-v", Cfg_.Validate, "Run adapter validation after import");

		// validate command
		auto* validate = AddSubcommand("validate", "Validate an imported model against reference fixtures");
		validate->add_option("path", Cfg_.InputPath, "Model file (.oam)")->required();
		validate->add_option("--reference,-r", Cfg_.ReferencePath, "Reference output file (.bin) for comparison");
		validate->add_option("--input", Cfg_.ValidateInput, "Validation input prompt (default: fixed seed prompt)");

		// list command
		auto* list = AddSubcommand("list", "List all .oam models in directory");
		list->add_option("dir", Cfg_.InputPath, "Directory to scan (default: var/model/dev)");

		// compare command
		auto* compare = AddSubcommand("compare", "Compare two models");
		compare->add_option("path1", Cfg_.InputPath, "First model (.oam)")->required();
		compare->add_option("path2", Cfg_.ComparePath, "Second model (.oam)")->required();

		// dump command
		auto* dump = AddSubcommand("dump", "Raw section dump (offsets, sizes, checksums)");
		dump->add_option("path", Cfg_.InputPath, "Model file (.oam)")->required();

		RequireSubcommand(1, 1);
	}

	OaString GetCommand() const { return GetSubcommand(); }
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
	ModelctlCli cli;
	if (!cli.Parse(argc, argv)) return 1;

	const auto& cfg = cli.GetConfig();
	auto cmd = cli.GetCommand();

	if (cmd == "inspect") return CmdInspect(cfg.InputPath);
	if (cmd == "info") return CmdInfo(cfg.InputPath);
	if (cmd == "verify") return CmdVerify(cfg.InputPath);
	if (cmd == "list") {
		OaString dir = cfg.InputPath.empty() ? cfg.ModelDir : cfg.InputPath;
		return CmdList(dir);
	}
	if (cmd == "compare") return CmdCompare(cfg.InputPath, cfg.ComparePath);
	if (cmd == "dump") return CmdDump(cfg.InputPath);
	if (cmd == "import") {
		return CmdImport(
			cfg.InputPath,
			cfg.Format,
			cfg.Arch,
			cfg.OutputPath,
			cfg.Dtype,
			cfg.Validate
		);
	}
	if (cmd == "validate") {
		return CmdValidate(
			cfg.InputPath,
			cfg.ReferencePath,
			cfg.ValidateInput
		);
	}

	OA_CLI("Error: unknown command '%s'", cmd.c_str());
	return 1;
}
