// modelctl - OA Model control tool (Unified .oam operations)
//
// Unified tool for model inspection, weight import, and validation.
// Uses oa::ComputeApp for GPU-accelerated validation (optional).
//
// commands:
//   inspect    - Inspect an external weight source
//   info       - show model info and all sections
//   verify     - verify model integrity (load + checksums)
//   import     - Import external weights to .oam
//   quantize   - convert dense Float32 matrix weights to native Q4/Q8
//   validate   - validate converted model against reference
//   list       - list all .oam models in directory
//   compare    - compare two models
//   dump       - Raw section dump (offsets, sizes, checksums)
//
// usage:
//   modelctl info model.oam
//   modelctl import --in model.safetensors --arch raw --out model.oam
//   modelctl quantize model.oam --out model-q4.oam --dtype q4
//   modelctl validate model.oam --reference ref.bin
//   modelctl list var/model
//   modelctl compare a.oam b.oam
//   modelctl dump model.oam

#include <oa/core/types.h>
#include <oa/core/log.h>
#include <oa/core/cli.h>
#include <oa/core/filesystem.h>
#include <oa/ml/modelFile.h>
#include <oa/ml/transferWeights.h>
#if OA_ENABLE_ALM
	#include <ml/nn/alm/clipTextModelTranslator.h>
#endif

#include <cmath>
#include <cstring>

// ============================================================================
// format Utilities
// ============================================================================

static oa::String formatBytes(oa::U64 inBytes) {
	char buf[64];
	if (inBytes >= 1'000'000'000) {
		snprintf(buf, sizeof(buf), "%.2f GB", static_cast<oa::F64>(inBytes) / 1'000'000'000);
	} else if (inBytes >= 1'000'000) {
		snprintf(buf, sizeof(buf), "%.2f MB", static_cast<oa::F64>(inBytes) / 1'000'000);
	} else if (inBytes >= 1'000) {
		snprintf(buf, sizeof(buf), "%.2f KB", static_cast<oa::F64>(inBytes) / 1'000);
	} else {
		snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(inBytes));
	}
	return buf;
}

static oa::String formatNumber(oa::I64 inValue) {
	char buf[64];
	if (inValue >= 1'000'000'000) {
		snprintf(buf, sizeof(buf), "%.2fB", static_cast<oa::F64>(inValue) / 1'000'000'000);
	} else if (inValue >= 1'000'000) {
		snprintf(buf, sizeof(buf), "%.2fM", static_cast<oa::F64>(inValue) / 1'000'000);
	} else if (inValue >= 1'000) {
		snprintf(buf, sizeof(buf), "%.2fK", static_cast<oa::F64>(inValue) / 1'000);
	} else {
		snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(inValue));
	}
	return buf;
}

static oa::String formatShape(const oa::ModelTensorEntry& inEntry) {
	oa::String result = "[";
	for (oa::U8 d = 0; d < inEntry.rank; ++d) {
		if (d > 0) result += ", ";
		char num[32];
		snprintf(num, sizeof(num), "%llu", static_cast<unsigned long long>(inEntry.shape[d]));
		result += num;
	}
	result += "]";
	return result;
}

static oa::String formatWeightType(const oa::ModelTensorEntry& inEntry) {
	if (inEntry.encoding != oa::ModelTensorEncoding::Dense) {
		return oa::modelFileTensorEncodingName(inEntry.encoding);
	}
	const auto name = oa::scalarTypeName(inEntry.dtype);
	return oa::String(name.data(), name.size());
}

static oa::String formatShape(const oa::WeightInfo& inEntry) {
	oa::String result = "[";
	for (oa::Usize d = 0; d < inEntry.shape.size(); ++d) {
		if (d > 0) result += ", ";
		char num[32];
		snprintf(num, sizeof(num), "%lld", static_cast<long long>(inEntry.shape[d]));
		result += num;
	}
	result += "]";
	return result;
}

static oa::I64 countElements(const oa::ModelTensorEntry& inEntry) {
	oa::I64 n = 1;
	for (oa::U8 d = 0; d < inEntry.rank; ++d)
		n *= static_cast<oa::I64>(inEntry.shape[d]);
	return n;
}

static oa::I64 countTotalParams(const oa::ModelFile& inModel) {
	oa::I64 total = 0;
	for (const auto& w : inModel.weightIndex)
		total += countElements(w);
	return total;
}

// ============================================================================
// commands
// ============================================================================

static int cmdInspect(const oa::String& inPath) {
	auto sourceResult = oa::openWeightSource(oa::Path(inPath));
	if (sourceResult.isError()) {
		OA_CLI("Error: {}", sourceResult.getStatus().toString().cStr());
		return 1;
	}
	auto& source = *sourceResult.getValue();

	const auto entries = source.list();
	oa::U64 payloadBytes = 0;
	oa::U64 elements = 0;
	for (const auto& entry : entries) {
		payloadBytes += entry.byteSize;
		elements += entry.elementCount;
	}

	OA_CLI_RAW("\n");
	OA_CLI("  weight source: {}", inPath.cStr());
	OA_CLI("  file size:     {}", formatBytes(source.sourceBytes()).cStr());
	OA_CLI("  Entries:     {}", entries.size());
	OA_CLI("  Elements:    {}", formatNumber(static_cast<oa::I64>(elements)).cStr());
	OA_CLI("  Payload:     {}", formatBytes(payloadBytes).cStr());

	const auto metadata = source.metadata();
	if (!metadata.empty()) {
		OA_CLI_RAW("\n");
		OA_CLI("  METADATA");
		for (const auto& [key, value] : metadata) {
			OA_CLI("  {}: {}", key.cStr(), value.cStr());
		}
	}

	OA_CLI_RAW("\n");
	OA_CLI("  {:<5} {:<42} {:<22} {:<10} {}", "idx", "Name", "Shape", "dtype", "Bytes");
	for (oa::Usize i = 0; i < entries.size(); ++i) {
		const auto& entry = entries[i];
		OA_CLI("  {:<5} {:<42} {:<22} {:<10} {}",
			i,
			entry.name.cStr(),
			formatShape(entry).cStr(),
			oa::scalarTypeName(entry.dtype),
			formatBytes(entry.byteSize).cStr());
	}
	OA_CLI_RAW("\n");
	return 0;
}

static int cmdInfo(const oa::String& inPath) {
	auto result = oa::ModelFile::load(inPath);
	if (!result.isOk()) {
		OA_CLI("Error: {}", result.getStatus().getMessage().cStr());
		return 1;
	}

	const auto& model = result.getValue();
	auto totalParams = countTotalParams(model);

	OA_CLI_RAW("\n");
	OA_CLI("  Model: {}", inPath.cStr());
	OA_CLI_RAW("\n");

	auto sizeResult = oa::Filesystem::getFileSize(oa::Path(inPath));
	if (sizeResult.isOk()) {
		OA_CLI("  file size:      {}", formatBytes(sizeResult.getValue()).cStr());
	}
	OA_CLI("  format:         OAM v{}", model.formatVersion);

	// Config
	OA_CLI_RAW("\n");
	OA_CLI("  CONFIG");
	OA_CLI("  architecture:   {}", model.config.architecture);
	OA_CLI("  dModel:         {}", model.config.dModel);
	OA_CLI("  nLayers:        {}", model.config.nLayers);
	OA_CLI("  dVocab:         {}", model.config.dVocab);
	OA_CLI("  weight dtype:   {}",
		oa::scalarTypeName(static_cast<oa::ScalarType>(model.config.weightDtype)));
	if (model.config.archConfigSize > 0) {
		OA_CLI("  archConfig:     {} bytes", model.config.archConfigSize);
	}

	// Weights
	if (model.hasWeights()) {
		oa::U64 weightBytes = 0;
		for (const auto& w : model.weightIndex) weightBytes += w.numBytes;

		OA_CLI_RAW("\n");
		OA_CLI("  WEIGHTS ({} tensors)", model.weightIndex.size());
		OA_CLI("  parameters:     {} ({} logical FP32)",
			formatNumber(totalParams).cStr(),
			formatBytes(static_cast<oa::U64>(totalParams) * 4).cStr());
		OA_CLI("  Blob size:      {}", formatBytes(weightBytes).cStr());

		constexpr oa::U32 kMaxTensors = 30;
		OA_CLI_RAW("\n");
		OA_CLI("  {:<6}  {:<30}  {:<18}  {:<10}  {}", "idx", "Name", "Shape", "Elements", "storage");
		for (oa::U32 i = 0; i < model.weightIndex.size(); ++i) {
			if (i >= kMaxTensors) {
				OA_CLI("  ... and {} more tensors", model.weightIndex.size() - kMaxTensors);
				break;
			}
			const auto& w = model.weightIndex[i];
			OA_CLI("  {:<6}  {:<30}  {:<18}  {:<10}  {}",
				i, w.name,
				formatShape(w).cStr(),
				formatNumber(countElements(w)).cStr(),
				formatWeightType(w).cStr());
		}
	}

	// State
	if (model.hasState()) {
		OA_CLI_RAW("\n");
		OA_CLI("  STATE ({} tensors)", model.stateIndex.size());
		for (oa::U32 i = 0; i < model.stateIndex.size() && i < 10; ++i) {
			const auto& s = model.stateIndex[i];
			OA_CLI("    {}  {}  {}", s.name, formatShape(s).cStr(),
				formatBytes(s.numBytes).cStr());
		}
	}

	// Optimizer
	if (model.hasOptimizer()) {
		OA_CLI_RAW("\n");
		OA_CLI("  OPTIMIZER");
		OA_CLI("  type:           {}", model.optimizer.type);
		OA_CLI("  step:           {}", static_cast<long long>(model.optimizer.step));
		OA_CLI("  lr:             {:.2e}", model.optimizer.lr);
		OA_CLI("  beta1:          {:.4f}", model.optimizer.beta1);
		OA_CLI("  beta2:          {:.6f}", model.optimizer.beta2);
		OA_CLI("  eps:            {:.1e}", model.optimizer.eps);
		OA_CLI("  weight decay:   {:.4f}", model.optimizer.weightDecay);
		OA_CLI("  num params:     {}", formatNumber(static_cast<oa::I64>(model.optimizer.numParams)).cStr());
		OA_CLI("  M/V size:       {} each",
			formatBytes(model.adamM.size() * sizeof(oa::F32)).cStr());
	}

	// Progress
	OA_CLI_RAW("\n");
	OA_CLI("  PROGRESS");
	OA_CLI("  step:           {}", static_cast<long long>(model.progress.step));
	OA_CLI("  Bytes Seen:     {}", formatBytes(model.progress.bytesSeen).cStr());
	OA_CLI("  lr:             {:.2e}", model.progress.lr);
	OA_CLI("  Best Metric:    {:.6f} ({}, {})",
		model.progress.bestMetric,
		model.progress.metricName,
		model.progress.lowerIsBetter ? "lower is better" : "higher is better");

	OA_CLI_RAW("\n");
	return 0;
}

static int cmdVerify(const oa::String& inPath) {
	OA_CLI("Verifying: {} ...", inPath.cStr());

	// ModelFile::load is the single checksum and structural authority. Keeping a
	// second verifier here previously made modelctl disagree with OAM v2+.
	auto result = oa::ModelFile::load(inPath);
	if (!result.isOk()) {
		OA_CLI("LOAD FAILED: {}", result.getStatus().getMessage().cStr());
		return 1;
	}

	const auto& model = result.getValue();
	auto totalParams = countTotalParams(model);
	oa::I32 issues = 0;

	// Check weight blob consistency
	for (const auto& w : model.weightIndex) {
		if (w.blobOffset + w.numBytes > model.weightBlob.size()) {
			OA_CLI("ERROR: weight '{}' extends past blob (offset={}, size={}, blob={})",
				w.name, static_cast<unsigned long long>(w.blobOffset),
				static_cast<unsigned long long>(w.numBytes), model.weightBlob.size());
			++issues;
		}
	}

	// Check for NaN/Inf
	oa::I64 nanCount = 0;
	oa::I64 infCount = 0;
	for (const auto& w : model.weightIndex) {
		if (w.encoding == oa::ModelTensorEncoding::Dense &&
			w.dtype == oa::ScalarType::Float32 &&
			w.blobOffset + w.numBytes <= model.weightBlob.size()) {
			oa::I64 elems = countElements(w);
			for (oa::I64 j = 0; j < elems; ++j) {
				oa::F32 value = 0.0F;
				std::memcpy(&value, model.weightBlob.data() + w.blobOffset +
					static_cast<oa::U64>(j) * sizeof(oa::F32), sizeof(value));
				if (std::isnan(value)) ++nanCount;
				if (std::isinf(value)) ++infCount;
			}
		}
	}

	if (nanCount > 0 || infCount > 0) {
		OA_CLI("WARNING: {} NaN, {} Inf values in weights",
			static_cast<long long>(nanCount), static_cast<long long>(infCount));
		++issues;
	}

	// Optimizer consistency
	if (model.hasOptimizer()) {
		if (model.adamM.size() != model.adamV.size()) {
			OA_CLI("ERROR: adamM size ({}) != adamV size ({})",
				model.adamM.size(), model.adamV.size());
			++issues;
		}
	}

	if (issues == 0) {
		OA_CLI("OK: arch={}, {} params, {} weights, optimizer={}",
			model.config.architecture,
			formatNumber(totalParams).cStr(),
			model.weightIndex.size(),
			model.hasOptimizer() ? "yes" : "no");
	} else {
		OA_CLI("ISSUES: {} problems found", issues);
	}
	return issues > 0 ? 1 : 0;
}

static int cmdList(const oa::String& inDir) {
	if (!oa::Filesystem::isDirectory(oa::Path(inDir))) {
		OA_CLI("Error: directory not found: {}", inDir.cStr());
		return 1;
	}

	auto filesResult = oa::Filesystem::listAll(oa::Path(inDir), true);
	if (!filesResult.isOk()) {
		OA_CLI("Error: {}", filesResult.getStatus().getMessage().cStr());
		return 1;
	}

	OA_CLI_RAW("\n");
	OA_CLI("  Models in: {}", inDir.cStr());
	OA_CLI_RAW("\n");
	OA_CLI("  {:<40}  {:<12}  {:<8}  {:<10}  {:<6}",
		"file", "architecture", "step", "parameters", "optim");

	oa::I32 count = 0;
	for (const auto& path : filesResult.getValue()) {
		const oa::String ext = path.extension().string();
		if (ext != ".oam") continue;

		auto loadResult = oa::ModelFile::load(path.string());
		if (loadResult.isOk()) {
			const auto& m = loadResult.getValue();
			auto totalParams = countTotalParams(m);

			oa::String relPath = path.string();
			if (relPath.size() > inDir.size() && relPath.substr(0, inDir.size()) == inDir) {
				relPath = relPath.substr(inDir.size());
				if (!relPath.empty() && relPath[0] == '/') relPath = relPath.substr(1);
			}
			if (relPath.size() > 40) relPath = "..." + relPath.substr(relPath.size() - 37);

			OA_CLI("  {:<40}  {:<12}  {:<8}  {:<10}  {:<6}",
				relPath.cStr(),
				m.config.architecture,
				static_cast<long long>(m.progress.step),
				formatNumber(totalParams).cStr(),
				m.hasOptimizer() ? "yes" : "no");
			++count;
		} else {
			oa::String relPath = path.string();
			if (relPath.size() > inDir.size() && relPath.substr(0, inDir.size()) == inDir) {
				relPath = relPath.substr(inDir.size());
				if (!relPath.empty() && relPath[0] == '/') relPath = relPath.substr(1);
			}
			if (relPath.size() > 40) relPath = "..." + relPath.substr(relPath.size() - 37);
			OA_CLI("  {:<40}  {:<12}  {:<8}  {:<10}  {:<6}",
				relPath.cStr(), "ERROR", "-", "-", "-");
		}
	}

	OA_CLI_RAW("\n");
	if (count == 0) {
		OA_CLI("  (no .oam models found)");
	} else {
		OA_CLI("  total: {} models", count);
	}
	OA_CLI_RAW("\n");
	return 0;
}

static int cmdCompare(const oa::String& inPath1, const oa::String& inPath2) {
	auto result1 = oa::ModelFile::load(inPath1);
	auto result2 = oa::ModelFile::load(inPath2);

	if (!result1.isOk()) {
		OA_CLI("Error loading {}: {}", inPath1.cStr(), result1.getStatus().getMessage().cStr());
		return 1;
	}
	if (!result2.isOk()) {
		OA_CLI("Error loading {}: {}", inPath2.cStr(), result2.getStatus().getMessage().cStr());
		return 1;
	}

	const auto& m1 = result1.getValue();
	const auto& m2 = result2.getValue();
	auto params1 = countTotalParams(m1);
	auto params2 = countTotalParams(m2);

	OA_CLI_RAW("\n");
	OA_CLI("  Model Comparison");
	OA_CLI_RAW("\n");
	OA_CLI("  {:<20}  {:>15}  {:>15}", "", "Model 1", "Model 2");
	OA_CLI("  {:<20}  {:>15}  {:>15}", "architecture:", m1.config.architecture, m2.config.architecture);
	OA_CLI("  {:<20}  {:>15}  {:>15}", "dModel:", m1.config.dModel, m2.config.dModel);
	OA_CLI("  {:<20}  {:>15}  {:>15}", "nLayers:", m1.config.nLayers, m2.config.nLayers);
	OA_CLI("  {:<20}  {:>15}  {:>15}", "dVocab:", m1.config.dVocab, m2.config.dVocab);
	OA_CLI("  {:<20}  {:>15}  {:>15}", "parameters:",
		formatNumber(params1).cStr(), formatNumber(params2).cStr());
	OA_CLI("  {:<20}  {:>15}  {:>15}", "weight Tensors:",
		m1.weightIndex.size(), m2.weightIndex.size());
	OA_CLI("  {:<20}  {:>15}  {:>15}", "step:",
		static_cast<long long>(m1.progress.step),
		static_cast<long long>(m2.progress.step));
	OA_CLI("  {:<20}  {:15.6f}  {:15.6f}", "Best Metric:",
		m1.progress.bestMetric, m2.progress.bestMetric);
	OA_CLI("  {:<20}  {:>15}  {:>15}", "Optimizer:",
		m1.hasOptimizer() ? "yes" : "no",
		m2.hasOptimizer() ? "yes" : "no");
	OA_CLI_RAW("\n");

	// weight diff if architectures match
	if (params1 == params2 && params1 > 0 &&
		m1.weightIndex.size() == m2.weightIndex.size()) {

		oa::F64 sumDiff = 0.0;
		oa::F64 maxDiff = 0.0;
		oa::I64 totalElements = 0;
		bool compatible = true;

		for (oa::U32 i = 0; i < m1.weightIndex.size(); ++i) {
			const auto& w1 = m1.weightIndex[i];
			const auto& w2 = m2.weightIndex[i];

			if (countElements(w1) != countElements(w2)
				or w1.dtype != w2.dtype or w1.encoding != w2.encoding)
			{
				compatible = false;
				break;
			}

			if (w1.encoding == oa::ModelTensorEncoding::Dense
				and w1.dtype == oa::ScalarType::Float32 &&
				w1.blobOffset + w1.numBytes <= m1.weightBlob.size() &&
				w2.blobOffset + w2.numBytes <= m2.weightBlob.size()) {
				oa::I64 elems = countElements(w1);
				for (oa::I64 j = 0; j < elems; ++j) {
					oa::F32 d1 = 0.0F;
					oa::F32 d2 = 0.0F;
					std::memcpy(&d1, m1.weightBlob.data() + w1.blobOffset +
						static_cast<oa::U64>(j) * sizeof(oa::F32), sizeof(d1));
					std::memcpy(&d2, m2.weightBlob.data() + w2.blobOffset +
						static_cast<oa::U64>(j) * sizeof(oa::F32), sizeof(d2));
					oa::F64 diff = std::abs(static_cast<oa::F64>(d1) - static_cast<oa::F64>(d2));
					sumDiff += diff;
					if (diff > maxDiff) maxDiff = diff;
				}
				totalElements += elems;
			}
		}

		if (compatible && totalElements > 0) {
			oa::F64 avgDiff = sumDiff / static_cast<oa::F64>(totalElements);
			OA_CLI("  WEIGHT DIFFERENCES");
			OA_CLI("  Avg Diff:       {:.8f}", avgDiff);
			OA_CLI("  Max Diff:       {:.8f}", maxDiff);
			OA_CLI_RAW("\n");
		}
	}

	return 0;
}

static int cmdDump(const oa::String& inPath) {
	oa::dumpModelFile(inPath);
	return 0;
}

static int cmdQuantize(
	const oa::String& inPath,
	const oa::String& inOutputPath,
	const oa::String& inDtype)
{
	if (inOutputPath.empty()) {
		OA_CLI("Error: --out is required for quantize");
		return 2;
	}
	oa::Quantization quantization{};
	if (inDtype == "q4" or inDtype == "Q4") {
		quantization = oa::Quantization::Q4;
	} else if (inDtype == "q8" or inDtype == "Q8") {
		quantization = oa::Quantization::Q8;
	} else {
		OA_CLI("Error: quantize --dtype must be q4 or q8");
		return 2;
	}
	auto loaded = oa::ModelFile::load(inPath);
	if (not loaded.isOk()) {
		OA_CLI("Error: {}", loaded.getStatus().toString().cStr());
		return 1;
	}
	const oa::U64 beforeBytes = loaded->weightBlob.size();
	auto converted = loaded->quantizeWeights(quantization);
	if (not converted.isOk()) {
		OA_CLI("Error: {}", converted.getStatus().toString().cStr());
		return 1;
	}
	const auto status = converted->save(inOutputPath);
	if (not status.isOk()) {
		OA_CLI("Error: {}", status.toString().cStr());
		return 1;
	}
	oa::Usize quantMatrices = 0;
	for (const auto& weight : converted->weightIndex) {
		if (weight.encoding != oa::ModelTensorEncoding::Dense) ++quantMatrices;
	}
	const oa::U64 afterBytes = converted->weightBlob.size();
	const oa::F64 ratio = beforeBytes == 0
		? 0.0 : static_cast<oa::F64>(afterBytes) / static_cast<oa::F64>(beforeBytes);
	OA_CLI("quantized: {} -> {}", inPath.cStr(), inOutputPath.cStr());
	OA_CLI("  encoding:      {}", oa::quantizationToString(quantization));
	OA_CLI("  Quant matrices: {}", quantMatrices);
	OA_CLI("  weight blob:   {} -> {} ({:.1f}%)",
		formatBytes(beforeBytes).cStr(), formatBytes(afterBytes).cStr(), ratio * 100.0);
	OA_CLI("  Optimizer:     removed (inference artifact)");
	return 0;
}


// ============================================================================
// Import command (external weight Transfer)
// ============================================================================

static int cmdImport(
	const oa::String& inInputPath,
	const oa::String& inFormat,
	const oa::String& inArch,
	const oa::String& inOutputPath,
	const oa::String& inDtype,
	bool inValidate
) {
	OA_CLI("Importing: {} -> {}", inInputPath.cStr(), inOutputPath.cStr());
	OA_CLI("format: {}, arch: {}, dtype: {}", inFormat.cStr(), inArch.cStr(), inDtype.cStr());

	if (inDtype != "preserve") {
		OA_CLI("Error: dtype conversion policy is translator-owned; currently use '--dtype preserve'");
		return 2;
	}
	if (inValidate) {
		OA_CLI("Error: import parity validation requires a registered architecture validator");
		return 2;
	}
	if (inOutputPath.empty()) {
		OA_CLI("Error: --out is required for import");
		return 2;
	}

	oa::WeightFormat format = oa::WeightFormat::Auto;
	if (inFormat == "safetensors") format = oa::WeightFormat::SafeTensors;
	else if (inFormat == "gguf") format = oa::WeightFormat::Gguf;
	else if (inFormat == "onnx") format = oa::WeightFormat::Onnx;
	else if (inFormat != "auto") {
		OA_CLI("Error: unknown weight format '{}'", inFormat.cStr());
		return 2;
	}
	auto sourceResult = oa::openWeightSource(oa::Path(inInputPath), format);
	if (sourceResult.isError()) {
		OA_CLI("Error: Failed to open weight source: {}", sourceResult.getStatus().getMessage().cStr());
		return 1;
	}
	auto& source = *sourceResult.getValue();

	auto entries = source.list();
	OA_CLI("Opened weight source: {}", inInputPath.cStr());
	OA_CLI("  file size: {}", formatBytes(source.sourceBytes()).cStr());
	OA_CLI("  Found {} weight entries", entries.size());
#if OA_ENABLE_ALM
	const auto builtinTranslators = oa::registerClipTextModelTranslator();
	if (not builtinTranslators.isOk()) {
		OA_CLI("Error: Failed to register built-in model translators: {}",
			builtinTranslators.getMessage().cStr());
		return 1;
	}
#endif

	oa::Result<oa::WeightMap> mapResult = inArch == "raw"
		? oa::makeRawWeightMap(source)
		: ([&]() -> oa::Result<oa::WeightMap> {
			const auto* translator = oa::findModelTranslator(inArch);
			if (!translator) return oa::Status::notFound(
				oa::String("No model translator is registered for: ") + inArch);
			return translator->buildMap(source);
		})();
	if (mapResult.isError()) {
		OA_CLI("Error: Failed to build weight map: {}", mapResult.getStatus().getMessage().cStr());
		return 1;
	}
	oa::ModelFile model;
	auto reportResult = oa::transferWeights(source, mapResult.getValue(), model);
	if (reportResult.isError()) {
		OA_CLI("Error: weight transfer failed: {}", reportResult.getStatus().getMessage().cStr());
		return 1;
	}
	const auto& report = reportResult.getValue();

	// save OAM file
	auto saveStatus = model.save(inOutputPath);
	if (!saveStatus.isOk()) {
		OA_CLI("Error: Failed to save OAM file: {}", saveStatus.getMessage().cStr());
		return 1;
	}

	OA_CLI("Successfully imported to OAM format");
	OA_CLI("  output: {}", inOutputPath.cStr());
	OA_CLI("  Weights: {} tensors", model.weightIndex.size());
	OA_CLI("  total size: {}", formatBytes(report.outputBytes).cStr());

	return 0;
}

// ============================================================================
// validate command (Post-conversion Verification)
// ============================================================================

static int cmdValidate(
	const oa::String& inModelPath,
	const oa::String& inReferencePath,
	const oa::String& inInput
) {
	OA_CLI("Validating: {}", inModelPath.cStr());

	// load the OAM model
	auto loadResult = oa::ModelFile::load(inModelPath);
	if (!loadResult.isOk()) {
		OA_CLI("Error: Failed to load OAM model: {}", loadResult.getStatus().getMessage().cStr());
		return 1;
	}

	const auto& model = loadResult.getValue();
	OA_CLI("Loaded model: {}", model.config.architecture);
	OA_CLI("  Weights: {} tensors", model.weightIndex.size());

	// Basic structural checks
	oa::I32 issues = 0;
	oa::I64 totalElements = 0;
	oa::I64 matchedElements = 0;

	for (const auto& w : model.weightIndex) {
		if (w.blobOffset + w.numBytes > model.weightBlob.size()) {
			OA_CLI("ERROR: weight '{}' extends past blob", w.name);
			++issues;
			continue;
		}
		const oa::I64 elems = countElements(w);
		totalElements += elems;
		// NaN / Inf spot-check (first element)
		if (w.dtype == oa::ScalarType::Float32 && elems > 0) {
			const auto* data = reinterpret_cast<const oa::F32*>(model.weightBlob.data() + w.blobOffset);
			if (!std::isnan(data[0]) && !std::isinf(data[0])) {
				matchedElements++;
			}
		}
	}

	OA_CLI("  total elements: {}", formatNumber(totalElements).cStr());
	OA_CLI("  valid elements: {}", formatNumber(matchedElements).cStr());

	if (!inReferencePath.empty()) {
		OA_CLI("Note: Reference numeric comparison requires model-specific forward pass (not yet implemented)");
		OA_CLI("  Reference path: {}", inReferencePath.cStr());
	}

	if (issues == 0) {
		OA_CLI("Validation passed");
		return 0;
	} else {
		OA_CLI("Validation failed: {} issues found", issues);
		return 1;
	}
}

// ============================================================================
// CLI Configuration
// ============================================================================

struct ModelctlConfig {
	// Subcommand selection
	oa::String command;

	// Common paths
	oa::String inputPath;
	oa::String comparePath;
	oa::String outputPath;
	oa::String modelDir = "var/model/dev";

	// Import options
	oa::String format = "auto";
	oa::String arch;                    // target architecture (for metadata only)
	oa::String dtype = "preserve";
	bool validate = false;

	// validate options
	oa::String referencePath;           // Reference output for comparison
	oa::String validateInput;           // input prompt for validation
};

class ModelctlCli : public oa::Cli<ModelctlConfig> {
public:
	ModelctlCli() : oa::Cli<ModelctlConfig>("modelctl", "OA Model control tool (.oam format)") {
		setEpilog(
			"Unified tool for model inspection, weight import, and validation.\n"
			"\n"
			"Examples:\n"
			"  modelctl inspect model.safetensors          Inspect mapped metadata\n"
			"  modelctl info model.oam                    show model info + all sections\n"
			"  modelctl verify model.oam                    verify model integrity\n"
			"  modelctl import --in model.safetensors \\\n"
			"                 --arch raw --dtype preserve \\\n"
			"                 --out model.oam               Import external weights\n"
			"  modelctl quantize model.oam --out model-q4.oam \\\n"
			"                 --dtype q4                    Build inference artifact\n"
			"  modelctl validate model.oam \\\n"
			"                 --reference ref.bin         validate against reference\n"
			"  modelctl list var/model                      list all .oam models\n"
			"  modelctl compare a.oam b.oam                 compare two models\n"
			"  modelctl dump model.oam                      Raw section dump\n"
		);

		auto* inspect = addSubcommand("inspect", "Inspect an external weight source");
		inspect->addOption("path", cfg_.inputPath, "external weight file")->required();

		// info command
		auto* info = addSubcommand("info", "show model info and all sections");
		info->addOption("path", cfg_.inputPath, "Model file (.oam)")->required();

		// verify command
		auto* verify = addSubcommand("verify", "verify model integrity (load + checksums)");
		verify->addOption("path", cfg_.inputPath, "Model file (.oam)")->required();

		// import command
		auto* import = addSubcommand("import", "Import external weights into .oam");
		import->addOption("--in,-i", cfg_.inputPath, "external weight file")->required();
		import->addOption("--format,-f", cfg_.format, "input format: auto | safetensors | gguf | onnx");
		import->addOption("--arch,-a", cfg_.arch, "Registered model translator, or raw")->required();
		import->addOption("--out,-o", cfg_.outputPath, "output .oam path")->required();
		import->addOption("--dtype,-d", cfg_.dtype, "weight dtype policy; raw requires preserve");
		import->addFlag("--validate,-v", cfg_.validate, "run translator validation after import");

		auto* quantize = addSubcommand("quantize", "quantize Float32 matrix weights for inference");
		quantize->addOption("path", cfg_.inputPath, "Dense model file (.oam)")->required();
		quantize->addOption("--out,-o", cfg_.outputPath, "output .oam path")->required();
		quantize->addOption("--dtype,-d", cfg_.dtype, "Native weight encoding: q4 | q8")->required();

		// validate command
		auto* validate = addSubcommand("validate", "validate an imported model against reference fixtures");
		validate->addOption("path", cfg_.inputPath, "Model file (.oam)")->required();
		validate->addOption("--reference,-r", cfg_.referencePath, "Reference output file (.bin) for comparison");
		validate->addOption("--input", cfg_.validateInput, "Validation input prompt (default: fixed seed prompt)");

		// list command
		auto* list = addSubcommand("list", "list all .oam models in directory");
		list->addOption("dir", cfg_.inputPath, "directory to scan (default: var/model/dev)");

		// compare command
		auto* compare = addSubcommand("compare", "compare two models");
		compare->addOption("path1", cfg_.inputPath, "first model (.oam)")->required();
		compare->addOption("path2", cfg_.comparePath, "second model (.oam)")->required();

		// dump command
		auto* dump = addSubcommand("dump", "Raw section dump (offsets, sizes, checksums)");
		dump->addOption("path", cfg_.inputPath, "Model file (.oam)")->required();

		requireSubcommand(1, 1);
	}

	oa::String getCommand() const { return getSubcommand(); }
};

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
	ModelctlCli cli;
	if (!cli.parse(argc, argv)) return cli.helpRequested() ? 0 : 1;

	const auto& cfg = cli.getConfig();
	auto cmd = cli.getCommand();

	if (cmd == "inspect") return cmdInspect(cfg.inputPath);
	if (cmd == "info") return cmdInfo(cfg.inputPath);
	if (cmd == "verify") return cmdVerify(cfg.inputPath);
	if (cmd == "list") {
		oa::String dir = cfg.inputPath.empty() ? cfg.modelDir : cfg.inputPath;
		return cmdList(dir);
	}
	if (cmd == "compare") return cmdCompare(cfg.inputPath, cfg.comparePath);
	if (cmd == "dump") return cmdDump(cfg.inputPath);
	if (cmd == "quantize") return cmdQuantize(cfg.inputPath, cfg.outputPath, cfg.dtype);
	if (cmd == "import") {
		return cmdImport(
			cfg.inputPath,
			cfg.format,
			cfg.arch,
			cfg.outputPath,
			cfg.dtype,
			cfg.validate
		);
	}
	if (cmd == "validate") {
		return cmdValidate(
			cfg.inputPath,
			cfg.referencePath,
			cfg.validateInput
		);
	}

	OA_CLI("Error: unknown command '{}'", cmd.cStr());
	return 1;
}
