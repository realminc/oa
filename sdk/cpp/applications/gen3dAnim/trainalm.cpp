// trainalm — two-stage trainer for the OaAlm motion tokenizer + Transformer LM.
//
// stage 1  train the temporal Conv1d VQ-VAE tokenizer (motion to discrete tokens)
// stage 2  tokenize every train clip, then train the autoregressive Transformer LM
//
// Uses oa::ItTraining with full callback pipeline: metrics, progress bar, summary,
// periodic checkpointing (save-best + save-every). ctrl+C for graceful exit.
//
// usage:
//   trainalm --config var/config/Alm.yaml
//   trainalm --dataset /path/to/Cmp --tok-steps 5000 --lm-steps 5000

#include <oa/runtime/app.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/cli.h>
#include <oa/core/log.h>
#include <oa/core/filesystem.h>
#include <oa/core/paths.h>
#include <oa/core/time.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/ml/optim.h>
#include <oa/ml/fnOptim.h>
#include <oa/ml/fnLoss.h>
#include <ml/fnLoss.h>
#include <oa/ml/itTraining.h>
#include <oa/ml/callbacks.h>
#include <oa/ml/checkpoint.h>
#include <oa/ml/metric.h>
#include <oa/ml/lrScheduler.h>
#include <data/dsHumanMl3d.h>
#include <core/streamText.h>

#include <algorithm>
#include <csignal>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

// Alm model headers (from Examples/Ml/Alm).
#include <ml/nn/alm/almConfig.h>
#include <ml/nn/alm/almAg.h>
#include <ml/nn/alm/almTokenizerAg.h>
#include <ml/nn/alm/almPriorAg.h>

// ── Config ──────────────────────────────────────────────────────────────────

struct PhaseConfig {
	oa::String id;
	oa::I32    epochs       = 0;
	oa::F32    lr           = 0.0F;
	oa::F32    minLrFrac    = 0.10F;
	oa::I32    warmupSteps  = 0;
};

struct AlmTrainConfig {
	oa::String name      = "Alm";
	oa::String dataset   = oa::Paths::data("humanMl3d/Cmp").string();
	oa::String split     = "train";
	oa::String valSplit  = "val";
	oa::I32    maxClips  = 0;
	oa::I32    valBatches = 0;  // 0 = complete held-out split
	oa::String modelDir  = oa::Paths::var("model/dev").string();
	oa::String precisionStr = "fp32";

	[[nodiscard]] oa::Precision precision() const {
		if (precisionStr == "fp32") return oa::Precision::FP32;
		if (precisionStr == "bf16") return oa::Precision::BF16;
		if (precisionStr == "fp64") return oa::Precision::FP64;
		return oa::Precision::FP32;
	}

	// stage selection: "both" (tok then lm), "tok" (tokenizer only),
	// "lm" (load tok, train lm), "export" (assemble saved best stages)
	oa::String stage = "both";

	// tokenizer architecture
	oa::I32 width      = 384;
	oa::I32 codeDim    = 256;
	oa::I32 numCodes   = 512;
	oa::I32 downT      = 2;
	oa::I32 depth      = 3;
	// VQ health: these three prevent codebook collapse.
	// commitBeta=0.02/emaDecay=0.999/deadThresh=1.0 collapsed the codebook to 1 code
	// (perplexity 1.0); 0.25/0.99/2.0 gives perplexity ~73 and 25× lower recon loss.
	oa::F32 commitBeta = 0.25F;
	oa::F32 emaDecay   = 0.99F;
	oa::F32 emaEps     = 1e-5F;
	oa::F32 deadThresh = 2.0F;

	// LM architecture
	oa::I32 dModel    = 384;
	oa::I32 numHeads  = 6;
	oa::I32 numLayers = 6;
	oa::I32 dFfn      = 1536;
	oa::I32 lmMaxSeqLen = 260;
	oa::String lmFfnType = "dense";
	oa::I32 lmMoeExperts = 4;
	oa::I32 lmMoeTopK = 2;
	oa::I32 lmMoeEvery = 2;
	oa::F32 lmMoeBalanceRate = 1e-3F;
	oa::F32 lmMoeAuxAlpha = 0.01F;
	oa::F32 lmMoeZBeta = 1e-3F;
	oa::String textConditioning = "clip";  // clip | none
	oa::String clipTextModel = "var/model/ref/ClipText/ClipText.oam";
	oa::String clipMerges = "var/model/ref/ClipText/merges.txt";

	// training schedule (epoch-based, keras style)
	oa::I32 tokEpochs = 50;
	oa::I32 lmEpochs  = 50;
	oa::I32 batchSize = 32;
	oa::I32 seqLen    = 64;
	oa::I32 lmSeqLen  = 64;
	oa::F32 tokLr     = 2e-4F;
	oa::F32 lmLr      = 1e-4F;
	oa::F32 tokMinLr  = 2e-5F;
	oa::F32 lmMinLr   = 1e-5F;
	oa::I32 tokWarmup = 500;
	oa::I32 lmWarmup  = 300;
	oa::F32 tokWeightDecay = 0.0F;
	oa::F32 lmWeightDecay  = 0.01F;
	oa::I64 seed      = 42;

	// Multiphase schedules (optional; when empty, use flat tok_epochs/lm_epochs)
	oa::Vec<PhaseConfig> tokPhases;
	oa::Vec<PhaseConfig> lmPhases;

	// callbacks
	oa::I64 ckptSaveEvery   = 0;       // 0 = epoch-end only; >0 adds mid-epoch saves
	oa::I32 ckptMaxKeep     = 5;
	bool  ckptRestoreBest = false;
};

class AlmTrainCli : public oa::Cli<AlmTrainConfig> {
public:
	AlmTrainCli()
		: oa::Cli<AlmTrainConfig>(
			"trainalm", "Train OaAlm tokenizer + Transformer LM (two-stage)") {
		addOption("--dataset",    cfg_.dataset,    "CMP dataset directory");
		addOption("--split",      cfg_.split,      "Dataset split");
		addOption("--val-split",  cfg_.valSplit,   "held-out validation split");
		addOption("--max-clips",  cfg_.maxClips,   "Max clips (0 = all)");
		addOption("--val-batches", cfg_.valBatches, "Validation batches per epoch (0=full split)");
		addOption("--model-dir",  cfg_.modelDir,   "checkpoint root");
		addOption("--name",       cfg_.name,       "Model name");
		addOption("--stage",      cfg_.stage,      "training stage: both | tok | lm | export");
		addOption("--width",      cfg_.width,      "tokenizer conv width");
		addOption("--code-dim",   cfg_.codeDim,    "codebook code dimension");
		addOption("--codes",      cfg_.numCodes,   "codebook size K");
		addOption("--down-t",     cfg_.downT,      "Temporal downsample stages");
		addOption("--depth",      cfg_.depth,      "residual blocks per stage");

		addOption("--dmodel",     cfg_.dModel,     "LM model dimension");
		addOption("--lm-heads",   cfg_.numHeads,   "LM attention heads");
		addOption("--lm-layers",  cfg_.numLayers,  "LM Transformer layers");
		addOption("--lm-ffn",     cfg_.dFfn,       "LM FFN hidden dimension");
		addOption("--lm-max-seq-len", cfg_.lmMaxSeqLen, "LM learned-position capacity");
		addOption("--lm-ffn-type", cfg_.lmFfnType, "LM FFN policy: dense | moe | hybrid");
		addOption("--lm-moe-experts", cfg_.lmMoeExperts, "LM MoE expert count");
		addOption("--lm-moe-top-k", cfg_.lmMoeTopK, "LM experts selected per token");
		addOption("--lm-moe-every", cfg_.lmMoeEvery, "Hybrid LM: use MoE every Nth layer");
		addOption("--lm-moe-balance-rate", cfg_.lmMoeBalanceRate, "MoE aux-loss-free routing-bias rate (0 disables)");
		addOption("--lm-moe-aux-alpha", cfg_.lmMoeAuxAlpha, "MoE Switch load-balancing loss coefficient");
		addOption("--lm-moe-z-beta", cfg_.lmMoeZBeta, "MoE router z-loss coefficient");
		addOption("--text-conditioning", cfg_.textConditioning,
			"LM caption conditioning: clip | none");
		addOption("--clip-text-model", cfg_.clipTextModel, "imported native oa::ClipTextAg .oam");
		addOption("--clip-merges", cfg_.clipMerges, "Pinned CLIP merges.txt tokenizer asset");

		addOption("--tok-epochs", cfg_.tokEpochs,  "tokenizer training epochs");
		addOption("--lm-epochs",  cfg_.lmEpochs,   "LM training epochs");
		addOption("--batch",      cfg_.batchSize,  "Batch size");
		addOption("--seq-len",    cfg_.seqLen,     "tokenizer window (frames)");
		addOption("--lm-seq-len", cfg_.lmSeqLen,   "LM token window");
		addOption("--tok-lr",     cfg_.tokLr,      "tokenizer learning rate");
		addOption("--lm-lr",      cfg_.lmLr,       "LM learning rate");
		addOption("--tok-min-lr", cfg_.tokMinLr,   "tokenizer min LR (cosine floor)");
		addOption("--lm-min-lr",  cfg_.lmMinLr,    "LM min LR (cosine floor)");
		addOption("--tok-warmup", cfg_.tokWarmup,  "tokenizer warmup steps");
		addOption("--lm-warmup",  cfg_.lmWarmup,   "LM warmup steps");
		addOption("--tok-wd",     cfg_.tokWeightDecay, "tokenizer weight decay");
		addOption("--lm-wd",      cfg_.lmWeightDecay,  "LM weight decay");
		addOption("--seed",       cfg_.seed,       "RNG seed");
		addOption("--precision",  cfg_.precisionStr, "fp32 | bf16 | fp64");

		addOption("--ckpt-save-every", cfg_.ckptSaveEvery,   "checkpoint interval (0=epoch-end only)");
		addOption("--ckpt-max-keep",  cfg_.ckptMaxKeep,     "Max incremental checkpoints");
		addOption("--ckpt-restore-best", cfg_.ckptRestoreBest, "Restore best weights at train end");
	}

	void loadYaml(const oa::Yaml::Node& inYaml) override {
		cfg_.name = oa::Yaml::get<oa::String>(inYaml, "name", cfg_.name);
		cfg_.stage = oa::Yaml::get<oa::String>(inYaml, "stage", cfg_.stage);

		const oa::Yaml::Node m = inYaml["model"];
		cfg_.width    = oa::Yaml::get<oa::I32>(m, "width",     cfg_.width);
		cfg_.codeDim  = oa::Yaml::get<oa::I32>(m, "code_dim",  cfg_.codeDim);
		cfg_.numCodes = oa::Yaml::get<oa::I32>(m, "num_codes", cfg_.numCodes);
		cfg_.downT    = oa::Yaml::get<oa::I32>(m, "down_t",    cfg_.downT);
		cfg_.depth      = oa::Yaml::get<oa::I32>(m, "depth",       cfg_.depth);
		cfg_.commitBeta = oa::Yaml::get<oa::F32>(m, "commit_beta", cfg_.commitBeta);
		cfg_.emaDecay   = oa::Yaml::get<oa::F32>(m, "ema_decay",   cfg_.emaDecay);
		cfg_.deadThresh = oa::Yaml::get<oa::F32>(m, "dead_thresh", cfg_.deadThresh);
		cfg_.dModel    = oa::Yaml::get<oa::I32>(m, "dmodel",    cfg_.dModel);
		cfg_.numHeads  = oa::Yaml::get<oa::I32>(m, "lm_heads",  cfg_.numHeads);
		cfg_.numLayers = oa::Yaml::get<oa::I32>(m, "lm_layers", cfg_.numLayers);
		cfg_.dFfn      = oa::Yaml::get<oa::I32>(m, "lm_ffn",    cfg_.dFfn);
		cfg_.lmMaxSeqLen = oa::Yaml::get<oa::I32>(m, "lm_max_seq_len", cfg_.lmMaxSeqLen);
		cfg_.lmFfnType = oa::Yaml::get<oa::String>(m, "lm_ffn_type", cfg_.lmFfnType);
		cfg_.lmMoeExperts = oa::Yaml::get<oa::I32>(m, "lm_moe_experts", cfg_.lmMoeExperts);
		cfg_.lmMoeTopK = oa::Yaml::get<oa::I32>(m, "lm_moe_top_k", cfg_.lmMoeTopK);
		cfg_.lmMoeEvery = oa::Yaml::get<oa::I32>(m, "lm_moe_every", cfg_.lmMoeEvery);
		cfg_.lmMoeBalanceRate = oa::Yaml::get<oa::F32>(m, "lm_moe_balance_rate", cfg_.lmMoeBalanceRate);
		cfg_.lmMoeAuxAlpha = oa::Yaml::get<oa::F32>(m, "lm_moe_aux_alpha", cfg_.lmMoeAuxAlpha);
		cfg_.lmMoeZBeta = oa::Yaml::get<oa::F32>(m, "lm_moe_z_beta", cfg_.lmMoeZBeta);
		cfg_.textConditioning = oa::Yaml::get<oa::String>(m, "text_conditioning", cfg_.textConditioning);
		cfg_.clipTextModel = oa::Yaml::get<oa::String>(m, "clip_text_model", cfg_.clipTextModel);
		cfg_.clipMerges = oa::Yaml::get<oa::String>(m, "clip_merges", cfg_.clipMerges);

		const oa::Yaml::Node t = inYaml["training"];
		cfg_.dataset  = oa::Yaml::get<oa::String>(t, "dataset",   cfg_.dataset);
		cfg_.split    = oa::Yaml::get<oa::String>(t, "split",     cfg_.split);
		cfg_.valSplit = oa::Yaml::get<oa::String>(t, "val_split", cfg_.valSplit);
		cfg_.maxClips = oa::Yaml::get<oa::I32>   (t, "max_clips", cfg_.maxClips);
		cfg_.valBatches = oa::Yaml::get<oa::I32> (t, "val_batches", cfg_.valBatches);
		cfg_.modelDir = oa::Yaml::get<oa::String>(t, "model_dir", cfg_.modelDir);
		cfg_.tokEpochs = oa::Yaml::get<oa::I32>   (t, "tok_epochs", cfg_.tokEpochs);
		cfg_.lmEpochs  = oa::Yaml::get<oa::I32>   (t, "lm_epochs",  cfg_.lmEpochs);
		cfg_.batchSize = oa::Yaml::get<oa::I32>  (t, "batch",     cfg_.batchSize);
		cfg_.seqLen   = oa::Yaml::get<oa::I32>   (t, "seq_len",   cfg_.seqLen);
		cfg_.lmSeqLen = oa::Yaml::get<oa::I32>   (t, "lm_seq_len", cfg_.lmSeqLen);
		cfg_.tokLr    = oa::Yaml::get<oa::F32>   (t, "tok_lr",    cfg_.tokLr);
		cfg_.lmLr     = oa::Yaml::get<oa::F32>   (t, "lm_lr",     cfg_.lmLr);
		cfg_.tokMinLr = oa::Yaml::get<oa::F32>   (t, "tok_min_lr", cfg_.tokMinLr);
		cfg_.lmMinLr  = oa::Yaml::get<oa::F32>   (t, "lm_min_lr",  cfg_.lmMinLr);
		cfg_.tokWarmup = oa::Yaml::get<oa::I32>  (t, "tok_warmup", cfg_.tokWarmup);
		cfg_.lmWarmup  = oa::Yaml::get<oa::I32>  (t, "lm_warmup",  cfg_.lmWarmup);
		cfg_.tokWeightDecay = oa::Yaml::get<oa::F32>(t, "tok_weight_decay", cfg_.tokWeightDecay);
		cfg_.lmWeightDecay  = oa::Yaml::get<oa::F32>(t, "lm_weight_decay",  cfg_.lmWeightDecay);
		cfg_.seed     = oa::Yaml::get<oa::I64>   (t, "seed",      cfg_.seed);
		cfg_.precisionStr = oa::Yaml::get<oa::String>(t, "precision", cfg_.precisionStr);

		// parse optional phase sequences
		loadPhases(t, "tok_phases", cfg_.tokPhases, cfg_.tokLr, cfg_.tokMinLr, cfg_.tokWarmup);
		loadPhases(t, "lm_phases",  cfg_.lmPhases,  cfg_.lmLr,  cfg_.lmMinLr,  cfg_.lmWarmup);

		const oa::Yaml::Node cb = inYaml["callbacks"];
		cfg_.ckptSaveEvery   = oa::Yaml::get<oa::I64>(cb, "ckpt_save_every", cfg_.ckptSaveEvery);
		cfg_.ckptMaxKeep     = oa::Yaml::get<oa::I32>(cb, "ckpt_max_keep",    cfg_.ckptMaxKeep);
		cfg_.ckptRestoreBest = oa::Yaml::get<bool> (cb, "ckpt_restore_best", cfg_.ckptRestoreBest);
	}

	static void loadPhases(const oa::Yaml::Node& inTraining, const oa::String& inKey,
		oa::Vec<PhaseConfig>& outPhases, oa::F32 inFallbackLr, oa::F32 inFallbackMinLr, oa::I32 inFallbackWarmup) {
		outPhases.clear();
		const oa::Yaml::Node seq = inTraining[oa::sdk::toStdString(inKey)];
		if (not(seq and seq.IsSequence())) return;
		for (const auto& item : seq) {
			PhaseConfig ph;
			ph.id          = oa::Yaml::get<oa::String>(item, "id", ph.id);
			ph.epochs      = oa::Yaml::get<oa::I32>(item, "epochs", ph.epochs);
			ph.lr          = oa::Yaml::get<oa::F32>(item, "lr", inFallbackLr);
			ph.minLrFrac   = oa::Yaml::get<oa::F32>(item, "min_lr_frac", ph.minLrFrac);
			ph.warmupSteps = oa::Yaml::get<oa::I32>(item, "warmup_steps", inFallbackWarmup);
			if (ph.epochs > 0) outPhases.pushBack(ph);
		}
	}

	static oa::SharedPtr<oa::LRScheduler> buildScheduler(
		const oa::Vec<PhaseConfig>& inPhases, oa::I64 inStepsPerEpoch,
		oa::F32 inFallbackLr, oa::F32 inFallbackMinLr, oa::I32 inFallbackWarmup, oa::I64 inTotalSteps) {
		if (inPhases.empty()) {
			return oa::makeShared<oa::LinearWarmupCosineScheduler>(
				inFallbackWarmup, static_cast<oa::I32>(inTotalSteps), inFallbackLr, inFallbackMinLr);
		}
		oa::Vec<oa::SharedPtr<oa::LRScheduler>> subs;
		oa::Vec<oa::U64> milestones;
		oa::U64 offset = 0;
		for (const auto& ph : inPhases) {
			const oa::I64 phaseSteps = static_cast<oa::I64>(ph.epochs) * inStepsPerEpoch;
			const oa::F32 minLr = ph.lr * ph.minLrFrac;
			subs.pushBack(oa::makeShared<oa::LinearWarmupCosineScheduler>(
				ph.warmupSteps, static_cast<oa::I32>(phaseSteps), ph.lr, minLr));
			offset += static_cast<oa::U64>(phaseSteps);
			milestones.pushBack(offset);
		}
		return oa::makeShared<oa::SequentialScheduler>(oa::move(subs), oa::move(milestones));
	}
};

// ── Host helpers ──────────────────────────────────────────────────────────────

namespace {

class OptimLrMetric final : public oa::Metric {
public:
	explicit OptimLrMetric(const oa::Optimizer& inOpt) : opt_(inOpt) {}
	void update(const oa::Matrix&, const oa::Matrix&) override {}
	void reset() override {}
	[[nodiscard]] oa::F64 result() const override { return opt_.getLr(); }
	[[nodiscard]] const char* name() const override { return "lr"; }
	oa::I32 render(char* outBuffer, oa::I32 inBufferSize, bool) const override {
		char value[96]{};
		formatter_.format(value, sizeof(value), result());
		return std::snprintf(outBuffer, static_cast<size_t>(inBufferSize), "lr: %s", value);
	}
private:
	const oa::Optimizer& opt_;
	mutable oa::MetricValueFormatter formatter_;
};

oa::Matrix makeI32(const std::vector<oa::I32>& h, const oa::MatrixShape& s) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(h.data()), h.size() * sizeof(oa::I32)),
		s, oa::ScalarType::Int32);
}

oa::Matrix makeF32(const std::vector<float>& h, const oa::MatrixShape& s) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(h.data()), h.size() * sizeof(float)),
		s, oa::FnMatrix::weightDtype());
}

std::vector<oa::F32> hostF32(const oa::Matrix& inMatrix) {
	auto& ctx = oa::ExecutionSession::getActive();
	if (inMatrix.getDtype() == oa::ScalarType::Float32) {
		const oa::F32* p = inMatrix.dataAs<const oa::F32>();
		return std::vector<oa::F32>(p, p + inMatrix.numElements());
	}
	oa::Matrix f32 = oa::FnMatrix::empty(inMatrix.getShape(), oa::ScalarType::Float32);
	oa::FnMatrix::castInto(inMatrix, f32);
	if (not ctx.submitAndWait().isOk()) return {};
	const oa::F32* p = f32.dataAs<const oa::F32>();
	return std::vector<oa::F32>(p, p + f32.numElements());
}

oa::Status writeNpyF32Atomic(const oa::Path& inPath, const oa::F32* inData,
	oa::Usize inRows, oa::Usize inCols) {
	if (inData == nullptr or inRows == 0 or inCols == 0)
		return oa::Status::invalidArgument("cannot write an empty CLIP feature array");
	std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
		+ std::to_string(inRows) + ", " + std::to_string(inCols) + "), }";
	const size_t prefixBytes = 10; // magic(6) + version(2) + uint16 header length
	const size_t padding = (64 - ((prefixBytes + header.size() + 1) % 64)) % 64;
	header.append(padding, ' ');
	header.push_back('\n');
	if (header.size() > std::numeric_limits<oa::U16>::max())
		return oa::Status::error(oa::StatusCode::OutOfRange, "NumPy header is too large");
	const oa::Usize payloadBytes = inRows * inCols * sizeof(oa::F32);
	oa::Vec<oa::U8> bytes(prefixBytes + header.size() + payloadBytes);
	const oa::U8 magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
	std::memcpy(bytes.data(), magic, sizeof(magic));
	const oa::U16 headerBytes = static_cast<oa::U16>(header.size());
	bytes[8] = static_cast<oa::U8>(headerBytes & 0xFFU);
	bytes[9] = static_cast<oa::U8>(headerBytes >> 8U);
	std::memcpy(bytes.data() + prefixBytes, header.data(), header.size());
	std::memcpy(bytes.data() + prefixBytes + header.size(), inData, payloadBytes);
	const oa::Path temporary(inPath.string() + ".tmp");
	OA_RETURN_IF_ERROR(oa::Filesystem::writeBinary(temporary,
		oa::Span<const oa::U8>(bytes.data(), bytes.size())));
	if (oa::Filesystem::exists(inPath)) OA_RETURN_IF_ERROR(oa::Filesystem::removeFile(inPath));
	return oa::Filesystem::move(temporary, inPath);
}

oa::Status writeTextAtomic(const oa::Path& inPath, oa::StringView inText) {
	const oa::Path temporary(inPath.string() + ".tmp");
	OA_RETURN_IF_ERROR(oa::Filesystem::writeText(temporary, inText));
	if (oa::Filesystem::exists(inPath)) OA_RETURN_IF_ERROR(oa::Filesystem::removeFile(inPath));
	return oa::Filesystem::move(temporary, inPath);
}

struct Lcg {
	oa::U64 s;
	explicit Lcg(oa::U64 seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
	oa::U32 u32() {
		s = s * 6364136223846793005ULL + 1442695040888963407ULL;
		return static_cast<oa::U32>(s >> 33);
	}
	float f01() { return static_cast<float>(u32()) / static_cast<float>(0xFFFFFFFFU >> 1); }
};

struct CodeUsage { oa::I32 unique; oa::F32 perplexity; };
CodeUsage codebookUsage(const oa::I32* inIds, size_t inCount, oa::I32 inK) {
	std::vector<oa::I64> hist(static_cast<size_t>(inK), 0);
	for (size_t i = 0; i < inCount; ++i) {
		const oa::I32 c = inIds[i];
		if (c >= 0 and c < inK) ++hist[static_cast<size_t>(c)];
	}
	oa::I32 uniq = 0; double ent = 0.0;
	const double n = static_cast<double>(inCount > 0 ? inCount : 1);
	for (oa::I32 k = 0; k < inK; ++k) {
		const oa::I64 h = hist[static_cast<size_t>(k)];
		if (h > 0) { ++uniq; const double p = static_cast<double>(h) / n; ent -= p * std::log(p); }
	}
	return {uniq, static_cast<oa::F32>(std::exp(ent))};
}

struct TokenCorpusStats {
	oa::I64 tokens = 0;
	oa::I32 unique = 0;
	oa::F64 unigramPpl = 0.0;
	oa::F64 bigramPpl = 0.0;
};

struct LmWindow {
	oa::I32 sequence = 0;
	oa::I32 start = 0;       // start in [SOM, motion..., EOM] next-token pairs
	oa::I32 valid = 0;       // valid next-token pairs; remainder is PAD/masked
};

std::vector<LmWindow> buildLmWindows(
	const std::vector<std::vector<oa::I32>>& inSequences, oa::I32 inWindowLen) {
	std::vector<LmWindow> out;
	for (oa::I32 seqIdx = 0; seqIdx < static_cast<oa::I32>(inSequences.size()); ++seqIdx) {
		// [SOM, c0, ..., cN, EOM] has codes+1 next-token pairs.
		const oa::I32 pairs = static_cast<oa::I32>(inSequences[static_cast<size_t>(seqIdx)].size()) + 1;
		if (pairs <= inWindowLen) {
			out.push_back({seqIdx, 0, pairs});
			continue;
		}
		// Preserve the previous all-start-position training density, but never
		// invent SOM/EOM at an interior window boundary.
		for (oa::I32 start = 0; start + inWindowLen <= pairs; ++start) {
			out.push_back({seqIdx, start, inWindowLen});
		}
	}
	return out;
}

void fillLmRow(const std::vector<oa::I32>& inCodes, const LmWindow& inWindow,
	oa::I32 inWindowLen, oa::I32 inSom, oa::I32 inEom, oa::I32 inPad,
	oa::I32* outInput, oa::I32* outTarget, float* outMask) {
	const oa::I32 streamLen = static_cast<oa::I32>(inCodes.size()) + 2;
	auto streamToken = [&](oa::I32 i) -> oa::I32 {
		if (i == 0) return inSom;
		if (i == streamLen - 1) return inEom;
		return inCodes[static_cast<size_t>(i - 1)];
	};
	for (oa::I32 t = 0; t < inWindowLen; ++t) {
		if (t < inWindow.valid) {
			outInput[t] = streamToken(inWindow.start + t);
			outTarget[t] = streamToken(inWindow.start + t + 1);
			outMask[t] = 1.0F;
		} else {
			outInput[t] = inPad;
			outTarget[t] = inPad;
			outMask[t] = 0.0F;
		}
	}
}

TokenCorpusStats measureTokenCorpus(const std::vector<std::vector<oa::I32>>& inSeqs, oa::I32 inK) {
	std::vector<oa::I64> unigram(static_cast<size_t>(inK), 0);
	std::vector<oa::I64> prevCount(static_cast<size_t>(inK), 0);
	std::vector<oa::I64> bigram(static_cast<size_t>(inK) * static_cast<size_t>(inK), 0);
	TokenCorpusStats out;
	oa::I64 pairs = 0;
	for (const auto& seq : inSeqs) {
		oa::I32 prev = -1;
		for (const oa::I32 tok : seq) {
			if (tok < 0 or tok >= inK) { prev = -1; continue; }
			++unigram[static_cast<size_t>(tok)];
			++out.tokens;
			if (prev >= 0) {
				++bigram[static_cast<size_t>(prev) * static_cast<size_t>(inK) + static_cast<size_t>(tok)];
				++prevCount[static_cast<size_t>(prev)];
				++pairs;
			}
			prev = tok;
		}
	}
	oa::F64 h1 = 0.0;
	for (const oa::I64 n : unigram) if (n > 0) {
		++out.unique;
		const oa::F64 p = static_cast<oa::F64>(n) / static_cast<oa::F64>(std::max<oa::I64>(out.tokens, 1));
		h1 -= p * std::log(p);
	}
	oa::F64 h2 = 0.0;
	if (pairs > 0) {
		for (oa::I32 prev = 0; prev < inK; ++prev) {
			const oa::I64 pn = prevCount[static_cast<size_t>(prev)];
			if (pn == 0) continue;
			for (oa::I32 tok = 0; tok < inK; ++tok) {
				const oa::I64 n = bigram[static_cast<size_t>(prev) * static_cast<size_t>(inK) + static_cast<size_t>(tok)];
				if (n == 0) continue;
				const oa::F64 joint = static_cast<oa::F64>(n) / static_cast<oa::F64>(pairs);
				h2 -= joint * std::log(static_cast<oa::F64>(n) / static_cast<oa::F64>(pn));
			}
		}
	}
	out.unigramPpl = std::exp(h1);
	out.bigramPpl = std::exp(h2);
	return out;
}

} // namespace

// ── Signal handling ──────────────────────────────────────────────────────────

static volatile sig_atomic_t gSigIntCount = 0;

static void onSigInt(int) {
	if (gSigIntCount == 0) {
		++gSigIntCount;
	} else {
		_exit(0);
	}
}

// ── tokenizer metrics callback ───────────────────────────────────────────────
//
// Tracks velocity loss, VQ commitment loss, and codebook usage (live codes +
// perplexity) as proper oa::CbTraining callbacks. The step lambda calls record()
// with the GPU matrix refs; after sync in next(), onStepEnd reads the scalar
// values and accumulates token IDs. onEpochEnd prints a TF-style summary:
//
//   epoch 3: codebook 48/64 live | ppl 31.2 | vel 0.023401 | commit 0.008123

class CbTokMetrics : public oa::CbTraining {
public:
	explicit CbTokMetrics(oa::I32 inNumCodes) : numCodes_(inNumCodes) {}

	// Called from step lambda — mirror each tensor into a persistent host-coherent
	// mailbox via a Copy kernel (exactly what oa::ItTraining does for its loss). The
	// transient vel/commit/ids buffers are recycled once the step's graph executes
	// and backward frees the forward activations; storing the raw handles and
	// reading .at(0) later raced with that recycling and returned the still-live
	// recon buffer for all three metrics. The Copy captures the value into a stable
	// buffer that survives to the post-sync read in onStepEnd.
	void record(const oa::Matrix& inVel, const oa::Matrix& inCommit, const oa::Matrix& inIds) {
		auto& ctx = oa::ExecutionSession::getActive();
		oa::BufferAccess rw[] = {oa::BufferAccess::Read, oa::BufferAccess::Write};
		auto mirror = [&](const oa::Matrix& inSrc, oa::Matrix& outDst) {
			if (not inSrc.hasStorage() or inSrc.numElements() == 0) return;
			if (not outDst.hasStorage() or outDst.numElements() != inSrc.numElements()
				or outDst.getDtype() != inSrc.getDtype()) {
				outDst = oa::FnMatrix::zeros(oa::MatrixShape{inSrc.numElements()}, inSrc.getDtype());
			}
			struct { oa::U32 count; } push{static_cast<oa::U32>(inSrc.numElements())};
			ctx.add( "Copy", {&inSrc, &outDst}, rw, &push, sizeof(push),
				(static_cast<oa::U32>(inSrc.numElements()) + 255u) / 256u);
		};
		mirror(inVel, velMailbox_);
		mirror(inCommit, commitMailbox_);
		mirror(inIds, idsMailbox_);
	}

	oa::MetricLoss velMetric{"vel"};
	oa::MetricLoss commitMetric{"commit"};

	[[nodiscard]] oa::Metric* velPtr() { return &velMetric; }
	[[nodiscard]] oa::Metric* commitPtr() { return &commitMetric; }

	void onEpochBegin(oa::ItTraining& inIter) override {
		(void)inIter;
		velMetric.reset();
		commitMetric.reset();
		epochIds_.clear();
	}

	void onStepEnd(oa::ItTraining&) override {
		// The iterator completes the step before callbacks, so every mailbox value
		// is exact and contributes once to the epoch metrics.
		if (velMailbox_.hasStorage()) velMetric.update(velMailbox_.at(0));
		if (commitMailbox_.hasStorage()) commitMetric.update(commitMailbox_.at(0));
		if (idsMailbox_.hasStorage()) {
			const oa::I64 n = idsMailbox_.numElements();
			const oa::I32* p = idsMailbox_.dataAs<const oa::I32>();
			if (p and n > 0) {
				epochIds_.insert(epochIds_.end(), p, p + static_cast<size_t>(n));
			}
		}
	}

	void onEpochEnd(oa::ItTraining& inIter) override {
		if (epochIds_.empty()) return;
		std::vector<oa::I64> hist(static_cast<size_t>(numCodes_), 0);
		for (oa::I32 id : epochIds_) {
			if (id >= 0 and id < numCodes_) ++hist[static_cast<size_t>(id)];
		}
		oa::I32 live = 0;
		oa::F64 entropy = 0.0;
		const oa::F64 total = static_cast<oa::F64>(epochIds_.size());
		for (oa::I32 k = 0; k < numCodes_; ++k) {
			const oa::I64 h = hist[static_cast<size_t>(k)];
			if (h > 0) {
				++live;
				const oa::F64 p = static_cast<oa::F64>(h) / total;
				entropy -= p * std::log(p);
			}
		}
		const oa::F64 perplexity = std::exp(entropy);
		std::printf("epoch %lld: codebook %d/%d live | ppl %.1f | vel %.6f | commit %.6f\n",
			static_cast<long long>(inIter.epoch()), live, numCodes_,
			perplexity, velMetric.result(), commitMetric.result());
	}

private:
	oa::I32 numCodes_;
	oa::Matrix velMailbox_;      // persistent host-coherent mirrors (see record)
	oa::Matrix commitMailbox_;
	oa::Matrix idsMailbox_;
	std::vector<oa::I32> epochIds_;
};

// Single autograd tokenizer path.
struct TokenizerBridge {
	oa::SharedPtr<oa::AlmTokenizerAg> ptr;

	oa::Matrix encode(const oa::Matrix& inX, oa::I32 inB, oa::I32 inT) {
		return ptr->encode(inX, inB, inT);
	}
	oa::ResidualVqResult quantize(const oa::Matrix& inZe) {
		return ptr->quantize(inZe);
	}
	oa::Matrix decode(const oa::Matrix& inZq, oa::I32 inB, oa::I32 inTokLen) {
		return ptr->decode(inZq, inB, inTokLen);
	}
	oa::Vec<oa::Matrix> tokenize(const oa::Matrix& inX, oa::I32 inB, oa::I32 inT) {
		return ptr->tokenize(inX, inB, inT);
	}
	oa::Matrix detokenize(const oa::Vec<oa::Matrix>& inIds, oa::I32 inB, oa::I32 inTokLen) {
		return ptr->detokenize(inIds, inB, inTokLen);
	}
	void emaUpdate(const oa::ResidualVqResult& inResult) {
		ptr->emaUpdate(inResult);
	}
	void seed(const oa::Matrix& inLatents) {
		ptr->seed(inLatents);
	}
	oa::I32 downsampleFactor() const {
		return ptr->downsampleFactor();
	}
	oa::Module& module() {
		return *ptr;
	}
};

// Single configurable Transformer-prior path.
struct LmBridge {
	oa::SharedPtr<oa::AlmPriorAg> ptr;

	oa::Matrix forward(const oa::Matrix& inIds) {
		return ptr->forward(inIds);
	}
	oa::Matrix forward(const oa::Matrix& inIds, const oa::Matrix& inTextFeatures) {
		return ptr->forwardConditioned(inIds, inTextFeatures);
	}
	oa::Matrix generate(oa::I32 inBatchSize, oa::F32 inTemp, oa::I32 inTopK, oa::F32 inTopP, oa::I32 inMaxLen, bool inUseCache) {
		return ptr->generate(inBatchSize, inTemp, inTopK, inTopP, inMaxLen, inUseCache);
	}
	oa::Module& module() {
		return *ptr;
	}
};

// Runs before validation/checkpoint callbacks so the aux-loss-free routing bias
// is updated from the just-completed TRAIN batch, never from a validation forward.
// It also makes collapse visible in every MoE/hybrid epoch.
class CbMoeRouting final : public oa::CbTraining {
public:
	explicit CbMoeRouting(oa::AlmPriorAg& inPrior) : prior_(inPrior) {}

	void onEpochBegin(oa::ItTraining&) override {
		entropySum_ = 0.0;
		maxLoadSum_ = 0.0;
		deadSum_ = 0;
		samples_ = 0;
		layers_ = 0;
	}

	void onStepEnd(oa::ItTraining&) override {
		auto stats = prior_.moeRouteStats();
		for (const auto& s : stats) {
			entropySum_ += s.entropy;
			maxLoadSum_ += s.maxLoadRatio;
			deadSum_ += s.deadExperts;
			++samples_;
		}
		layers_ = static_cast<oa::I32>(stats.size());
		prior_.updateMoeRoutingBias();
	}

	void onEpochEnd(oa::ItTraining&) override {
		if (samples_ == 0) return;
		const auto samples = static_cast<oa::F64>(samples_);
		std::printf("MoE routing: %d layers · entropy %.3f · max-load %.2fx · dead %.2f/layer\n",
			layers_, entropySum_ / samples, maxLoadSum_ / samples,
			static_cast<oa::F64>(deadSum_) / samples);
	}

private:
	oa::AlmPriorAg& prior_;
	oa::F64 entropySum_ = 0.0;
	oa::F64 maxLoadSum_ = 0.0;
	oa::I64 deadSum_ = 0;
	oa::I64 samples_ = 0;
	oa::I32 layers_ = 0;
};

// ── App ─────────────────────────────────────────────────────────────────────

struct TrainAlmApp : oa::ComputeApp {
	AlmTrainCli cli;

	// stage state (kept on the app so tick can drive both stages)
	oa::I32 currentStage_ = 0;  // 0=tokenizer, 1=tokenize, 2=LM, 3=done, 4=export saved stages

	// Dataset
	oa::DsCombatMotionProcessed* ds_ = nullptr;
	oa::DsCombatMotionProcessed* valDs_ = nullptr;
	oa::I32 featDim_ = 0;
	oa::I32 numJoints_ = 0;
	oa::I32 numClips_ = 0;

	// tokenizer
	TokenizerBridge tok_;
	oa::UniquePtr<oa::AdamW> tokOpt_;
	oa::Vec<oa::Parameter*> tokParams_;
	std::vector<std::pair<oa::I32, oa::I32>> tokWindows_;
	std::vector<std::pair<oa::I32, oa::I32>> tokValWindows_;
	size_t tokCursor_ = 0;

	// LM
	LmBridge lm_;
	oa::UniquePtr<oa::AdamW> lmOpt_;
	oa::Vec<oa::Parameter*> lmParams_;
	std::vector<std::vector<oa::I32>> tokenSequences_;
	std::vector<std::vector<oa::I32>> valTokenSequences_;
	std::vector<LmWindow> lmWindows_;
	std::vector<LmWindow> lmValWindows_;
	oa::I32 lmTextFeatureDim_ = 0;

	// oa::ItTraining iterators (one per stage)
	oa::UniquePtr<oa::ItTraining> tokIter_;
	oa::UniquePtr<oa::ItTraining> lmIter_;
	oa::I64 tokStepsPerEpoch_ = 0;
	oa::I64 lmStepsPerEpoch_  = 0;

	// LR schedulers (base-class pointer holds either single or sequential)
	oa::SharedPtr<oa::LRScheduler> tokSched_;
	oa::SharedPtr<oa::LRScheduler> lmSched_;
	oa::UniquePtr<oa::CbLrScheduler> tokSchedCb_;
	oa::UniquePtr<oa::CbLrScheduler> lmSchedCb_;

	// callbacks — tokenizer stage
	oa::UniquePtr<oa::MetricLoss> tokLossMetric_;
	oa::UniquePtr<CbTokMetrics> tokExtraCb_;
	oa::UniquePtr<oa::CbProgressBar> tokBar_;
	oa::UniquePtr<oa::CbValidation> tokValidationCb_;
	oa::UniquePtr<oa::CbSummary>  tokSummary_;
	oa::UniquePtr<oa::CheckpointManager> tokMgr_;
	oa::UniquePtr<oa::CbCheckpoint> tokCkptCb_;

	// callbacks — LM stage
	oa::UniquePtr<oa::MetricLoss> lmLossMetric_;
	oa::UniquePtr<OptimLrMetric> lmLrMetric_;
	oa::UniquePtr<oa::CbProgressBar> lmBar_;
	oa::UniquePtr<CbMoeRouting> lmMoeRoutingCb_;
	oa::UniquePtr<oa::CbValidation> lmValidationCb_;
	oa::UniquePtr<oa::CbSummary>  lmSummary_;
	oa::UniquePtr<oa::CheckpointManager> lmMgr_;
	oa::UniquePtr<oa::CbCheckpoint> lmCkptCb_;

	oa::Bool exitRequested_ = false;

	oa::AlmPriorConfig makeLmConfig(oa::I32 inTextFeatureDim) const {
		const auto& c = cli.getConfig();
		oa::AlmPriorConfig cfg;
		cfg.syncVocab(c.numCodes);
		cfg.dModel = c.dModel;
		cfg.numHeads = c.numHeads;
		cfg.numLayers = c.numLayers;
		cfg.dFfn = c.dFfn;
		cfg.textFeatureDim = inTextFeatureDim;
		cfg.seqLen = c.lmSeqLen + 1;
		cfg.maxSeqLen = c.lmMaxSeqLen;
		if (c.lmFfnType == "dense") cfg.ffnType = oa::AlmFfnType::Dense;
		else if (c.lmFfnType == "moe") cfg.ffnType = oa::AlmFfnType::Moe;
		else if (c.lmFfnType == "hybrid") cfg.ffnType = oa::AlmFfnType::Hybrid;
		cfg.moeNumExperts = c.lmMoeExperts;
		cfg.moeExpertsPerToken = c.lmMoeTopK;
		cfg.moeEvery = c.lmMoeEvery;
		cfg.moeBalanceRate = c.lmMoeBalanceRate;
		cfg.moeAuxLossAlpha = c.lmMoeAuxAlpha;
		cfg.moeRouterZLossBeta = c.lmMoeZBeta;
		return cfg;
	}

	oa::Status saveAlmBundle() {
		const auto& c = cli.getConfig();
		if (not tok_.ptr or not lm_.ptr)
			return oa::Status::invalidArgument("ALM tokenizer and prior must both be loaded");
		const oa::String bundleDir = c.modelDir + "/" + c.name;
		const oa::String bundlePath = bundleDir + "/" + c.name + ".oam";
		(void)oa::Filesystem::createDirectories(oa::Path(bundleDir));
		const oa::String textEncoder = lmTextFeatureDim_ > 0
			? oa::String("openai/clip-vit-large-patch14") : oa::String();
		oa::SharedPtr<oa::AlmAg> alm;
		if (lmTextFeatureDim_ > 0) {
			auto clip = oa::ClipTextAg::loadArchive(engine(), c.clipTextModel);
			if (clip.isError()) return clip.getStatus();
			auto merges = oa::Filesystem::readBinary(oa::Path(c.clipMerges));
			if (merges.isError()) return merges.getStatus();
			const auto& bytes = merges.getValue();
			alm = oa::makeShared<oa::AlmAg>(tok_.ptr, lm_.ptr,
				oa::move(clip.getValue()), oa::Span<const oa::U8>(bytes.data(), bytes.size()), textEncoder);
		} else {
			alm = oa::makeShared<oa::AlmAg>(tok_.ptr, lm_.ptr, textEncoder);
		}
		const oa::Status status = alm->saveBundle(engine(), bundlePath);
		if (status.isOk()) {
			std::printf("Saved OaAlm: %s\n", bundlePath.cStr());
			std::printf("  stage checkpoints: %s · %s\n",
				tokMgr_->masterPath().cStr(), lmMgr_->masterPath().cStr());
			std::fflush(stdout);
		}
		return status;
	}

	oa::Status exportSavedStages() {
		const auto& c = cli.getConfig();
		lmTextFeatureDim_ = c.textConditioning == "clip"
			? oa::ClipTextConfig::viTL14().projectionDim : 0;
		if (c.textConditioning != "clip" and c.textConditioning != "none")
			return oa::Status::invalidArgument(oa::String("unknown text_conditioning: ") + c.textConditioning);
		const oa::AlmPriorConfig lmCfg = makeLmConfig(lmTextFeatureDim_);
		if (c.lmFfnType != "dense" and c.lmFfnType != "moe" and c.lmFfnType != "hybrid")
			return oa::Status::invalidArgument(oa::String("unknown lm_ffn_type: ") + c.lmFfnType);
		lm_.ptr = oa::makeShared<oa::AlmPriorAg>(lmCfg);
		lmParams_ = lm_.module().allParameterPtrs();
		lmOpt_ = oa::makeUnique<oa::AdamW>(lmParams_, c.lmLr, 0.9F, 0.99F, 1e-8F, c.lmWeightDecay);
		lmMgr_ = oa::makeUnique<oa::CheckpointManager>(engine(), oa::CheckpointManagerConfig{
			.dir = c.modelDir, .modelName = c.name + "prior",
			.maxKeep = c.ckptMaxKeep, .metricName = "val_loss", .lowerIsBetter = true});
		const oa::String priorPath = lmMgr_->masterPath();
		const oa::Status loaded = lm_.module().load(engine(), priorPath, *lmOpt_);
		if (not loaded.isOk()) return oa::Status::notFound(
			oa::String("failed to load ALM prior from ") + priorPath + ": " + loaded.getMessage());
		(void)oa::ExecutionSession::getActive().submitAndWait();
		const oa::Status saved = saveAlmBundle();
		if (saved.isOk()) currentStage_ = 3;
		return saved;
	}

	oa::Status bakeNativeClipFeatures() {
		const auto& c = cli.getConfig();
		if (not oa::Filesystem::isFile(oa::Path(c.clipTextModel)) or
			not oa::Filesystem::isFile(oa::Path(c.clipMerges))) {
			return oa::Status::notFound(oa::String("native CLIP assets are missing: ")
				+ c.clipTextModel + ", " + c.clipMerges);
		}

		struct FeatureRecord { oa::String id; oa::Usize offset = 0; oa::Usize count = 0; };
		oa::Vec<FeatureRecord> records;
		oa::Vec<oa::String> prompts;
		prompts.pushBack(oa::String()); // unconditional feature row
		std::unordered_set<std::string> seen;
		auto append = [&](const oa::DsCombatMotionProcessed& dataset) {
			for (oa::I32 clip = 0; clip < dataset.numClips(); ++clip) {
				const oa::String& id = dataset.clipId(clip);
				if (not seen.emplace(id.cStr()).second) continue;
				const auto& captions = dataset.clipCaptions(clip);
				if (captions.empty()) continue;
				FeatureRecord record{id, prompts.size(), captions.size()};
				for (const auto& caption : captions) prompts.pushBack(caption.text);
				records.pushBack(oa::move(record));
			}
		};
		append(*ds_);
		if (valDs_) append(*valDs_);
		oa::DsCombatMotionProcessed test(c.dataset, "test", c.maxClips);
		if (test.ok()) append(test);
		if (records.empty()) return oa::Status::error("CMP has no caption records to bake");

		auto clipResult = oa::ClipTextAg::loadArchive(engine(), c.clipTextModel);
		if (clipResult.isError()) return clipResult.getStatus();
		auto clip = oa::move(clipResult.getValue());
		oa::ClipTokenizer tokenizer;
		OA_RETURN_IF_ERROR(tokenizer.loadMerges(oa::Path(c.clipMerges)));
		const oa::I32 contextLength = clip->config().contextLength;
		const oa::I32 dim = clip->config().projectionDim;
		constexpr oa::I32 batchSize = 16;
		std::vector<oa::F32> features(prompts.size() * static_cast<oa::Usize>(dim));
		auto& ctx = oa::ExecutionSession::getActive();
		std::printf("Native CLIP bake: %zu captions + unconditional · batch=%d · vulkan\n",
			prompts.size() - 1, batchSize);
		for (oa::Usize start = 0; start < prompts.size(); start += batchSize) {
			const oa::I32 count = static_cast<oa::I32>(
				std::min<oa::Usize>(batchSize, prompts.size() - start));
			auto encoded = tokenizer.encode(
				oa::Span<const oa::String>(prompts.data() + start, static_cast<oa::Usize>(count)),
				contextLength, true);
			if (encoded.isError()) return encoded.getStatus();
			const auto& batch = encoded.getValue();
			auto ids = oa::FnMatrix::fromInt32(
				oa::Span<const oa::I32>(batch.tokenIds.data(), batch.tokenIds.size()),
				oa::MatrixShape{count, contextLength}, oa::ScalarType::Int32);
			auto eos = oa::FnMatrix::fromInt32(
				oa::Span<const oa::I32>(batch.flatEosRows.data(), batch.flatEosRows.size()),
				oa::MatrixShape{count}, oa::ScalarType::Int32);
			oa::GradNo noGrad;
			auto output = clip->forwardTokens(ids, eos);
			OA_RETURN_IF_ERROR(ctx.submitAndWait());
			const auto host = hostF32(output);
			if (host.size() != static_cast<size_t>(count) * dim)
				return oa::Status::error("native CLIP returned an invalid feature shape");
			std::memcpy(features.data() + start * static_cast<oa::Usize>(dim),
				host.data(), host.size() * sizeof(oa::F32));
			ctx.clear();
			const oa::Usize done = start + static_cast<oa::Usize>(count);
			std::printf("  encoded %zu/%zu captions\r", done, prompts.size());
			std::fflush(stdout);
		}
		std::printf("\n");

		const oa::Path outDir = oa::Path(c.dataset) / "text_feats";
		OA_RETURN_IF_ERROR(oa::Filesystem::createDirectories(outDir));
		OA_RETURN_IF_ERROR(writeNpyF32Atomic(outDir / "uncond.npy",
			features.data(), 1, static_cast<oa::Usize>(dim)));
		for (const auto& record : records) {
			OA_RETURN_IF_ERROR(writeNpyF32Atomic(outDir / oa::Path(record.id + ".npy"),
				features.data() + record.offset * static_cast<oa::Usize>(dim),
				record.count, static_cast<oa::Usize>(dim)));
		}
		const oa::String manifest = oa::String("{\n")
			+ "  \"format\": \"oa_clip_text_v1\",\n"
			+ "  \"model\": \"openai/clip-vit-large-patch14\",\n"
			+ "  \"feature\": \"CLIPTextModelWithProjection.text_embeds\",\n"
			+ "  \"dtype\": \"float32\",\n"
			+ "  \"dim\": " + oa::toString(static_cast<oa::I64>(dim)) + ",\n"
			+ "  \"caption_order\": \"texts/<id>.txt line order\",\n"
			+ "  \"max_length\": " + oa::toString(static_cast<oa::I64>(contextLength)) + ",\n"
			+ "  \"producer\": \"oa::ClipTextAg/vulkan\"\n"
			+ "}\n";
		OA_RETURN_IF_ERROR(writeTextAtomic(outDir / "manifest.json", manifest));
		std::printf("Native CLIP cache: %zu clips -> %s\n",
			records.size(), outDir.string().cStr());
		return oa::Status::ok();
	}

	oa::Status reloadDatasetsWithTextFeatures() {
		const auto& c = cli.getConfig();
		const oa::I32 expectedClips = numClips_;
		delete ds_;
		delete valDs_;
		ds_ = new oa::DsCombatMotionProcessed(c.dataset, c.split, c.maxClips);
		valDs_ = new oa::DsCombatMotionProcessed(c.dataset, c.valSplit, c.maxClips);
		if (not ds_->ok() or ds_->numClips() != expectedClips)
			return oa::Status::error("failed to reload training data after native CLIP bake");
		if (not valDs_->ok()) {
			delete valDs_;
			valDs_ = nullptr;
		}
		return oa::Status::ok();
	}

	int setup(int argc, char** argv) override {
#if defined(_WIN32)
		std::signal(SIGINT, onSigInt);
		std::signal(SIGTERM, onSigInt);
#else
		struct sigaction sa{};
		sa.sa_handler = onSigInt;
		sigemptyset(&sa.sa_mask);
		sa.sa_flags = SA_RESTART;
		sigaction(SIGINT, &sa, nullptr);
		sigaction(SIGTERM, &sa, nullptr);
#endif
		if (not cli.parse(argc, argv)) { isRunning = false; }
		const auto& c = cli.getConfig();
		engineConfig_.precision = c.precision();
		return 0;
	}

	oa::Status init() override {
		const auto& c = cli.getConfig();
		auto& ctx = oa::ExecutionSession::getActive();

		ds_ = new oa::DsCombatMotionProcessed(c.dataset, c.split, c.maxClips);
		if (not ds_->ok()) {
			OaLogError(oa::LogComponent::Ml, "trainalm: failed to load CMP from %s", c.dataset.cStr());
			isRunning = false; return oa::Status::ok();
		}
		featDim_   = ds_->featDim();
		numJoints_ = ds_->numJoints();
		numClips_  = ds_->numClips();
		valDs_ = new oa::DsCombatMotionProcessed(c.dataset, c.valSplit, c.maxClips);
		if (not valDs_->ok()) {
			OaLogWarn(oa::LogComponent::Ml,
				"trainalm: validation split '%s' unavailable; val_loss/checkpoint selection disabled",
				c.valSplit.cStr());
			delete valDs_;
			valDs_ = nullptr;
		}

		oa::FnMatrix::setRngSeed(static_cast<oa::U64>(c.seed));

		// ── Build tokenizer ──
		oa::AlmTokenizerConfig tokCfg;
		tokCfg.inputDim   = featDim_;
		tokCfg.width      = c.width;
		tokCfg.codeDim    = c.codeDim;
		tokCfg.numCodes   = c.numCodes;
		tokCfg.downT      = c.downT;
		tokCfg.depth      = c.depth;
		tokCfg.commitBeta = c.commitBeta;
		tokCfg.emaDecay   = c.emaDecay;
		tokCfg.emaEps     = c.emaEps;
		tokCfg.deadThresh = c.deadThresh;
		tok_.ptr = oa::makeShared<oa::AlmTokenizerAg>(tokCfg);

		const oa::I32 B = c.batchSize;
		const oa::I32 T = c.seqLen;
		const oa::I32 N = B * T;

		// seed codebook
		if (N >= c.numCodes) {
			std::vector<float> seed(static_cast<size_t>(B * T) * featDim_);
			for (oa::I32 b = 0; b < B; ++b) {
				const oa::I32 clipIdx = b % numClips_;
				const oa::I32 frames = ds_->clipFrames(clipIdx);
				const oa::I32 start = frames > T ? (frames - T) / 2 : 0;
				const oa::F32* src = ds_->clipData(clipIdx) + static_cast<size_t>(start) * featDim_;
				float* dst = seed.data() + static_cast<size_t>(b) * T * featDim_;
				std::memcpy(dst, src, static_cast<size_t>(std::min(T, frames)) * featDim_ * sizeof(float));
			}
			auto seedX = makeF32(seed, oa::MatrixShape{B, T, featDim_});
			auto z0 = tok_.encode(seedX, B, T);
			tok_.seed(z0);
			ctx.clear();
		}

		// Build sliding windows
		for (oa::I32 ci = 0; ci < numClips_; ++ci) {
			const oa::I32 frames = ds_->clipFrames(ci);
			if (frames < T) continue;
			oa::I32 lastStart = -1;
			for (oa::I32 s = 0; s + T <= frames; s += T / 2) {
				tokWindows_.emplace_back(ci, s);
				lastStart = s;
			}
			// include the exact clip tail when the overlap stride does not land on it.
			// This covers every real frame without introducing padded VQ assignments.
			const oa::I32 tailStart = frames - T;
			if (tailStart != lastStart) tokWindows_.emplace_back(ci, tailStart);
		}
		if (valDs_) {
			for (oa::I32 ci = 0; ci < valDs_->numClips(); ++ci) {
				const oa::I32 frames = valDs_->clipFrames(ci);
				if (frames < T) continue;
				oa::I32 lastStart = -1;
				for (oa::I32 s = 0; s + T <= frames; s += T / 2) {
					tokValWindows_.emplace_back(ci, s);
					lastStart = s;
				}
				const oa::I32 tailStart = frames - T;
				if (tailStart != lastStart) tokValWindows_.emplace_back(ci, tailStart);
			}
			if (tokValWindows_.empty()) {
				OaLogWarn(oa::LogComponent::Ml, "trainalm: validation split has no tokenizer windows");
			}
		}
		if (tokWindows_.empty()) {
			OaLogError(oa::LogComponent::Ml, "trainalm: no clip long enough for seq_len %d", T);
			isRunning = false; return oa::Status::ok();
		}
		{ Lcg r(0xABCD); for (size_t i = tokWindows_.size(); i > 1; --i) std::swap(tokWindows_[i - 1], tokWindows_[r.u32() % i]); }

		// ── tokenizer optimizer + oa::ItTraining ──
		tokParams_ = tok_.module().allParameterPtrs();
		tokOpt_ = oa::makeUnique<oa::AdamW>(tokParams_, c.tokLr, 0.9F, 0.99F, 1e-8F, c.tokWeightDecay);

		tokLossMetric_ = oa::makeUnique<oa::MetricLoss>("recon");

		tokExtraCb_ = oa::makeUnique<CbTokMetrics>(c.numCodes);

		tokBar_ = oa::makeUnique<oa::CbProgressBar>(10);
		tokBar_->addMetric(tokLossMetric_.get());
		tokBar_->addMetric(tokExtraCb_->velPtr());
		tokBar_->addMetric(tokExtraCb_->commitPtr());

		if (not tokValWindows_.empty()) {
			tokValidationCb_ = oa::makeUnique<oa::CbValidation>(
				[this](oa::ItTraining& inIter) { return evaluateTokenizer(inIter); });
		}

		tokSummary_ = oa::makeUnique<oa::CbSummary>(true);
		if (tokValidationCb_) tokSummary_->setValidationMetric(tokValidationCb_->metricPtr());

		tokMgr_ = oa::makeUnique<oa::CheckpointManager>(engine(), oa::CheckpointManagerConfig{
			.dir           = c.modelDir,
			.modelName     = c.name + "Tok",
			.maxKeep       = c.ckptMaxKeep,
			.metricName    = tokValidationCb_ ? oa::String("val_loss") : oa::String("recon"),
			.lowerIsBetter = true,
		});
		(void)oa::Filesystem::createDirectories(oa::Path(tokMgr_->modelDir()));
		tokCkptCb_ = oa::makeUnique<oa::CbCheckpoint>(
			*tokMgr_, tok_.module(), *tokOpt_, c.ckptSaveEvery,
			tokValidationCb_ ? tokValidationCb_->metricPtr() : nullptr, c.ckptRestoreBest);

		oa::ItTrainingConfig tokIterCfg;
		const oa::I64 tokStepsPerEpoch = static_cast<oa::I64>(tokWindows_.size() + static_cast<size_t>(B) - 1) / B;
		tokStepsPerEpoch_ = tokStepsPerEpoch;
		const oa::I32 tokTotalEpochs = c.tokPhases.empty() ? c.tokEpochs
			: [&c]() { oa::I32 sum = 0; for (const auto& ph : c.tokPhases) sum += ph.epochs; return sum; }();
		tokIterCfg.totalSteps     = static_cast<oa::I64>(tokTotalEpochs) * tokStepsPerEpoch;
		tokIterCfg.stepsPerEpoch  = tokStepsPerEpoch;
		tokIterCfg.batchSize      = B;
		tokIterCfg.sequenceLength = T;
		tokIterCfg.sequenceUnit   = "frame";
		tokIterCfg.timerName      = "tokenizer_step";
		tokIterCfg.metrics        = {tokLossMetric_.get()};
		// LR scheduler: phased or single warmup+cosine
		tokSched_ = cli.buildScheduler(c.tokPhases, tokStepsPerEpoch,
			c.tokLr, c.tokMinLr, c.tokWarmup, tokIterCfg.totalSteps);
		tokOpt_->setLr(tokSched_->getLr(1));
		tokSchedCb_ = oa::makeUnique<oa::CbLrScheduler>(*tokSched_, *tokOpt_);

		tokIterCfg.callbacks = {tokExtraCb_.get(), tokBar_.get()};
		if (tokValidationCb_) tokIterCfg.callbacks.push_back(tokValidationCb_.get());
		tokIterCfg.callbacks.push_back(tokCkptCb_.get());
		tokIterCfg.callbacks.push_back(tokSchedCb_.get());
		tokIterCfg.callbacks.push_back(tokSummary_.get());

		tokIter_ = oa::makeUnique<oa::ItTraining>(engine(), *tokOpt_, tokIterCfg);

		// ── stage selection ──
		if (c.stage == "lm" or c.stage == "export") {
			// load the saved tokenizer for LM training or bundle recovery.
			const oa::String tokPath = tokMgr_->masterPath();
			auto tokParams = tok_.module().allParameterPtrs();
			auto dummyOpt = oa::makeUnique<oa::AdamW>(tokParams, 0.0F);
			auto st = tok_.module().load(engine(), tokPath, *dummyOpt);
			if (not st.isOk()) {
				OaLogError(oa::LogComponent::Ml, "trainalm: failed to load tokenizer from %s: %s",
					tokPath.cStr(), st.getMessage().cStr());
				isRunning = false; return oa::Status::ok();
			}
			std::printf("Loaded tokenizer: %s\n", tokPath.cStr());
			(void)ctx.submitAndWait();
			currentStage_ = c.stage == "export" ? 4 : 1;
		} else if (c.stage != "both" and c.stage != "tok") {
			OaLogError(oa::LogComponent::Ml,
				"trainalm: unknown stage '%s' (expected both, tok, lm, or export)", c.stage.cStr());
			isRunning = false; return oa::Status::ok();
		}

		// ── print header ──
		std::printf("\ntrainalm — OaAlm tokenizer + Transformer LM (%s FFN)\n", c.lmFfnType.cStr());
		std::printf("  data: %s · %d clips · featDim %d · %d joints\n",
			c.dataset.cStr(), numClips_, featDim_, numJoints_);
		if (valDs_) std::printf("  validation: %s · %d clips · %zu tokenizer windows\n",
			c.valSplit.cStr(), valDs_->numClips(), tokValWindows_.size());
		std::printf("tokenizer: %d epochs × %lld steps/epoch · B=%d · T=%d · lr %.1e · ckpt %s\n",
			c.tokEpochs, static_cast<long long>(tokStepsPerEpoch),
			B, T, static_cast<double>(c.tokLr),
			c.ckptSaveEvery > 0 ? "step+epoch" : "epoch-end");
		std::fflush(stdout);

		return oa::Status::ok();
	}

	oa::Status tick() override {
		// ── Graceful exit ──
		if (gSigIntCount > 0) {
			gSigIntCount = 0;
			if (not exitRequested_) {
				exitRequested_ = true;
				OaLogInfo(oa::LogComponent::Ml, "Interrupted. Press ctrl+C again to exit.");
				return oa::Status::ok();
			}
			OaLogInfo(oa::LogComponent::Ml, "Exiting...");
			if (tokIter_) (void)tokIter_->finish();
			if (lmIter_) {
				const oa::Status finish = lmIter_->finish();
				if (finish.isOk() and lm_.ptr) {
					OaLogInfo(oa::LogComponent::Ml,
						"Publishing best completed ALM checkpoint before exit...");
					const oa::Status saved = saveAlmBundle();
					if (not saved.isOk()) OaLogError(oa::LogComponent::Ml,
						"trainalm: interrupted ALM bundle save failed: %s",
						saved.getMessage().cStr());
				}
			}
			isRunning = false;
			return oa::Status::ok();
		}

		if (currentStage_ == 0) return tickTokenizer();
		if (currentStage_ == 1) return startLM();
		if (currentStage_ == 2) return tickLM();
		if (currentStage_ == 4) {
			const oa::Status status = exportSavedStages();
			if (not status.isOk()) OaLogError(oa::LogComponent::Ml,
				"trainalm: saved-stage export failed: %s", status.getMessage().cStr());
			isRunning = false;
			return oa::Status::ok();
		}
		isRunning = false;
		return oa::Status::ok();
	}

	oa::Status tickTokenizer() {
		const auto& c = cli.getConfig();
		auto& ctx = oa::ExecutionSession::getActive();
		const oa::I32 B = c.batchSize;
		const oa::I32 T = c.seqLen;
		const oa::I32 tokLen = T / tok_.downsampleFactor();

		if (tokIter_->isDone()) {
			(void)tokIter_->finish();
			std::printf("tokenizer training complete.\n");
			std::fflush(stdout);

			// Dataset-wide codebook usage
			{
				std::vector<oa::I32> allTok;
				for (oa::I32 ci = 0; ci < numClips_; ++ci) {
					const oa::I32 frames = ds_->clipFrames(ci);
					if (frames < T) continue;
					std::vector<float> clip(static_cast<size_t>(frames) * featDim_);
					std::memcpy(clip.data(), ds_->clipData(ci), clip.size() * sizeof(float));
					auto x = makeF32(clip, oa::MatrixShape{1, frames, featDim_});
					auto ids = tok_.tokenize(x, 1, frames)[0];
					(void)ctx.submitAndWait();
					const oa::I64 n = ids.numElements();
					const oa::I32* p = ids.dataAs<const oa::I32>();
					allTok.insert(allTok.end(), p, p + n);
					ctx.clear();
				}
				const auto u = codebookUsage(allTok.data(), allTok.size(), c.numCodes);
				std::printf("codebook usage (dataset): %d/%d live | perplexity %.1f | %zu tokens\n",
					u.unique, c.numCodes, static_cast<double>(u.perplexity), allTok.size());
				std::fflush(stdout);
			}

			currentStage_ = (c.stage == "tok") ? 3 : 1;
			return oa::Status::ok();
		}

		// ── One tokenizer step ──
		tokIter_->step([&]() {
			const size_t cursor = tokCursor_;
			tokCursor_ = (tokCursor_ + static_cast<size_t>(B)) % tokWindows_.size();

			std::vector<float> batch(static_cast<size_t>(B * T) * featDim_);
			for (oa::I32 b = 0; b < B; ++b) {
				const auto& w = tokWindows_[(cursor + static_cast<size_t>(b)) % tokWindows_.size()];
				const oa::F32* src = ds_->clipData(w.first) + static_cast<size_t>(w.second) * featDim_;
				float* dst = batch.data() + static_cast<size_t>(b) * T * featDim_;
				std::memcpy(dst, src, static_cast<size_t>(T) * featDim_ * sizeof(float));
			}
			auto X = makeF32(batch, oa::MatrixShape{B, T, featDim_});

			tokOpt_->zeroGrad();
			// autograd nodes are attached during forward only while a tape is active.
			// Keep it alive across the complete Ag forward/loss construction.
			auto tape = oa::makeUnique<oa::GradientTape>();
			auto z = tok_.encode(X, B, T);
			auto q = tok_.quantize(z);
			auto rec = tok_.decode(q.quantized, B, tokLen);
			auto xFlat = X.reshape(oa::MatrixShape{static_cast<oa::I64>(B) * T, featDim_});
			auto recon = oa::FnLoss::smoothL1Mean(rec, xFlat);

			auto rec3d = rec.reshape(oa::MatrixShape{B, T, featDim_});
			auto velLoss = oa::FnLoss::velSmoothL1(rec3d, X);

			auto loss = recon + velLoss + q.commitLoss;
			tape->backward(loss);

			if (tokIter_->stepCount() == 0) {
				(void)ctx.submitAndWait();
				std::printf("[DEBUG] X dtype=%d shape=(%lld,%lld,%lld) first=%f\n",
					static_cast<int>(X.getDtype()), X.size(0), X.size(1), X.size(2), X.at(0));
				std::printf("[DEBUG] rec dtype=%d numel=%lld first=%f rms=%f\n",
					static_cast<int>(rec.getDtype()), rec.numElements(), rec.at(0),
					std::sqrt(static_cast<double>(oa::FnMatrix::sum(oa::FnMatrix::mul(rec, rec)).at(0)) / rec.numElements()));
				std::printf("[DEBUG] recon dtype=%d val=%f commit=%f\n",
					static_cast<int>(recon.getDtype()), recon.at(0), q.commitLoss.at(0));
				std::fflush(stdout);
			}

			tokIter_->recordLoss(recon);
			tokExtraCb_->record(velLoss, q.commitLoss, q.idx[0]);

			tok_.emaUpdate(q);
		});

		const oa::I64 step = tokIter_->stepCount();
		const float lv = tokIter_->lastLoss();
		if (not std::isfinite(lv)) {
			OaLogError(oa::LogComponent::Ml, "trainalm: tokenizer diverged at step %lld",
				static_cast<long long>(step));
			tokIter_->requestStop();
		}

		return oa::Status::ok();
	}

	oa::ValidationResult evaluateTokenizer(oa::ItTraining&) {
		const auto& c = cli.getConfig();
		const oa::I32 B = c.batchSize;
		const oa::I32 T = c.seqLen;
		const oa::I32 tokLen = T / tok_.downsampleFactor();
		const oa::I64 allBatches = static_cast<oa::I64>(
			(tokValWindows_.size() + static_cast<size_t>(B) - 1) / static_cast<size_t>(B));
		const oa::I64 batches = c.valBatches > 0
			? std::min<oa::I64>(allBatches, c.valBatches) : allBatches;
		oa::F64 weightedLoss = 0.0;
		oa::F64 weightedVel = 0.0;
		oa::F64 weightedMpjpeCm = 0.0;
		oa::F64 footSkateCm = 0.0;
		oa::I64 footSkateFrames = 0;
		oa::I64 contactCorrect = 0;
		oa::I64 contactTotal = 0;
		oa::I64 samples = 0;
		std::vector<oa::I32> validationTokens;
		auto& ctx = oa::ExecutionSession::getActive();

		for (oa::I64 batchIdx = 0; batchIdx < batches; ++batchIdx) {
			const size_t begin = static_cast<size_t>(batchIdx) * static_cast<size_t>(B);
			const oa::I32 n = static_cast<oa::I32>(
				std::min<size_t>(static_cast<size_t>(B), tokValWindows_.size() - begin));
			ctx.clear();
			std::vector<float> batch(static_cast<size_t>(n * T) * featDim_);
			for (oa::I32 b = 0; b < n; ++b) {
				const auto& w = tokValWindows_[begin + static_cast<size_t>(b)];
				const oa::F32* src = valDs_->clipData(w.first)
					+ static_cast<size_t>(w.second) * featDim_;
				float* dst = batch.data() + static_cast<size_t>(b) * T * featDim_;
				std::memcpy(dst, src, static_cast<size_t>(T) * featDim_ * sizeof(float));
			}
			auto x = makeF32(batch, oa::MatrixShape{n, T, featDim_});
			auto z = tok_.encode(x, n, T);
			auto q = tok_.quantize(z);
			auto rec = tok_.decode(q.quantized, n, tokLen);
			auto xFlat = x.reshape(oa::MatrixShape{static_cast<oa::I64>(n) * T, featDim_});
			auto loss = oa::FnLoss::smoothL1Mean(rec, xFlat);
			auto vel = oa::FnLoss::velSmoothL1(
				rec.reshape(oa::MatrixShape{n, T, featDim_}), x);
			if (not ctx.submitAndWait().isOk()) return {};
			weightedLoss += static_cast<oa::F64>(loss.at(0)) * n;
			weightedVel += static_cast<oa::F64>(vel.at(0)) * n;
			const auto& tokenIds = q.idx[0];
			const oa::I32* tokenData = tokenIds.dataAs<const oa::I32>();
			validationTokens.insert(validationTokens.end(), tokenData,
				tokenData + tokenIds.numElements());

			auto pred = hostF32(rec);
			if (pred.size() != batch.size()) return {};
			valDs_->denormalize(pred.data(), static_cast<oa::I64>(n) * T);
			valDs_->denormalize(batch.data(), static_cast<oa::I64>(n) * T);
			for (oa::I32 b = 0; b < n; ++b) {
				const size_t featureOffset = static_cast<size_t>(b) * T * featDim_;
				auto predWorld = oa::humanMl3dRecoverWorldJoints(
					oa::Span<const oa::F32>(pred.data() + featureOffset,
						static_cast<size_t>(T) * featDim_), T, featDim_);
				auto targetWorld = oa::humanMl3dRecoverWorldJoints(
					oa::Span<const oa::F32>(batch.data() + featureOffset,
						static_cast<size_t>(T) * featDim_), T, featDim_);
				weightedMpjpeCm += oa::humanMl3dMpjpeCm(
					oa::Span<const oa::F32>(predWorld.data(), predWorld.size()),
					oa::Span<const oa::F32>(targetWorld.data(), targetWorld.size()));

				for (oa::I32 t = 0; t < T; ++t) {
					for (oa::I32 cidx = 0; cidx < 4; ++cidx) {
						const size_t k = featureOffset + static_cast<size_t>(t) * featDim_
							+ static_cast<size_t>(featDim_ - 4 + cidx);
						contactCorrect += (pred[k] >= 0.5F) == (batch[k] >= 0.5F) ? 1 : 0;
						++contactTotal;
					}
					if (t == 0) continue;
					for (oa::I32 foot = 0; foot < 2; ++foot) {
						const size_t currFeat = featureOffset + static_cast<size_t>(t) * featDim_;
						const size_t prevFeat = currFeat - featDim_;
						const oa::I32 contactBase = featDim_ - 4 + foot * 2;
						const bool planted = (batch[currFeat + contactBase] >= 0.5F or
							batch[currFeat + contactBase + 1] >= 0.5F) and
							(batch[prevFeat + contactBase] >= 0.5F or
							 batch[prevFeat + contactBase + 1] >= 0.5F);
						if (not planted) continue;
						const oa::I32 joint = foot == 0 ? 10 : 11;
						const size_t curr = (static_cast<size_t>(t) * valDs_->numJoints() + joint) * 3;
						const size_t prev = (static_cast<size_t>(t - 1) * valDs_->numJoints() + joint) * 3;
						const oa::F64 dx = predWorld[curr] - predWorld[prev];
						const oa::F64 dz = predWorld[curr + 2] - predWorld[prev + 2];
						footSkateCm += 100.0 * std::sqrt(dx * dx + dz * dz);
						++footSkateFrames;
					}
				}
			}
			samples += n;
		}
		if (samples > 0) {
			const auto usage = codebookUsage(validationTokens.data(), validationTokens.size(), c.numCodes);
			std::printf("  Validation tokenizer: vel %.6f · MPJPE %.3f cm · contact %.2f%% · foot skate %.3f cm/frame\n"
				"    codebook: %d/%d live · perplexity %.2f · %zu tokens\n",
				weightedVel / static_cast<oa::F64>(samples),
				weightedMpjpeCm / static_cast<oa::F64>(samples),
				contactTotal > 0 ? 100.0 * static_cast<oa::F64>(contactCorrect) / contactTotal : 0.0,
				footSkateFrames > 0 ? footSkateCm / footSkateFrames : 0.0,
				usage.unique, c.numCodes, static_cast<double>(usage.perplexity),
				validationTokens.size());
		}
		ctx.clear();
		return {.loss = samples > 0 ? weightedLoss / static_cast<oa::F64>(samples)
			: std::numeric_limits<oa::F64>::quiet_NaN(), .batches = batches, .samples = samples};
	}

	oa::Status startLM() {
		const auto& c = cli.getConfig();
		auto& ctx = oa::ExecutionSession::getActive();
		const oa::I32 B = c.batchSize;
		const oa::I32 lmTokLen = c.lmSeqLen;
		if (c.textConditioning == "clip") {
			auto cacheComplete = [](const oa::DsCombatMotionProcessed* dataset) {
				if (dataset == nullptr) return true;
				if (dataset->textFeatureDim() != oa::ClipTextConfig::viTL14().projectionDim or
					dataset->textFeatureFormat() != "oa_clip_text_v1" or
					dataset->textFeatureModel() != "openai/clip-vit-large-patch14") return false;
				for (oa::I32 clip = 0; clip < dataset->numClips(); ++clip)
					if (dataset->clipTextFeatureCount(clip) !=
						static_cast<oa::I32>(dataset->clipCaptions(clip).size())) return false;
				return true;
			};
			const bool cacheReady = cacheComplete(ds_) and cacheComplete(valDs_);
			if (not cacheReady) {
				OaLogInfo(oa::LogComponent::Ml,
					"trainalm: CLIP caption cache missing/incompatible; baking with native oa::ClipTextAg");
				const oa::Status baked = bakeNativeClipFeatures();
				if (not baked.isOk()) {
					OaLogError(oa::LogComponent::Ml, "trainalm: native CLIP bake failed: %s",
						baked.getMessage().cStr());
					isRunning = false; return oa::Status::ok();
				}
				const oa::Status reloaded = reloadDatasetsWithTextFeatures();
				if (not reloaded.isOk()) {
					OaLogError(oa::LogComponent::Ml, "trainalm: %s", reloaded.getMessage().cStr());
					isRunning = false; return oa::Status::ok();
				}
			}
			lmTextFeatureDim_ = ds_->textFeatureDim();
			if (lmTextFeatureDim_ <= 0 or ds_->textFeatureFormat() != "oa_clip_text_v1"
				or ds_->textFeatureModel().empty()) {
				OaLogError(oa::LogComponent::Ml,
					"trainalm: native CLIP cache validation failed in %s/text_feats",
					c.dataset.cStr());
				isRunning = false; return oa::Status::ok();
			}
			if (ds_->textFeatureModel() != "openai/clip-vit-large-patch14" or
				lmTextFeatureDim_ != oa::ClipTextConfig::viTL14().projectionDim) {
				OaLogError(oa::LogComponent::Ml,
					"trainalm: dataset CLIP contract '%s'/%d does not match native openai/clip-vit-large-patch14/%d",
					ds_->textFeatureModel().cStr(), lmTextFeatureDim_,
					oa::ClipTextConfig::viTL14().projectionDim);
				isRunning = false; return oa::Status::ok();
			}
			if (not oa::Filesystem::isFile(oa::Path(c.clipTextModel)) or
				not oa::Filesystem::isFile(oa::Path(c.clipMerges))) {
				OaLogError(oa::LogComponent::Ml,
					"trainalm: native CLIP assets are missing (%s, %s); import them before LM training",
					c.clipTextModel.cStr(), c.clipMerges.cStr());
				isRunning = false; return oa::Status::ok();
			}
			for (oa::I32 ci = 0; ci < numClips_; ++ci) {
				if (ds_->clipTextFeatureCount(ci) != static_cast<oa::I32>(ds_->clipCaptions(ci).size())) {
					OaLogError(oa::LogComponent::Ml,
						"trainalm: clip %s lacks one CLIP feature per caption",
						ds_->clipId(ci).cStr());
					isRunning = false; return oa::Status::ok();
				}
			}
			if (valDs_) {
				if (valDs_->textFeatureDim() != lmTextFeatureDim_ or
					valDs_->textFeatureFormat() != ds_->textFeatureFormat() or
					valDs_->textFeatureModel() != ds_->textFeatureModel()) {
					OaLogError(oa::LogComponent::Ml,
						"trainalm: train/validation CLIP feature contracts differ");
					isRunning = false; return oa::Status::ok();
				}
				for (oa::I32 ci = 0; ci < valDs_->numClips(); ++ci) {
					if (valDs_->clipTextFeatureCount(ci) !=
						static_cast<oa::I32>(valDs_->clipCaptions(ci).size())) {
						OaLogError(oa::LogComponent::Ml,
							"trainalm: validation clip %s lacks one CLIP feature per caption",
							valDs_->clipId(ci).cStr());
						isRunning = false; return oa::Status::ok();
					}
				}
			}
		} else if (c.textConditioning == "none") {
			lmTextFeatureDim_ = 0;
		} else {
			OaLogError(oa::LogComponent::Ml, "trainalm: unknown text_conditioning '%s'",
				c.textConditioning.cStr());
			isRunning = false; return oa::Status::ok();
		}

		// tokenize every clip. Short clips are first-class LM examples and are
		// padded only after tokenization; the loss mask excludes their PAD tail.
		tokenSequences_.clear();
		for (oa::I32 ci = 0; ci < numClips_; ++ci) {
			const oa::I32 frames = ds_->clipFrames(ci);
			std::vector<float> clip(static_cast<size_t>(frames) * featDim_);
			std::memcpy(clip.data(), ds_->clipData(ci), clip.size() * sizeof(float));
			auto x = makeF32(clip, oa::MatrixShape{1, frames, featDim_});
			auto ids = tok_.tokenize(x, 1, frames)[0];
			(void)ctx.submitAndWait();
			const oa::I64 n = ids.numElements();
			const oa::I32* p = ids.dataAs<const oa::I32>();
			tokenSequences_.emplace_back(p, p + n);
			ctx.clear();
		}
		if (tokenSequences_.empty()) {
			OaLogError(oa::LogComponent::Ml, "trainalm: tokenizer produced no LM sequences");
			isRunning = false; return oa::Status::ok();
		}
		lmWindows_ = buildLmWindows(tokenSequences_, lmTokLen + 1);
		std::printf("Tokenized all %zu clips for LM training · %zu true-boundary windows\n",
			tokenSequences_.size(), lmWindows_.size());
		const auto corpusStats = measureTokenCorpus(tokenSequences_, c.numCodes);
		std::printf("LM token corpus: %lld tokens · %d/%d codes · unigram ppl %.2f · bigram ppl %.2f\n",
			static_cast<long long>(corpusStats.tokens), corpusStats.unique, c.numCodes,
			corpusStats.unigramPpl, corpusStats.bigramPpl);
		std::fflush(stdout);

		// tokenize the held-out split independently with the identical padded,
		// true-boundary prediction contract.
		valTokenSequences_.clear();
		lmValWindows_.clear();
		if (valDs_) {
			for (oa::I32 ci = 0; ci < valDs_->numClips(); ++ci) {
				const oa::I32 frames = valDs_->clipFrames(ci);
				std::vector<float> clip(static_cast<size_t>(frames) * featDim_);
				std::memcpy(clip.data(), valDs_->clipData(ci), clip.size() * sizeof(float));
				auto x = makeF32(clip, oa::MatrixShape{1, frames, featDim_});
				auto ids = tok_.tokenize(x, 1, frames)[0];
				(void)ctx.submitAndWait();
				const oa::I64 n = ids.numElements();
				const oa::I32* p = ids.dataAs<const oa::I32>();
				valTokenSequences_.emplace_back(p, p + n);
				ctx.clear();
			}
			lmValWindows_ = buildLmWindows(valTokenSequences_, lmTokLen + 1);
			std::printf("LM validation: %zu clips · %zu true-boundary windows\n",
				valTokenSequences_.size(), lmValWindows_.size());
			if (lmValWindows_.empty()) {
				OaLogWarn(oa::LogComponent::Ml, "trainalm: validation split has no LM windows");
			}
		}

		// Build LM
		oa::AlmPriorConfig lmCfg = makeLmConfig(lmTextFeatureDim_);
		if (c.lmFfnType != "dense" and c.lmFfnType != "moe" and c.lmFfnType != "hybrid") {
			OaLogError(oa::LogComponent::Ml, "trainalm: unknown lm_ffn_type '%s'", c.lmFfnType.cStr());
			isRunning = false; return oa::Status::ok();
		}
		lm_.ptr = oa::makeShared<oa::AlmPriorAg>(lmCfg);
		(void)ctx.submitAndWait();

		lmParams_ = lm_.module().allParameterPtrs();
		lmOpt_ = oa::makeUnique<oa::AdamW>(lmParams_, c.lmLr, 0.9F, 0.99F, 1e-8F, c.lmWeightDecay);

		lmLossMetric_    = oa::makeUnique<oa::MetricLoss>("cross_entropy");
		lmLrMetric_      = oa::makeUnique<OptimLrMetric>(*lmOpt_);

		lmBar_ = oa::makeUnique<oa::CbProgressBar>(10);
		lmBar_->addMetric(lmLossMetric_.get());
		lmBar_->addMetric(lmLrMetric_.get());
		if (lmCfg.ffnType != oa::AlmFfnType::Dense) {
			lmMoeRoutingCb_ = oa::makeUnique<CbMoeRouting>(*lm_.ptr);
		}
		if (not lmValWindows_.empty()) {
			lmValidationCb_ = oa::makeUnique<oa::CbValidation>(
				[this](oa::ItTraining& inIter) { return evaluateLm(inIter); });
		}

		lmSummary_ = oa::makeUnique<oa::CbSummary>(true);
		if (lmValidationCb_) lmSummary_->setValidationMetric(lmValidationCb_->metricPtr());

		lmMgr_ = oa::makeUnique<oa::CheckpointManager>(engine(), oa::CheckpointManagerConfig{
			.dir           = c.modelDir,
			// Do not reuse the retired Mamba-era AlmLm namespace: generic module
			// loading is permissive about missing names, so architecture migrations
			// need a fresh checkpoint root to prevent a misleading partial load.
			.modelName     = c.name + "prior",
			.maxKeep       = c.ckptMaxKeep,
			.metricName    = lmValidationCb_ ? oa::String("val_loss") : oa::String("cross_entropy"),
			.lowerIsBetter = true,
		});
		(void)oa::Filesystem::createDirectories(oa::Path(lmMgr_->modelDir()));
		lmCkptCb_ = oa::makeUnique<oa::CbCheckpoint>(
			*lmMgr_, lm_.module(), *lmOpt_, c.ckptSaveEvery,
			lmValidationCb_ ? lmValidationCb_->metricPtr() : nullptr, c.ckptRestoreBest);

		oa::ItTrainingConfig lmIterCfg;
		// Treat every valid start position in each token sequence as a distinct training
		// example so the LM sees many more steps than just ceil(num_clips / batch).
		const oa::I64 totalWindows = static_cast<oa::I64>(lmWindows_.size());
		const oa::I64 lmStepsPerEpoch = std::max<oa::I64>(1,
			(totalWindows + B - 1) / B);
		lmStepsPerEpoch_ = lmStepsPerEpoch;
		const oa::I32 lmTotalEpochs = c.lmPhases.empty() ? c.lmEpochs
			: [&c]() { oa::I32 sum = 0; for (const auto& ph : c.lmPhases) sum += ph.epochs; return sum; }();
		lmIterCfg.totalSteps     = static_cast<oa::I64>(lmTotalEpochs) * lmStepsPerEpoch;
		lmIterCfg.stepsPerEpoch  = lmStepsPerEpoch;
		// One sample is one sequence; sequence length derives token throughput.
		lmIterCfg.batchSize      = B;
		lmIterCfg.sequenceLength = lmTokLen + 1 + (lmTextFeatureDim_ > 0 ? 1 : 0);
		lmIterCfg.sequenceUnit   = "token";
		lmIterCfg.timerName      = "lm_step";
		lmIterCfg.metrics        = {lmLossMetric_.get()};
		// LR scheduler: phased or single warmup+cosine
		lmSched_ = cli.buildScheduler(c.lmPhases, lmStepsPerEpoch,
			c.lmLr, c.lmMinLr, c.lmWarmup, lmIterCfg.totalSteps);
		// Start on the scheduler's first value. Previously step 1 used the optimizer's
		// peak LR, then step 2 abruptly fell to the bottom of the warmup ramp.
		lmOpt_->setLr(lmSched_->getLr(1));
		lmSchedCb_ = oa::makeUnique<oa::CbLrScheduler>(*lmSched_, *lmOpt_);

		lmIterCfg.callbacks = {};
		if (lmMoeRoutingCb_) lmIterCfg.callbacks.push_back(lmMoeRoutingCb_.get());
		lmIterCfg.callbacks.push_back(lmBar_.get());
		if (lmValidationCb_) lmIterCfg.callbacks.push_back(lmValidationCb_.get());
		lmIterCfg.callbacks.push_back(lmCkptCb_.get());
		lmIterCfg.callbacks.push_back(lmSchedCb_.get());
		lmIterCfg.callbacks.push_back(lmSummary_.get());

		lmIter_ = oa::makeUnique<oa::ItTraining>(engine(), *lmOpt_, lmIterCfg);

		const oa::String textLabel = lmTextFeatureDim_ > 0
			? ds_->textFeatureModel() + oa::String("-")
				+ oa::toString(static_cast<oa::I64>(lmTextFeatureDim_))
			: oa::String("none");
		std::printf("LM: %d epochs × %lld steps/epoch · B=%d · tokLen=%d · text=%s · lr %.1e · ckpt %s\n",
			lmTotalEpochs, static_cast<long long>(lmStepsPerEpoch),
			B, lmTokLen, textLabel.cStr(), static_cast<double>(c.lmLr),
			c.ckptSaveEvery > 0 ? "step+epoch" : "epoch-end");
		std::fflush(stdout);

		currentStage_ = 2;
		return oa::Status::ok();
	}

	oa::ValidationResult evaluateLm(oa::ItTraining&) {
		const auto& c = cli.getConfig();
		const oa::I32 B = c.batchSize;
		const oa::I32 lmTokLen = c.lmSeqLen;
		const oa::I64 allBatches = static_cast<oa::I64>(
			(lmValWindows_.size() + static_cast<size_t>(B) - 1) / static_cast<size_t>(B));
		const oa::I64 batches = c.valBatches > 0
			? std::min<oa::I64>(allBatches, c.valBatches) : allBatches;
		oa::F64 weightedLoss = 0.0;
		oa::I64 correctTokens = 0;
		oa::I64 correctEos = 0;
		oa::I64 eosTokens = 0;
		oa::I64 samples = 0;
		auto& ctx = oa::ExecutionSession::getActive();

		for (oa::I64 batchIdx = 0; batchIdx < batches; ++batchIdx) {
			const size_t begin = static_cast<size_t>(batchIdx) * static_cast<size_t>(B);
			const oa::I32 n = static_cast<oa::I32>(
				std::min<size_t>(static_cast<size_t>(B), lmValWindows_.size() - begin));
			std::vector<oa::I32> inputHost(static_cast<size_t>(n) * (lmTokLen + 1));
			std::vector<oa::I32> targetHost(static_cast<size_t>(n) * (lmTokLen + 1));
			std::vector<float> maskHost(static_cast<size_t>(n) * (lmTokLen + 1));
			std::vector<float> eosMaskHost(static_cast<size_t>(n) * (lmTokLen + 1));
			oa::I32 validCount = 0;
			std::vector<float> textHost;
			if (lmTextFeatureDim_ > 0) textHost.resize(static_cast<size_t>(n) * lmTextFeatureDim_);
			for (oa::I32 b = 0; b < n; ++b) {
				const auto& window = lmValWindows_[begin + static_cast<size_t>(b)];
				const auto& seq = valTokenSequences_[static_cast<size_t>(window.sequence)];
				const size_t row = static_cast<size_t>(b) * (lmTokLen + 1);
				fillLmRow(seq, window, lmTokLen + 1, c.numCodes,
					c.numCodes + 1, c.numCodes + 2, inputHost.data() + row,
					targetHost.data() + row, maskHost.data() + row);
				validCount += window.valid;
				for (oa::I32 t = 0; t < lmTokLen + 1; ++t) {
					const bool isEos = maskHost[row + static_cast<size_t>(t)] != 0.0F
						and targetHost[row + static_cast<size_t>(t)] == c.numCodes + 1;
					eosMaskHost[row + static_cast<size_t>(t)] = isEos ? 1.0F : 0.0F;
					eosTokens += isEos ? 1 : 0;
				}
				if (lmTextFeatureDim_ > 0) {
					const oa::F32* feature = valDs_->clipTextFeatureData(window.sequence);
					std::memcpy(textHost.data() + static_cast<size_t>(b) * lmTextFeatureDim_,
						feature, static_cast<size_t>(lmTextFeatureDim_) * sizeof(float));
				}
			}
			ctx.clear();
			auto inputIds = makeI32(inputHost, oa::MatrixShape{n, lmTokLen + 1});
			auto targetIds = makeI32(targetHost, oa::MatrixShape{n, lmTokLen + 1});
			auto lossMask = makeF32(maskHost, oa::MatrixShape{static_cast<oa::I64>(n) * (lmTokLen + 1)});
			auto eosMask = makeF32(eosMaskHost, oa::MatrixShape{static_cast<oa::I64>(n) * (lmTokLen + 1)});
			oa::Matrix logits;
			if (lmTextFeatureDim_ > 0) {
				auto textFeatures = makeF32(textHost, oa::MatrixShape{n, lmTextFeatureDim_});
				logits = lm_.forward(inputIds, textFeatures);
			} else {
				logits = lm_.forward(inputIds);
			}
			auto logitsFlat = logits.reshape(oa::MatrixShape{
				static_cast<oa::I64>(n) * (lmTokLen + 1), c.numCodes + 3});
			auto targetFlat = targetIds.reshape(oa::MatrixShape{
				static_cast<oa::I64>(n) * (lmTokLen + 1)});
			auto loss = oa::FnLoss::maskedCrossEntropy(logitsFlat, targetFlat, lossMask, validCount);
			auto correct = oa::FnMatrix::maskedCategoricalAccuracyCount(
				logitsFlat, targetFlat, lossMask);
			auto eosCorrect = oa::FnMatrix::maskedCategoricalAccuracyCount(
				logitsFlat, targetFlat, eosMask);
			if (not ctx.submitAndWait().isOk()) return {};
			weightedLoss += static_cast<oa::F64>(loss.at(0)) * validCount;
			correctTokens += correct.dataAs<const oa::U32>()[0];
			correctEos += eosCorrect.dataAs<const oa::U32>()[0];
			samples += validCount;
		}
		if (samples > 0) {
			const oa::F64 meanLoss = weightedLoss / static_cast<oa::F64>(samples);
			std::printf("  Validation LM: perplexity %.3f · token accuracy %.2f%% · EOS accuracy %.2f%%\n",
				std::exp(meanLoss),
				100.0 * static_cast<oa::F64>(correctTokens) / samples,
				eosTokens > 0 ? 100.0 * static_cast<oa::F64>(correctEos) / eosTokens : 0.0);
		}
		ctx.clear();
		return {.loss = samples > 0 ? weightedLoss / static_cast<oa::F64>(samples)
			: std::numeric_limits<oa::F64>::quiet_NaN(), .batches = batches, .samples = samples};
	}

	oa::Status tickLM() {
		const auto& c = cli.getConfig();
		const oa::I32 B = c.batchSize;
		const oa::I32 lmTokLen = c.lmSeqLen;

		if (lmIter_->isDone()) {
			(void)lmIter_->finish();
			const oa::Status bundleStatus = saveAlmBundle();
			if (not bundleStatus.isOk()) {
				OaLogError(oa::LogComponent::Ml, "trainalm: failed to save ALM bundle: %s",
					bundleStatus.getMessage().cStr());
				isRunning = false;
				return oa::Status::ok();
			}
			std::printf("LM training complete.\n");
			std::fflush(stdout);
			currentStage_ = 3;
			return oa::Status::ok();
		}

		lmIter_->step([&]() {
			std::vector<oa::I32> inputHost(static_cast<size_t>(B) * (lmTokLen + 1));
			std::vector<oa::I32> targetHost(static_cast<size_t>(B) * (lmTokLen + 1));
			std::vector<float> maskHost(static_cast<size_t>(B) * (lmTokLen + 1));
			oa::I32 validCount = 0;
			std::vector<float> textHost;
			if (lmTextFeatureDim_ > 0) textHost.resize(static_cast<size_t>(B) * lmTextFeatureDim_);
			for (oa::I32 b = 0; b < B; ++b) {
				const auto& window = lmWindows_[static_cast<size_t>(
					(lmIter_->stepCount() * B + b) % static_cast<oa::I64>(lmWindows_.size()))];
				const auto& seq = tokenSequences_[static_cast<size_t>(window.sequence)];
				const size_t row = static_cast<size_t>(b) * (lmTokLen + 1);
				fillLmRow(seq, window, lmTokLen + 1, c.numCodes,
					c.numCodes + 1, c.numCodes + 2, inputHost.data() + row,
					targetHost.data() + row, maskHost.data() + row);
				validCount += window.valid;
				if (lmTextFeatureDim_ > 0) {
					const oa::I32 count = ds_->clipTextFeatureCount(window.sequence);
					const oa::I32 caption = static_cast<oa::I32>((c.seed + lmIter_->epoch()
						+ window.sequence) % count);
					const oa::F32* feature = ds_->clipTextFeatureData(window.sequence)
						+ static_cast<size_t>(caption) * lmTextFeatureDim_;
					std::memcpy(textHost.data() + static_cast<size_t>(b) * lmTextFeatureDim_,
						feature, static_cast<size_t>(lmTextFeatureDim_) * sizeof(float));
				}
			}
			auto inputIds = makeI32(inputHost, oa::MatrixShape{B, lmTokLen + 1});
			auto targetIds = makeI32(targetHost, oa::MatrixShape{B, lmTokLen + 1});
			auto lossMask = makeF32(maskHost, oa::MatrixShape{static_cast<oa::I64>(B) * (lmTokLen + 1)});

			lmOpt_->zeroGrad();
			// The tape must exist before forward; creating it only at backward time
			// leaves the graph unrecorded and the model pinned at ln(vocab) CE.
			auto tape = oa::makeUnique<oa::GradientTape>();
			oa::Matrix logits;
			if (lmTextFeatureDim_ > 0) {
				auto textFeatures = makeF32(textHost, oa::MatrixShape{B, lmTextFeatureDim_});
				logits = lm_.forward(inputIds, textFeatures);
			} else {
				logits = lm_.forward(inputIds);
			}
			auto logitsFlat = logits.reshape(oa::MatrixShape{
				static_cast<oa::I64>(B) * (lmTokLen + 1), c.numCodes + 3});
			auto targetFlat = targetIds.reshape(oa::MatrixShape{
				static_cast<oa::I64>(B) * (lmTokLen + 1)});
			auto ce = oa::FnLoss::maskedCrossEntropy(logitsFlat, targetFlat, lossMask, validCount);
			lmIter_->recordLoss(ce);
			auto aux = lm_.ptr->moeAuxLoss();
			tape->backward(aux.isEmpty() ? ce : oa::FnMatrix::add(ce, aux));

		});

		const oa::I64 step = lmIter_->stepCount();
		const float lv = lmIter_->lastLoss();
		if (not std::isfinite(lv)) {
			OaLogError(oa::LogComponent::Ml, "trainalm: LM diverged at step %lld",
				static_cast<long long>(step));
			lmIter_->requestStop();
		}

		return oa::Status::ok();
	}

	void shutdown() override {
		// release GPU-holding members before the explicit engine close.
		// oa::ComputeApp::main closes rt after shutdown returns; if these
		// smart pointers are still alive when the app struct is destroyed
		// (after main), their destructors would otherwise observe an engine
		// whose vulkan resources have already been released.
		tokSchedCb_.reset();
		lmSchedCb_.reset();
		tokSched_.reset();
		lmSched_.reset();
		tokCkptCb_.reset();
		lmCkptCb_.reset();
		tokMgr_.reset();
		lmMgr_.reset();
		tokSummary_.reset();
		lmSummary_.reset();
		tokValidationCb_.reset();
		lmValidationCb_.reset();
		lmMoeRoutingCb_.reset();
		tokBar_.reset();
		lmBar_.reset();
		tokExtraCb_.reset();
		tokLossMetric_.reset();
		lmLossMetric_.reset();
		lmLrMetric_.reset();
		tokIter_.reset();
		lmIter_.reset();
		tokOpt_.reset();
		lmOpt_.reset();
		tok_.ptr = {};
		lm_.ptr = {};
		tokParams_.clear();
		lmParams_.clear();
		tokenSequences_.clear();
		valTokenSequences_.clear();
		tokWindows_.clear();
		tokValWindows_.clear();
		lmValWindows_.clear();

		delete ds_;
		ds_ = nullptr;
		delete valDs_;
		valDs_ = nullptr;
	}
};

int main(int argc, char** argv) {
	TrainAlmApp app;
	return app.main(argc, argv);
}
