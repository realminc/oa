// trainalm — two-stage trainer for the OaAlm motion tokenizer + Transformer LM.
//
// Stage 1  train the temporal Conv1d VQ-VAE tokenizer (motion to discrete tokens)
// Stage 2  tokenize every train clip, then train the autoregressive Transformer LM
//
// Uses OaItTraining with full callback pipeline: metrics, progress bar, summary,
// periodic checkpointing (save-best + save-every). Ctrl+C for graceful exit.
//
// Usage:
//   trainalm --config var/config/Alm.yaml
//   trainalm --dataset /path/to/Cmp --tok-steps 5000 --lm-steps 5000

#include <Oa/Runtime/App.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Cli.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/FileIo.h>
#include <Oa/Core/Time.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Ml/FnOptim.h>
#include <Oa/Ml/FnLoss.h>
#include <Ml/FnLoss.h>
#include <Oa/Ml/ItTraining.h>
#include <Oa/Ml/Callbacks.h>
#include <Oa/Ml/Checkpoint.h>
#include <Oa/Ml/Metric.h>
#include <Oa/Ml/LrScheduler.h>
#include <Oa/Data/DsHumanMl3d.h>

#include <algorithm>
#include <csignal>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

// Alm model headers (from Examples/Ml/Alm).
#include <Ml/Nn/Alm/AlmConfig.h>
#include <Ml/Nn/Alm/AlmAg.h>
#include <Ml/Nn/Alm/AlmTokenizerAg.h>
#include <Ml/Nn/Alm/AlmPriorAg.h>

// ── Config ──────────────────────────────────────────────────────────────────

struct PhaseConfig {
	OaString Id;
	OaI32    Epochs       = 0;
	OaF32    Lr           = 0.0F;
	OaF32    MinLrFrac    = 0.10F;
	OaI32    WarmupSteps  = 0;
};

struct AlmTrainConfig {
	OaString Name      = "Alm";
	OaString Dataset   = "../dataset/gen/3d/anim/ds/Cmp";
	OaString Split     = "train";
	OaString ValSplit  = "val";
	OaI32    MaxClips  = 0;
	OaI32    ValBatches = 0;  // 0 = complete held-out split
	OaString ModelDir  = OaFileIo::GetVarDir("model/dev").String();
	OaString PrecisionStr = "fp32";

	[[nodiscard]] OaPrecision Precision() const {
		if (PrecisionStr == "fp32") return OaPrecision::FP32;
		if (PrecisionStr == "bf16") return OaPrecision::BF16;
		if (PrecisionStr == "tf32") return OaPrecision::TF32;
		if (PrecisionStr == "fp16") return OaPrecision::FP16;
		return OaPrecision::FP32;
	}

	// Stage selection: "both" (tok then lm), "tok" (tokenizer only),
	// "lm" (load tok, train lm), "export" (assemble saved best stages)
	OaString Stage = "both";

	// Tokenizer architecture
	OaI32 Width      = 384;
	OaI32 CodeDim    = 256;
	OaI32 NumCodes   = 512;
	OaI32 DownT      = 2;
	OaI32 Depth      = 3;
	// VQ health: these three prevent codebook collapse.
	// CommitBeta=0.02/EmaDecay=0.999/DeadThresh=1.0 collapsed the codebook to 1 code
	// (perplexity 1.0); 0.25/0.99/2.0 gives perplexity ~73 and 25× lower recon loss.
	OaF32 CommitBeta = 0.25F;
	OaF32 EmaDecay   = 0.99F;
	OaF32 EmaEps     = 1e-5F;
	OaF32 DeadThresh = 2.0F;

	// LM architecture
	OaI32 DModel    = 384;
	OaI32 NumHeads  = 6;
	OaI32 NumLayers = 6;
	OaI32 DFfn      = 1536;
	OaI32 LmMaxSeqLen = 260;
	OaString LmFfnType = "dense";
	OaI32 LmMoeExperts = 4;
	OaI32 LmMoeTopK = 2;
	OaI32 LmMoeEvery = 2;
	OaF32 LmMoeBalanceRate = 1e-3F;
	OaF32 LmMoeAuxAlpha = 0.01F;
	OaF32 LmMoeZBeta = 1e-3F;
	OaString TextConditioning = "clip";  // clip | none
	OaString ClipTextModel = "var/model/ref/ClipText/ClipText.oam";
	OaString ClipMerges = "var/model/ref/ClipText/merges.txt";

	// Training schedule (epoch-based, Keras style)
	OaI32 TokEpochs = 50;
	OaI32 LmEpochs  = 50;
	OaI32 BatchSize = 32;
	OaI32 SeqLen    = 64;
	OaI32 LmSeqLen  = 64;
	OaF32 TokLr     = 2e-4F;
	OaF32 LmLr      = 1e-4F;
	OaF32 TokMinLr  = 2e-5F;
	OaF32 LmMinLr   = 1e-5F;
	OaI32 TokWarmup = 500;
	OaI32 LmWarmup  = 300;
	OaF32 TokWeightDecay = 0.0F;
	OaF32 LmWeightDecay  = 0.01F;
	OaI64 Seed      = 42;

	// Multiphase schedules (optional; when empty, use flat tok_epochs/lm_epochs)
	OaVec<PhaseConfig> TokPhases;
	OaVec<PhaseConfig> LmPhases;

	// Callbacks
	OaI64 CkptSaveEvery   = 0;       // 0 = epoch-end only; >0 adds mid-epoch saves
	OaI32 CkptMaxKeep     = 5;
	bool  CkptRestoreBest = false;
};

class AlmTrainCli : public OaCli<AlmTrainConfig> {
public:
	AlmTrainCli()
		: OaCli("trainalm", "Train OaAlm tokenizer + Transformer LM (two-stage)") {
		AddOption("--dataset",    Cfg_.Dataset,    "CMP dataset directory");
		AddOption("--split",      Cfg_.Split,      "Dataset split");
		AddOption("--val-split",  Cfg_.ValSplit,   "Held-out validation split");
		AddOption("--max-clips",  Cfg_.MaxClips,   "Max clips (0 = all)");
		AddOption("--val-batches", Cfg_.ValBatches, "Validation batches per epoch (0=full split)");
		AddOption("--model-dir",  Cfg_.ModelDir,   "Checkpoint root");
		AddOption("--name",       Cfg_.Name,       "Model name");
		AddOption("--stage",      Cfg_.Stage,      "Training stage: both | tok | lm | export");
		AddOption("--width",      Cfg_.Width,      "Tokenizer conv width");
		AddOption("--code-dim",   Cfg_.CodeDim,    "Codebook code dimension");
		AddOption("--codes",      Cfg_.NumCodes,   "Codebook size K");
		AddOption("--down-t",     Cfg_.DownT,      "Temporal downsample stages");
		AddOption("--depth",      Cfg_.Depth,      "Residual blocks per stage");

		AddOption("--dmodel",     Cfg_.DModel,     "LM model dimension");
		AddOption("--lm-heads",   Cfg_.NumHeads,   "LM attention heads");
		AddOption("--lm-layers",  Cfg_.NumLayers,  "LM Transformer layers");
		AddOption("--lm-ffn",     Cfg_.DFfn,       "LM FFN hidden dimension");
		AddOption("--lm-max-seq-len", Cfg_.LmMaxSeqLen, "LM learned-position capacity");
		AddOption("--lm-ffn-type", Cfg_.LmFfnType, "LM FFN policy: dense | moe | hybrid");
		AddOption("--lm-moe-experts", Cfg_.LmMoeExperts, "LM MoE expert count");
		AddOption("--lm-moe-top-k", Cfg_.LmMoeTopK, "LM experts selected per token");
		AddOption("--lm-moe-every", Cfg_.LmMoeEvery, "Hybrid LM: use MoE every Nth layer");
		AddOption("--lm-moe-balance-rate", Cfg_.LmMoeBalanceRate, "MoE aux-loss-free routing-bias rate (0 disables)");
		AddOption("--lm-moe-aux-alpha", Cfg_.LmMoeAuxAlpha, "MoE Switch load-balancing loss coefficient");
		AddOption("--lm-moe-z-beta", Cfg_.LmMoeZBeta, "MoE router z-loss coefficient");
		AddOption("--text-conditioning", Cfg_.TextConditioning,
			"LM caption conditioning: clip | none");
		AddOption("--clip-text-model", Cfg_.ClipTextModel, "Imported native OaClipTextAg .oam");
		AddOption("--clip-merges", Cfg_.ClipMerges, "Pinned CLIP merges.txt tokenizer asset");

		AddOption("--tok-epochs", Cfg_.TokEpochs,  "Tokenizer training epochs");
		AddOption("--lm-epochs",  Cfg_.LmEpochs,   "LM training epochs");
		AddOption("--batch",      Cfg_.BatchSize,  "Batch size");
		AddOption("--seq-len",    Cfg_.SeqLen,     "Tokenizer window (frames)");
		AddOption("--lm-seq-len", Cfg_.LmSeqLen,   "LM token window");
		AddOption("--tok-lr",     Cfg_.TokLr,      "Tokenizer learning rate");
		AddOption("--lm-lr",      Cfg_.LmLr,       "LM learning rate");
		AddOption("--tok-min-lr", Cfg_.TokMinLr,   "Tokenizer min LR (cosine floor)");
		AddOption("--lm-min-lr",  Cfg_.LmMinLr,    "LM min LR (cosine floor)");
		AddOption("--tok-warmup", Cfg_.TokWarmup,  "Tokenizer warmup steps");
		AddOption("--lm-warmup",  Cfg_.LmWarmup,   "LM warmup steps");
		AddOption("--tok-wd",     Cfg_.TokWeightDecay, "Tokenizer weight decay");
		AddOption("--lm-wd",      Cfg_.LmWeightDecay,  "LM weight decay");
		AddOption("--seed",       Cfg_.Seed,       "RNG seed");
		AddOption("--precision",  Cfg_.PrecisionStr, "fp32 | bf16 | tf32 | fp16");

		AddOption("--ckpt-save-every", Cfg_.CkptSaveEvery,   "Checkpoint interval (0=epoch-end only)");
		AddOption("--ckpt-max-keep",  Cfg_.CkptMaxKeep,     "Max incremental checkpoints");
		AddOption("--ckpt-restore-best", Cfg_.CkptRestoreBest, "Restore best weights at train end");
	}

	void LoadYaml(const OaYaml::Node& InYaml) override {
		Cfg_.Name = OaYaml::Get<OaString>(InYaml, "name", Cfg_.Name);
		Cfg_.Stage = OaYaml::Get<OaString>(InYaml, "stage", Cfg_.Stage);

		const OaYaml::Node m = InYaml["model"];
		Cfg_.Width    = OaYaml::Get<OaI32>(m, "width",     Cfg_.Width);
		Cfg_.CodeDim  = OaYaml::Get<OaI32>(m, "code_dim",  Cfg_.CodeDim);
		Cfg_.NumCodes = OaYaml::Get<OaI32>(m, "num_codes", Cfg_.NumCodes);
		Cfg_.DownT    = OaYaml::Get<OaI32>(m, "down_t",    Cfg_.DownT);
		Cfg_.Depth      = OaYaml::Get<OaI32>(m, "depth",       Cfg_.Depth);
		Cfg_.CommitBeta = OaYaml::Get<OaF32>(m, "commit_beta", Cfg_.CommitBeta);
		Cfg_.EmaDecay   = OaYaml::Get<OaF32>(m, "ema_decay",   Cfg_.EmaDecay);
		Cfg_.DeadThresh = OaYaml::Get<OaF32>(m, "dead_thresh", Cfg_.DeadThresh);
		Cfg_.DModel    = OaYaml::Get<OaI32>(m, "dmodel",    Cfg_.DModel);
		Cfg_.NumHeads  = OaYaml::Get<OaI32>(m, "lm_heads",  Cfg_.NumHeads);
		Cfg_.NumLayers = OaYaml::Get<OaI32>(m, "lm_layers", Cfg_.NumLayers);
		Cfg_.DFfn      = OaYaml::Get<OaI32>(m, "lm_ffn",    Cfg_.DFfn);
		Cfg_.LmMaxSeqLen = OaYaml::Get<OaI32>(m, "lm_max_seq_len", Cfg_.LmMaxSeqLen);
		Cfg_.LmFfnType = OaYaml::Get<OaString>(m, "lm_ffn_type", Cfg_.LmFfnType);
		Cfg_.LmMoeExperts = OaYaml::Get<OaI32>(m, "lm_moe_experts", Cfg_.LmMoeExperts);
		Cfg_.LmMoeTopK = OaYaml::Get<OaI32>(m, "lm_moe_top_k", Cfg_.LmMoeTopK);
		Cfg_.LmMoeEvery = OaYaml::Get<OaI32>(m, "lm_moe_every", Cfg_.LmMoeEvery);
		Cfg_.LmMoeBalanceRate = OaYaml::Get<OaF32>(m, "lm_moe_balance_rate", Cfg_.LmMoeBalanceRate);
		Cfg_.LmMoeAuxAlpha = OaYaml::Get<OaF32>(m, "lm_moe_aux_alpha", Cfg_.LmMoeAuxAlpha);
		Cfg_.LmMoeZBeta = OaYaml::Get<OaF32>(m, "lm_moe_z_beta", Cfg_.LmMoeZBeta);
		Cfg_.TextConditioning = OaYaml::Get<OaString>(m, "text_conditioning", Cfg_.TextConditioning);
		Cfg_.ClipTextModel = OaYaml::Get<OaString>(m, "clip_text_model", Cfg_.ClipTextModel);
		Cfg_.ClipMerges = OaYaml::Get<OaString>(m, "clip_merges", Cfg_.ClipMerges);

		const OaYaml::Node t = InYaml["training"];
		Cfg_.Dataset  = OaYaml::Get<OaString>(t, "dataset",   Cfg_.Dataset);
		Cfg_.Split    = OaYaml::Get<OaString>(t, "split",     Cfg_.Split);
		Cfg_.ValSplit = OaYaml::Get<OaString>(t, "val_split", Cfg_.ValSplit);
		Cfg_.MaxClips = OaYaml::Get<OaI32>   (t, "max_clips", Cfg_.MaxClips);
		Cfg_.ValBatches = OaYaml::Get<OaI32> (t, "val_batches", Cfg_.ValBatches);
		Cfg_.ModelDir = OaYaml::Get<OaString>(t, "model_dir", Cfg_.ModelDir);
		Cfg_.TokEpochs = OaYaml::Get<OaI32>   (t, "tok_epochs", Cfg_.TokEpochs);
		Cfg_.LmEpochs  = OaYaml::Get<OaI32>   (t, "lm_epochs",  Cfg_.LmEpochs);
		Cfg_.BatchSize = OaYaml::Get<OaI32>  (t, "batch",     Cfg_.BatchSize);
		Cfg_.SeqLen   = OaYaml::Get<OaI32>   (t, "seq_len",   Cfg_.SeqLen);
		Cfg_.LmSeqLen = OaYaml::Get<OaI32>   (t, "lm_seq_len", Cfg_.LmSeqLen);
		Cfg_.TokLr    = OaYaml::Get<OaF32>   (t, "tok_lr",    Cfg_.TokLr);
		Cfg_.LmLr     = OaYaml::Get<OaF32>   (t, "lm_lr",     Cfg_.LmLr);
		Cfg_.TokMinLr = OaYaml::Get<OaF32>   (t, "tok_min_lr", Cfg_.TokMinLr);
		Cfg_.LmMinLr  = OaYaml::Get<OaF32>   (t, "lm_min_lr",  Cfg_.LmMinLr);
		Cfg_.TokWarmup = OaYaml::Get<OaI32>  (t, "tok_warmup", Cfg_.TokWarmup);
		Cfg_.LmWarmup  = OaYaml::Get<OaI32>  (t, "lm_warmup",  Cfg_.LmWarmup);
		Cfg_.TokWeightDecay = OaYaml::Get<OaF32>(t, "tok_weight_decay", Cfg_.TokWeightDecay);
		Cfg_.LmWeightDecay  = OaYaml::Get<OaF32>(t, "lm_weight_decay",  Cfg_.LmWeightDecay);
		Cfg_.Seed     = OaYaml::Get<OaI64>   (t, "seed",      Cfg_.Seed);
		Cfg_.PrecisionStr = OaYaml::Get<OaString>(t, "precision", Cfg_.PrecisionStr);

		// Parse optional phase sequences
		LoadPhases(t, "tok_phases", Cfg_.TokPhases, Cfg_.TokLr, Cfg_.TokMinLr, Cfg_.TokWarmup);
		LoadPhases(t, "lm_phases",  Cfg_.LmPhases,  Cfg_.LmLr,  Cfg_.LmMinLr,  Cfg_.LmWarmup);

		const OaYaml::Node cb = InYaml["callbacks"];
		Cfg_.CkptSaveEvery   = OaYaml::Get<OaI64>(cb, "ckpt_save_every", Cfg_.CkptSaveEvery);
		Cfg_.CkptMaxKeep     = OaYaml::Get<OaI32>(cb, "ckpt_max_keep",    Cfg_.CkptMaxKeep);
		Cfg_.CkptRestoreBest = OaYaml::Get<bool> (cb, "ckpt_restore_best", Cfg_.CkptRestoreBest);
	}

	static void LoadPhases(const OaYaml::Node& InTraining, const OaString& InKey,
		OaVec<PhaseConfig>& OutPhases, OaF32 InFallbackLr, OaF32 InFallbackMinLr, OaI32 InFallbackWarmup) {
		OutPhases.Clear();
		const OaYaml::Node seq = InTraining[InKey.StdStr()];
		if (not(seq and seq.IsSequence())) return;
		for (const auto& item : seq) {
			PhaseConfig ph;
			ph.Id          = OaYaml::Get<OaString>(item, "id", ph.Id);
			ph.Epochs      = OaYaml::Get<OaI32>(item, "epochs", ph.Epochs);
			ph.Lr          = OaYaml::Get<OaF32>(item, "lr", InFallbackLr);
			ph.MinLrFrac   = OaYaml::Get<OaF32>(item, "min_lr_frac", ph.MinLrFrac);
			ph.WarmupSteps = OaYaml::Get<OaI32>(item, "warmup_steps", InFallbackWarmup);
			if (ph.Epochs > 0) OutPhases.PushBack(ph);
		}
	}

	static OaSharedPtr<OaLRScheduler> BuildScheduler(
		const OaVec<PhaseConfig>& InPhases, OaI64 InStepsPerEpoch,
		OaF32 InFallbackLr, OaF32 InFallbackMinLr, OaI32 InFallbackWarmup, OaI64 InTotalSteps) {
		if (InPhases.Empty()) {
			return OaMakeSharedPtr<OaLinearWarmupCosineScheduler>(
				InFallbackWarmup, static_cast<OaI32>(InTotalSteps), InFallbackLr, InFallbackMinLr);
		}
		OaVec<OaSharedPtr<OaLRScheduler>> subs;
		OaVec<OaU64> milestones;
		OaU64 offset = 0;
		for (const auto& ph : InPhases) {
			const OaI64 phaseSteps = static_cast<OaI64>(ph.Epochs) * InStepsPerEpoch;
			const OaF32 minLr = ph.Lr * ph.MinLrFrac;
			subs.PushBack(OaMakeSharedPtr<OaLinearWarmupCosineScheduler>(
				ph.WarmupSteps, static_cast<OaI32>(phaseSteps), ph.Lr, minLr));
			offset += static_cast<OaU64>(phaseSteps);
			milestones.PushBack(offset);
		}
		return OaMakeSharedPtr<OaSequentialScheduler>(std::move(subs), std::move(milestones));
	}
};

// ── Host helpers ──────────────────────────────────────────────────────────────

namespace {

class OptimLrMetric final : public OaMetric {
public:
	explicit OptimLrMetric(const OaOptimizer& InOpt) : Opt_(InOpt) {}
	void Update(const OaMatrix&, const OaMatrix&) override {}
	void Reset() override {}
	[[nodiscard]] OaF64 Result() const override { return Opt_.GetLr(); }
	[[nodiscard]] const char* Name() const override { return "lr"; }
	OaI32 Render(char* OutBuffer, OaI32 InBufferSize, bool) const override {
		char value[96]{};
		Formatter_.Format(value, sizeof(value), Result());
		return std::snprintf(OutBuffer, static_cast<size_t>(InBufferSize), "lr: %s", value);
	}
private:
	const OaOptimizer& Opt_;
	mutable OaMetricValueFormatter Formatter_;
};

OaMatrix MakeI32(const std::vector<OaI32>& h, const OaMatrixShape& s) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(h.data()), h.size() * sizeof(OaI32)),
		s, OaScalarType::Int32);
}

OaMatrix MakeF32(const std::vector<float>& h, const OaMatrixShape& s) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(h.data()), h.size() * sizeof(float)),
		s, OaFnMatrix::GetWeightDtype());
}

std::vector<OaF32> HostF32(const OaMatrix& InMatrix) {
	auto& ctx = OaContext::GetDefault();
	if (InMatrix.GetDtype() == OaScalarType::Float32) {
		const OaF32* p = InMatrix.DataAs<const OaF32>();
		return std::vector<OaF32>(p, p + InMatrix.NumElements());
	}
	OaMatrix f32 = OaFnMatrix::Empty(InMatrix.GetShape(), OaScalarType::Float32);
	OaFnMatrix::CastInto(InMatrix, f32);
	if (not ctx.Execute().IsOk() or not ctx.Sync().IsOk()) return {};
	const OaF32* p = f32.DataAs<const OaF32>();
	return std::vector<OaF32>(p, p + f32.NumElements());
}

OaStatus WriteNpyF32Atomic(const OaPath& InPath, const OaF32* InData,
	OaUsize InRows, OaUsize InCols) {
	if (InData == nullptr or InRows == 0 or InCols == 0)
		return OaStatus::InvalidArgument("cannot write an empty CLIP feature array");
	std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
		+ std::to_string(InRows) + ", " + std::to_string(InCols) + "), }";
	const size_t prefixBytes = 10; // magic(6) + version(2) + uint16 header length
	const size_t padding = (64 - ((prefixBytes + header.size() + 1) % 64)) % 64;
	header.append(padding, ' ');
	header.push_back('\n');
	if (header.size() > std::numeric_limits<OaU16>::max())
		return OaStatus::Error(OaStatusCode::OutOfRange, "NumPy header is too large");
	const OaUsize payloadBytes = InRows * InCols * sizeof(OaF32);
	OaVec<OaU8> bytes(prefixBytes + header.size() + payloadBytes);
	const OaU8 magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
	std::memcpy(bytes.Data(), magic, sizeof(magic));
	const OaU16 headerBytes = static_cast<OaU16>(header.size());
	bytes[8] = static_cast<OaU8>(headerBytes & 0xFFU);
	bytes[9] = static_cast<OaU8>(headerBytes >> 8U);
	std::memcpy(bytes.Data() + prefixBytes, header.data(), header.size());
	std::memcpy(bytes.Data() + prefixBytes + header.size(), InData, payloadBytes);
	const OaPath temporary(InPath.String() + ".tmp");
	OA_RETURN_IF_ERROR(OaFileIo::WriteBinary(temporary,
		OaSpan<const OaU8>(bytes.Data(), bytes.Size())));
	if (OaFileIo::Exists(InPath)) OA_RETURN_IF_ERROR(OaFileIo::RemoveFile(InPath));
	return OaFileIo::Move(temporary, InPath);
}

OaStatus WriteTextAtomic(const OaPath& InPath, OaStringView InText) {
	const OaPath temporary(InPath.String() + ".tmp");
	OA_RETURN_IF_ERROR(OaFileIo::WriteText(temporary, InText));
	if (OaFileIo::Exists(InPath)) OA_RETURN_IF_ERROR(OaFileIo::RemoveFile(InPath));
	return OaFileIo::Move(temporary, InPath);
}

struct Lcg {
	OaU64 s;
	explicit Lcg(OaU64 seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
	OaU32 U32() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<OaU32>(s >> 33); }
	float F01() { return static_cast<float>(U32()) / static_cast<float>(0xFFFFFFFFU >> 1); }
};

struct CodeUsage { OaI32 Unique; OaF32 Perplexity; };
CodeUsage CodebookUsage(const OaI32* InIds, size_t InCount, OaI32 InK) {
	std::vector<OaI64> hist(static_cast<size_t>(InK), 0);
	for (size_t i = 0; i < InCount; ++i) {
		const OaI32 c = InIds[i];
		if (c >= 0 and c < InK) ++hist[static_cast<size_t>(c)];
	}
	OaI32 uniq = 0; double ent = 0.0;
	const double n = static_cast<double>(InCount > 0 ? InCount : 1);
	for (OaI32 k = 0; k < InK; ++k) {
		const OaI64 h = hist[static_cast<size_t>(k)];
		if (h > 0) { ++uniq; const double p = static_cast<double>(h) / n; ent -= p * std::log(p); }
	}
	return {uniq, static_cast<OaF32>(std::exp(ent))};
}

struct TokenCorpusStats {
	OaI64 Tokens = 0;
	OaI32 Unique = 0;
	OaF64 UnigramPpl = 0.0;
	OaF64 BigramPpl = 0.0;
};

struct LmWindow {
	OaI32 Sequence = 0;
	OaI32 Start = 0;       // start in [SOM, motion..., EOM] next-token pairs
	OaI32 Valid = 0;       // valid next-token pairs; remainder is PAD/masked
};

std::vector<LmWindow> BuildLmWindows(
	const std::vector<std::vector<OaI32>>& InSequences, OaI32 InWindowLen) {
	std::vector<LmWindow> out;
	for (OaI32 seqIdx = 0; seqIdx < static_cast<OaI32>(InSequences.size()); ++seqIdx) {
		// [SOM, c0, ..., cN, EOM] has codes+1 next-token pairs.
		const OaI32 pairs = static_cast<OaI32>(InSequences[static_cast<size_t>(seqIdx)].size()) + 1;
		if (pairs <= InWindowLen) {
			out.push_back({seqIdx, 0, pairs});
			continue;
		}
		// Preserve the previous all-start-position training density, but never
		// invent SOM/EOM at an interior window boundary.
		for (OaI32 start = 0; start + InWindowLen <= pairs; ++start) {
			out.push_back({seqIdx, start, InWindowLen});
		}
	}
	return out;
}

void FillLmRow(const std::vector<OaI32>& InCodes, const LmWindow& InWindow,
	OaI32 InWindowLen, OaI32 InSom, OaI32 InEom, OaI32 InPad,
	OaI32* OutInput, OaI32* OutTarget, float* OutMask) {
	const OaI32 streamLen = static_cast<OaI32>(InCodes.size()) + 2;
	auto streamToken = [&](OaI32 i) -> OaI32 {
		if (i == 0) return InSom;
		if (i == streamLen - 1) return InEom;
		return InCodes[static_cast<size_t>(i - 1)];
	};
	for (OaI32 t = 0; t < InWindowLen; ++t) {
		if (t < InWindow.Valid) {
			OutInput[t] = streamToken(InWindow.Start + t);
			OutTarget[t] = streamToken(InWindow.Start + t + 1);
			OutMask[t] = 1.0F;
		} else {
			OutInput[t] = InPad;
			OutTarget[t] = InPad;
			OutMask[t] = 0.0F;
		}
	}
}

TokenCorpusStats MeasureTokenCorpus(const std::vector<std::vector<OaI32>>& InSeqs, OaI32 InK) {
	std::vector<OaI64> unigram(static_cast<size_t>(InK), 0);
	std::vector<OaI64> prevCount(static_cast<size_t>(InK), 0);
	std::vector<OaI64> bigram(static_cast<size_t>(InK) * static_cast<size_t>(InK), 0);
	TokenCorpusStats out;
	OaI64 pairs = 0;
	for (const auto& seq : InSeqs) {
		OaI32 prev = -1;
		for (const OaI32 tok : seq) {
			if (tok < 0 or tok >= InK) { prev = -1; continue; }
			++unigram[static_cast<size_t>(tok)];
			++out.Tokens;
			if (prev >= 0) {
				++bigram[static_cast<size_t>(prev) * static_cast<size_t>(InK) + static_cast<size_t>(tok)];
				++prevCount[static_cast<size_t>(prev)];
				++pairs;
			}
			prev = tok;
		}
	}
	OaF64 h1 = 0.0;
	for (const OaI64 n : unigram) if (n > 0) {
		++out.Unique;
		const OaF64 p = static_cast<OaF64>(n) / static_cast<OaF64>(std::max<OaI64>(out.Tokens, 1));
		h1 -= p * std::log(p);
	}
	OaF64 h2 = 0.0;
	if (pairs > 0) {
		for (OaI32 prev = 0; prev < InK; ++prev) {
			const OaI64 pn = prevCount[static_cast<size_t>(prev)];
			if (pn == 0) continue;
			for (OaI32 tok = 0; tok < InK; ++tok) {
				const OaI64 n = bigram[static_cast<size_t>(prev) * static_cast<size_t>(InK) + static_cast<size_t>(tok)];
				if (n == 0) continue;
				const OaF64 joint = static_cast<OaF64>(n) / static_cast<OaF64>(pairs);
				h2 -= joint * std::log(static_cast<OaF64>(n) / static_cast<OaF64>(pn));
			}
		}
	}
	out.UnigramPpl = std::exp(h1);
	out.BigramPpl = std::exp(h2);
	return out;
}

} // namespace

// ── Signal handling ──────────────────────────────────────────────────────────

static volatile sig_atomic_t gSigIntCount = 0;

static void OnSigInt(int) {
	if (gSigIntCount == 0) {
		++gSigIntCount;
	} else {
		_exit(0);
	}
}

// ── Tokenizer metrics callback ───────────────────────────────────────────────
//
// Tracks velocity loss, VQ commitment loss, and codebook usage (live codes +
// perplexity) as proper OaCbTraining callbacks. The Step lambda calls Record()
// with the GPU matrix refs; after sync in Next(), OnStepEnd reads the scalar
// values and accumulates token IDs. OnEpochEnd prints a TF-style summary:
//
//   Epoch 3: codebook 48/64 live | ppl 31.2 | vel 0.023401 | commit 0.008123

class OaCbTokMetrics : public OaCbTraining {
public:
	explicit OaCbTokMetrics(OaI32 InNumCodes) : NumCodes_(InNumCodes) {}

	// Called from Step lambda — mirror each tensor into a persistent host-coherent
	// mailbox via a Copy kernel (exactly what OaItTraining does for its loss). The
	// transient vel/commit/ids buffers are recycled once the step's graph executes
	// and backward frees the forward activations; storing the raw handles and
	// reading .At(0) later raced with that recycling and returned the still-live
	// recon buffer for all three metrics. The Copy captures the value into a stable
	// buffer that survives to the post-sync read in OnStepEnd.
	void Record(const OaMatrix& InVel, const OaMatrix& InCommit, const OaMatrix& InIds) {
		auto& ctx = OaContext::GetDefault();
		OaBufferAccess rw[] = {OaBufferAccess::Read, OaBufferAccess::Write};
		auto mirror = [&](const OaMatrix& InSrc, OaMatrix& OutDst) {
			if (not InSrc.HasStorage() or InSrc.NumElements() == 0) return;
			if (not OutDst.HasStorage() or OutDst.NumElements() != InSrc.NumElements()
				or OutDst.GetDtype() != InSrc.GetDtype()) {
				OutDst = OaFnMatrix::Zeros(OaMatrixShape{InSrc.NumElements()}, InSrc.GetDtype());
			}
			struct { OaU32 Count; } push{static_cast<OaU32>(InSrc.NumElements())};
			ctx.Add("Copy", {&InSrc, &OutDst}, rw, &push, sizeof(push),
				(static_cast<OaU32>(InSrc.NumElements()) + 255u) / 256u);
		};
		mirror(InVel, VelMailbox_);
		mirror(InCommit, CommitMailbox_);
		mirror(InIds, IdsMailbox_);
	}

	OaMetricLoss VelMetric{"vel"};
	OaMetricLoss CommitMetric{"commit"};

	[[nodiscard]] OaMetric* VelPtr() { return &VelMetric; }
	[[nodiscard]] OaMetric* CommitPtr() { return &CommitMetric; }

	void OnEpochBegin(OaItTraining& InIter) override {
		(void)InIter;
		VelMetric.Reset();
		CommitMetric.Reset();
		EpochIds_.clear();
	}

	void OnStepEnd(OaItTraining&) override {
		// The iterator completes the step before callbacks, so every mailbox value
		// is exact and contributes once to the epoch metrics.
		if (VelMailbox_.HasStorage()) VelMetric.Update(VelMailbox_.At(0));
		if (CommitMailbox_.HasStorage()) CommitMetric.Update(CommitMailbox_.At(0));
		if (IdsMailbox_.HasStorage()) {
			const OaI64 n = IdsMailbox_.NumElements();
			const OaI32* p = IdsMailbox_.DataAs<const OaI32>();
			if (p and n > 0) {
				EpochIds_.insert(EpochIds_.end(), p, p + static_cast<size_t>(n));
			}
		}
	}

	void OnEpochEnd(OaItTraining& InIter) override {
		if (EpochIds_.empty()) return;
		std::vector<OaI64> hist(static_cast<size_t>(NumCodes_), 0);
		for (OaI32 id : EpochIds_) {
			if (id >= 0 and id < NumCodes_) ++hist[static_cast<size_t>(id)];
		}
		OaI32 live = 0;
		OaF64 entropy = 0.0;
		const OaF64 total = static_cast<OaF64>(EpochIds_.size());
		for (OaI32 k = 0; k < NumCodes_; ++k) {
			const OaI64 h = hist[static_cast<size_t>(k)];
			if (h > 0) {
				++live;
				const OaF64 p = static_cast<OaF64>(h) / total;
				entropy -= p * std::log(p);
			}
		}
		const OaF64 perplexity = std::exp(entropy);
		std::printf("Epoch %lld: codebook %d/%d live | ppl %.1f | vel %.6f | commit %.6f\n",
			static_cast<long long>(InIter.Epoch()), live, NumCodes_,
			perplexity, VelMetric.Result(), CommitMetric.Result());
	}

private:
	OaI32 NumCodes_;
	OaMatrix VelMailbox_;      // persistent host-coherent mirrors (see Record)
	OaMatrix CommitMailbox_;
	OaMatrix IdsMailbox_;
	std::vector<OaI32> EpochIds_;
};

// Single autograd tokenizer path.
struct TokenizerBridge {
	OaSharedPtr<OaAlmTokenizerAg> Ptr;

	OaMatrix Encode(const OaMatrix& InX, OaI32 InB, OaI32 InT) {
		return Ptr->Encode(InX, InB, InT);
	}
	OaResidualVqResult Quantize(const OaMatrix& InZe) {
		return Ptr->Quantize(InZe);
	}
	OaMatrix Decode(const OaMatrix& InZq, OaI32 InB, OaI32 InTokLen) {
		return Ptr->Decode(InZq, InB, InTokLen);
	}
	OaVec<OaMatrix> Tokenize(const OaMatrix& InX, OaI32 InB, OaI32 InT) {
		return Ptr->Tokenize(InX, InB, InT);
	}
	OaMatrix Detokenize(const OaVec<OaMatrix>& InIds, OaI32 InB, OaI32 InTokLen) {
		return Ptr->Detokenize(InIds, InB, InTokLen);
	}
	void EmaUpdate(const OaResidualVqResult& InResult) {
		Ptr->EmaUpdate(InResult);
	}
	void Seed(const OaMatrix& InLatents) {
		Ptr->Seed(InLatents);
	}
	OaI32 DownsampleFactor() const {
		return Ptr->DownsampleFactor();
	}
	OaModule& Module() {
		return *Ptr;
	}
};

// Single configurable Transformer-prior path.
struct LmBridge {
	OaSharedPtr<OaAlmPriorAg> Ptr;

	OaMatrix Forward(const OaMatrix& InIds) {
		return Ptr->Forward(InIds);
	}
	OaMatrix Forward(const OaMatrix& InIds, const OaMatrix& InTextFeatures) {
		return Ptr->ForwardConditioned(InIds, InTextFeatures);
	}
	OaMatrix Generate(OaI32 InBatchSize, OaF32 InTemp, OaI32 InTopK, OaF32 InTopP, OaI32 InMaxLen, bool InUseCache) {
		return Ptr->Generate(InBatchSize, InTemp, InTopK, InTopP, InMaxLen, InUseCache);
	}
	OaModule& Module() {
		return *Ptr;
	}
};

// Runs before validation/checkpoint callbacks so the aux-loss-free routing bias
// is updated from the just-completed TRAIN batch, never from a validation forward.
// It also makes collapse visible in every MoE/hybrid epoch.
class OaCbMoeRouting final : public OaCbTraining {
public:
	explicit OaCbMoeRouting(OaAlmPriorAg& InPrior) : Prior_(InPrior) {}

	void OnEpochBegin(OaItTraining&) override {
		EntropySum_ = 0.0;
		MaxLoadSum_ = 0.0;
		DeadSum_ = 0;
		Samples_ = 0;
		Layers_ = 0;
	}

	void OnStepEnd(OaItTraining&) override {
		auto stats = Prior_.MoeRouteStats();
		for (const auto& s : stats) {
			EntropySum_ += s.Entropy;
			MaxLoadSum_ += s.MaxLoadRatio;
			DeadSum_ += s.DeadExperts;
			++Samples_;
		}
		Layers_ = static_cast<OaI32>(stats.Size());
		Prior_.UpdateMoeRoutingBias();
	}

	void OnEpochEnd(OaItTraining&) override {
		if (Samples_ == 0) return;
		const auto samples = static_cast<OaF64>(Samples_);
		std::printf("MoE routing: %d layers · entropy %.3f · max-load %.2fx · dead %.2f/layer\n",
			Layers_, EntropySum_ / samples, MaxLoadSum_ / samples,
			static_cast<OaF64>(DeadSum_) / samples);
	}

private:
	OaAlmPriorAg& Prior_;
	OaF64 EntropySum_ = 0.0;
	OaF64 MaxLoadSum_ = 0.0;
	OaI64 DeadSum_ = 0;
	OaI64 Samples_ = 0;
	OaI32 Layers_ = 0;
};

// ── App ─────────────────────────────────────────────────────────────────────

struct TrainAlmApp : OaComputeApp {
	AlmTrainCli Cli;

	// Stage state (kept on the app so Tick can drive both stages)
	OaI32 CurrentStage_ = 0;  // 0=tokenizer, 1=tokenize, 2=LM, 3=done, 4=export saved stages

	// Dataset
	OaDsCmp* Ds_ = nullptr;
	OaDsCmp* ValDs_ = nullptr;
	OaI32 FeatDim_ = 0;
	OaI32 NumJoints_ = 0;
	OaI32 NumClips_ = 0;

	// Tokenizer
	TokenizerBridge Tok_;
	OaUniquePtr<OaAdamW> TokOpt_;
	OaVec<OaParameter*> TokParams_;
	std::vector<std::pair<OaI32, OaI32>> TokWindows_;
	std::vector<std::pair<OaI32, OaI32>> TokValWindows_;
	size_t TokCursor_ = 0;

	// LM
	LmBridge Lm_;
	OaUniquePtr<OaAdamW> LmOpt_;
	OaVec<OaParameter*> LmParams_;
	std::vector<std::vector<OaI32>> TokenSequences_;
	std::vector<std::vector<OaI32>> ValTokenSequences_;
	std::vector<LmWindow> LmWindows_;
	std::vector<LmWindow> LmValWindows_;
	OaI32 LmTextFeatureDim_ = 0;

	// OaItTraining iterators (one per stage)
	OaUniquePtr<OaItTraining> TokIter_;
	OaUniquePtr<OaItTraining> LmIter_;
	OaI64 TokStepsPerEpoch_ = 0;
	OaI64 LmStepsPerEpoch_  = 0;

	// LR schedulers (base-class pointer holds either single or sequential)
	OaSharedPtr<OaLRScheduler> TokSched_;
	OaSharedPtr<OaLRScheduler> LmSched_;
	OaUniquePtr<OaCbLrScheduler> TokSchedCb_;
	OaUniquePtr<OaCbLrScheduler> LmSchedCb_;

	// Callbacks — tokenizer stage
	OaUniquePtr<OaMetricLoss> TokLossMetric_;
	OaUniquePtr<OaCbTokMetrics> TokExtraCb_;
	OaUniquePtr<OaCbProgressBar> TokBar_;
	OaUniquePtr<OaCbValidation> TokValidationCb_;
	OaUniquePtr<OaCbSummary>  TokSummary_;
	OaUniquePtr<OaCheckpointManager> TokMgr_;
	OaUniquePtr<OaCbCheckpoint> TokCkptCb_;

	// Callbacks — LM stage
	OaUniquePtr<OaMetricLoss> LmLossMetric_;
	OaUniquePtr<OptimLrMetric> LmLrMetric_;
	OaUniquePtr<OaCbProgressBar> LmBar_;
	OaUniquePtr<OaCbMoeRouting> LmMoeRoutingCb_;
	OaUniquePtr<OaCbValidation> LmValidationCb_;
	OaUniquePtr<OaCbSummary>  LmSummary_;
	OaUniquePtr<OaCheckpointManager> LmMgr_;
	OaUniquePtr<OaCbCheckpoint> LmCkptCb_;

	OaBool ExitRequested_ = false;

	OaAlmPriorConfig MakeLmConfig(OaI32 InTextFeatureDim) const {
		const auto& c = Cli.GetConfig();
		OaAlmPriorConfig cfg;
		cfg.SyncVocab(c.NumCodes);
		cfg.DModel = c.DModel;
		cfg.NumHeads = c.NumHeads;
		cfg.NumLayers = c.NumLayers;
		cfg.DFfn = c.DFfn;
		cfg.TextFeatureDim = InTextFeatureDim;
		cfg.SeqLen = c.LmSeqLen + 1;
		cfg.MaxSeqLen = c.LmMaxSeqLen;
		if (c.LmFfnType == "dense") cfg.FfnType = OaAlmFfnType::Dense;
		else if (c.LmFfnType == "moe") cfg.FfnType = OaAlmFfnType::Moe;
		else if (c.LmFfnType == "hybrid") cfg.FfnType = OaAlmFfnType::Hybrid;
		cfg.MoeNumExperts = c.LmMoeExperts;
		cfg.MoeExpertsPerToken = c.LmMoeTopK;
		cfg.MoeEvery = c.LmMoeEvery;
		cfg.MoeBalanceRate = c.LmMoeBalanceRate;
		cfg.MoeAuxLossAlpha = c.LmMoeAuxAlpha;
		cfg.MoeRouterZLossBeta = c.LmMoeZBeta;
		return cfg;
	}

	OaStatus SaveAlmBundle() {
		const auto& c = Cli.GetConfig();
		if (not Tok_.Ptr or not Lm_.Ptr)
			return OaStatus::InvalidArgument("ALM tokenizer and prior must both be loaded");
		const OaString bundleDir = c.ModelDir + "/" + c.Name;
		const OaString bundlePath = bundleDir + "/" + c.Name + ".oam";
		(void)OaFileIo::CreateDirectories(OaPath(bundleDir));
		const OaString textEncoder = LmTextFeatureDim_ > 0
			? OaString("openai/clip-vit-large-patch14") : OaString();
		OaSharedPtr<OaAlmAg> alm;
		if (LmTextFeatureDim_ > 0) {
			auto clip = OaClipTextAg::LoadOam(c.ClipTextModel);
			if (clip.IsError()) return clip.GetStatus();
			auto merges = OaFileIo::ReadBinary(OaPath(c.ClipMerges));
			if (merges.IsError()) return merges.GetStatus();
			const auto& bytes = merges.GetValue();
			alm = OaMakeSharedPtr<OaAlmAg>(Tok_.Ptr, Lm_.Ptr,
				OaStdMove(clip.GetValue()), OaSpan<const OaU8>(bytes.Data(), bytes.Size()), textEncoder);
		} else {
			alm = OaMakeSharedPtr<OaAlmAg>(Tok_.Ptr, Lm_.Ptr, textEncoder);
		}
		const OaStatus status = alm->SaveBundle(bundlePath);
		if (status.IsOk()) {
			std::printf("Saved OaAlm: %s\n", bundlePath.CStr());
			std::printf("  stage checkpoints: %s · %s\n",
				TokMgr_->GetMasterPath().CStr(), LmMgr_->GetMasterPath().CStr());
			std::fflush(stdout);
		}
		return status;
	}

	OaStatus ExportSavedStages() {
		const auto& c = Cli.GetConfig();
		LmTextFeatureDim_ = c.TextConditioning == "clip"
			? OaClipTextConfig::ViTL14().ProjectionDim : 0;
		if (c.TextConditioning != "clip" and c.TextConditioning != "none")
			return OaStatus::InvalidArgument(OaString("unknown text_conditioning: ") + c.TextConditioning);
		const OaAlmPriorConfig lmCfg = MakeLmConfig(LmTextFeatureDim_);
		if (c.LmFfnType != "dense" and c.LmFfnType != "moe" and c.LmFfnType != "hybrid")
			return OaStatus::InvalidArgument(OaString("unknown lm_ffn_type: ") + c.LmFfnType);
		Lm_.Ptr = OaMakeSharedPtr<OaAlmPriorAg>(lmCfg);
		LmParams_ = Lm_.Module().AllParameterPtrs();
		LmOpt_ = OaMakeUniquePtr<OaAdamW>(LmParams_, c.LmLr, 0.9F, 0.99F, 1e-8F, c.LmWeightDecay);
		LmMgr_ = OaMakeUniquePtr<OaCheckpointManager>(OaCheckpointManagerConfig{
			.Dir = c.ModelDir, .ModelName = c.Name + "Prior",
			.MaxKeep = c.CkptMaxKeep, .MetricName = "val_loss", .LowerIsBetter = true});
		const OaString priorPath = LmMgr_->GetMasterPath();
		const OaStatus loaded = Lm_.Module().Load(priorPath, *LmOpt_);
		if (not loaded.IsOk()) return OaStatus::NotFound(
			OaString("failed to load ALM prior from ") + priorPath + ": " + loaded.GetMessage());
		(void)OaContext::GetDefault().Execute();
		(void)OaContext::GetDefault().Sync();
		const OaStatus saved = SaveAlmBundle();
		if (saved.IsOk()) CurrentStage_ = 3;
		return saved;
	}

	OaStatus BakeNativeClipFeatures() {
		const auto& c = Cli.GetConfig();
		if (not OaFileIo::IsFile(OaPath(c.ClipTextModel)) or
			not OaFileIo::IsFile(OaPath(c.ClipMerges))) {
			return OaStatus::NotFound(OaString("native CLIP assets are missing: ")
				+ c.ClipTextModel + ", " + c.ClipMerges);
		}

		struct Record { OaString Id; OaUsize Offset = 0; OaUsize Count = 0; };
		OaVec<Record> records;
		OaVec<OaString> prompts;
		prompts.PushBack(OaString()); // unconditional feature row
		std::unordered_set<std::string> seen;
		auto append = [&](const OaDsCmp& dataset) {
			for (OaI32 clip = 0; clip < dataset.NumClips(); ++clip) {
				const OaString& id = dataset.ClipId(clip);
				if (not seen.emplace(id.CStr()).second) continue;
				const auto& captions = dataset.ClipCaptions(clip);
				if (captions.Empty()) continue;
				Record record{id, prompts.Size(), captions.Size()};
				for (const auto& caption : captions) prompts.PushBack(caption.Text);
				records.PushBack(OaStdMove(record));
			}
		};
		append(*Ds_);
		if (ValDs_) append(*ValDs_);
		OaDsCmp test(c.Dataset, "test", c.MaxClips);
		if (test.Ok()) append(test);
		if (records.Empty()) return OaStatus::Error("CMP has no caption records to bake");

		auto clipResult = OaClipTextAg::LoadOam(c.ClipTextModel);
		if (clipResult.IsError()) return clipResult.GetStatus();
		auto clip = OaStdMove(clipResult.GetValue());
		OaClipTokenizer tokenizer;
		OA_RETURN_IF_ERROR(tokenizer.LoadMerges(OaPath(c.ClipMerges)));
		const OaI32 contextLength = clip->Config().ContextLength;
		const OaI32 dim = clip->Config().ProjectionDim;
		constexpr OaI32 batchSize = 16;
		std::vector<OaF32> features(prompts.Size() * static_cast<OaUsize>(dim));
		auto& ctx = OaContext::GetDefault();
		std::printf("Native CLIP bake: %zu captions + unconditional · batch=%d · Vulkan\n",
			prompts.Size() - 1, batchSize);
		for (OaUsize start = 0; start < prompts.Size(); start += batchSize) {
			const OaI32 count = static_cast<OaI32>(
				std::min<OaUsize>(batchSize, prompts.Size() - start));
			auto encoded = tokenizer.Encode(
				OaSpan<const OaString>(prompts.Data() + start, static_cast<OaUsize>(count)),
				contextLength, true);
			if (encoded.IsError()) return encoded.GetStatus();
			const auto& batch = encoded.GetValue();
			auto ids = OaFnMatrix::FromInt32(
				OaSpan<const OaI32>(batch.TokenIds.Data(), batch.TokenIds.Size()),
				OaMatrixShape{count, contextLength}, OaScalarType::Int32);
			auto eos = OaFnMatrix::FromInt32(
				OaSpan<const OaI32>(batch.FlatEosRows.Data(), batch.FlatEosRows.Size()),
				OaMatrixShape{count}, OaScalarType::Int32);
			OaGradNo noGrad;
			auto output = clip->ForwardTokens(ids, eos);
			OA_RETURN_IF_ERROR(ctx.Execute());
			OA_RETURN_IF_ERROR(ctx.Sync());
			const auto host = HostF32(output);
			if (host.size() != static_cast<size_t>(count) * dim)
				return OaStatus::Error("native CLIP returned an invalid feature shape");
			std::memcpy(features.data() + start * static_cast<OaUsize>(dim),
				host.data(), host.size() * sizeof(OaF32));
			ctx.Clear();
			const OaUsize done = start + static_cast<OaUsize>(count);
			std::printf("  encoded %zu/%zu captions\r", done, prompts.Size());
			std::fflush(stdout);
		}
		std::printf("\n");

		const OaPath outDir = OaPath(c.Dataset) / "text_feats";
		OA_RETURN_IF_ERROR(OaFileIo::CreateDirectories(outDir));
		OA_RETURN_IF_ERROR(WriteNpyF32Atomic(outDir / "uncond.npy",
			features.data(), 1, static_cast<OaUsize>(dim)));
		for (const auto& record : records) {
			OA_RETURN_IF_ERROR(WriteNpyF32Atomic(outDir / (record.Id + ".npy"),
				features.data() + record.Offset * static_cast<OaUsize>(dim),
				record.Count, static_cast<OaUsize>(dim)));
		}
		const OaString manifest = OaString("{\n")
			+ "  \"format\": \"oa_clip_text_v1\",\n"
			+ "  \"model\": \"openai/clip-vit-large-patch14\",\n"
			+ "  \"feature\": \"CLIPTextModelWithProjection.text_embeds\",\n"
			+ "  \"dtype\": \"float32\",\n"
			+ "  \"dim\": " + OaToString(static_cast<OaI64>(dim)) + ",\n"
			+ "  \"caption_order\": \"texts/<id>.txt line order\",\n"
			+ "  \"max_length\": " + OaToString(static_cast<OaI64>(contextLength)) + ",\n"
			+ "  \"producer\": \"OaClipTextAg/Vulkan\"\n"
			+ "}\n";
		OA_RETURN_IF_ERROR(WriteTextAtomic(outDir / "manifest.json", manifest));
		std::printf("Native CLIP cache: %zu clips -> %s\n",
			records.Size(), outDir.String().CStr());
		return OaStatus::Ok();
	}

	OaStatus ReloadDatasetsWithTextFeatures() {
		const auto& c = Cli.GetConfig();
		const OaI32 expectedClips = NumClips_;
		delete Ds_;
		delete ValDs_;
		Ds_ = new OaDsCmp(c.Dataset, c.Split, c.MaxClips);
		ValDs_ = new OaDsCmp(c.Dataset, c.ValSplit, c.MaxClips);
		if (not Ds_->Ok() or Ds_->NumClips() != expectedClips)
			return OaStatus::Error("failed to reload training data after native CLIP bake");
		if (not ValDs_->Ok()) {
			delete ValDs_;
			ValDs_ = nullptr;
		}
		return OaStatus::Ok();
	}

	int Setup(int argc, char** argv) override {
#if defined(_WIN32)
		std::signal(SIGINT, OnSigInt);
		std::signal(SIGTERM, OnSigInt);
#else
		struct sigaction sa{};
		sa.sa_handler = OnSigInt;
		sigemptyset(&sa.sa_mask);
		sa.sa_flags = SA_RESTART;
		sigaction(SIGINT, &sa, nullptr);
		sigaction(SIGTERM, &sa, nullptr);
#endif
		if (not Cli.Parse(argc, argv)) { IsRunning = false; }
		const auto& c = Cli.GetConfig();
		EngineConfig_.Precision = c.Precision();
		return 0;
	}

	OaStatus Init() override {
		const auto& c = Cli.GetConfig();
		auto& ctx = OaContext::GetDefault();

		Ds_ = new OaDsCmp(c.Dataset, c.Split, c.MaxClips);
		if (not Ds_->Ok()) {
			OA_LOG_ERROR(OaLogComponent::ML, "trainalm: failed to load CMP from %s", c.Dataset.CStr());
			IsRunning = false; return OaStatus::Ok();
		}
		FeatDim_   = Ds_->FeatDim();
		NumJoints_ = Ds_->NumJoints();
		NumClips_  = Ds_->NumClips();
		ValDs_ = new OaDsCmp(c.Dataset, c.ValSplit, c.MaxClips);
		if (not ValDs_->Ok()) {
			OA_LOG_WARN(OaLogComponent::ML,
				"trainalm: validation split '%s' unavailable; val_loss/checkpoint selection disabled",
				c.ValSplit.CStr());
			delete ValDs_;
			ValDs_ = nullptr;
		}

		OaFnMatrix::SetRngSeed(static_cast<OaU64>(c.Seed));

		// ── Build tokenizer ──
		OaAlmTokenizerConfig tokCfg;
		tokCfg.InputDim   = FeatDim_;
		tokCfg.Width      = c.Width;
		tokCfg.CodeDim    = c.CodeDim;
		tokCfg.NumCodes   = c.NumCodes;
		tokCfg.DownT      = c.DownT;
		tokCfg.Depth      = c.Depth;
		tokCfg.CommitBeta = c.CommitBeta;
		tokCfg.EmaDecay   = c.EmaDecay;
		tokCfg.EmaEps     = c.EmaEps;
		tokCfg.DeadThresh = c.DeadThresh;
		Tok_.Ptr = OaMakeSharedPtr<OaAlmTokenizerAg>(tokCfg);

		const OaI32 B = c.BatchSize;
		const OaI32 T = c.SeqLen;
		const OaI32 N = B * T;

		// Seed codebook
		if (N >= c.NumCodes) {
			std::vector<float> seed(static_cast<size_t>(B * T) * FeatDim_);
			for (OaI32 b = 0; b < B; ++b) {
				const OaI32 clipIdx = b % NumClips_;
				const OaI32 frames = Ds_->ClipFrames(clipIdx);
				const OaI32 start = frames > T ? (frames - T) / 2 : 0;
				const OaF32* src = Ds_->ClipData(clipIdx) + static_cast<size_t>(start) * FeatDim_;
				float* dst = seed.data() + static_cast<size_t>(b) * T * FeatDim_;
				std::memcpy(dst, src, static_cast<size_t>(std::min(T, frames)) * FeatDim_ * sizeof(float));
			}
			auto seedX = MakeF32(seed, OaMatrixShape{B, T, FeatDim_});
			auto z0 = Tok_.Encode(seedX, B, T);
			Tok_.Seed(z0);
			ctx.Clear();
		}

		// Build sliding windows
		for (OaI32 ci = 0; ci < NumClips_; ++ci) {
			const OaI32 frames = Ds_->ClipFrames(ci);
			if (frames < T) continue;
			OaI32 lastStart = -1;
			for (OaI32 s = 0; s + T <= frames; s += T / 2) {
				TokWindows_.emplace_back(ci, s);
				lastStart = s;
			}
			// Include the exact clip tail when the overlap stride does not land on it.
			// This covers every real frame without introducing padded VQ assignments.
			const OaI32 tailStart = frames - T;
			if (tailStart != lastStart) TokWindows_.emplace_back(ci, tailStart);
		}
		if (ValDs_) {
			for (OaI32 ci = 0; ci < ValDs_->NumClips(); ++ci) {
				const OaI32 frames = ValDs_->ClipFrames(ci);
				if (frames < T) continue;
				OaI32 lastStart = -1;
				for (OaI32 s = 0; s + T <= frames; s += T / 2) {
					TokValWindows_.emplace_back(ci, s);
					lastStart = s;
				}
				const OaI32 tailStart = frames - T;
				if (tailStart != lastStart) TokValWindows_.emplace_back(ci, tailStart);
			}
			if (TokValWindows_.empty()) {
				OA_LOG_WARN(OaLogComponent::ML, "trainalm: validation split has no tokenizer windows");
			}
		}
		if (TokWindows_.empty()) {
			OA_LOG_ERROR(OaLogComponent::ML, "trainalm: no clip long enough for seq_len %d", T);
			IsRunning = false; return OaStatus::Ok();
		}
		{ Lcg r(0xABCD); for (size_t i = TokWindows_.size(); i > 1; --i) std::swap(TokWindows_[i - 1], TokWindows_[r.U32() % i]); }

		// ── Tokenizer optimizer + OaItTraining ──
		TokParams_ = Tok_.Module().AllParameterPtrs();
		TokOpt_ = OaMakeUniquePtr<OaAdamW>(TokParams_, c.TokLr, 0.9F, 0.99F, 1e-8F, c.TokWeightDecay);

		TokLossMetric_ = OaMakeUniquePtr<OaMetricLoss>("recon");

		TokExtraCb_ = OaMakeUniquePtr<OaCbTokMetrics>(c.NumCodes);

		TokBar_ = OaMakeUniquePtr<OaCbProgressBar>(10);
		TokBar_->AddMetric(TokLossMetric_.get());
		TokBar_->AddMetric(TokExtraCb_->VelPtr());
		TokBar_->AddMetric(TokExtraCb_->CommitPtr());

		if (not TokValWindows_.empty()) {
			TokValidationCb_ = OaMakeUniquePtr<OaCbValidation>(
				[this](OaItTraining& InIter) { return EvaluateTokenizer(InIter); });
		}

		TokSummary_ = OaMakeUniquePtr<OaCbSummary>(true);
		if (TokValidationCb_) TokSummary_->SetValidationMetric(TokValidationCb_->MetricPtr());

		TokMgr_ = OaMakeUniquePtr<OaCheckpointManager>(OaCheckpointManagerConfig{
			.Dir           = c.ModelDir,
			.ModelName     = c.Name + "Tok",
			.MaxKeep       = c.CkptMaxKeep,
			.MetricName    = TokValidationCb_ ? OaString("val_loss") : OaString("recon"),
			.LowerIsBetter = true,
		});
		(void)OaFileIo::CreateDirectories(OaPath(TokMgr_->GetModelDir()));
		TokCkptCb_ = OaMakeUniquePtr<OaCbCheckpoint>(
			*TokMgr_, Tok_.Module(), *TokOpt_, c.CkptSaveEvery,
			TokValidationCb_ ? TokValidationCb_->MetricPtr() : nullptr, c.CkptRestoreBest);

		OaItTrainingConfig tokIterCfg;
		const OaI64 tokStepsPerEpoch = static_cast<OaI64>(TokWindows_.size() + static_cast<size_t>(B) - 1) / B;
		TokStepsPerEpoch_ = tokStepsPerEpoch;
		const OaI32 tokTotalEpochs = c.TokPhases.Empty() ? c.TokEpochs
			: [&c]() { OaI32 sum = 0; for (const auto& ph : c.TokPhases) sum += ph.Epochs; return sum; }();
		tokIterCfg.TotalSteps     = static_cast<OaI64>(tokTotalEpochs) * tokStepsPerEpoch;
		tokIterCfg.StepsPerEpoch  = tokStepsPerEpoch;
		tokIterCfg.BatchSize      = B;
		tokIterCfg.SequenceLength = T;
		tokIterCfg.SequenceUnit   = "frame";
		tokIterCfg.TimerName      = "tokenizer_step";
		tokIterCfg.Metrics        = {TokLossMetric_.get()};
		// LR scheduler: phased or single warmup+cosine
		TokSched_ = Cli.BuildScheduler(c.TokPhases, tokStepsPerEpoch,
			c.TokLr, c.TokMinLr, c.TokWarmup, tokIterCfg.TotalSteps);
		TokOpt_->SetLr(TokSched_->GetLr(1));
		TokSchedCb_ = OaMakeUniquePtr<OaCbLrScheduler>(*TokSched_, *TokOpt_);

		tokIterCfg.Callbacks = {TokExtraCb_.get(), TokBar_.get()};
		if (TokValidationCb_) tokIterCfg.Callbacks.push_back(TokValidationCb_.get());
		tokIterCfg.Callbacks.push_back(TokCkptCb_.get());
		tokIterCfg.Callbacks.push_back(TokSchedCb_.get());
		tokIterCfg.Callbacks.push_back(TokSummary_.get());

		TokIter_ = OaMakeUniquePtr<OaItTraining>(*TokOpt_, tokIterCfg);

		// ── Stage selection ──
		if (c.Stage == "lm" or c.Stage == "export") {
			// Load the saved tokenizer for LM training or bundle recovery.
			const OaString tokPath = TokMgr_->GetMasterPath();
			auto tokParams = Tok_.Module().AllParameterPtrs();
			auto dummyOpt = OaMakeUniquePtr<OaAdamW>(tokParams, 0.0F);
			auto st = Tok_.Module().Load(tokPath, *dummyOpt);
			if (not st.IsOk()) {
				OA_LOG_ERROR(OaLogComponent::ML, "trainalm: failed to load tokenizer from %s: %s",
					tokPath.CStr(), st.GetMessage().CStr());
				IsRunning = false; return OaStatus::Ok();
			}
			std::printf("Loaded tokenizer: %s\n", tokPath.CStr());
			(void)ctx.Execute(); (void)ctx.Sync();
			CurrentStage_ = c.Stage == "export" ? 4 : 1;
		} else if (c.Stage != "both" and c.Stage != "tok") {
			OA_LOG_ERROR(OaLogComponent::ML,
				"trainalm: unknown stage '%s' (expected both, tok, lm, or export)", c.Stage.CStr());
			IsRunning = false; return OaStatus::Ok();
		}

		// ── Print header ──
		std::printf("\ntrainalm — OaAlm tokenizer + Transformer LM (%s FFN)\n", c.LmFfnType.CStr());
		std::printf("  data: %s · %d clips · featDim %d · %d joints\n",
			c.Dataset.CStr(), NumClips_, FeatDim_, NumJoints_);
		if (ValDs_) std::printf("  validation: %s · %d clips · %zu tokenizer windows\n",
			c.ValSplit.CStr(), ValDs_->NumClips(), TokValWindows_.size());
		std::printf("Tokenizer: %d epochs × %lld steps/epoch · B=%d · T=%d · lr %.1e · ckpt %s\n",
			c.TokEpochs, static_cast<long long>(tokStepsPerEpoch),
			B, T, static_cast<double>(c.TokLr),
			c.CkptSaveEvery > 0 ? "step+epoch" : "epoch-end");
		std::fflush(stdout);

		return OaStatus::Ok();
	}

	OaStatus Tick() override {
		// ── Graceful exit ──
		if (gSigIntCount > 0) {
			gSigIntCount = 0;
			if (not ExitRequested_) {
				ExitRequested_ = true;
				OA_LOG_INFO(OaLogComponent::ML, "Interrupted. Press Ctrl+C again to exit.");
				return OaStatus::Ok();
			}
			OA_LOG_INFO(OaLogComponent::ML, "Exiting...");
			if (TokIter_) (void)TokIter_->Finish();
			if (LmIter_) {
				const OaStatus finish = LmIter_->Finish();
				if (finish.IsOk() and Lm_.Ptr) {
					OA_LOG_INFO(OaLogComponent::ML,
						"Publishing best completed ALM checkpoint before exit...");
					const OaStatus saved = SaveAlmBundle();
					if (not saved.IsOk()) OA_LOG_ERROR(OaLogComponent::ML,
						"trainalm: interrupted ALM bundle save failed: %s",
						saved.GetMessage().CStr());
				}
			}
			IsRunning = false;
			return OaStatus::Ok();
		}

		if (CurrentStage_ == 0) return TickTokenizer();
		if (CurrentStage_ == 1) return StartLM();
		if (CurrentStage_ == 2) return TickLM();
		if (CurrentStage_ == 4) {
			const OaStatus status = ExportSavedStages();
			if (not status.IsOk()) OA_LOG_ERROR(OaLogComponent::ML,
				"trainalm: saved-stage export failed: %s", status.GetMessage().CStr());
			IsRunning = false;
			return OaStatus::Ok();
		}
		IsRunning = false;
		return OaStatus::Ok();
	}

	OaStatus TickTokenizer() {
		const auto& c = Cli.GetConfig();
		auto& ctx = OaContext::GetDefault();
		const OaI32 B = c.BatchSize;
		const OaI32 T = c.SeqLen;
		const OaI32 tokLen = T / Tok_.DownsampleFactor();

		if (TokIter_->IsDone()) {
			(void)TokIter_->Finish();
			std::printf("Tokenizer training complete.\n");
			std::fflush(stdout);

			// Dataset-wide codebook usage
			{
				std::vector<OaI32> allTok;
				for (OaI32 ci = 0; ci < NumClips_; ++ci) {
					const OaI32 frames = Ds_->ClipFrames(ci);
					if (frames < T) continue;
					std::vector<float> clip(static_cast<size_t>(frames) * FeatDim_);
					std::memcpy(clip.data(), Ds_->ClipData(ci), clip.size() * sizeof(float));
					auto x = MakeF32(clip, OaMatrixShape{1, frames, FeatDim_});
					auto ids = Tok_.Tokenize(x, 1, frames)[0];
					(void)ctx.Execute(); (void)ctx.Sync();
					const OaI64 n = ids.NumElements();
					const OaI32* p = ids.DataAs<const OaI32>();
					allTok.insert(allTok.end(), p, p + n);
					ctx.Clear();
				}
				const auto u = CodebookUsage(allTok.data(), allTok.size(), c.NumCodes);
				std::printf("Codebook usage (dataset): %d/%d live | perplexity %.1f | %zu tokens\n",
					u.Unique, c.NumCodes, static_cast<double>(u.Perplexity), allTok.size());
				std::fflush(stdout);
			}

			CurrentStage_ = (c.Stage == "tok") ? 3 : 1;
			return OaStatus::Ok();
		}

		// ── One tokenizer step ──
		TokIter_->Step([&]() {
			const size_t cursor = TokCursor_;
			TokCursor_ = (TokCursor_ + static_cast<size_t>(B)) % TokWindows_.size();

			std::vector<float> batch(static_cast<size_t>(B * T) * FeatDim_);
			for (OaI32 b = 0; b < B; ++b) {
				const auto& w = TokWindows_[(cursor + static_cast<size_t>(b)) % TokWindows_.size()];
				const OaF32* src = Ds_->ClipData(w.first) + static_cast<size_t>(w.second) * FeatDim_;
				float* dst = batch.data() + static_cast<size_t>(b) * T * FeatDim_;
				std::memcpy(dst, src, static_cast<size_t>(T) * FeatDim_ * sizeof(float));
			}
			auto X = MakeF32(batch, OaMatrixShape{B, T, FeatDim_});

			TokOpt_->ZeroGrad();
			// Autograd nodes are attached during Forward only while a tape is active.
			// Keep it alive across the complete Ag forward/loss construction.
			auto tape = OaMakeUniquePtr<OaGradientTape>();
			auto z = Tok_.Encode(X, B, T);
			auto q = Tok_.Quantize(z);
			auto rec = Tok_.Decode(q.Quantized, B, tokLen);
			auto xFlat = X.Reshape(OaMatrixShape{static_cast<OaI64>(B) * T, FeatDim_});
			auto recon = OaFnLoss::SmoothL1Mean(rec, xFlat);

			auto rec3d = rec.Reshape(OaMatrixShape{B, T, FeatDim_});
			auto velLoss = OaFnLoss::VelSmoothL1(rec3d, X);

			auto loss = recon + velLoss + q.CommitLoss;
			tape->Backward(loss);

			if (TokIter_->StepCount() == 0) {
				(void)ctx.Execute(); (void)ctx.Sync();
				std::printf("[DEBUG] X dtype=%d shape=(%lld,%lld,%lld) first=%f\n",
					static_cast<int>(X.GetDtype()), X.Size(0), X.Size(1), X.Size(2), X.At(0));
				std::printf("[DEBUG] rec dtype=%d numel=%lld first=%f rms=%f\n",
					static_cast<int>(rec.GetDtype()), rec.NumElements(), rec.At(0),
					std::sqrt(static_cast<double>(OaFnMatrix::Sum(OaFnMatrix::Mul(rec, rec)).At(0)) / rec.NumElements()));
				std::printf("[DEBUG] recon dtype=%d val=%f commit=%f\n",
					static_cast<int>(recon.GetDtype()), recon.At(0), q.CommitLoss.At(0));
				std::fflush(stdout);
			}

			TokIter_->RecordLoss(recon);
			TokExtraCb_->Record(velLoss, q.CommitLoss, q.Idx[0]);

			Tok_.EmaUpdate(q);
		});

		const OaI64 step = TokIter_->StepCount();
		const float lv = TokIter_->LastLoss();
		if (not std::isfinite(lv)) {
			OA_LOG_ERROR(OaLogComponent::ML, "trainalm: tokenizer diverged at step %lld",
				static_cast<long long>(step));
			TokIter_->RequestStop();
		}

		return OaStatus::Ok();
	}

	OaValidationResult EvaluateTokenizer(OaItTraining&) {
		const auto& c = Cli.GetConfig();
		const OaI32 B = c.BatchSize;
		const OaI32 T = c.SeqLen;
		const OaI32 tokLen = T / Tok_.DownsampleFactor();
		const OaI64 allBatches = static_cast<OaI64>(
			(TokValWindows_.size() + static_cast<size_t>(B) - 1) / static_cast<size_t>(B));
		const OaI64 batches = c.ValBatches > 0
			? std::min<OaI64>(allBatches, c.ValBatches) : allBatches;
		OaF64 weightedLoss = 0.0;
		OaF64 weightedVel = 0.0;
		OaF64 weightedMpjpeCm = 0.0;
		OaF64 footSkateCm = 0.0;
		OaI64 footSkateFrames = 0;
		OaI64 contactCorrect = 0;
		OaI64 contactTotal = 0;
		OaI64 samples = 0;
		std::vector<OaI32> validationTokens;
		auto& ctx = OaContext::GetDefault();

		for (OaI64 batchIdx = 0; batchIdx < batches; ++batchIdx) {
			const size_t begin = static_cast<size_t>(batchIdx) * static_cast<size_t>(B);
			const OaI32 n = static_cast<OaI32>(
				std::min<size_t>(static_cast<size_t>(B), TokValWindows_.size() - begin));
			ctx.Clear();
			std::vector<float> batch(static_cast<size_t>(n * T) * FeatDim_);
			for (OaI32 b = 0; b < n; ++b) {
				const auto& w = TokValWindows_[begin + static_cast<size_t>(b)];
				const OaF32* src = ValDs_->ClipData(w.first)
					+ static_cast<size_t>(w.second) * FeatDim_;
				float* dst = batch.data() + static_cast<size_t>(b) * T * FeatDim_;
				std::memcpy(dst, src, static_cast<size_t>(T) * FeatDim_ * sizeof(float));
			}
			auto x = MakeF32(batch, OaMatrixShape{n, T, FeatDim_});
			auto z = Tok_.Encode(x, n, T);
			auto q = Tok_.Quantize(z);
			auto rec = Tok_.Decode(q.Quantized, n, tokLen);
			auto xFlat = x.Reshape(OaMatrixShape{static_cast<OaI64>(n) * T, FeatDim_});
			auto loss = OaFnLoss::SmoothL1Mean(rec, xFlat);
			auto vel = OaFnLoss::VelSmoothL1(
				rec.Reshape(OaMatrixShape{n, T, FeatDim_}), x);
			if (not ctx.Execute().IsOk() or not ctx.Sync().IsOk()) return {};
			weightedLoss += static_cast<OaF64>(loss.At(0)) * n;
			weightedVel += static_cast<OaF64>(vel.At(0)) * n;
			const auto& tokenIds = q.Idx[0];
			const OaI32* tokenData = tokenIds.DataAs<const OaI32>();
			validationTokens.insert(validationTokens.end(), tokenData,
				tokenData + tokenIds.NumElements());

			auto pred = HostF32(rec);
			if (pred.size() != batch.size()) return {};
			ValDs_->Denormalize(pred.data(), static_cast<OaI64>(n) * T);
			ValDs_->Denormalize(batch.data(), static_cast<OaI64>(n) * T);
			for (OaI32 b = 0; b < n; ++b) {
				const size_t featureOffset = static_cast<size_t>(b) * T * FeatDim_;
				auto predWorld = OaHumanMl3dRecoverWorldJoints(
					OaSpan<const OaF32>(pred.data() + featureOffset,
						static_cast<size_t>(T) * FeatDim_), T, FeatDim_);
				auto targetWorld = OaHumanMl3dRecoverWorldJoints(
					OaSpan<const OaF32>(batch.data() + featureOffset,
						static_cast<size_t>(T) * FeatDim_), T, FeatDim_);
				weightedMpjpeCm += OaHumanMl3dMpjpeCm(
					OaSpan<const OaF32>(predWorld.Data(), predWorld.Size()),
					OaSpan<const OaF32>(targetWorld.Data(), targetWorld.Size()));

				for (OaI32 t = 0; t < T; ++t) {
					for (OaI32 cidx = 0; cidx < 4; ++cidx) {
						const size_t k = featureOffset + static_cast<size_t>(t) * FeatDim_
							+ static_cast<size_t>(FeatDim_ - 4 + cidx);
						contactCorrect += (pred[k] >= 0.5F) == (batch[k] >= 0.5F) ? 1 : 0;
						++contactTotal;
					}
					if (t == 0) continue;
					for (OaI32 foot = 0; foot < 2; ++foot) {
						const size_t currFeat = featureOffset + static_cast<size_t>(t) * FeatDim_;
						const size_t prevFeat = currFeat - FeatDim_;
						const OaI32 contactBase = FeatDim_ - 4 + foot * 2;
						const bool planted = (batch[currFeat + contactBase] >= 0.5F or
							batch[currFeat + contactBase + 1] >= 0.5F) and
							(batch[prevFeat + contactBase] >= 0.5F or
							 batch[prevFeat + contactBase + 1] >= 0.5F);
						if (not planted) continue;
						const OaI32 joint = foot == 0 ? 10 : 11;
						const size_t curr = (static_cast<size_t>(t) * ValDs_->NumJoints() + joint) * 3;
						const size_t prev = (static_cast<size_t>(t - 1) * ValDs_->NumJoints() + joint) * 3;
						const OaF64 dx = predWorld[curr] - predWorld[prev];
						const OaF64 dz = predWorld[curr + 2] - predWorld[prev + 2];
						footSkateCm += 100.0 * std::sqrt(dx * dx + dz * dz);
						++footSkateFrames;
					}
				}
			}
			samples += n;
		}
		if (samples > 0) {
			const auto usage = CodebookUsage(validationTokens.data(), validationTokens.size(), c.NumCodes);
			std::printf("  Validation tokenizer: vel %.6f · MPJPE %.3f cm · contact %.2f%% · foot skate %.3f cm/frame\n"
				"    Codebook: %d/%d live · perplexity %.2f · %zu tokens\n",
				weightedVel / static_cast<OaF64>(samples),
				weightedMpjpeCm / static_cast<OaF64>(samples),
				contactTotal > 0 ? 100.0 * static_cast<OaF64>(contactCorrect) / contactTotal : 0.0,
				footSkateFrames > 0 ? footSkateCm / footSkateFrames : 0.0,
				usage.Unique, c.NumCodes, static_cast<double>(usage.Perplexity),
				validationTokens.size());
		}
		ctx.Clear();
		return {.Loss = samples > 0 ? weightedLoss / static_cast<OaF64>(samples)
			: std::numeric_limits<OaF64>::quiet_NaN(), .Batches = batches, .Samples = samples};
	}

	OaStatus StartLM() {
		const auto& c = Cli.GetConfig();
		auto& ctx = OaContext::GetDefault();
		const OaI32 B = c.BatchSize;
		const OaI32 lmTokLen = c.LmSeqLen;
		if (c.TextConditioning == "clip") {
			auto cacheComplete = [](const OaDsCmp* dataset) {
				if (dataset == nullptr) return true;
				if (dataset->TextFeatureDim() != OaClipTextConfig::ViTL14().ProjectionDim or
					dataset->TextFeatureFormat() != "oa_clip_text_v1" or
					dataset->TextFeatureModel() != "openai/clip-vit-large-patch14") return false;
				for (OaI32 clip = 0; clip < dataset->NumClips(); ++clip)
					if (dataset->ClipTextFeatureCount(clip) !=
						static_cast<OaI32>(dataset->ClipCaptions(clip).Size())) return false;
				return true;
			};
			const bool cacheReady = cacheComplete(Ds_) and cacheComplete(ValDs_);
			if (not cacheReady) {
				OA_LOG_INFO(OaLogComponent::ML,
					"trainalm: CLIP caption cache missing/incompatible; baking with native OaClipTextAg");
				const OaStatus baked = BakeNativeClipFeatures();
				if (not baked.IsOk()) {
					OA_LOG_ERROR(OaLogComponent::ML, "trainalm: native CLIP bake failed: %s",
						baked.GetMessage().CStr());
					IsRunning = false; return OaStatus::Ok();
				}
				const OaStatus reloaded = ReloadDatasetsWithTextFeatures();
				if (not reloaded.IsOk()) {
					OA_LOG_ERROR(OaLogComponent::ML, "trainalm: %s", reloaded.GetMessage().CStr());
					IsRunning = false; return OaStatus::Ok();
				}
			}
			LmTextFeatureDim_ = Ds_->TextFeatureDim();
			if (LmTextFeatureDim_ <= 0 or Ds_->TextFeatureFormat() != "oa_clip_text_v1"
				or Ds_->TextFeatureModel().Empty()) {
				OA_LOG_ERROR(OaLogComponent::ML,
					"trainalm: native CLIP cache validation failed in %s/text_feats",
					c.Dataset.CStr());
				IsRunning = false; return OaStatus::Ok();
			}
			if (Ds_->TextFeatureModel() != "openai/clip-vit-large-patch14" or
				LmTextFeatureDim_ != OaClipTextConfig::ViTL14().ProjectionDim) {
				OA_LOG_ERROR(OaLogComponent::ML,
					"trainalm: dataset CLIP contract '%s'/%d does not match native openai/clip-vit-large-patch14/%d",
					Ds_->TextFeatureModel().CStr(), LmTextFeatureDim_,
					OaClipTextConfig::ViTL14().ProjectionDim);
				IsRunning = false; return OaStatus::Ok();
			}
			if (not OaFileIo::IsFile(OaPath(c.ClipTextModel)) or
				not OaFileIo::IsFile(OaPath(c.ClipMerges))) {
				OA_LOG_ERROR(OaLogComponent::ML,
					"trainalm: native CLIP assets are missing (%s, %s); import them before LM training",
					c.ClipTextModel.CStr(), c.ClipMerges.CStr());
				IsRunning = false; return OaStatus::Ok();
			}
			for (OaI32 ci = 0; ci < NumClips_; ++ci) {
				if (Ds_->ClipTextFeatureCount(ci) != static_cast<OaI32>(Ds_->ClipCaptions(ci).Size())) {
					OA_LOG_ERROR(OaLogComponent::ML,
						"trainalm: clip %s lacks one CLIP feature per caption",
						Ds_->ClipId(ci).CStr());
					IsRunning = false; return OaStatus::Ok();
				}
			}
			if (ValDs_) {
				if (ValDs_->TextFeatureDim() != LmTextFeatureDim_ or
					ValDs_->TextFeatureFormat() != Ds_->TextFeatureFormat() or
					ValDs_->TextFeatureModel() != Ds_->TextFeatureModel()) {
					OA_LOG_ERROR(OaLogComponent::ML,
						"trainalm: train/validation CLIP feature contracts differ");
					IsRunning = false; return OaStatus::Ok();
				}
				for (OaI32 ci = 0; ci < ValDs_->NumClips(); ++ci) {
					if (ValDs_->ClipTextFeatureCount(ci) !=
						static_cast<OaI32>(ValDs_->ClipCaptions(ci).Size())) {
						OA_LOG_ERROR(OaLogComponent::ML,
							"trainalm: validation clip %s lacks one CLIP feature per caption",
							ValDs_->ClipId(ci).CStr());
						IsRunning = false; return OaStatus::Ok();
					}
				}
			}
		} else if (c.TextConditioning == "none") {
			LmTextFeatureDim_ = 0;
		} else {
			OA_LOG_ERROR(OaLogComponent::ML, "trainalm: unknown text_conditioning '%s'",
				c.TextConditioning.CStr());
			IsRunning = false; return OaStatus::Ok();
		}

		// Tokenize every clip. Short clips are first-class LM examples and are
		// padded only after tokenization; the loss mask excludes their PAD tail.
		TokenSequences_.clear();
		for (OaI32 ci = 0; ci < NumClips_; ++ci) {
			const OaI32 frames = Ds_->ClipFrames(ci);
			std::vector<float> clip(static_cast<size_t>(frames) * FeatDim_);
			std::memcpy(clip.data(), Ds_->ClipData(ci), clip.size() * sizeof(float));
			auto x = MakeF32(clip, OaMatrixShape{1, frames, FeatDim_});
			auto ids = Tok_.Tokenize(x, 1, frames)[0];
			(void)ctx.Execute(); (void)ctx.Sync();
			const OaI64 n = ids.NumElements();
			const OaI32* p = ids.DataAs<const OaI32>();
			TokenSequences_.emplace_back(p, p + n);
			ctx.Clear();
		}
		if (TokenSequences_.empty()) {
			OA_LOG_ERROR(OaLogComponent::ML, "trainalm: tokenizer produced no LM sequences");
			IsRunning = false; return OaStatus::Ok();
		}
		LmWindows_ = BuildLmWindows(TokenSequences_, lmTokLen + 1);
		std::printf("Tokenized all %zu clips for LM training · %zu true-boundary windows\n",
			TokenSequences_.size(), LmWindows_.size());
		const auto corpusStats = MeasureTokenCorpus(TokenSequences_, c.NumCodes);
		std::printf("LM token corpus: %lld tokens · %d/%d codes · unigram ppl %.2f · bigram ppl %.2f\n",
			static_cast<long long>(corpusStats.Tokens), corpusStats.Unique, c.NumCodes,
			corpusStats.UnigramPpl, corpusStats.BigramPpl);
		std::fflush(stdout);

		// Tokenize the held-out split independently with the identical padded,
		// true-boundary prediction contract.
		ValTokenSequences_.clear();
		LmValWindows_.clear();
		if (ValDs_) {
			for (OaI32 ci = 0; ci < ValDs_->NumClips(); ++ci) {
				const OaI32 frames = ValDs_->ClipFrames(ci);
				std::vector<float> clip(static_cast<size_t>(frames) * FeatDim_);
				std::memcpy(clip.data(), ValDs_->ClipData(ci), clip.size() * sizeof(float));
				auto x = MakeF32(clip, OaMatrixShape{1, frames, FeatDim_});
				auto ids = Tok_.Tokenize(x, 1, frames)[0];
				(void)ctx.Execute(); (void)ctx.Sync();
				const OaI64 n = ids.NumElements();
				const OaI32* p = ids.DataAs<const OaI32>();
				ValTokenSequences_.emplace_back(p, p + n);
				ctx.Clear();
			}
			LmValWindows_ = BuildLmWindows(ValTokenSequences_, lmTokLen + 1);
			std::printf("LM validation: %zu clips · %zu true-boundary windows\n",
				ValTokenSequences_.size(), LmValWindows_.size());
			if (LmValWindows_.empty()) {
				OA_LOG_WARN(OaLogComponent::ML, "trainalm: validation split has no LM windows");
			}
		}

		// Build LM
		OaAlmPriorConfig lmCfg = MakeLmConfig(LmTextFeatureDim_);
		if (c.LmFfnType != "dense" and c.LmFfnType != "moe" and c.LmFfnType != "hybrid") {
			OA_LOG_ERROR(OaLogComponent::ML, "trainalm: unknown lm_ffn_type '%s'", c.LmFfnType.CStr());
			IsRunning = false; return OaStatus::Ok();
		}
		Lm_.Ptr = OaMakeSharedPtr<OaAlmPriorAg>(lmCfg);
		(void)ctx.Execute(); (void)ctx.Sync();

		LmParams_ = Lm_.Module().AllParameterPtrs();
		LmOpt_ = OaMakeUniquePtr<OaAdamW>(LmParams_, c.LmLr, 0.9F, 0.99F, 1e-8F, c.LmWeightDecay);

		LmLossMetric_    = OaMakeUniquePtr<OaMetricLoss>("cross_entropy");
		LmLrMetric_      = OaMakeUniquePtr<OptimLrMetric>(*LmOpt_);

		LmBar_ = OaMakeUniquePtr<OaCbProgressBar>(10);
		LmBar_->AddMetric(LmLossMetric_.get());
		LmBar_->AddMetric(LmLrMetric_.get());
		if (lmCfg.FfnType != OaAlmFfnType::Dense) {
			LmMoeRoutingCb_ = OaMakeUniquePtr<OaCbMoeRouting>(*Lm_.Ptr);
		}
		if (not LmValWindows_.empty()) {
			LmValidationCb_ = OaMakeUniquePtr<OaCbValidation>(
				[this](OaItTraining& InIter) { return EvaluateLm(InIter); });
		}

		LmSummary_ = OaMakeUniquePtr<OaCbSummary>(true);
		if (LmValidationCb_) LmSummary_->SetValidationMetric(LmValidationCb_->MetricPtr());

		LmMgr_ = OaMakeUniquePtr<OaCheckpointManager>(OaCheckpointManagerConfig{
			.Dir           = c.ModelDir,
			// Do not reuse the retired Mamba-era AlmLm namespace: generic module
			// loading is permissive about missing names, so architecture migrations
			// need a fresh checkpoint root to prevent a misleading partial load.
			.ModelName     = c.Name + "Prior",
			.MaxKeep       = c.CkptMaxKeep,
			.MetricName    = LmValidationCb_ ? OaString("val_loss") : OaString("cross_entropy"),
			.LowerIsBetter = true,
		});
		(void)OaFileIo::CreateDirectories(OaPath(LmMgr_->GetModelDir()));
		LmCkptCb_ = OaMakeUniquePtr<OaCbCheckpoint>(
			*LmMgr_, Lm_.Module(), *LmOpt_, c.CkptSaveEvery,
			LmValidationCb_ ? LmValidationCb_->MetricPtr() : nullptr, c.CkptRestoreBest);

		OaItTrainingConfig lmIterCfg;
		// Treat every valid start position in each token sequence as a distinct training
		// example so the LM sees many more steps than just ceil(num_clips / batch).
		const OaI64 totalWindows = static_cast<OaI64>(LmWindows_.size());
		const OaI64 lmStepsPerEpoch = std::max<OaI64>(1,
			(totalWindows + B - 1) / B);
		LmStepsPerEpoch_ = lmStepsPerEpoch;
		const OaI32 lmTotalEpochs = c.LmPhases.Empty() ? c.LmEpochs
			: [&c]() { OaI32 sum = 0; for (const auto& ph : c.LmPhases) sum += ph.Epochs; return sum; }();
		lmIterCfg.TotalSteps     = static_cast<OaI64>(lmTotalEpochs) * lmStepsPerEpoch;
		lmIterCfg.StepsPerEpoch  = lmStepsPerEpoch;
		// One sample is one sequence; sequence length derives token throughput.
		lmIterCfg.BatchSize      = B;
		lmIterCfg.SequenceLength = lmTokLen + 1 + (LmTextFeatureDim_ > 0 ? 1 : 0);
		lmIterCfg.SequenceUnit   = "token";
		lmIterCfg.TimerName      = "lm_step";
		lmIterCfg.Metrics        = {LmLossMetric_.get()};
		// LR scheduler: phased or single warmup+cosine
		LmSched_ = Cli.BuildScheduler(c.LmPhases, lmStepsPerEpoch,
			c.LmLr, c.LmMinLr, c.LmWarmup, lmIterCfg.TotalSteps);
		// Start on the scheduler's first value. Previously step 1 used the optimizer's
		// peak LR, then step 2 abruptly fell to the bottom of the warmup ramp.
		LmOpt_->SetLr(LmSched_->GetLr(1));
		LmSchedCb_ = OaMakeUniquePtr<OaCbLrScheduler>(*LmSched_, *LmOpt_);

		lmIterCfg.Callbacks = {};
		if (LmMoeRoutingCb_) lmIterCfg.Callbacks.push_back(LmMoeRoutingCb_.get());
		lmIterCfg.Callbacks.push_back(LmBar_.get());
		if (LmValidationCb_) lmIterCfg.Callbacks.push_back(LmValidationCb_.get());
		lmIterCfg.Callbacks.push_back(LmCkptCb_.get());
		lmIterCfg.Callbacks.push_back(LmSchedCb_.get());
		lmIterCfg.Callbacks.push_back(LmSummary_.get());

		LmIter_ = OaMakeUniquePtr<OaItTraining>(*LmOpt_, lmIterCfg);

		const OaString textLabel = LmTextFeatureDim_ > 0
			? Ds_->TextFeatureModel() + OaString("-")
				+ OaToString(static_cast<OaI64>(LmTextFeatureDim_))
			: OaString("none");
		std::printf("LM: %d epochs × %lld steps/epoch · B=%d · tokLen=%d · text=%s · lr %.1e · ckpt %s\n",
			lmTotalEpochs, static_cast<long long>(lmStepsPerEpoch),
			B, lmTokLen, textLabel.CStr(), static_cast<double>(c.LmLr),
			c.CkptSaveEvery > 0 ? "step+epoch" : "epoch-end");
		std::fflush(stdout);

		CurrentStage_ = 2;
		return OaStatus::Ok();
	}

	OaValidationResult EvaluateLm(OaItTraining&) {
		const auto& c = Cli.GetConfig();
		const OaI32 B = c.BatchSize;
		const OaI32 lmTokLen = c.LmSeqLen;
		const OaI64 allBatches = static_cast<OaI64>(
			(LmValWindows_.size() + static_cast<size_t>(B) - 1) / static_cast<size_t>(B));
		const OaI64 batches = c.ValBatches > 0
			? std::min<OaI64>(allBatches, c.ValBatches) : allBatches;
		OaF64 weightedLoss = 0.0;
		OaI64 correctTokens = 0;
		OaI64 correctEos = 0;
		OaI64 eosTokens = 0;
		OaI64 samples = 0;
		auto& ctx = OaContext::GetDefault();

		for (OaI64 batchIdx = 0; batchIdx < batches; ++batchIdx) {
			const size_t begin = static_cast<size_t>(batchIdx) * static_cast<size_t>(B);
			const OaI32 n = static_cast<OaI32>(
				std::min<size_t>(static_cast<size_t>(B), LmValWindows_.size() - begin));
			std::vector<OaI32> inputHost(static_cast<size_t>(n) * (lmTokLen + 1));
			std::vector<OaI32> targetHost(static_cast<size_t>(n) * (lmTokLen + 1));
			std::vector<float> maskHost(static_cast<size_t>(n) * (lmTokLen + 1));
			std::vector<float> eosMaskHost(static_cast<size_t>(n) * (lmTokLen + 1));
			OaI32 validCount = 0;
			std::vector<float> textHost;
			if (LmTextFeatureDim_ > 0) textHost.resize(static_cast<size_t>(n) * LmTextFeatureDim_);
			for (OaI32 b = 0; b < n; ++b) {
				const auto& window = LmValWindows_[begin + static_cast<size_t>(b)];
				const auto& seq = ValTokenSequences_[static_cast<size_t>(window.Sequence)];
				const size_t row = static_cast<size_t>(b) * (lmTokLen + 1);
				FillLmRow(seq, window, lmTokLen + 1, c.NumCodes,
					c.NumCodes + 1, c.NumCodes + 2, inputHost.data() + row,
					targetHost.data() + row, maskHost.data() + row);
				validCount += window.Valid;
				for (OaI32 t = 0; t < lmTokLen + 1; ++t) {
					const bool isEos = maskHost[row + static_cast<size_t>(t)] != 0.0F
						and targetHost[row + static_cast<size_t>(t)] == c.NumCodes + 1;
					eosMaskHost[row + static_cast<size_t>(t)] = isEos ? 1.0F : 0.0F;
					eosTokens += isEos ? 1 : 0;
				}
				if (LmTextFeatureDim_ > 0) {
					const OaF32* feature = ValDs_->ClipTextFeatureData(window.Sequence);
					std::memcpy(textHost.data() + static_cast<size_t>(b) * LmTextFeatureDim_,
						feature, static_cast<size_t>(LmTextFeatureDim_) * sizeof(float));
				}
			}
			ctx.Clear();
			auto inputIds = MakeI32(inputHost, OaMatrixShape{n, lmTokLen + 1});
			auto targetIds = MakeI32(targetHost, OaMatrixShape{n, lmTokLen + 1});
			auto lossMask = MakeF32(maskHost, OaMatrixShape{static_cast<OaI64>(n) * (lmTokLen + 1)});
			auto eosMask = MakeF32(eosMaskHost, OaMatrixShape{static_cast<OaI64>(n) * (lmTokLen + 1)});
			OaMatrix logits;
			if (LmTextFeatureDim_ > 0) {
				auto textFeatures = MakeF32(textHost, OaMatrixShape{n, LmTextFeatureDim_});
				logits = Lm_.Forward(inputIds, textFeatures);
			} else {
				logits = Lm_.Forward(inputIds);
			}
			auto logitsFlat = logits.Reshape(OaMatrixShape{
				static_cast<OaI64>(n) * (lmTokLen + 1), c.NumCodes + 3});
			auto targetFlat = targetIds.Reshape(OaMatrixShape{
				static_cast<OaI64>(n) * (lmTokLen + 1)});
			auto loss = OaFnLoss::MaskedCrossEntropy(logitsFlat, targetFlat, lossMask, validCount);
			auto correct = OaFnMatrix::MaskedCategoricalAccuracyCount(
				logitsFlat, targetFlat, lossMask);
			auto eosCorrect = OaFnMatrix::MaskedCategoricalAccuracyCount(
				logitsFlat, targetFlat, eosMask);
			if (not ctx.Execute().IsOk() or not ctx.Sync().IsOk()) return {};
			weightedLoss += static_cast<OaF64>(loss.At(0)) * validCount;
			correctTokens += correct.DataAs<const OaU32>()[0];
			correctEos += eosCorrect.DataAs<const OaU32>()[0];
			samples += validCount;
		}
		if (samples > 0) {
			const OaF64 meanLoss = weightedLoss / static_cast<OaF64>(samples);
			std::printf("  Validation LM: perplexity %.3f · token accuracy %.2f%% · EOS accuracy %.2f%%\n",
				std::exp(meanLoss),
				100.0 * static_cast<OaF64>(correctTokens) / samples,
				eosTokens > 0 ? 100.0 * static_cast<OaF64>(correctEos) / eosTokens : 0.0);
		}
		ctx.Clear();
		return {.Loss = samples > 0 ? weightedLoss / static_cast<OaF64>(samples)
			: std::numeric_limits<OaF64>::quiet_NaN(), .Batches = batches, .Samples = samples};
	}

	OaStatus TickLM() {
		const auto& c = Cli.GetConfig();
		const OaI32 B = c.BatchSize;
		const OaI32 lmTokLen = c.LmSeqLen;

		if (LmIter_->IsDone()) {
			(void)LmIter_->Finish();
			const OaStatus bundleStatus = SaveAlmBundle();
			if (not bundleStatus.IsOk()) {
				OA_LOG_ERROR(OaLogComponent::ML, "trainalm: failed to save ALM bundle: %s",
					bundleStatus.GetMessage().CStr());
				IsRunning = false;
				return OaStatus::Ok();
			}
			std::printf("LM training complete.\n");
			std::fflush(stdout);
			CurrentStage_ = 3;
			return OaStatus::Ok();
		}

		LmIter_->Step([&]() {
			std::vector<OaI32> inputHost(static_cast<size_t>(B) * (lmTokLen + 1));
			std::vector<OaI32> targetHost(static_cast<size_t>(B) * (lmTokLen + 1));
			std::vector<float> maskHost(static_cast<size_t>(B) * (lmTokLen + 1));
			OaI32 validCount = 0;
			std::vector<float> textHost;
			if (LmTextFeatureDim_ > 0) textHost.resize(static_cast<size_t>(B) * LmTextFeatureDim_);
			for (OaI32 b = 0; b < B; ++b) {
				const auto& window = LmWindows_[static_cast<size_t>(
					(LmIter_->StepCount() * B + b) % static_cast<OaI64>(LmWindows_.size()))];
				const auto& seq = TokenSequences_[static_cast<size_t>(window.Sequence)];
				const size_t row = static_cast<size_t>(b) * (lmTokLen + 1);
				FillLmRow(seq, window, lmTokLen + 1, c.NumCodes,
					c.NumCodes + 1, c.NumCodes + 2, inputHost.data() + row,
					targetHost.data() + row, maskHost.data() + row);
				validCount += window.Valid;
				if (LmTextFeatureDim_ > 0) {
					const OaI32 count = Ds_->ClipTextFeatureCount(window.Sequence);
					const OaI32 caption = static_cast<OaI32>((c.Seed + LmIter_->Epoch()
						+ window.Sequence) % count);
					const OaF32* feature = Ds_->ClipTextFeatureData(window.Sequence)
						+ static_cast<size_t>(caption) * LmTextFeatureDim_;
					std::memcpy(textHost.data() + static_cast<size_t>(b) * LmTextFeatureDim_,
						feature, static_cast<size_t>(LmTextFeatureDim_) * sizeof(float));
				}
			}
			auto inputIds = MakeI32(inputHost, OaMatrixShape{B, lmTokLen + 1});
			auto targetIds = MakeI32(targetHost, OaMatrixShape{B, lmTokLen + 1});
			auto lossMask = MakeF32(maskHost, OaMatrixShape{static_cast<OaI64>(B) * (lmTokLen + 1)});

			LmOpt_->ZeroGrad();
			// The tape must exist before Forward; creating it only at Backward time
			// leaves the graph unrecorded and the model pinned at ln(vocab) CE.
			auto tape = OaMakeUniquePtr<OaGradientTape>();
			OaMatrix logits;
			if (LmTextFeatureDim_ > 0) {
				auto textFeatures = MakeF32(textHost, OaMatrixShape{B, LmTextFeatureDim_});
				logits = Lm_.Forward(inputIds, textFeatures);
			} else {
				logits = Lm_.Forward(inputIds);
			}
			auto logitsFlat = logits.Reshape(OaMatrixShape{
				static_cast<OaI64>(B) * (lmTokLen + 1), c.NumCodes + 3});
			auto targetFlat = targetIds.Reshape(OaMatrixShape{
				static_cast<OaI64>(B) * (lmTokLen + 1)});
			auto ce = OaFnLoss::MaskedCrossEntropy(logitsFlat, targetFlat, lossMask, validCount);
			LmIter_->RecordLoss(ce);
			auto aux = Lm_.Ptr->MoeAuxLoss();
			tape->Backward(aux.IsEmpty() ? ce : OaFnMatrix::Add(ce, aux));

		});

		const OaI64 step = LmIter_->StepCount();
		const float lv = LmIter_->LastLoss();
		if (not std::isfinite(lv)) {
			OA_LOG_ERROR(OaLogComponent::ML, "trainalm: LM diverged at step %lld",
				static_cast<long long>(step));
			LmIter_->RequestStop();
		}

		return OaStatus::Ok();
	}

	void Shutdown() override {
		// Release GPU-holding members before the explicit engine close.
		// OaComputeApp::Main closes Rt after Shutdown returns; if these
		// smart pointers are still alive when the app struct is destroyed
		// (after Main), their destructors would otherwise observe an engine
		// whose Vulkan resources have already been released.
		TokSchedCb_.Reset();
		LmSchedCb_.Reset();
		TokSched_.Reset();
		LmSched_.Reset();
		TokCkptCb_.Reset();
		LmCkptCb_.Reset();
		TokMgr_.Reset();
		LmMgr_.Reset();
		TokSummary_.Reset();
		LmSummary_.Reset();
		TokValidationCb_.Reset();
		LmValidationCb_.Reset();
		LmMoeRoutingCb_.Reset();
		TokBar_.Reset();
		LmBar_.Reset();
		TokExtraCb_.Reset();
		TokLossMetric_.Reset();
		LmLossMetric_.Reset();
		LmLrMetric_.Reset();
		TokIter_.Reset();
		LmIter_.Reset();
		TokOpt_.Reset();
		LmOpt_.Reset();
		Tok_.Ptr = {};
		Lm_.Ptr = {};
		TokParams_.Clear();
		LmParams_.Clear();
		TokenSequences_.clear();
		ValTokenSequences_.clear();
		TokWindows_.clear();
		TokValWindows_.clear();
		LmValWindows_.clear();

		delete Ds_;
		Ds_ = nullptr;
		delete ValDs_;
		ValDs_ = nullptr;
	}
};

int main(int argc, char** argv) {
	TrainAlmApp app;
	return app.Main(argc, argv);
}
