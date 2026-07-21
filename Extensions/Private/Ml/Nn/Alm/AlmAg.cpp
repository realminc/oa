#include <Ml/Nn/Alm/AlmAg.h>

#include <Oa/Ml/Oam.h>
#include <Oa/Runtime/Context.h>

#include <cstring>
#include <stdexcept>

namespace {

constexpr OaU32 kAlmAgBundleMagic = 0x47414D41U; // "AMAG"
constexpr OaU32 kAlmAgBundleVersion = 3;

#pragma pack(push, 1)
struct OaAlmAgBundleConfigV3 {
	OaU32 Magic = kAlmAgBundleMagic;
	OaU32 Version = kAlmAgBundleVersion;
	OaI32 InputDim = 0;
	OaI32 Width = 0;
	OaI32 CodeDim = 0;
	OaI32 NumCodes = 0;
	OaI32 DownT = 0;
	OaI32 Depth = 0;
	OaF32 CommitBeta = 0.0F;
	OaF32 EmaDecay = 0.0F;
	OaF32 EmaEps = 0.0F;
	OaF32 DeadThresh = 0.0F;
	OaI32 DModel = 0;
	OaI32 NumHeads = 0;
	OaI32 NumLayers = 0;
	OaI32 DFfn = 0;
	OaI32 TextFeatureDim = 0;
	OaU8 FfnType = 0;
	OaI32 MoeNumExperts = 0;
	OaI32 MoeExpertsPerToken = 0;
	OaI32 MoeEvery = 0;
	OaF32 MoeBalanceRate = 0.0F;
	OaF32 MoeAuxLossAlpha = 0.0F;
	OaF32 MoeRouterZLossBeta = 0.0F;
	OaI32 SeqLen = 0;
	OaI32 MaxSeqLen = 0;
	OaI32 MaxGenLen = 0;
	OaU32 ClipMergesBytes = 0;
	char TextEncoder[96] = {};
};
#pragma pack(pop)

OaAlmAgBundleConfigV3 EncodeConfig(const OaAlmAgConfig& InConfig) {
	OaAlmAgBundleConfigV3 out;
	const auto& t = InConfig.Tokenizer;
	const auto& p = InConfig.Prior;
	out.InputDim = t.InputDim;
	out.Width = t.Width;
	out.CodeDim = t.CodeDim;
	out.NumCodes = t.NumCodes;
	out.DownT = t.DownT;
	out.Depth = t.Depth;
	out.CommitBeta = t.CommitBeta;
	out.EmaDecay = t.EmaDecay;
	out.EmaEps = t.EmaEps;
	out.DeadThresh = t.DeadThresh;
	out.DModel = p.DModel;
	out.NumHeads = p.NumHeads;
	out.NumLayers = p.NumLayers;
	out.DFfn = p.DFfn;
	out.TextFeatureDim = p.TextFeatureDim;
	out.FfnType = static_cast<OaU8>(p.FfnType);
	out.MoeNumExperts = p.MoeNumExperts;
	out.MoeExpertsPerToken = p.MoeExpertsPerToken;
	out.MoeEvery = p.MoeEvery;
	out.MoeBalanceRate = p.MoeBalanceRate;
	out.MoeAuxLossAlpha = p.MoeAuxLossAlpha;
	out.MoeRouterZLossBeta = p.MoeRouterZLossBeta;
	out.SeqLen = p.SeqLen;
	out.MaxSeqLen = p.MaxSeqLen;
	out.MaxGenLen = p.MaxGenLen;
	out.ClipMergesBytes = InConfig.ClipMergesBytes;
	std::strncpy(out.TextEncoder, InConfig.TextEncoder.CStr(), sizeof(out.TextEncoder) - 1);
	return out;
}

OaAlmAgConfig DecodeConfig(const OaAlmAgBundleConfigV3& InConfig) {
	OaAlmAgConfig out;
	auto& t = out.Tokenizer;
	auto& p = out.Prior;
	t.InputDim = InConfig.InputDim;
	t.Width = InConfig.Width;
	t.CodeDim = InConfig.CodeDim;
	t.NumCodes = InConfig.NumCodes;
	t.DownT = InConfig.DownT;
	t.Depth = InConfig.Depth;
	t.CommitBeta = InConfig.CommitBeta;
	t.EmaDecay = InConfig.EmaDecay;
	t.EmaEps = InConfig.EmaEps;
	t.DeadThresh = InConfig.DeadThresh;
	p.SyncVocab(t.NumCodes);
	p.DModel = InConfig.DModel;
	p.NumHeads = InConfig.NumHeads;
	p.NumLayers = InConfig.NumLayers;
	p.DFfn = InConfig.DFfn;
	p.TextFeatureDim = InConfig.TextFeatureDim;
	p.FfnType = static_cast<OaAlmFfnType>(InConfig.FfnType);
	p.MoeNumExperts = InConfig.MoeNumExperts;
	p.MoeExpertsPerToken = InConfig.MoeExpertsPerToken;
	p.MoeEvery = InConfig.MoeEvery;
	p.MoeBalanceRate = InConfig.MoeBalanceRate;
	p.MoeAuxLossAlpha = InConfig.MoeAuxLossAlpha;
	p.MoeRouterZLossBeta = InConfig.MoeRouterZLossBeta;
	p.SeqLen = InConfig.SeqLen;
	p.MaxSeqLen = InConfig.MaxSeqLen;
	p.MaxGenLen = InConfig.MaxGenLen;
	out.ClipMergesBytes = InConfig.ClipMergesBytes;
	out.TextEncoder = InConfig.TextEncoder;
	return out;
}

OaStatus ValidateConfig(const OaAlmAgConfig& InConfig) {
	if (InConfig.Tokenizer.NumCodes <= 0 or
		InConfig.Prior.NumCodes != InConfig.Tokenizer.NumCodes or
		InConfig.Prior.VocabSize != InConfig.Tokenizer.NumCodes + 3) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaAlmAg tokenizer/prior vocabulary contract does not match");
	}
	if (InConfig.Prior.TextFeatureDim > 0 and InConfig.TextEncoder.Empty()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaAlmAg conditioned prior requires an exact text encoder identity");
	}
	if (InConfig.Prior.NumHeads <= 0 or
		InConfig.Prior.DModel % InConfig.Prior.NumHeads != 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaAlmAg prior DModel must be divisible by a positive attention-head count");
	}
	if (InConfig.ClipMergesBytes > 0 and
		(InConfig.Prior.TextFeatureDim != OaClipTextConfig::ViTL14().ProjectionDim or
		 InConfig.TextEncoder != "openai/clip-vit-large-patch14")) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"native OaAlmAg CLIP bundle requires the pinned ViT-L/14 identity and feature dimension");
	}
	return OaStatus::Ok();
}

OaStatus ValidateWeights(OaModule& InModule, const OamModel& InOam) {
	const auto expected = InModule.AllNamedParameterPtrs();
	if (InOam.WeightIndex.Size() != expected.Size()) {
		return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
			"OaAlmAg bundle parameter count does not match its architecture");
	}
	for (const auto& named : expected) {
		const OamTensorEntry* entry = InOam.FindWeight(named.Path.CStr());
		if (entry == nullptr or entry->Dtype != named.Param->Data.GetDtype() or
			entry->Rank != named.Param->Data.Rank() or
			entry->NumBytes != static_cast<OaU64>(named.Param->Data.ByteSize())) {
			return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
				OaString("OaAlmAg bundle tensor mismatch: ") + named.Path);
		}
		for (OaI32 d = 0; d < named.Param->Data.Rank(); ++d) {
			if (entry->Shape[d] != static_cast<OaU64>(named.Param->Data.Size(d))) {
				return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
					OaString("OaAlmAg bundle shape mismatch: ") + named.Path);
			}
		}
	}
	struct NamedBuffer { OaString Path; const OaModuleBuffer* Buffer = nullptr; };
	OaVec<NamedBuffer> buffers;
	auto collect = [&](auto&& self, const OaModule& module, const OaString& prefix) -> void {
		for (const auto& buffer : module.Buffers()) {
			if (not buffer.Persistent or buffer.Data.IsEmpty()) continue;
			buffers.PushBack({prefix.Empty() ? buffer.Name : prefix + "." + buffer.Name, &buffer});
		}
		for (const auto& child : module.Children()) {
			const OaString path = prefix.Empty() ? child.Name : prefix + "." + child.Name;
			self(self, *child.Module, path);
		}
	};
	collect(collect, InModule, OaString());
	if (InOam.StateIndex.Size() != buffers.Size()) {
		return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
			"OaAlmAg bundle persistent-state count does not match its architecture");
	}
	for (const auto& named : buffers) {
		const OamTensorEntry* entry = InOam.FindState(named.Path.CStr());
		const OaMatrix& data = named.Buffer->Data;
		if (entry == nullptr or entry->Dtype != data.GetDtype() or
			entry->Rank != data.Rank() or entry->NumBytes != static_cast<OaU64>(data.ByteSize())) {
			return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
				OaString("OaAlmAg bundle state mismatch: ") + named.Path);
		}
		for (OaI32 d = 0; d < data.Rank(); ++d) {
			if (entry->Shape[d] != static_cast<OaU64>(data.Size(d))) {
				return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
					OaString("OaAlmAg bundle state shape mismatch: ") + named.Path);
			}
		}
	}
	return OaStatus::Ok();
}

} // namespace

OaAlmAg::OaAlmAg(const OaAlmAgConfig& InConfig) : Config_(InConfig) {
	const OaStatus valid = ValidateConfig(Config_);
	if (not valid.IsOk()) throw std::invalid_argument(valid.GetMessage().CStr());
	Tokenizer_ = OaMakeSharedPtr<OaAlmTokenizerAg>(Config_.Tokenizer);
	Prior_ = OaMakeSharedPtr<OaAlmPriorAg>(Config_.Prior);
	if (Config_.ClipMergesBytes > 0) {
		TextEncoder_ = OaMakeSharedPtr<OaClipTextAg>(OaClipTextConfig::ViTL14());
		RegisterBuffer("text_tokenizer_merges", OaFnMatrix::Empty(
			OaMatrixShape{Config_.ClipMergesBytes}, OaScalarType::UInt8), true);
	}
	RegisterChildren();
}

OaAlmAg::OaAlmAg(OaSharedPtr<OaAlmTokenizerAg> InTokenizer,
	OaSharedPtr<OaAlmPriorAg> InPrior, OaStringView InTextEncoder)
	: Tokenizer_(std::move(InTokenizer)), Prior_(std::move(InPrior)) {
	if (not Tokenizer_ or not Prior_) throw std::invalid_argument("OaAlmAg children cannot be null");
	Config_.Tokenizer = Tokenizer_->Config();
	Config_.Prior = Prior_->GetConfig();
	Config_.TextEncoder = OaString(InTextEncoder);
	const OaStatus valid = ValidateConfig(Config_);
	if (not valid.IsOk()) throw std::invalid_argument(valid.GetMessage().CStr());
	RegisterChildren();
}

OaAlmAg::OaAlmAg(OaSharedPtr<OaAlmTokenizerAg> InTokenizer,
	OaSharedPtr<OaAlmPriorAg> InPrior, OaSharedPtr<OaClipTextAg> InTextEncoder,
	OaSpan<const OaU8> InClipMerges, OaStringView InTextEncoderIdentity)
	: Tokenizer_(std::move(InTokenizer)), Prior_(std::move(InPrior)),
	  TextEncoder_(std::move(InTextEncoder)) {
	if (not Tokenizer_ or not Prior_ or not TextEncoder_ or InClipMerges.Empty())
		throw std::invalid_argument("native OaAlmAg children and CLIP merges cannot be empty");
	Config_.Tokenizer = Tokenizer_->Config();
	Config_.Prior = Prior_->GetConfig();
	Config_.TextEncoder = OaString(InTextEncoderIdentity);
	Config_.ClipMergesBytes = static_cast<OaU32>(InClipMerges.Size());
	const OaStatus valid = ValidateConfig(Config_);
	if (not valid.IsOk()) throw std::invalid_argument(valid.GetMessage().CStr());
	RegisterBuffer("text_tokenizer_merges", OaFnMatrix::FromBytes(
		InClipMerges, OaMatrixShape{Config_.ClipMergesBytes}, OaScalarType::UInt8), true);
	RegisterChildren();
}

void OaAlmAg::RegisterChildren() {
	RegisterModule("tokenizer", Tokenizer_);
	RegisterModule("prior", Prior_);
	if (TextEncoder_) RegisterModule("text_encoder", TextEncoder_);
}

OaMatrix OaAlmAg::Forward(const OaMatrix& InTokenIds) {
	return Prior_->Forward(InTokenIds);
}

OaMatrix OaAlmAg::ForwardConditioned(
	const OaMatrix& InTokenIds, const OaMatrix& InTextFeatures) {
	return Prior_->ForwardConditioned(InTokenIds, InTextFeatures);
}

OaVec<OaMatrix> OaAlmAg::Tokenize(
	const OaMatrix& InMotion, OaI32 InBatch, OaI32 InFrames) {
	return Tokenizer_->Tokenize(InMotion, InBatch, InFrames);
}

OaMatrix OaAlmAg::Detokenize(
	const OaVec<OaMatrix>& InTokenIds, OaI32 InBatch, OaI32 InTokenLength) {
	return Tokenizer_->Detokenize(InTokenIds, InBatch, InTokenLength);
}

OaMatrix OaAlmAg::GenerateMotion(OaI32 InBatchSize, OaF32 InTemperature,
	OaI32 InTopK, OaF32 InTopP, OaI32 InMaxTokens) {
	auto tokens = Prior_->Generate(
		InBatchSize, InTemperature, InTopK, InTopP, InMaxTokens);
	return Prior_->DecodeToMotion(tokens, *Tokenizer_);
}

OaMatrix OaAlmAg::GenerateMotionConditioned(const OaMatrix& InTextFeatures,
	OaF32 InTemperature, OaI32 InTopK, OaF32 InTopP, OaI32 InMaxTokens) {
	auto tokens = Prior_->GenerateConditioned(
		InTextFeatures, InTemperature, InTopK, InTopP, InMaxTokens);
	return Prior_->DecodeToMotion(tokens, *Tokenizer_);
}

OaResult<OaMatrix> OaAlmAg::EncodePrompt(OaStringView InPrompt) {
	const OaModuleBuffer* mergesBuffer = nullptr;
	for (const auto& buffer : Buffers_) {
		if (buffer.Name == "text_tokenizer_merges") {
			mergesBuffer = &buffer;
			break;
		}
	}
	if (not TextEncoder_ or mergesBuffer == nullptr)
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaAlmAg bundle has no native CLIP text encoder/tokenizer");
	if (not ClipTokenizer_) {
		ClipTokenizer_ = OaMakeUniquePtr<OaClipTokenizer>();
		const auto& merges = mergesBuffer->Data;
		auto& ctx = OaContext::GetDefault();
		OA_RETURN_IF_ERROR(ctx.Execute());
		OA_RETURN_IF_ERROR(ctx.Sync());
		OA_RETURN_IF_ERROR(ClipTokenizer_->LoadMerges(OaSpan<const OaU8>(
			merges.DataAs<const OaU8>(), static_cast<OaUsize>(merges.NumElements()))));
	}
	const OaString prompt(InPrompt);
	auto encoded = ClipTokenizer_->Encode(OaSpan<const OaString>(&prompt, 1),
		TextEncoder_->Config().ContextLength, true);
	if (encoded.IsError()) return encoded.GetStatus();
	const auto& batch = encoded.GetValue();
	auto ids = OaFnMatrix::FromInt32(
		OaSpan<const OaI32>(batch.TokenIds.Data(), batch.TokenIds.Size()),
		OaMatrixShape{1, batch.ContextLength}, OaScalarType::Int32);
	auto eos = OaFnMatrix::FromInt32(
		OaSpan<const OaI32>(batch.FlatEosRows.Data(), batch.FlatEosRows.Size()),
		OaMatrixShape{1}, OaScalarType::Int32);
	return TextEncoder_->ForwardTokens(ids, eos);
}

OaResult<OaMatrix> OaAlmAg::GenerateMotionPrompt(OaStringView InPrompt,
	OaF32 InTemperature, OaI32 InTopK, OaF32 InTopP, OaI32 InMaxTokens) {
	auto feature = EncodePrompt(InPrompt);
	if (feature.IsError()) return feature.GetStatus();
	return GenerateMotionConditioned(feature.GetValue(), InTemperature, InTopK, InTopP, InMaxTokens);
}

OaStatus OaAlmAg::SaveBundle(const OaString& InPath) const {
	OamModel oam;
	std::strncpy(oam.Config.Architecture, "OaAlmAg", sizeof(oam.Config.Architecture) - 1);
	oam.Config.ConfigVersion = kAlmAgBundleVersion;
	oam.Config.DModel = static_cast<OaU32>(Config_.Prior.DModel);
	oam.Config.NLayers = static_cast<OaU32>(Config_.Prior.NumLayers);
	oam.Config.DVocab = static_cast<OaU32>(Config_.Prior.VocabSize);
	oam.Config.WeightDtype = static_cast<OaU8>(OaFnMatrix::GetWeightDtype());
	const auto arch = EncodeConfig(Config_);
	oam.ArchConfig.Resize(sizeof(arch));
	std::memcpy(oam.ArchConfig.Data(), &arch, sizeof(arch));
	oam.Config.ArchConfigSize = static_cast<OaU32>(oam.ArchConfig.Size());
	OA_RETURN_IF_ERROR(SaveTo(oam));
	return oam.Save(InPath);
}

OaResult<OaSharedPtr<OaAlmAg>> OaAlmAg::LoadBundle(const OaString& InPath) {
	auto loaded = OamModel::Load(InPath);
	if (not loaded.IsOk()) return loaded.GetStatus();
	auto oam = std::move(loaded).GetValue();
	if (std::strncmp(oam.Config.Architecture, "OaAlmAg", sizeof(oam.Config.Architecture)) != 0 or
		oam.Config.ConfigVersion != kAlmAgBundleVersion or
		oam.ArchConfig.Size() != sizeof(OaAlmAgBundleConfigV3)) {
		return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
			"checkpoint is not a supported OaAlmAg bundle");
	}
	OaAlmAgBundleConfigV3 encoded;
	std::memcpy(&encoded, oam.ArchConfig.Data(), sizeof(encoded));
	if (encoded.Magic != kAlmAgBundleMagic or encoded.Version != kAlmAgBundleVersion) {
		return OaStatus::Error(OaStatusCode::CheckpointCorrupt,
			"OaAlmAg architecture payload has an invalid version");
	}
	auto config = DecodeConfig(encoded);
	const OaStatus valid = ValidateConfig(config);
	if (not valid.IsOk()) return valid;
	auto model = OaMakeSharedPtr<OaAlmAg>(config);
	const OaStatus weights = ValidateWeights(*model, oam);
	if (not weights.IsOk()) return weights;
	// Constructors enqueue deferred parameter initialization. Drain it before the
	// host checkpoint copy so a later Execute cannot overwrite restored bytes.
	auto& ctx = OaContext::GetDefault();
	const OaStatus execute = ctx.Execute();
	if (not execute.IsOk()) return execute;
	const OaStatus sync = ctx.Sync();
	if (not sync.IsOk()) return sync;
	ctx.Clear();
	OA_RETURN_IF_ERROR(model->LoadFrom(oam));
	return model;
}
