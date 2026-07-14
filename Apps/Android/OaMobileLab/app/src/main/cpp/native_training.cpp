#include <adrenotools/driver.h>
#include <adrenotools/priv.h>
#include <android/log.h>
#include <jni.h>
#include <vulkan/vulkan.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/FnLoss.h>
#include <Oa/Ml/ItTraining.h>
#include <Oa/Ml/Metric.h>
#include <Oa/Ml/NlpSuite.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVk.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

constexpr char OaMobileLogTag[] = "OA";
std::atomic_bool OaMobileCancelRequested{false};

[[noreturn]] static void OaMobileFail(const std::string& InMessage) {
	__android_log_print(ANDROID_LOG_ERROR, OaMobileLogTag, "%s", InMessage.c_str());
	throw std::runtime_error(InMessage);
}

class OaMobileJavaString {
public:
	OaMobileJavaString(JNIEnv* InEnvironment, jstring InValue)
		: Environment_(InEnvironment)
		, Value_(InValue) {
		Characters_ = InValue == nullptr
			? nullptr
			: InEnvironment->GetStringUTFChars(InValue, nullptr);
	}

	~OaMobileJavaString() {
		if (Characters_ != nullptr) {
			Environment_->ReleaseStringUTFChars(Value_, Characters_);
		}
	}

	[[nodiscard]] std::string Get() const {
		return Characters_ == nullptr ? std::string{} : std::string(Characters_);
	}

private:
	JNIEnv* Environment_ = nullptr;
	jstring Value_ = nullptr;
	const char* Characters_ = nullptr;
};

class OaMobileVulkanLibrary {
public:
	explicit OaMobileVulkanLibrary(void* InHandle)
		: Handle_(InHandle) {
	}

	OaMobileVulkanLibrary(const OaMobileVulkanLibrary&) = delete;
	OaMobileVulkanLibrary& operator=(const OaMobileVulkanLibrary&) = delete;

	~OaMobileVulkanLibrary() {
		if (Handle_ != nullptr) {
			dlclose(Handle_);
		}
	}

	[[nodiscard]] void* Get() const { return Handle_; }

private:
	void* Handle_ = nullptr;
};

static OaMobileVulkanLibrary OaMobileOpenTurnip(
	std::string InDriverDirectory,
	const std::string& InNativeLibraryDirectory,
	const std::string& InCacheDirectory) {
	if (not InDriverDirectory.empty() and InDriverDirectory.back() != '/') {
		InDriverDirectory.push_back('/');
	}
	const std::string temporaryDirectory = InCacheDirectory + "/adrenotools-training";
	(void)mkdir(temporaryDirectory.c_str(), 0700);
	void* handle = adrenotools_open_libvulkan(
		RTLD_NOW | RTLD_LOCAL,
		ADRENOTOOLS_DRIVER_CUSTOM,
		temporaryDirectory.c_str(),
		InNativeLibraryDirectory.c_str(),
		InDriverDirectory.c_str(),
		"libvulkan_freedreno.so",
		nullptr,
		nullptr);
	if (handle == nullptr) {
		const char* error = dlerror();
		OaMobileFail("Could not open bundled Turnip: " +
			(error == nullptr ? std::string("unknown dlopen error") : std::string(error)));
	}
	return OaMobileVulkanLibrary(handle);
}

template <typename Function>
static Function OaMobileLoadExport(void* InLibrary, const char* InName) {
	auto function = reinterpret_cast<Function>(dlsym(InLibrary, InName));
	if (function == nullptr) {
		OaMobileFail(std::string("Missing Vulkan export ") + InName);
	}
	return function;
}

class OaMobileVkScope {
public:
	OaMobileVkScope() = default;
	OaMobileVkScope(const OaMobileVkScope&) = delete;
	OaMobileVkScope& operator=(const OaMobileVkScope&) = delete;
	~OaMobileVkScope() { OaVkFinalize(); }
};

class OaMobileProgressCallback {
public:
	OaMobileProgressCallback(JNIEnv* InEnvironment, jobject InCallback)
		: Environment_(InEnvironment)
		, Callback_(InCallback) {
		if (Callback_ == nullptr) {
			return;
		}
		jclass callbackClass = Environment_->GetObjectClass(Callback_);
		Method_ = Environment_->GetMethodID(callbackClass, "onNativeProgress", "(IIFDD)V");
		Environment_->DeleteLocalRef(callbackClass);
		if (Method_ == nullptr) {
			Environment_->ExceptionClear();
			OaMobileFail("Training callback is missing onNativeProgress(IIFDD)");
		}
	}

	void Send(OaI32 InStep, OaI32 InTotal, OaF32 InLoss, OaF64 InGpuMs, OaF64 InWallMs) {
		if (Callback_ == nullptr or Method_ == nullptr) {
			return;
		}
		Environment_->CallVoidMethod(
			Callback_, Method_, InStep, InTotal, InLoss,
			static_cast<jdouble>(InGpuMs), static_cast<jdouble>(InWallMs));
		if (Environment_->ExceptionCheck()) {
			Environment_->ExceptionDescribe();
			Environment_->ExceptionClear();
			__android_log_print(
				ANDROID_LOG_WARN, OaMobileLogTag, "Training progress callback failed");
		}
	}

private:
	JNIEnv* Environment_ = nullptr;
	jobject Callback_ = nullptr;
	jmethodID Method_ = nullptr;
};

static OaNlpArchitecture OaMobileParseArchitecture(const std::string& InValue) {
	if (InValue == "rnn") {
		return OaNlpArchitecture::Rnn;
	}
	if (InValue == "transformer") {
		return OaNlpArchitecture::Transformer;
	}
	if (InValue == "moe") {
		return OaNlpArchitecture::MoeTransformer;
	}
	if (InValue == "mamba3") {
		return OaNlpArchitecture::Mamba3;
	}
	return OaNlpArchitecture::Gru;
}

static OaNlpTokenizerKind OaMobileParseTokenizer(const std::string& InValue) {
	if (InValue == "bpe") {
		return OaNlpTokenizerKind::Bpe;
	}
	if (InValue == "char") {
		return OaNlpTokenizerKind::Char;
	}
	return OaNlpTokenizerKind::Byte;
}

static std::string OaMobileEscapeText(const OaString& InValue) {
	std::ostringstream escaped;
	for (const unsigned char value : InValue) {
		if (value >= 32 and value <= 126 and value != '\\') {
			escaped << static_cast<char>(value);
		} else if (value == '\\') {
			escaped << "\\\\";
		} else {
			escaped << "\\x" << std::hex << std::setw(2) << std::setfill('0')
				<< static_cast<unsigned int>(value) << std::dec;
		}
	}
	return escaped.str();
}

static OaU64 OaMobileParameterFingerprint(OaNlpSuiteModel& InModel) {
	auto& context = OaContext::GetDefault();
	const auto execute = context.Execute();
	if (not execute.IsOk()) {
		OaMobileFail("Parameter fingerprint execute failed");
	}
	const auto sync = context.Sync();
	if (not sync.IsOk()) {
		OaMobileFail("Parameter fingerprint sync failed");
	}

	// Exact FNV-1a over the serialized parameter order. This turns a vague
	// generation mismatch into a precise persistence boundary: either checkpoint
	// load changed model bytes, or identical bytes executed differently.
	OaU64 hash = 1469598103934665603ULL;
	for (const auto* parameter : InModel.AllParameterPtrs()) {
		const OaU64 bytes = static_cast<OaU64>(parameter->Data.ByteSize());
		OaVec<OaU8> host(static_cast<OaUsize>(bytes));
		const auto copy = OaFnMatrix::CopyToHost(parameter->Data, host.Data(), bytes);
		if (not copy.IsOk()) {
			OaMobileFail("Parameter fingerprint readback failed");
		}
		for (const OaU8 value : host) {
			hash ^= value;
			hash *= 1099511628211ULL;
		}
	}
	return hash;
}

struct OaMobileGenerationQuality {
	bool Alphabet = false;
	OaI32 CorpusPrefix = 0;
	OaF32 CorpusNgramCoverage = 0.0F;

	[[nodiscard]] bool Pass() const {
		// This is an intentional overfit tutorial, not an open-ended language
		// benchmark. A successful sample must enter a real continuation of the
		// fixed corpus and remain corpus-like. Merely producing printable lower-
		// case bytes is not evidence of learning ("/xeze/asae" exposed that hole).
		return Alphabet and CorpusPrefix >= 16 and CorpusNgramCoverage >= 0.90F;
	}
};

static OaMobileGenerationQuality OaMobileMeasureGenerationQuality(
	const OaString& InValue) {
	OaMobileGenerationQuality quality;
	const OaUsize promptLength = std::strlen(OaNlpSuiteGenerationPrompt);
	const OaUsize expectedLength = promptLength +
		static_cast<OaUsize>(OaNlpSuiteGenerationSourceUnits);
	if (InValue.size() != expectedLength) {
		return quality;
	}
	quality.Alphabet = true;
	for (OaUsize index = promptLength; index < InValue.size(); ++index) {
		const unsigned char value = InValue[index];
		if (value != ' ' and (value < 'a' or value > 'z')) {
			quality.Alphabet = false;
			return quality;
		}
	}

	const std::string value(InValue.Data(), InValue.Size());
	const std::string corpus = OaNlpSuiteSampler::Corpus();
	const std::string prompt = OaNlpSuiteGenerationPrompt;
	const std::string continuation = value.substr(promptLength);
	for (OaUsize found = corpus.find(prompt);
		found != std::string::npos;
		found = corpus.find(prompt, found + 1)) {
		const OaUsize corpusStart = found + prompt.size();
		OaI32 matched = 0;
		while (static_cast<OaUsize>(matched) < continuation.size() and
			corpusStart + static_cast<OaUsize>(matched) < corpus.size() and
			continuation[static_cast<OaUsize>(matched)] ==
				corpus[corpusStart + static_cast<OaUsize>(matched)]) {
			++matched;
		}
		quality.CorpusPrefix = std::max(quality.CorpusPrefix, matched);
	}

	constexpr OaUsize ngramLength = 8;
	OaI32 supported = 0;
	OaI32 total = 0;
	if (value.size() >= ngramLength) {
		for (OaUsize index = 0; index + ngramLength <= value.size(); ++index) {
			++total;
			if (corpus.find(value.substr(index, ngramLength)) != std::string::npos) {
				++supported;
			}
		}
	}
	quality.CorpusNgramCoverage = total > 0
		? static_cast<OaF32>(supported) / static_cast<OaF32>(total)
		: 0.0F;
	return quality;
}

static OaString OaMobileGenerateGreedy(
	OaNlpSuiteModel& InModel,
	OaNlpSuiteSampler& InSampler,
	const OaNlpSuiteRecipe& InRecipe) {
	auto& runtimeContext = OaContext::GetDefault();
	// Match the desktop suite exactly. RNN and GRU deliberately use the causal
	// full-window path there; their single-token path remains a module API but is
	// not the cross-driver tutorial oracle. Mamba-3 is the only Byte tutorial
	// whose canonical generation uses its persistent Step state.
	if (InRecipe.Architecture() == OaNlpArchitecture::Mamba3 and
		InModel.SupportsStatefulGeneration()) {
		InModel.ResetGenerationState(1);
		const auto resetExecute = runtimeContext.Execute();
		if (not resetExecute.IsOk()) {
			OaMobileFail("Recurrent generation-state reset execute failed");
		}
		const auto resetSync = runtimeContext.Sync();
		if (not resetSync.IsOk()) {
			OaMobileFail("Recurrent generation-state reset sync failed");
		}
		const auto prompt = InSampler.Encode(OaNlpSuiteGenerationPrompt);
		OaMatrix logits;
		for (const OaI32 token : prompt) {
			// Keep the token object explicit through execution. The graph retains
			// its buffer owner, but this also makes the recurrent step boundary and
			// host-side lifetime unambiguous during mobile diagnostics.
			auto input = InSampler.InputStepMatrix(token);
			logits = InModel.ForwardGenerationStep(input);
			const auto execute = runtimeContext.Execute();
			if (not execute.IsOk()) {
				OaMobileFail("Recurrent prompt-step execute failed");
			}
			const auto sync = runtimeContext.Sync();
			if (not sync.IsOk()) {
				OaMobileFail("Recurrent prompt-step sync failed");
			}
		}
		if (logits.NumElements() == 0) {
			OaMobileFail("Recurrent generation prompt encoded to no tokens");
		}

		OaString output(OaNlpSuiteGenerationPrompt);
		OaI32 generatedSourceUnits = 0;
		for (OaI32 index = 0;
			index < OaNlpSuiteGenerationSourceUnits and
			generatedSourceUnits < OaNlpSuiteGenerationSourceUnits;
			++index) {
			auto row = logits.Reshape(OaMatrixShape{InRecipe.VocabSize()});
			const OaI32 next = static_cast<OaI32>(OaFnMatrix::Argmax(row));
			OaVec<OaI32> nextToken{next};
			const OaString decoded = InSampler.Decode(nextToken);
			output += decoded;
			generatedSourceUnits += static_cast<OaI32>(decoded.size());
			if (index + 1 < OaNlpSuiteGenerationSourceUnits and
				generatedSourceUnits < OaNlpSuiteGenerationSourceUnits) {
				auto input = InSampler.InputStepMatrix(next);
				logits = InModel.ForwardGenerationStep(input);
				const auto stepExecute = runtimeContext.Execute();
				if (not stepExecute.IsOk()) {
					OaMobileFail("Recurrent token-step execute failed");
				}
				const auto stepSync = runtimeContext.Sync();
				if (not stepSync.IsOk()) {
					OaMobileFail("Recurrent token-step sync failed");
				}
			}
		}
		const OaUsize targetLength = std::strlen(OaNlpSuiteGenerationPrompt) +
			static_cast<OaUsize>(OaNlpSuiteGenerationSourceUnits);
		return output.size() > targetLength
			? output.substr(0, targetLength)
			: output;
	}

	const OaI32 contextLength = InRecipe.ContextLength();
	const OaI32 padToken = InRecipe.Tokenizer() == OaNlpTokenizerKind::Char ? 26 : 0;
	OaVec<OaI32> context(contextLength, padToken);
	const auto prompt = InSampler.Encode(OaNlpSuiteGenerationPrompt);
	const OaI32 copyCount = std::min(
		static_cast<OaI32>(prompt.Size()), contextLength);
	for (OaI32 index = 0; index < copyCount; ++index) {
		context[index] = prompt[index];
	}

	OaI32 filled = std::max(copyCount, 1);
	OaI32 logitRow = filled - 1;
	OaString output(OaNlpSuiteGenerationPrompt);
	OaI32 generatedSourceUnits = 0;

	// Byte, Char, and BPE all generate the same amount of source text. A BPE
	// token can decode to several bytes, so a fixed token count is not the
	// desktop suite's contract and produced misleadingly longer BPE samples.
	for (OaI32 index = 0;
		index < OaNlpSuiteGenerationSourceUnits and
		generatedSourceUnits < OaNlpSuiteGenerationSourceUnits;
		++index) {
		// Keep the source object explicit through the lazy graph execution. The
		// graph owns its buffer too; the local primarily makes this decode boundary
		// auditable alongside the synchronized recurrent step path above.
		auto input = InSampler.InputMatrix(context);
		auto logits = InModel.Forward(input);
		auto row = OaFnMatrix::Reshape(
			OaFnMatrix::Slice(logits, 0, logitRow, logitRow + 1),
			OaMatrixShape{InRecipe.VocabSize()});
		const OaI32 next = static_cast<OaI32>(OaFnMatrix::Argmax(row));
		OaVec<OaI32> nextToken(1, next);
		const OaString decoded = InSampler.Decode(nextToken);
		output += decoded;
		generatedSourceUnits += static_cast<OaI32>(decoded.size());

		if (filled < contextLength) {
			context[filled] = next;
			++filled;
			logitRow = filled - 1;
		} else {
			for (OaI32 token = 1; token < contextLength; ++token) {
				context[token - 1] = context[token];
			}
			context[contextLength - 1] = next;
			logitRow = contextLength - 1;
		}
	}
	const OaUsize targetLength = std::strlen(OaNlpSuiteGenerationPrompt) +
		static_cast<OaUsize>(OaNlpSuiteGenerationSourceUnits);
	return output.size() > targetLength ? output.substr(0, targetLength) : output;
}

static std::string OaMobileRunTraining(
	JNIEnv* InEnvironment,
	jobject InCallback,
	const std::string& InDriverDirectory,
	const std::string& InNativeLibraryDirectory,
	const std::string& InCacheDirectory,
	const std::string& InCheckpointPath,
	const OaNlpSuiteRecipe& InRecipe,
	OaI32 InTotalSteps,
	OaI32 InBatchSize,
	bool InResume) {
	OaMobileCancelRequested.store(false, std::memory_order_release);
	OaMobileProgressCallback progress(InEnvironment, InCallback);
	auto turnip = OaMobileOpenTurnip(
		InDriverDirectory, InNativeLibraryDirectory, InCacheDirectory);
	auto getInstanceProcAddr = OaMobileLoadExport<PFN_vkGetInstanceProcAddr>(
		turnip.Get(), "vkGetInstanceProcAddr");
	OaVkInitCustom(getInstanceProcAddr);
	OaMobileVkScope vkScope;

	setenv("OA_VAR_DIR", InCacheDirectory.c_str(), 1);
	setenv("OA_DISABLE_GRU_SCAN", "1", 1);
	setenv("OA_DISABLE_RNN_SCAN", "1", 1);
	OaEngineConfig config;
	config.DevicePref = OaDevicePreference::Integrated;
	config.Precision = OaPrecision::FP32;
	config.NumericMode = OaNumericMode::Stable;
	config.EnableValidation = false;
	config.EnablePipelineCache = true;
	config.PreloadEmbeddedPipelines = false;
	config.PipelineCacheDir = OaString((InCacheDirectory + "/oa-vk").c_str());
	config.AppName = "OaMobileLab";

	auto engineResult = OaComputeEngine::Create(config);
	if (not engineResult.IsOk()) {
		OaMobileFail("OA engine creation failed: " +
			std::string(engineResult.GetStatus().ToString().c_str()));
	}
	auto engine = std::move(engineResult).GetValue();

	OaF32 initialLoss = 0.0F;
	OaF32 finalLoss = 0.0F;
	OaI64 completedSteps = 0;
	OaI64 optimizerStep = 0;
	OaI64 parameterCount = 0;
	OaI64 lastSourceUnits = 0;
	OaGpuTimingStats gpuStats;
	OaF64 wallMsPerStep = 0.0;
	OaF64 sourceUnitsPerSecond = 0.0;
	OaF32 finalAccuracy = 0.0F;
	std::string generated;
	bool checkpointSaved = false;
	bool checkpointRoundTrip = false;
	bool generationQualityEvaluated = false;
	bool generationQualityPassed = false;
	OaMobileGenerationQuality generationQuality;
	OaU64 parameterFingerprint = 0;
	const std::string deviceName = engine->Device.Info.Hardware.DeviceName.c_str();

	{
		// A hardware comparison cannot also be an initialization lottery. Philox
		// remains GPU-native; this only fixes its host-provided seed so desktop and
		// Android start from the same reproducible parameter stream.
		OaFnMatrix::SetRngSeed(OaNlpSuiteRngSeed);
		auto model = OaMakeSharedPtr<OaNlpSuiteModel>(InRecipe);
		auto parameters = model->AllParameterPtrs();
		auto optimizer = OaMakeUniquePtr<OaAdamW>(parameters, InRecipe.LearningRate());
		parameterCount = model->NumParameters();

		if (InResume) {
			const auto load = model->Load(InCheckpointPath.c_str(), *optimizer);
			if (not load.IsOk()) {
				OaMobileFail("Checkpoint load failed: " +
					std::string(load.ToString().c_str()));
			}
		}

		OaNlpSuiteSampler sampler(InRecipe, InBatchSize);
		OaItTraining training(*optimizer, OaItTrainingConfig{
			.TotalSteps = InTotalSteps,
			.BatchSize = InBatchSize,
			.SequenceLength = InRecipe.ContextLength(),
			.SequenceUnit = "token",
			.SourceUnit = InRecipe.Tokenizer() == OaNlpTokenizerKind::Char
				? "character"
				: "byte",
			.TimerName = InRecipe.TimerName(),
		});

		OaMatrix batchInput;
		OaMatrix batchTarget;
		while (not training.IsDone()) {
			if (OaMobileCancelRequested.load(std::memory_order_acquire)) {
				training.RequestStop();
				break;
			}
			sampler.Next(batchInput, batchTarget);
			lastSourceUnits = sampler.LastSourceUnits();
			training.RecordSourceUnits(lastSourceUnits);
			optimizer->ZeroGrad();
			OaGradientTape tape;
			auto logits = model->Forward(batchInput);
			auto loss = OaFnLoss::CrossEntropy(
				logits,
				batchTarget.Reshape(OaMatrixShape{batchTarget.NumElements()}));
			tape.Backward(loss);
			training.Next(loss);
			if (not training.LastStatus().IsOk()) {
				OaMobileFail("Training step failed: " +
					std::string(training.LastStatus().ToString().c_str()));
			}
			if (not training.HasLossSample()) {
				OaMobileFail("Training step completed without an exact loss sample");
			}

			completedSteps = training.Index();
			finalLoss = training.LiveLoss();
			if (completedSteps == 1) {
				initialLoss = finalLoss;
			}
			progress.Send(
				static_cast<OaI32>(completedSteps), InTotalSteps, finalLoss,
				training.LastGpuMs(), training.WallMsPerStep());
		}

		const auto finish = training.Finish();
		if (not finish.IsOk()) {
			OaMobileFail("Training finish failed: " +
				std::string(finish.ToString().c_str()));
		}
		gpuStats = training.GpuTimingStats();
		wallMsPerStep = training.WallMsPerStep();
		sourceUnitsPerSecond = training.WallSourceUnitsPerSecond();
		optimizerStep = optimizer->GetStep();
		if (completedSteps > 0) {
			finalAccuracy = 100.0F * OaFnMetric::Accuracy(
				model->Forward(batchInput), batchTarget);
		}

		const auto save = model->Save(InCheckpointPath.c_str(), *optimizer);
		if (not save.IsOk()) {
			OaMobileFail("Checkpoint save failed: " +
				std::string(save.ToString().c_str()));
		}
		checkpointSaved = true;
		parameterFingerprint = OaMobileParameterFingerprint(*model);

		if (not OaMobileCancelRequested.load(std::memory_order_acquire) and
			completedSteps > 0) {
			OaNlpSuiteSampler generationSampler(InRecipe, 1);
			const OaString generatedText = OaMobileGenerateGreedy(
				*model, generationSampler, InRecipe);
			const OaU64 afterFirstGenerationFingerprint =
				OaMobileParameterFingerprint(*model);
			if (afterFirstGenerationFingerprint != parameterFingerprint) {
				OaMobileFail("Inference mutated parameter bytes");
			}
			OaNlpSuiteSampler repeatSampler(InRecipe, 1);
			const OaString repeatedText = OaMobileGenerateGreedy(
				*model, repeatSampler, InRecipe);
			if (repeatedText != generatedText) {
				OaMobileFail(
					"Repeated deterministic generation changed: first='" +
					OaMobileEscapeText(generatedText) + "' repeat='" +
					OaMobileEscapeText(repeatedText) + "'");
			}
			generationQualityEvaluated =
				optimizerStep >= OaNlpSuiteTrainingSteps;
			generationQuality = OaMobileMeasureGenerationQuality(generatedText);
			generationQualityPassed = generationQuality.Pass();

			// Match the desktop tutorial's complete checkpoint gate: reconstruct a
			// fresh module/optimizer, reload the file, and require deterministic
			// fixed-prompt generation to survive the round trip exactly.
			auto reloaded = OaMakeSharedPtr<OaNlpSuiteModel>(InRecipe);
			auto reloadParameters = reloaded->AllParameterPtrs();
			auto reloadOptimizer = OaMakeUniquePtr<OaAdamW>(
				reloadParameters, InRecipe.LearningRate());
			const auto load = reloaded->Load(InCheckpointPath.c_str(), *reloadOptimizer);
			if (not load.IsOk()) {
				OaMobileFail("Checkpoint reload failed: " +
					std::string(load.ToString().c_str()));
			}
			const OaU64 reloadedFingerprint =
				OaMobileParameterFingerprint(*reloaded);
			if (reloadedFingerprint != parameterFingerprint) {
				std::ostringstream message;
				message << "Checkpoint reload changed parameter bytes: original=0x"
					<< std::hex << parameterFingerprint << " reloaded=0x"
					<< reloadedFingerprint;
				OaMobileFail(message.str());
			}
			OaNlpSuiteSampler reloadSampler(InRecipe, 1);
			const OaString reloadedText = OaMobileGenerateGreedy(
				*reloaded, reloadSampler, InRecipe);
			if (reloadedText != generatedText) {
				OaMobileFail(
					"Checkpoint reload changed deterministic generation with identical "
					"parameter bytes: original='" + OaMobileEscapeText(generatedText) +
					"' reloaded='" + OaMobileEscapeText(reloadedText) + "'");
			}
			checkpointRoundTrip = true;
			// Do not present random smoke-test output as learned language. The
			// sample becomes user-facing only after the checkpoint has accumulated
			// the desktop-equivalent 300 optimizer steps. Fresh shorter runs still
			// exercise checkpoint determinism without presenting random language.
			if (generationQualityEvaluated) {
				generated = OaMobileEscapeText(generatedText);
			}
		}
	}

	auto& runtimeContext = OaContext::GetDefault();
	const auto sync = runtimeContext.Sync();
	if (not sync.IsOk()) {
		OaMobileFail("Final runtime sync failed: " +
			std::string(sync.ToString().c_str()));
	}
	runtimeContext.Clear();

	const bool cancelled = OaMobileCancelRequested.load(std::memory_order_acquire);
	const OaF64 positions = static_cast<OaF64>(InBatchSize) * InRecipe.ContextLength();
	const OaF64 sourceUnitsPerToken = positions > 0.0
		? static_cast<OaF64>(lastSourceUnits) / positions
		: 1.0;
	const OaF64 bitsPerSourceUnit = sourceUnitsPerToken > 0.0
		? finalLoss / std::log(2.0F) / sourceUnitsPerToken
		: 0.0;
	std::ostringstream report;
	report << "OaMobileLab(\n"
		<< "  Driver: Mesa Turnip 26.1.4 (per-app)\n"
		<< "  Device: " << deviceName << '\n'
		<< "  Precision: FP32 / stable\n"
		<< "  Tokenizer: " << InRecipe.TokenizerName()
		<< " / vocab " << InRecipe.VocabSize() << '\n'
		<< "  Architecture: " << InRecipe.ArchitectureName() << '\n'
		<< "  Model: " << InRecipe.ModelDescription() << '\n'
		<< "  Parameters: " << parameterCount << '\n'
		<< "  Context: " << InRecipe.ContextLength() << " tokens\n"
		<< "  Batch: " << InBatchSize << '\n'
		<< "  Steps: " << completedSteps << '/' << InTotalSteps << '\n'
		<< "  OptimizerStep: " << optimizerStep << '\n'
		<< "  Resume: " << (InResume ? "yes" : "no") << '\n'
		<< "  Cancelled: " << (cancelled ? "yes" : "no") << '\n'
		<< std::fixed << std::setprecision(4)
		<< "  InitialLoss: " << initialLoss << '\n'
		<< "  FinalLoss: " << finalLoss << '\n'
		<< "  BitsPerSourceUnit: " << bitsPerSourceUnit << '\n'
		<< std::setprecision(2)
		<< "  GpuMeanMs: " << gpuStats.MeanMs << '\n'
		<< "  GpuP95Ms: " << gpuStats.P95Ms << '\n'
		<< "  WallMeanMs: " << wallMsPerStep << '\n'
		<< "  SourceUnitsPerSecond: " << sourceUnitsPerSecond << '\n'
		<< "  Checkpoint: " << (checkpointSaved ? InCheckpointPath : "not saved") << '\n'
		<< "  ParameterFingerprint: 0x" << std::hex << parameterFingerprint << std::dec << '\n'
		<< "  CheckpointRoundTrip: " << (checkpointRoundTrip ? "PASS" : "not run") << '\n'
		<< "  GenerationQuality: "
		<< (generationQualityEvaluated
			? (generationQualityPassed ? "PASS" : "FAIL")
			: "not evaluated (<300 optimizer steps)") << '\n';
	if (generationQualityEvaluated) {
		report << "  CorpusPrefixMatch: " << generationQuality.CorpusPrefix
			<< " characters\n"
			<< std::fixed << std::setprecision(1)
			<< "  Corpus8GramCoverage: "
			<< 100.0F * generationQuality.CorpusNgramCoverage << "%\n";
	}
	if (not generated.empty()) {
		report << "\nEvaluation:\n"
			<< "  Random-loss baseline ln(" << InRecipe.VocabSize() << ") = "
			<< std::setprecision(4)
			<< std::log(static_cast<double>(InRecipe.VocabSize())) << '\n'
			<< std::setprecision(1)
			<< "  Accuracy: " << finalAccuracy << "%\n"
			<< "\nGeneration:\n"
			<< "  Prompt: '" << OaNlpSuiteGenerationPrompt << "'\n"
			<< "  Generated: '" << generated << "'\n";
	}
	report << ")\n";

	__android_log_print(
		ANDROID_LOG_INFO,
		OaMobileLogTag,
		"%s/%s finished: %lld/%d steps, loss %.4f -> %.4f",
		InRecipe.TokenizerId(),
		InRecipe.ArchitectureId(),
		static_cast<long long>(completedSteps),
		InTotalSteps,
		initialLoss,
		finalLoss);
	return report.str();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_oa_mobilelab_TrainingService_nativeRunTraining(
	JNIEnv* InEnvironment,
	jclass,
	jstring InDriverDirectory,
	jstring InNativeLibraryDirectory,
	jstring InCacheDirectory,
	jstring InCheckpointPath,
	jstring InArchitecture,
	jstring InTokenizer,
	jint InSteps,
	jint InBatchSize,
	jboolean InResume,
	jobject InCallback) {
	const std::string architecture = OaMobileJavaString(
		InEnvironment, InArchitecture).Get();
	const std::string tokenizer = OaMobileJavaString(InEnvironment, InTokenizer).Get();
	const OaNlpSuiteRecipe recipe(
		OaMobileParseArchitecture(architecture),
		OaMobileParseTokenizer(tokenizer));
	try {
		const std::string report = OaMobileRunTraining(
			InEnvironment,
			InCallback,
			OaMobileJavaString(InEnvironment, InDriverDirectory).Get(),
			OaMobileJavaString(InEnvironment, InNativeLibraryDirectory).Get(),
			OaMobileJavaString(InEnvironment, InCacheDirectory).Get(),
			OaMobileJavaString(InEnvironment, InCheckpointPath).Get(),
			recipe,
			std::max<jint>(1, InSteps),
			std::max<jint>(1, InBatchSize),
			InResume == JNI_TRUE);
		return InEnvironment->NewStringUTF(report.c_str());
	} catch (const std::exception& error) {
		__android_log_print(
			ANDROID_LOG_ERROR, OaMobileLogTag, "%s/%s fatal: %s",
			recipe.TokenizerId(), recipe.ArchitectureId(), error.what());
		const std::string report =
			"OaMobileLab(\n  Tokenizer: " + std::string(recipe.TokenizerName()) +
			"\n  Architecture: " + recipe.ArchitectureName() +
			"\n  Fatal: " + error.what() + "\n)\n";
		return InEnvironment->NewStringUTF(report.c_str());
	}
}

extern "C" JNIEXPORT void JNICALL
Java_com_oa_mobilelab_TrainingService_nativeRequestCancel(JNIEnv*, jclass) {
	OaMobileCancelRequested.store(true, std::memory_order_release);
}
