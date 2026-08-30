#include <adrenotools/driver.h>
#include <adrenotools/priv.h>
#include <android/log.h>
#include <jni.h>
#include <vulkan/vulkan.h>

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnLoss.h>
#include <oa/ml/itTraining.h>
#include <oa/ml/metric.h>
#include <ml/nlpSuite.h>
#include <oa/ml/optim.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/core/std.h>

#include <cstdlib>
#include <dlfcn.h>
#include <sys/stat.h>

constexpr char OaMobileLogTag[] = "OA";
oa::Atomic<bool> OaMobileCancelRequested{false};

[[nodiscard]] static oa::Status OaMobileSubmitAndWait(oa::ExecutionSession& inContext) {
	if (inContext.nodeCount() == 0U) return oa::Status::ok();
	auto submitted = inContext.submit();
	if (not submitted.isOk()) return submitted.getStatus();
	return inContext.wait(submitted.getValue());
}

struct OaMobileError {
	oa::String message;
};

[[noreturn]] static void OaMobileFail(oa::String inMessage) {
	__android_log_print(ANDROID_LOG_ERROR, OaMobileLogTag, "%s", inMessage.cStr());
	throw OaMobileError{oa::move(inMessage)};
}

class OaMobileJavaString {
public:
	OaMobileJavaString(JNIEnv* inEnvironment, jstring inValue)
		: environment_(inEnvironment)
		, value_(inValue) {
		characters_ = inValue == nullptr
			? nullptr
			: inEnvironment->GetStringUTFChars(inValue, nullptr);
	}

	~OaMobileJavaString() {
		if (characters_ != nullptr) {
			environment_->ReleaseStringUTFChars(value_, characters_);
		}
	}

	[[nodiscard]] oa::String get() const {
		return characters_ == nullptr ? oa::String{} : oa::String(characters_);
	}

private:
	JNIEnv* environment_ = nullptr;
	jstring value_ = nullptr;
	const char* characters_ = nullptr;
};

class OaMobileVulkanLibrary {
public:
	explicit OaMobileVulkanLibrary(void* inHandle)
		: handle_(inHandle) {
	}

	OaMobileVulkanLibrary(const OaMobileVulkanLibrary&) = delete;
	OaMobileVulkanLibrary& operator=(const OaMobileVulkanLibrary&) = delete;

	~OaMobileVulkanLibrary() {
		if (handle_ != nullptr) {
			dlclose(handle_);
		}
	}

	[[nodiscard]] void* get() const { return handle_; }

private:
	void* handle_ = nullptr;
};

static OaMobileVulkanLibrary OaMobileOpenTurnip(
	oa::String inDriverDirectory,
	const oa::String& inNativeLibraryDirectory,
	const oa::String& inCacheDirectory) {
	if (not inDriverDirectory.empty() and inDriverDirectory.back() != '/') {
		inDriverDirectory.pushBack('/');
	}
	const oa::String temporaryDirectory = inCacheDirectory + "/adrenotools-training";
	(void)mkdir(temporaryDirectory.cStr(), 0700);
	void* handle = adrenotools_open_libvulkan(
		RTLD_NOW | RTLD_LOCAL,
		ADRENOTOOLS_DRIVER_CUSTOM,
		temporaryDirectory.cStr(),
		inNativeLibraryDirectory.cStr(),
		inDriverDirectory.cStr(),
		"libvulkan_freedreno.so",
		nullptr,
		nullptr);
	if (handle == nullptr) {
		const char* error = dlerror();
		OaMobileFail("Could not open bundled Turnip: " +
			(error == nullptr ? oa::String("unknown dlopen error") : oa::String(error)));
	}
	return OaMobileVulkanLibrary(handle);
}

template <typename Function>
static Function OaMobileLoadExport(void* inLibrary, const char* inName) {
	auto function = reinterpret_cast<Function>(dlsym(inLibrary, inName));
	if (function == nullptr) {
		OaMobileFail(oa::String("Missing vulkan export ") + inName);
	}
	return function;
}

class OaMobileProgressCallback {
public:
	OaMobileProgressCallback(JNIEnv* inEnvironment, jobject inCallback)
		: environment_(inEnvironment)
		, callback_(inCallback) {
		if (callback_ == nullptr) {
			return;
		}
		jclass callbackClass = environment_->GetObjectClass(callback_);
		method_ = environment_->GetMethodID(callbackClass, "onNativeProgress", "(IIFDD)V");
		environment_->DeleteLocalRef(callbackClass);
		if (method_ == nullptr) {
			environment_->ExceptionClear();
			OaMobileFail("training callback is missing onNativeProgress(IIFDD)");
		}
	}

	void send(oa::I32 inStep, oa::I32 inTotal, oa::F32 inLoss, oa::F64 inGpuMs, oa::F64 inWallMs) {
		if (callback_ == nullptr or method_ == nullptr) {
			return;
		}
		environment_->CallVoidMethod(
			callback_, method_, inStep, inTotal, inLoss,
			static_cast<jdouble>(inGpuMs), static_cast<jdouble>(inWallMs));
		if (environment_->ExceptionCheck()) {
			environment_->ExceptionDescribe();
			environment_->ExceptionClear();
			__android_log_print(
				ANDROID_LOG_WARN, OaMobileLogTag, "training progress callback failed");
		}
	}

private:
	JNIEnv* environment_ = nullptr;
	jobject callback_ = nullptr;
	jmethodID method_ = nullptr;
};

static oa::NlpArchitecture OaMobileParseArchitecture(const oa::String& inValue) {
	if (inValue == "rnn") {
		return oa::NlpArchitecture::Rnn;
	}
	if (inValue == "transformer") {
		return oa::NlpArchitecture::Transformer;
	}
	if (inValue == "moe") {
		return oa::NlpArchitecture::MoeTransformer;
	}
	if (inValue == "mamba3") {
		return oa::NlpArchitecture::Mamba3;
	}
	return oa::NlpArchitecture::Gru;
}

static oa::NlpTokenizerKind OaMobileParseTokenizer(const oa::String& inValue) {
	if (inValue == "bpe") {
		return oa::NlpTokenizerKind::Bpe;
	}
	if (inValue == "char") {
		return oa::NlpTokenizerKind::Char;
	}
	return oa::NlpTokenizerKind::Byte;
}

static oa::String OaMobileEscapeText(const oa::String& inValue) {
	oa::String escaped;
	escaped.reserve(inValue.size());
	for (const unsigned char value : inValue) {
		if (value >= 32 and value <= 126 and value != '\\') {
			escaped.pushBack(static_cast<char>(value));
		} else if (value == '\\') {
			escaped += "\\\\";
		} else {
			escaped += oa::format("\\x{:02x}", static_cast<oa::U32>(value));
		}
	}
	return escaped;
}

static oa::U64 OaMobileParameterFingerprint(oa::NlpSuiteModel& inModel) {
	auto& context = oa::ExecutionSession::getActive();
	const auto completion = OaMobileSubmitAndWait(context);
	if (not completion.isOk()) {
		OaMobileFail("Parameter fingerprint submission failed");
	}

	// exact FNV-1a over the serialized parameter order. This turns a vague
	// generation mismatch into a precise persistence boundary: either checkpoint
	// load changed model bytes, or identical bytes executed differently.
	oa::U64 hash = 1469598103934665603ULL;
	for (const auto* parameter : inModel.allParameterPtrs()) {
		const oa::U64 bytes = static_cast<oa::U64>(parameter->data.byteSize());
		oa::Vector<oa::U8> host(static_cast<oa::Usize>(bytes));
		const auto copy = oa::FnMatrix::copyToHost(parameter->data, host.data(), bytes);
		if (not copy.isOk()) {
			OaMobileFail("Parameter fingerprint readback failed");
		}
		for (const oa::U8 value : host) {
			hash ^= value;
			hash *= 1099511628211ULL;
		}
	}
	return hash;
}

struct OaMobileGenerationQuality {
	bool alphabet = false;
	oa::I32 corpusPrefix = 0;
	oa::F32 corpusNgramCoverage = 0.0F;

	[[nodiscard]] bool pass() const {
		// This is an intentional overfit tutorial, not an open-ended language
		// benchmark. A successful sample must enter a real continuation of the
		// fixed corpus and remain corpus-like. Merely producing printable lower-
		// case bytes is not evidence of learning ("/xeze/asae" exposed that hole).
		return alphabet and corpusPrefix >= 16 and corpusNgramCoverage >= 0.90F;
	}
};

static OaMobileGenerationQuality OaMobileMeasureGenerationQuality(
	const oa::String& inValue) {
	OaMobileGenerationQuality quality;
	const oa::Usize promptLength = oa::strlen(oa::NlpSuiteGenerationPrompt);
	const oa::Usize expectedLength = promptLength +
		static_cast<oa::Usize>(oa::NlpSuiteGenerationSourceUnits);
	if (inValue.size() != expectedLength) {
		return quality;
	}
	quality.alphabet = true;
	for (oa::Usize index = promptLength; index < inValue.size(); ++index) {
		const unsigned char value = inValue[index];
		if (value != ' ' and (value < 'a' or value > 'z')) {
			quality.alphabet = false;
			return quality;
		}
	}

	const oa::String value(inValue.data(), inValue.size());
	const oa::String corpus = oa::NlpSuiteSampler::corpus();
	const oa::String prompt = oa::NlpSuiteGenerationPrompt;
	const oa::String continuation = value.substr(promptLength);
	for (oa::Usize found = corpus.find(prompt);
		found != oa::String::Npos;
		found = corpus.find(prompt, found + 1)) {
		const oa::Usize corpusStart = found + prompt.size();
		oa::I32 matched = 0;
		while (static_cast<oa::Usize>(matched) < continuation.size() and
			corpusStart + static_cast<oa::Usize>(matched) < corpus.size() and
			continuation[static_cast<oa::Usize>(matched)] ==
				corpus[corpusStart + static_cast<oa::Usize>(matched)]) {
			++matched;
		}
		quality.corpusPrefix = oa::max(quality.corpusPrefix, matched);
	}

	constexpr oa::Usize ngramLength = 8;
	oa::I32 supported = 0;
	oa::I32 total = 0;
	if (value.size() >= ngramLength) {
		for (oa::Usize index = 0; index + ngramLength <= value.size(); ++index) {
			++total;
			if (corpus.find(value.substr(index, ngramLength)) != oa::String::Npos) {
				++supported;
			}
		}
	}
	quality.corpusNgramCoverage = total > 0
		? static_cast<oa::F32>(supported) / static_cast<oa::F32>(total)
		: 0.0F;
	return quality;
}

static oa::String OaMobileGenerateGreedy(
	oa::NlpSuiteModel& inModel,
	oa::NlpSuiteSampler& inSampler,
	const oa::NlpSuiteRecipe& inRecipe) {
	auto& runtimeContext = oa::ExecutionSession::getActive();
	// Match the desktop suite exactly. RNN and GRU deliberately use the causal
	// full-window path there; their single-token path remains a module API but is
	// not the cross-driver tutorial oracle. Mamba-3 is the only Byte tutorial
	// whose canonical generation uses its persistent step state.
	if (inRecipe.architecture() == oa::NlpArchitecture::Mamba3 and
		inModel.supportsStatefulGeneration()) {
		inModel.resetGenerationState(1);
		const auto resetCompletion = OaMobileSubmitAndWait(runtimeContext);
		if (not resetCompletion.isOk()) {
			OaMobileFail("Recurrent generation-state reset submission failed");
		}
		const auto prompt = inSampler.encode(oa::NlpSuiteGenerationPrompt);
		oa::Matrix logits;
		for (const oa::I32 token : prompt) {
			// Keep the token object explicit through execution. The graph retains
			// its buffer owner, but this also makes the recurrent step boundary and
			// host-side lifetime unambiguous during mobile diagnostics.
			auto input = inSampler.inputStepMatrix(token);
			logits = inModel.forwardGenerationStep(input);
			const auto completion = OaMobileSubmitAndWait(runtimeContext);
			if (not completion.isOk()) {
				OaMobileFail("Recurrent prompt-step submission failed");
			}
		}
		if (logits.numElements() == 0) {
			OaMobileFail("Recurrent generation prompt encoded to no tokens");
		}

		oa::String output(oa::NlpSuiteGenerationPrompt);
		oa::I32 generatedSourceUnits = 0;
		for (oa::I32 index = 0;
			index < oa::NlpSuiteGenerationSourceUnits and
			generatedSourceUnits < oa::NlpSuiteGenerationSourceUnits;
			++index) {
			auto row = logits.reshape(oa::MatrixShape{inRecipe.vocabSize()});
			const oa::I32 next = static_cast<oa::I32>(oa::FnMatrix::argmax(row));
			oa::Vector<oa::I32> nextToken{next};
			const oa::String decoded = inSampler.decode(nextToken);
			output += decoded;
			generatedSourceUnits += static_cast<oa::I32>(decoded.size());
			if (index + 1 < oa::NlpSuiteGenerationSourceUnits and
				generatedSourceUnits < oa::NlpSuiteGenerationSourceUnits) {
				auto input = inSampler.inputStepMatrix(next);
				logits = inModel.forwardGenerationStep(input);
				const auto stepCompletion = OaMobileSubmitAndWait(runtimeContext);
				if (not stepCompletion.isOk()) {
					OaMobileFail("Recurrent token-step submission failed");
				}
			}
		}
		const oa::Usize targetLength = oa::strlen(oa::NlpSuiteGenerationPrompt) +
			static_cast<oa::Usize>(oa::NlpSuiteGenerationSourceUnits);
		return output.size() > targetLength
			? output.substr(0, targetLength)
			: output;
	}

	const oa::I32 contextLength = inRecipe.contextLength();
	const oa::I32 padToken = inRecipe.tokenizer() == oa::NlpTokenizerKind::Char ? 26 : 0;
	oa::Vector<oa::I32> context(contextLength, padToken);
	const auto prompt = inSampler.encode(oa::NlpSuiteGenerationPrompt);
	const oa::I32 copyCount = oa::min(
		static_cast<oa::I32>(prompt.size()), contextLength);
	for (oa::I32 index = 0; index < copyCount; ++index) {
		context[index] = prompt[index];
	}

	oa::I32 filled = oa::max(copyCount, 1);
	oa::I32 logitRow = filled - 1;
	oa::String output(oa::NlpSuiteGenerationPrompt);
	oa::I32 generatedSourceUnits = 0;

	// Byte, Char, and BPE all generate the same amount of source text. A BPE
	// token can decode to several bytes, so a fixed token count is not the
	// desktop suite's contract and produced misleadingly longer BPE samples.
	for (oa::I32 index = 0;
		index < oa::NlpSuiteGenerationSourceUnits and
		generatedSourceUnits < oa::NlpSuiteGenerationSourceUnits;
		++index) {
		// Keep the source object explicit through the lazy graph execution. The
		// graph owns its buffer too; the local primarily makes this decode boundary
		// auditable alongside the synchronized recurrent step path above.
		auto input = inSampler.inputMatrix(context);
		auto logits = inModel.forward(input);
		auto row = oa::FnMatrix::reshape(
			oa::FnMatrix::slice(logits, 0, logitRow, logitRow + 1),
			oa::MatrixShape{inRecipe.vocabSize()});
		const oa::I32 next = static_cast<oa::I32>(oa::FnMatrix::argmax(row));
		oa::Vector<oa::I32> nextToken(1, next);
		const oa::String decoded = inSampler.decode(nextToken);
		output += decoded;
		generatedSourceUnits += static_cast<oa::I32>(decoded.size());

		if (filled < contextLength) {
			context[filled] = next;
			++filled;
			logitRow = filled - 1;
		} else {
			for (oa::I32 token = 1; token < contextLength; ++token) {
				context[token - 1] = context[token];
			}
			context[contextLength - 1] = next;
			logitRow = contextLength - 1;
		}
	}
	const oa::Usize targetLength = oa::strlen(oa::NlpSuiteGenerationPrompt) +
		static_cast<oa::Usize>(oa::NlpSuiteGenerationSourceUnits);
	return output.size() > targetLength ? output.substr(0, targetLength) : output;
}

static oa::String OaMobileRunTraining(
	JNIEnv* inEnvironment,
	jobject inCallback,
	const oa::String& inDriverDirectory,
	const oa::String& inNativeLibraryDirectory,
	const oa::String& inCacheDirectory,
	const oa::String& inCheckpointPath,
	const oa::NlpSuiteRecipe& inRecipe,
	oa::I32 inTotalSteps,
	oa::I32 inBatchSize,
	bool inResume) {
	OaMobileCancelRequested.store(false, oa::MemoryOrder::Release);
	OaMobileProgressCallback progress(inEnvironment, inCallback);
	auto turnip = OaMobileOpenTurnip(
		inDriverDirectory, inNativeLibraryDirectory, inCacheDirectory);

	setenv("OA_VAR_DIR", inCacheDirectory.cStr(), 1);
	setenv("OA_DISABLE_GRU_SCAN", "1", 1);
	setenv("OA_DISABLE_RNN_SCAN", "1", 1);
	oa::EngineConfig config;
	config.devicePref = oa::DevicePreference::Integrated;
	config.precision = oa::Precision::FP32;
	config.numericMode = oa::NumericMode::Stable;
	config.enableValidation = false;
	config.enablePipelineCache = true;
	config.preloadEmbeddedPipelines = false;
	config.pipelineCacheDir = oa::String((inCacheDirectory + "/oa-vk").cStr());
	config.appName = "OaMobileLab";
	// Route the Turnip/Adreno per-app loader through EngineConfig instead of
	// calling VKL directly. The engine installs it before loader initialization.
	config.vulkanLoaderProcAddr = OaMobileLoadExport<PFN_vkGetInstanceProcAddr>(
		turnip.get(), "vkGetInstanceProcAddr");

	auto engineResult = oa::Engine::create(config);
	if (not engineResult.isOk()) {
		OaMobileFail("OA engine creation failed: " +
			oa::String(engineResult.getStatus().toString().cStr()));
	}
	auto engine = oa::move(engineResult).getValue();

	oa::F32 initialLoss = 0.0F;
	oa::F32 finalLoss = 0.0F;
	oa::I64 completedSteps = 0;
	oa::I64 optimizerStep = 0;
	oa::I64 parameterCount = 0;
	oa::I64 lastSourceUnits = 0;
	oa::GpuTimingStats gpuStats;
	oa::F64 wallMsPerStep = 0.0;
	oa::F64 sourceUnitsPerSecond = 0.0;
	oa::F32 finalAccuracy = 0.0F;
	oa::String generated;
	bool checkpointSaved = false;
	bool checkpointRoundTrip = false;
	bool generationQualityEvaluated = false;
	bool generationQualityPassed = false;
	OaMobileGenerationQuality generationQuality;
	oa::U64 parameterFingerprint = 0;
	const oa::StringView deviceNameView = engine->deviceName();
	const oa::String deviceName(deviceNameView.data(), deviceNameView.size());

	{
		// A hardware comparison cannot also be an initialization lottery. Philox
		// remains GPU-native; this only fixes its host-provided seed so desktop and
		// android start from the same reproducible parameter stream.
		oa::FnMatrix::setRngSeed(oa::NlpSuiteRngSeed);
		auto model = oa::makeShared<oa::NlpSuiteModel>(inRecipe);
		auto parameters = model->allParameterPtrs();
		auto optimizer = oa::makeUnique<oa::AdamW>(parameters, inRecipe.learningRate());
		parameterCount = model->numParameters();

		if (inResume) {
			const auto load = model->load(*engine, inCheckpointPath.cStr(), *optimizer);
			if (not load.isOk()) {
				OaMobileFail("checkpoint load failed: " +
					oa::String(load.toString().cStr()));
			}
		}

		oa::NlpSuiteSampler sampler(inRecipe, inBatchSize);
		oa::ItTraining training(*engine, *optimizer, oa::ItTrainingConfig{
			.totalSteps = inTotalSteps,
			.batchSize = inBatchSize,
			.sequenceLength = inRecipe.contextLength(),
			.sequenceUnit = "token",
			.sourceUnit = inRecipe.tokenizer() == oa::NlpTokenizerKind::Char
				? "character"
				: "byte",
			.timerName = inRecipe.timerName(),
		});

		oa::Matrix batchInput;
		oa::Matrix batchTarget;
		while (not training.isDone()) {
			if (OaMobileCancelRequested.load(oa::MemoryOrder::Acquire)) {
				training.requestStop();
				break;
			}
			sampler.next(batchInput, batchTarget);
			lastSourceUnits = sampler.lastSourceUnits();
			training.recordSourceUnits(lastSourceUnits);
			optimizer->zeroGrad();
			oa::GradientTape tape;
			auto logits = model->forward(batchInput);
			auto loss = oa::FnLoss::crossEntropy(
				logits,
				batchTarget.reshape(oa::MatrixShape{batchTarget.numElements()}));
			tape.backward(loss);
			training.next(loss);
			if (not training.lastStatus().isOk()) {
				OaMobileFail("training step failed: " +
					oa::String(training.lastStatus().toString().cStr()));
			}
			if (not training.hasLossSample()) {
				OaMobileFail("training step completed without an exact loss sample");
			}

			completedSteps = training.index();
			finalLoss = training.lastLoss();
			if (completedSteps == 1) {
				initialLoss = finalLoss;
			}
			progress.send(
				static_cast<oa::I32>(completedSteps), inTotalSteps, finalLoss,
				training.lastGpuMs(), training.wallMsPerStep());
		}

		const auto finish = training.finish();
		if (not finish.isOk()) {
			OaMobileFail("training finish failed: " +
				oa::String(finish.toString().cStr()));
		}
		gpuStats = training.gpuTimingStats();
		wallMsPerStep = training.wallMsPerStep();
		sourceUnitsPerSecond = training.wallSourceUnitsPerSecond();
		optimizerStep = optimizer->getStep();
		if (completedSteps > 0) {
			finalAccuracy = 100.0F * oa::FnMetric::accuracy(
				model->forward(batchInput), batchTarget);
		}

		const auto save = model->save(*engine, inCheckpointPath.cStr(), *optimizer);
		if (not save.isOk()) {
			OaMobileFail("checkpoint save failed: " +
				oa::String(save.toString().cStr()));
		}
		checkpointSaved = true;
		parameterFingerprint = OaMobileParameterFingerprint(*model);

		if (not OaMobileCancelRequested.load(oa::MemoryOrder::Acquire) and
			completedSteps > 0) {
			oa::NlpSuiteSampler generationSampler(inRecipe, 1);
			const oa::String generatedText = OaMobileGenerateGreedy(
				*model, generationSampler, inRecipe);
			const oa::U64 afterFirstGenerationFingerprint =
				OaMobileParameterFingerprint(*model);
			if (afterFirstGenerationFingerprint != parameterFingerprint) {
				OaMobileFail("Inference mutated parameter bytes");
			}
			oa::NlpSuiteSampler repeatSampler(inRecipe, 1);
			const oa::String repeatedText = OaMobileGenerateGreedy(
				*model, repeatSampler, inRecipe);
			if (repeatedText != generatedText) {
				OaMobileFail(
					"Repeated deterministic generation changed: first='" +
					OaMobileEscapeText(generatedText) + "' repeat='" +
					OaMobileEscapeText(repeatedText) + "'");
			}
			generationQualityEvaluated =
				optimizerStep >= oa::NlpSuiteTrainingSteps;
			generationQuality = OaMobileMeasureGenerationQuality(generatedText);
			generationQualityPassed = generationQuality.pass();

			// Match the desktop tutorial's complete checkpoint gate: reconstruct a
			// fresh module/optimizer, reload the file, and require deterministic
			// fixed-prompt generation to survive the round trip exactly.
			auto reloaded = oa::makeShared<oa::NlpSuiteModel>(inRecipe);
			auto reloadParameters = reloaded->allParameterPtrs();
			auto reloadOptimizer = oa::makeUnique<oa::AdamW>(
				reloadParameters, inRecipe.learningRate());
			const auto load = reloaded->load(
				*engine, inCheckpointPath.cStr(), *reloadOptimizer);
			if (not load.isOk()) {
				OaMobileFail("checkpoint reload failed: " +
					oa::String(load.toString().cStr()));
			}
			const oa::U64 reloadedFingerprint =
				OaMobileParameterFingerprint(*reloaded);
			if (reloadedFingerprint != parameterFingerprint) {
				OaMobileFail(oa::format(
					"checkpoint reload changed parameter bytes: original=0x{:x} reloaded=0x{:x}",
					parameterFingerprint, reloadedFingerprint));
			}
			oa::NlpSuiteSampler reloadSampler(inRecipe, 1);
			const oa::String reloadedText = OaMobileGenerateGreedy(
				*reloaded, reloadSampler, inRecipe);
			if (reloadedText != generatedText) {
				OaMobileFail(
					"checkpoint reload changed deterministic generation with identical "
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

	auto& runtimeContext = oa::ExecutionSession::getActive();
	const auto completion = OaMobileSubmitAndWait(runtimeContext);
	if (not completion.isOk()) {
		OaMobileFail("Final runtime submission failed: " +
			oa::String(completion.toString().cStr()));
	}
	runtimeContext.clear();

	const bool cancelled = OaMobileCancelRequested.load(oa::MemoryOrder::Acquire);
	const oa::F64 positions = static_cast<oa::F64>(inBatchSize) * inRecipe.contextLength();
	const oa::F64 sourceUnitsPerToken = positions > 0.0
		? static_cast<oa::F64>(lastSourceUnits) / positions
		: 1.0;
	const oa::F64 bitsPerSourceUnit = sourceUnitsPerToken > 0.0
		? finalLoss / oa::log(2.0F) / sourceUnitsPerToken
		: 0.0;
	oa::String report = oa::format(
		"OaMobileLab(\n"
		"  Driver: Mesa Turnip 26.1.4 (per-app)\n"
		"  Device: {}\n"
		"  Precision: FP32 / stable\n"
		"  tokenizer: {} / vocab {}\n"
		"  architecture: {}\n"
		"  Model: {}\n"
		"  parameters: {}\n"
		"  context: {} tokens\n"
		"  Batch: {}\n"
		"  steps: {}/{}\n"
		"  optimizerStep: {}\n"
		"  Resume: {}\n"
		"  Cancelled: {}\n"
		"  initialLoss: {:.4f}\n"
		"  finalLoss: {:.4f}\n"
		"  BitsPerSourceUnit: {:.4f}\n"
		"  GpuMeanMs: {:.2f}\n"
		"  GpuP95Ms: {:.2f}\n"
		"  WallMeanMs: {:.2f}\n"
		"  SourceUnitsPerSecond: {:.2f}\n"
		"  checkpoint: {}\n"
		"  ParameterFingerprint: 0x{:x}\n"
		"  checkpointRoundTrip: {}\n"
		"  GenerationQuality: {}\n",
		deviceName,
		inRecipe.tokenizerName(), inRecipe.vocabSize(),
		inRecipe.architectureName(), inRecipe.modelDescription(), parameterCount,
		inRecipe.contextLength(), inBatchSize, completedSteps, inTotalSteps,
		optimizerStep, inResume ? "yes" : "no", cancelled ? "yes" : "no",
		initialLoss, finalLoss, bitsPerSourceUnit, gpuStats.meanMs, gpuStats.p95Ms,
		wallMsPerStep, sourceUnitsPerSecond,
		checkpointSaved ? inCheckpointPath.view() : oa::StringView("not saved"),
		parameterFingerprint, checkpointRoundTrip ? "PASS" : "not run",
		generationQualityEvaluated
			? (generationQualityPassed ? "PASS" : "FAIL")
			: "not evaluated (<300 optimizer steps)");
	if (generationQualityEvaluated) {
		report += oa::format(
			"  CorpusPrefixMatch: {} characters\n"
			"  Corpus8GramCoverage: {:.1f}%\n",
			generationQuality.corpusPrefix,
			100.0F * generationQuality.corpusNgramCoverage);
	}
	if (not generated.empty()) {
		report += oa::format(
			"\nEvaluation:\n"
			"  Random-loss baseline ln({}) = {:.4f}\n"
			"  Accuracy: {:.1f}%\n"
			"\nGeneration:\n"
			"  prompt: '{}'\n"
			"  generated: '{}'\n",
			inRecipe.vocabSize(),
			oa::log(static_cast<oa::F64>(inRecipe.vocabSize())), finalAccuracy,
			oa::NlpSuiteGenerationPrompt, generated);
	}
	report += ")\n";

	__android_log_print(
		ANDROID_LOG_INFO,
		OaMobileLogTag,
		"%s/%s finished: %lld/%d steps, loss %.4f -> %.4f",
		inRecipe.tokenizerId(),
		inRecipe.architectureId(),
		static_cast<long long>(completedSteps),
		inTotalSteps,
		initialLoss,
		finalLoss);
	return report;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_oa_mobilelab_TrainingService_nativeRunTraining(
	JNIEnv* inEnvironment,
	jclass,
	jstring inDriverDirectory,
	jstring inNativeLibraryDirectory,
	jstring inCacheDirectory,
	jstring inCheckpointPath,
	jstring inArchitecture,
	jstring inTokenizer,
	jint inSteps,
	jint inBatchSize,
	jboolean inResume,
	jobject inCallback) {
	const oa::String architecture = OaMobileJavaString(
		inEnvironment, inArchitecture).get();
	const oa::String tokenizer = OaMobileJavaString(inEnvironment, inTokenizer).get();
	const oa::NlpSuiteRecipe recipe(
		OaMobileParseArchitecture(architecture),
		OaMobileParseTokenizer(tokenizer));
	try {
		const oa::String report = OaMobileRunTraining(
			inEnvironment,
			inCallback,
			OaMobileJavaString(inEnvironment, inDriverDirectory).get(),
			OaMobileJavaString(inEnvironment, inNativeLibraryDirectory).get(),
			OaMobileJavaString(inEnvironment, inCacheDirectory).get(),
			OaMobileJavaString(inEnvironment, inCheckpointPath).get(),
			recipe,
			oa::max<jint>(1, inSteps),
			oa::max<jint>(1, inBatchSize),
			inResume == JNI_TRUE);
		return inEnvironment->NewStringUTF(report.cStr());
	} catch (const OaMobileError& error) {
		__android_log_print(
			ANDROID_LOG_ERROR, OaMobileLogTag, "%s/%s fatal: %s",
			recipe.tokenizerId(), recipe.architectureId(), error.message.cStr());
		const oa::String report =
			"OaMobileLab(\n  tokenizer: " + oa::String(recipe.tokenizerName()) +
			"\n  architecture: " + recipe.architectureName() +
			"\n  Fatal: " + error.message + "\n)\n";
		return inEnvironment->NewStringUTF(report.cStr());
	}
}

extern "C" JNIEXPORT void JNICALL
Java_com_oa_mobilelab_TrainingService_nativeRequestCancel(JNIEnv*, jclass) {
	OaMobileCancelRequested.store(true, oa::MemoryOrder::Release);
}
