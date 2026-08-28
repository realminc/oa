// OA_DOC_BEGIN: ml-transformer
#include <oa/oa.h>

#include <algorithm>
#include <cstdio>

namespace {
constexpr oa::I32 VocabSize = 300;
constexpr oa::I32 ContextLength = 16;
constexpr oa::I32 ModelWidth = 32;
constexpr oa::I32 HiddenWidth = 64;
constexpr oa::I32 BatchSize = 64;
constexpr oa::I32 TrainingSteps = 300;
constexpr const char* Corpus =
	"to be or not to be that is the question whether tis nobler in the mind "
	"to suffer the slings and arrows of outrageous fortune or to take arms "
	"against a sea of troubles and by opposing end them to be or not to be "
	"that is the question whether tis nobler in the mind to suffer the slings "
	"and arrows of outrageous fortune or to take arms against a sea of "
	"troubles and by opposing end them to be or not to be that is the question "
	"whether tis nobler in the mind to suffer the slings and arrows of "
	"outrageous fortune or to take arms against a sea of troubles and by "
	"opposing end them ";

oa::Matrix tokenMatrix(const oa::Vector<oa::I32>& tokens, oa::I32 batch) {
	return oa::FnMatrix::fromInt32(
		oa::Span<const oa::I32>(tokens.data(), tokens.size()),
		{batch, ContextLength}, oa::ScalarType::UInt32
	);
}

void nextBatch(const oa::Vector<oa::I32>& tokens, oa::I64& cursor,
	oa::Matrix& input, oa::Matrix& target) {
	const oa::I64 limit = static_cast<oa::I64>(tokens.size()) - ContextLength - 1;
	oa::Vector<oa::I32> x(static_cast<oa::Usize>(BatchSize * ContextLength));
	oa::Vector<oa::I32> y(x.size());
	for (oa::I32 batch = 0; batch < BatchSize; ++batch) {
		const oa::I64 start = (cursor + static_cast<oa::I64>(batch) * 7) % limit;
		for (oa::I32 position = 0; position < ContextLength; ++position) {
			const auto output = static_cast<oa::Usize>(batch * ContextLength + position);
			x[output] = tokens[static_cast<oa::Usize>(start + position)];
			y[output] = tokens[static_cast<oa::Usize>(start + position + 1)];
		}
	}
	cursor = (cursor + BatchSize) % limit;
	input = tokenMatrix(x, BatchSize);
	target = tokenMatrix(y, BatchSize);
}

oa::String generate(oa::NnTransformer& model, const oa::BpeTokenizer& tokenizer) {
	const oa::String prompt = "to be";
	auto promptTokens = tokenizer.encode(prompt.cStr());
	oa::Vector<oa::I32> context(static_cast<oa::Usize>(ContextLength), 0);
	const auto copied = std::min<oa::Usize>(promptTokens.size(), context.size());
	for (oa::Usize index = 0; index < copied; ++index) context[index] = promptTokens[index];
	oa::I32 filled = std::max<oa::I32>(1, static_cast<oa::I32>(copied));
	oa::String result = prompt;
	for (oa::I32 index = 0; index < 32; ++index) {
		auto logits = model.forward(tokenMatrix(context, 1));
		auto row = oa::FnMatrix::slice(logits, 0, filled - 1, filled);
		const auto next = static_cast<oa::I32>(oa::FnMatrix::argmax(row.reshape({VocabSize})));
		result += tokenizer.decode({next});
		if (filled < ContextLength) {
			context[static_cast<oa::Usize>(filled++)] = next;
		} else {
			for (oa::Usize token = 1; token < context.size(); ++token) context[token - 1] = context[token];
			context[ContextLength - 1] = next;
		}
	}
	return result;
}
} // namespace

OA_MAIN("ExampleMlTransformer") {
	oa::FnMatrix::setRngSeed(20260714ULL);
	oa::BpeTokenizer tokenizer(VocabSize);
	tokenizer.train(Corpus, VocabSize - 256);
	if (tokenizer.vocabSize() != VocabSize) return 1;
	const auto corpusTokens = tokenizer.encode(Corpus);

	oa::NnTransformer model(VocabSize, ContextLength, ModelWidth, HiddenWidth);
	auto parameters = model.allParameterPtrs();
	oa::AdamW optimizer(parameters, 0.01F);
	oa::MetricLoss lossMetric;
	oa::CbProgressBar progress;
	oa::CbSummary summary;
	progress.addMetric(&lossMetric);
	oa::ItTrainingConfig config;
	config.totalSteps = TrainingSteps;
	config.batchSize = BatchSize;
	config.sequenceLength = ContextLength;
	config.sequenceUnit = "token";
	config.timerName = "example_transformer_step";
	oa::ItTraining training(engine, optimizer, config);
	training.addMetric(&lossMetric);
	training.addCallback(&progress);
	training.addCallback(&summary);

	std::printf("\nOA SDK Example — BPE Transformer · all-position LM\n");
	std::printf("Tokenizer: byte BPE · vocab=%d · context=%d\n", VocabSize, ContextLength);
	std::printf(
		"Model: NnTransformer(width=%d, hidden=%d, layers=1, heads=1)\n",
		ModelWidth,
		HiddenWidth
	);
	std::printf("Params: %lld · AdamW(lr=0.01)\n", static_cast<long long>(model.numParameters()));
	std::printf(
		"Training: %d steps · batch=%d · sequence=%d tokens\n",
		TrainingSteps,
		BatchSize,
		ContextLength
	);

	oa::I64 cursor = 0;
	oa::Matrix input;
	oa::Matrix target;
	oa::F32 initialLoss = 0.0F;
	while (not training.isDone()) {
		nextBatch(corpusTokens, cursor, input, target);
		optimizer.zeroGrad();
		oa::GradientTape tape;
		auto logits = model.forward(input);
		auto loss = oa::FnLoss::crossEntropy(logits, target.reshape({target.numElements()}));
		tape.backward(loss);
		training.next(loss);
		if (training.index() == 1) {
			initialLoss = training.lastLoss();
		}
	}
	if (not training.finish().isOk()) {
		return 1;
	}

	const oa::F32 finalLoss = training.lastLoss();
	if (not (finalLoss < initialLoss)) {
		return 1;
	}
	const auto generated = generate(model, tokenizer);
	std::printf("Transformer training verified: vocab=300, steps=300\n");
	std::printf("Loss: %.4f -> %.4f\n", initialLoss, finalLoss);
	std::printf("Generated: %s\n", generated.cStr());
	return 0;
}
// OA_DOC_END: ml-transformer
