# OA_DOC_BEGIN: ml-transformer
import oa

vocabSize = 300
contextLength = 16
modelWidth = 32
hiddenWidth = 64
batchSize = 64
trainingSteps = 300
corpus = (
	"to be or not to be that is the question whether tis nobler in the mind "
	"to suffer the slings and arrows of outrageous fortune or to take arms "
	"against a sea of troubles and by opposing end them to be or not to be "
	"that is the question whether tis nobler in the mind to suffer the slings "
	"and arrows of outrageous fortune or to take arms against a sea of "
	"troubles and by opposing end them to be or not to be that is the question "
	"whether tis nobler in the mind to suffer the slings and arrows of "
	"outrageous fortune or to take arms against a sea of troubles and by "
	"opposing end them "
)

oa.FnMatrix.setRngSeed(20260714)
tokenizer = oa.BpeTokenizer(vocabSize)
tokenizer.train(corpus, vocabSize - 256)
assert tokenizer.vocabSize() == vocabSize
corpusTokens = tokenizer.encode(corpus)

model = oa.NnTransformer(
	vocabSize, contextLength, modelWidth, hiddenWidth,
)
parameters = model.allParameterPtrs()

def tokenMatrix(tokens, batch):
	return oa.FnMatrix.fromInt32(tokens, [batch, contextLength], oa.ScalarType.UInt32)

def nextBatch(tokens, cursor):
	limit = len(tokens) - contextLength - 1
	x, y = [], []
	for batch in range(batchSize):
		start = (cursor + batch * 7) % limit
		x.extend(tokens[start:start + contextLength])
		y.extend(tokens[start + 1:start + contextLength + 1])
	return tokenMatrix(x, batchSize), tokenMatrix(y, batchSize), (cursor + batchSize) % limit

def generate(tokenizer):
	prompt = "to be"
	context = (tokenizer.encode(prompt) + [0] * contextLength)[:contextLength]
	filled = max(1, min(len(tokenizer.encode(prompt)), contextLength))
	result = prompt
	for _ in range(32):
		logits = model.forward(tokenMatrix(context, 1))
		row = oa.FnMatrix.slice(logits, 0, filled - 1, filled)
		nextToken = int(oa.FnMatrix.argmax(row.reshape([vocabSize])))
		result += tokenizer.decode([nextToken])
		if filled < contextLength:
			context[filled] = nextToken
			filled += 1
		else:
			context = context[1:] + [nextToken]
	return result

optimizer = oa.AdamW(parameters, 0.01)
lossMetric = oa.MetricLoss()
progress = oa.CbProgressBar()
summary = oa.CbSummary()
progress.addMetric(lossMetric)
config = oa.ItTrainingConfig()
config.totalSteps = trainingSteps
config.batchSize = batchSize
config.sequenceLength = contextLength
config.sequenceUnit = "token"
config.timerName = "example_transformer_step"
training = oa.ItTraining(optimizer, config)
training.addMetric(lossMetric)
training.addCallback(progress)
training.addCallback(summary)

print("\nOA SDK Example — BPE Transformer · all-position LM")
print(f"Tokenizer: byte BPE · vocab={vocabSize} · context={contextLength}")
print(f"Model: NnTransformer(width={modelWidth}, hidden={hiddenWidth}, layers=1, heads=1)")
print(f"Params: {model.numParameters()} · AdamW(lr=0.01)")
print(f"Training: {trainingSteps} steps · batch={batchSize} · sequence={contextLength} tokens")

cursor = 0
initialLoss = 0.0
while not training.isDone():
	input, target, cursor = nextBatch(corpusTokens, cursor)
	optimizer.zeroGrad()
	with oa.GradientTape() as tape:
		logits = model.forward(input)
		loss = oa.FnLoss.crossEntropy(logits, target.reshape([target.numElements()]))
		tape.backward(loss)
	training.next(loss)
	if training.index() == 1:
		initialLoss = training.lastLoss()
training.finish()

finalLoss = training.lastLoss()
assert finalLoss < initialLoss
print("Transformer training verified: vocab=300, steps=300")
print(f"Loss: {initialLoss:.4f} -> {finalLoss:.4f}")
print(f"Generated: {generate(tokenizer)}")
# OA_DOC_END: ml-transformer
