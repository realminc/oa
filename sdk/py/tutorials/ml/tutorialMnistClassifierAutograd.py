#!/usr/bin/env python3
"""
OA Python Tutorial — Fashion-MNIST Image Classification (Implicit Autograd)

Python port of sdk/cpp/tutorials/ml/tutorialMnistClassifierAg.cpp.

Model: Linear(784→128, ReLU) → Linear(128→10)
Loss: CrossEntropy
Optimizer: AdamW
Backward: Implicit autograd via GradientTape

Architecture:
  Input: 28×28 grayscale images (784 pixels)
  Hidden: 128 neurons with ReLU activation
  Output: 10 classes (Fashion-MNIST categories)
  Params: 101,770
  Batch: 64
  Epochs: 5
  Optimizer: AdamW(lr=0.001, weight_decay=0.01)

Dataset: Fashion-MNIST IDX files
  Train: 60,000 images
  Test: 10,000 images
  Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
"""

import sys
import os
import array
import struct

# pyright: reportWildcardImportFromLibrary=false
from oa import *

# ─── Hyperparameters ─────────────────────────────────────────────────────────

INPUT_DIM = 784
HIDDEN_DIM = 128
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001

CLASSES = [
	"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
	"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ─── IDX Dataset Loader ──────────────────────────────────────────────────────

def readIdxImages(filepath):
	"""Read IDX image file format (Fashion-MNIST)."""
	with open(filepath, 'rb') as f:
		magic = struct.unpack('>I', f.read(4))[0]
		if magic != 2051:
			raise ValueError(f"Invalid magic number {magic} in {filepath}")
		numImages = struct.unpack('>I', f.read(4))[0]
		rows = struct.unpack('>I', f.read(4))[0]
		cols = struct.unpack('>I', f.read(4))[0]
		data = array.array('B', f.read())
		return data, numImages, rows, cols

def readIdxLabels(filepath):
	"""Read IDX label file format (Fashion-MNIST)."""
	with open(filepath, 'rb') as f:
		magic = struct.unpack('>I', f.read(4))[0]
		if magic != 2049:
			raise ValueError(f"Invalid magic number {magic} in {filepath}")
		numLabels = struct.unpack('>I', f.read(4))[0]
		data = array.array('B', f.read())
		return data, numLabels

class MnistDataLoader:
	"""Simple MNIST data loader with batching."""
	def __init__(self, dataDir, split, batchSize, shuffle=False):
		self.batchSize = batchSize
		self.shuffle = shuffle

		if split == "train":
			imgFile = os.path.join(dataDir, "train-images-idx3-ubyte")
			lblFile = os.path.join(dataDir, "train-labels-idx1-ubyte")
		else:  # test
			imgFile = os.path.join(dataDir, "t10k-images-idx3-ubyte")
			lblFile = os.path.join(dataDir, "t10k-labels-idx1-ubyte")

		self.images, self.numImages, self.rows, self.cols = readIdxImages(imgFile)
		self.labels, self.numLabels = readIdxLabels(lblFile)

		if self.numImages != self.numLabels:
			raise ValueError(f"Image count {self.numImages} != label count {self.numLabels}")

		self.indices = list(range(self.numImages))
		self.cursor = 0

		if self.shuffle:
			import random
			random.shuffle(self.indices)

	def nextBatch(self):
		"""Get next batch of images and labels."""
		if self.cursor >= self.numImages:
			return None, None

		end = min(self.cursor + self.batchSize, self.numImages)
		actualBatch = end - self.cursor

		# Gather batch. Images must be uploaded as Float32 — Scale/matmul on a
		# UInt8 matrix silently produces garbage (only gather/CrossEntropy read
		# integer tensors correctly), so pixels go up as floats via FromFloats and
		# the model's Scale(1/255) normalizes them. Labels stay bytes: CrossEntropy
		# reads integer class indices directly.
		batchImages = array.array('f')
		batchLabels = array.array('B')

		for i in range(self.cursor, end):
			idx = self.indices[i]
			start = idx * self.rows * self.cols
			endImg = start + self.rows * self.cols
			batchImages.extend(float(p) for p in self.images[start:endImg])
			batchLabels.append(self.labels[idx])

		self.cursor = end

		# Convert to Matrix
		x = FnMatrix.fromFloats(batchImages, actualBatch, INPUT_DIM)
		y = FnMatrix.fromBytes(batchLabels, actualBatch, oa.ScalarType.UInt8)

		return x, y

	def reset(self):
		"""Reset cursor to beginning."""
		self.cursor = 0
		if self.shuffle:
			import random
			random.shuffle(self.indices)

# ─── Model: Linear(784→128, ReLU) → Linear(128→10) ──────────────────────────

class MnistClassifier:
	def __init__(self):
		wd = FnMatrix.weightDtype()

		# Hidden layer (Linear + ReLU)
		self.fc1 = Linear(INPUT_DIM, HIDDEN_DIM)
		self.fc1.setActivation(oa.Activation.Relu)
		self._fc1P = self.fc1.parameters()
		self._fc1P[0].data = FnMatrix.randKaimingUniform(HIDDEN_DIM, INPUT_DIM, wd)
		self._fc1P[0].data.setRequiresGrad(True)
		self._fc1P[1].data.setRequiresGrad(True)

		# Output layer
		self.fc2 = Linear(HIDDEN_DIM, NUM_CLASSES)
		self._fc2P = self.fc2.parameters()
		self._fc2P[0].data = FnMatrix.randGlorotUniform(NUM_CLASSES, HIDDEN_DIM, wd)
		self._fc2P[0].data.setRequiresGrad(True)
		self._fc2P[1].data.setRequiresGrad(True)

	def parameters(self):
		return self._fc1P + self._fc2P

	def forward(self, x):
		"""Forward pass — returns logits [B, 10]."""
		xNorm = FnMatrix.scale(x, 1.0 / 255.0)  # Normalize [0,255] → [0,1]
		h = self.fc1.forward(xNorm)         # [B, 128] (LinearRelu fused)
		return self.fc2.forward(h)           # [B, 10]

# ─── Inference helpers ───────────────────────────────────────────────────────

def argmaxRow(probs, row, numClasses):
	"""Host-side argmax for a single row."""
	best = 0
	bestVal = probs[row * numClasses]
	for i in range(1, numClasses):
		v = probs[row * numClasses + i]
		if v > bestVal:
			bestVal = v
			best = i
	return best, bestVal

def predict(model, x):
	"""Get predictions for a batch."""
	logits = model.forward(x)
	probs = FnMatrix.softmax(logits, -1)

	batch = int(x.size(0))
	flat = FnMatrix.copyToHost(probs)

	predictions = []
	for i in range(batch):
		classIdx, confidence = argmaxRow(flat, i, NUM_CLASSES)
		predictions.append((classIdx, confidence * 100.0))

	return predictions

def evalAccuracy(model, loader):
	"""Evaluate accuracy on full dataset."""
	correct = 0
	total = 0

	loader.reset()
	while True:
		x, y = loader.nextBatch()
		if x is None:
			break

		preds = predict(model, x)
		labels = FnMatrix.copyToHost(y)

		batch = len(preds)
		for i in range(batch):
			if preds[i][0] == labels[i]:
				correct += 1
		total += batch

	loader.reset()
	return 100.0 * correct / total if total > 0 else 0.0

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
	dataDir = os.fspath(Paths.data("fashionMnist"))

	if not os.path.exists(dataDir):
		print(f"Fashion-MNIST not found at: {dataDir}")
		print("Run: python3 tools/data/manage.py fetch fashionMnist")
		sys.exit(1)

	print()
	print("╔══════════════════════════════════════════════════════════════════╗")
	print("║  OA Python Tutorial — Fashion-MNIST Classification (Autograd)   ║")
	print("╚══════════════════════════════════════════════════════════════════╝")
	print()

	# Enable autograd globally BEFORE initializing engine
	FnAutograd.setEnabled(True)

	if not initComputeEngine():
		print("Failed to initialize OA compute engine")
		sys.exit(1)

	# Load dataset
	trainLoader = MnistDataLoader(dataDir, "train", BATCH_SIZE, shuffle=True)
	testLoader = MnistDataLoader(dataDir, "test", 100, shuffle=False)

	print(f"Dataset: {trainLoader.numImages} train / {testLoader.numImages} test, "
		  f"{trainLoader.rows}×{trainLoader.cols} grayscale, {NUM_CLASSES} classes\n")

	# Model + optimizer
	model = MnistClassifier()
	params = model.parameters()
	adam = AdamW(params, LR)

	nParams = sum(p.data.numElements() for p in params)
	print(f"Model: {INPUT_DIM} → Linear({HIDDEN_DIM}) + ReLU → Linear({NUM_CLASSES})")
	print(f"Params: {nParams}    Optimizer: AdamW(lr={LR})    Loss: CrossEntropy\n")

	# Training loop
	stepsPerEpoch = trainLoader.numImages // BATCH_SIZE
	totalSteps = EPOCHS * stepsPerEpoch

	progressBar = CbProgressBar()
	summary = CbSummary()
	lossMetric = MetricLoss()

	progressBar.addMetric(lossMetric)

	config = ItTrainingConfig()
	config.totalSteps = totalSteps
	config.stepsPerEpoch = stepsPerEpoch
	config.batchSize = BATCH_SIZE

	loop = ItTraining(adam, config)
	loop.addMetric(lossMetric)
	loop.addCallback(progressBar)
	loop.addCallback(summary)

	print(f"Training: {EPOCHS} epochs × {stepsPerEpoch} steps/epoch · batch={BATCH_SIZE}")

	initialLoss = 0.0

	while not loop.isDone():
		x, y = trainLoader.nextBatch()
		if x is None:
			trainLoader.reset()
			x, y = trainLoader.nextBatch()

		adam.zeroGrad()  # implicit-autograd accumulates, so clear each step
		tape = GradientTape()
		logits = model.forward(x)
		loss = FnLoss.crossEntropy(logits, y)
		tape.backward(loss)
		loop.next(loss)

		if loop.index() == 1:
			initialLoss = loop.lastLoss()

	loop.finish()
	lastLoss = loop.lastLoss()

	# Evaluation
	testAcc = evalAccuracy(model, testLoader)
	print(f"Test accuracy: {testAcc:.2f}% (over {testLoader.numImages} samples)\n")

	# Show predictions on first 10 test samples
	print("Predictions on the first 10 test samples:")
	print("  # | Actual              | Predicted           | Conf   ")
	print("  ──┼─────────────────────┼─────────────────────┼────────")

	testLoader.reset()
	x10, y10 = testLoader.nextBatch()
	preds = predict(model, x10)
	labels = FnMatrix.copyToHost(y10)

	for i in range(min(10, len(preds))):
		actual = labels[i]
		predClass, confidence = preds[i]
		check = "✓" if actual == predClass else "✗"
		print(f"  {i} | {CLASSES[actual]:<19} | {CLASSES[predClass]:<19} | {confidence:5.1f}% {check}")

	print()

	# Assertions
	assert initialLoss > 0.0, "Initial loss must be positive"
	assert lastLoss < initialLoss, f"Loss must decrease: {lastLoss} >= {initialLoss}"
	assert testAcc > 70.0, f"Test accuracy should exceed 70%, got {testAcc:.2f}%"

	print(f"✓ Training converged successfully")
	print(f"  Initial loss: {initialLoss:.4f}")
	print(f"  Final loss: {lastLoss:.4f}")
	print(f"  Test accuracy: {testAcc:.2f}%")

if __name__ == "__main__":
	main()
