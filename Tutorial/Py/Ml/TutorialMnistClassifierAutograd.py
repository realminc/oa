#!/usr/bin/env python3
"""
OA Python Tutorial — Fashion-MNIST Image Classification (Implicit Autograd)

Python port of Tutorial/Ml/TutorialMnistClassifierAg.cpp.

Model: Linear(784→128, ReLU) → Linear(128→10)
Loss: CrossEntropy
Optimizer: AdamW
Backward: Implicit autograd via OaGradientTape

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

def read_idx_images(filepath):
	"""Read IDX image file format (Fashion-MNIST)."""
	with open(filepath, 'rb') as f:
		magic = struct.unpack('>I', f.read(4))[0]
		if magic != 2051:
			raise ValueError(f"Invalid magic number {magic} in {filepath}")
		num_images = struct.unpack('>I', f.read(4))[0]
		rows = struct.unpack('>I', f.read(4))[0]
		cols = struct.unpack('>I', f.read(4))[0]
		data = array.array('B', f.read())
		return data, num_images, rows, cols

def read_idx_labels(filepath):
	"""Read IDX label file format (Fashion-MNIST)."""
	with open(filepath, 'rb') as f:
		magic = struct.unpack('>I', f.read(4))[0]
		if magic != 2049:
			raise ValueError(f"Invalid magic number {magic} in {filepath}")
		num_labels = struct.unpack('>I', f.read(4))[0]
		data = array.array('B', f.read())
		return data, num_labels

class MnistDataLoader:
	"""Simple MNIST data loader with batching."""
	def __init__(self, data_dir, split, batch_size, shuffle=False):
		self.batch_size = batch_size
		self.shuffle = shuffle

		if split == "train":
			img_file = os.path.join(data_dir, "train-images-idx3-ubyte")
			lbl_file = os.path.join(data_dir, "train-labels-idx1-ubyte")
		else:  # test
			img_file = os.path.join(data_dir, "t10k-images-idx3-ubyte")
			lbl_file = os.path.join(data_dir, "t10k-labels-idx1-ubyte")

		self.images, self.num_images, self.rows, self.cols = read_idx_images(img_file)
		self.labels, self.num_labels = read_idx_labels(lbl_file)

		if self.num_images != self.num_labels:
			raise ValueError(f"Image count {self.num_images} != label count {self.num_labels}")

		self.indices = list(range(self.num_images))
		self.cursor = 0

		if self.shuffle:
			import random
			random.shuffle(self.indices)

	def next_batch(self):
		"""Get next batch of images and labels."""
		if self.cursor >= self.num_images:
			return None, None

		end = min(self.cursor + self.batch_size, self.num_images)
		actual_batch = end - self.cursor

		# Gather batch. Images must be uploaded as Float32 — Scale/matmul on a
		# UInt8 matrix silently produces garbage (only gather/CrossEntropy read
		# integer tensors correctly), so pixels go up as floats via FromFloats and
		# the model's Scale(1/255) normalizes them. Labels stay bytes: CrossEntropy
		# reads integer class indices directly.
		batch_images = array.array('f')
		batch_labels = array.array('B')

		for i in range(self.cursor, end):
			idx = self.indices[i]
			start = idx * self.rows * self.cols
			end_img = start + self.rows * self.cols
			batch_images.extend(float(p) for p in self.images[start:end_img])
			batch_labels.append(self.labels[idx])

		self.cursor = end

		# Convert to OaMatrix
		x = OaFnMatrix.FromFloats(batch_images, actual_batch, INPUT_DIM)
		y = OaFnMatrix.FromBytes(batch_labels, actual_batch, OaScalarType.UInt8)

		return x, y

	def reset(self):
		"""Reset cursor to beginning."""
		self.cursor = 0
		if self.shuffle:
			import random
			random.shuffle(self.indices)

# ─── Model: Linear(784→128, ReLU) → Linear(128→10) ──────────────────────────

class OaMnistClassifier:
	def __init__(self):
		wd = OaFnMatrix.GetWeightDtype()

		# Hidden layer (Linear + ReLU)
		self.fc1 = OaLinear(INPUT_DIM, HIDDEN_DIM)
		self.fc1.SetActivation(OaActivation.Relu)
		self._fc1_p = self.fc1.Parameters()
		self._fc1_p[0].Data = OaFnMatrix.RandKaimingUniform(HIDDEN_DIM, INPUT_DIM, wd)
		self._fc1_p[0].Data.SetRequiresGrad(True)
		self._fc1_p[1].Data.SetRequiresGrad(True)

		# Output layer
		self.fc2 = OaLinear(HIDDEN_DIM, NUM_CLASSES)
		self._fc2_p = self.fc2.Parameters()
		self._fc2_p[0].Data = OaFnMatrix.RandGlorotUniform(NUM_CLASSES, HIDDEN_DIM, wd)
		self._fc2_p[0].Data.SetRequiresGrad(True)
		self._fc2_p[1].Data.SetRequiresGrad(True)

	def parameters(self):
		return self._fc1_p + self._fc2_p

	def forward(self, x):
		"""Forward pass — returns logits [B, 10]."""
		x_norm = OaFnMatrix.Scale(x, 1.0 / 255.0)  # Normalize [0,255] → [0,1]
		h = self.fc1.Forward(x_norm)         # [B, 128] (LinearRelu fused)
		return self.fc2.Forward(h)           # [B, 10]

# ─── Inference helpers ───────────────────────────────────────────────────────

def argmax_row(probs, row, num_classes):
	"""Host-side argmax for a single row."""
	best = 0
	best_val = probs[row * num_classes]
	for i in range(1, num_classes):
		v = probs[row * num_classes + i]
		if v > best_val:
			best_val = v
			best = i
	return best, best_val

def predict(model, x):
	"""Get predictions for a batch."""
	logits = model.forward(x)
	probs = OaFnMatrix.Softmax(logits, -1)
	ctx = OaContextGetDefault()
	ctx.Execute()
	ctx.Sync()

	batch = int(x.Size(0))
	flat = OaFnMatrix.CopyToHost(probs)

	predictions = []
	for i in range(batch):
		class_idx, confidence = argmax_row(flat, i, NUM_CLASSES)
		predictions.append((class_idx, confidence * 100.0))

	return predictions

def eval_accuracy(model, loader):
	"""Evaluate accuracy on full dataset."""
	correct = 0
	total = 0

	loader.reset()
	while True:
		x, y = loader.next_batch()
		if x is None:
			break

		preds = predict(model, x)
		labels = OaFnMatrix.CopyToHost(y)

		batch = len(preds)
		for i in range(batch):
			if preds[i][0] == labels[i]:
				correct += 1
		total += batch

	loader.reset()
	return 100.0 * correct / total if total > 0 else 0.0

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
	# Get dataset path
	data_dir = os.getenv("OA_MNIST_DATA")
	if not data_dir:
		data_dir = os.path.join(os.getcwd(), "Data", "FashionMNIST", "raw")

	if not os.path.exists(data_dir):
		print(f"Fashion-MNIST not found at: {data_dir}")
		print("Set OA_MNIST_DATA environment variable to dataset location")
		sys.exit(1)

	print()
	print("╔══════════════════════════════════════════════════════════════════╗")
	print("║  OA Python Tutorial — Fashion-MNIST Classification (Autograd)   ║")
	print("╚══════════════════════════════════════════════════════════════════╝")
	print()

	# Enable autograd globally BEFORE initializing engine
	OaFnAutograd.SetEnabled(True)

	if not OaInitComputeEngine():
		print("Failed to initialize OA compute engine")
		sys.exit(1)

	# Load dataset
	train_loader = MnistDataLoader(data_dir, "train", BATCH_SIZE, shuffle=True)
	test_loader = MnistDataLoader(data_dir, "test", 100, shuffle=False)

	print(f"Dataset: {train_loader.num_images} train / {test_loader.num_images} test, "
		  f"{train_loader.rows}×{train_loader.cols} grayscale, {NUM_CLASSES} classes\n")

	# Model + optimizer
	model = OaMnistClassifier()
	params = model.parameters()
	adam = OaAdamW(params, LR)

	n_params = sum(p.Data.NumElements() for p in params)
	print(f"Model: {INPUT_DIM} → Linear({HIDDEN_DIM}) + ReLU → Linear({NUM_CLASSES})")
	print(f"Params: {n_params}    Optimizer: AdamW(lr={LR})    Loss: CrossEntropy\n")

	# Training loop
	steps_per_epoch = train_loader.num_images // BATCH_SIZE
	total_steps = EPOCHS * steps_per_epoch

	progress_bar = OaCbProgressBar()
	summary = OaCbSummary()
	loss_metric = OaMetricLoss()

	progress_bar.AddMetric(loss_metric)

	config = OaItTrainingConfig()
	config.TotalSteps = total_steps
	config.StepsPerEpoch = steps_per_epoch
	config.BatchSize = BATCH_SIZE

	loop = OaItTraining(adam, config)
	loop.AddMetric(loss_metric)
	loop.AddCallback(progress_bar)
	loop.AddCallback(summary)

	print(f"Training: {EPOCHS} epochs × {steps_per_epoch} steps/epoch · batch={BATCH_SIZE}")

	initial_loss = 0.0

	while not loop.IsDone():
		x, y = train_loader.next_batch()
		if x is None:
			train_loader.reset()
			x, y = train_loader.next_batch()

		adam.ZeroGrad()  # implicit-autograd accumulates, so clear each step
		tape = OaGradientTape()
		logits = model.forward(x)
		loss = OaFnLoss.CrossEntropy(logits, y)
		tape.Backward(loss)
		loop.Next(loss)

		if loop.Index() == 1:
			initial_loss = loop.LiveLoss()

	loop.Finish()
	last_loss = loop.LastLoss()

	# Evaluation
	test_acc = eval_accuracy(model, test_loader)
	print(f"Test accuracy: {test_acc:.2f}% (over {test_loader.num_images} samples)\n")

	# Show predictions on first 10 test samples
	print("Predictions on the first 10 test samples:")
	print("  # | Actual              | Predicted           | Conf   ")
	print("  ──┼─────────────────────┼─────────────────────┼────────")

	test_loader.reset()
	x10, y10 = test_loader.next_batch()
	preds = predict(model, x10)
	labels = OaFnMatrix.CopyToHost(y10)

	for i in range(min(10, len(preds))):
		actual = labels[i]
		pred_class, confidence = preds[i]
		check = "✓" if actual == pred_class else "✗"
		print(f"  {i} | {CLASSES[actual]:<19} | {CLASSES[pred_class]:<19} | {confidence:5.1f}% {check}")

	print()

	# Assertions
	assert initial_loss > 0.0, "Initial loss must be positive"
	assert last_loss < initial_loss, f"Loss must decrease: {last_loss} >= {initial_loss}"
	assert test_acc > 70.0, f"Test accuracy should exceed 70%, got {test_acc:.2f}%"

	print(f"✓ Training converged successfully")
	print(f"  Initial loss: {initial_loss:.4f}")
	print(f"  Final loss: {last_loss:.4f}")
	print(f"  Test accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
	main()
