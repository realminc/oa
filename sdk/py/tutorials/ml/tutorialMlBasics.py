#!/usr/bin/env python3
"""Fit a one-layer regression model with OA autograd and Sgd.
"""

# pyright: reportWildcardImportFromLibrary=false
from oa import *


x = FnMatrix.fromFloats([-2.0, -1.0, 0.0, 1.0, 2.0], [5, 1])
target = FnMatrix.fromFloats([-3.0, -1.0, 1.0, 3.0, 5.0], [5, 1])

FnAutograd.setEnabled(True)
model = Linear(1, 1)
optimizer = Sgd(model.parameters(), lr=0.05)
config = ItTrainingConfig()
config.totalSteps = 80
config.batchSize = 5
training = ItTraining(optimizer, config)

initialLoss = 0.0
while not training.isDone():
	optimizer.zeroGrad()
	tape = GradientTape()
	prediction = model.forward(x)
	loss = FnLoss.mse(prediction, target)
	tape.backward(loss)
	del tape
	training.next(loss)
	if training.index() == 1:
		initialLoss = training.lastLoss()

training.finish()
prediction = model.forward(x)
values = FnMatrix.copyToHost(prediction)

assert training.lastLoss() < initialLoss * 0.01
assert max(abs(a - b) for a, b in zip(values, [-3, -1, 1, 3, 5])) < 0.1
print(f"loss: {initialLoss:.6f} -> {training.lastLoss():.6f}")
print(values)
