#!/usr/bin/env python3
"""Fit a one-layer regression model with OA autograd and SGD.
"""

# pyright: reportWildcardImportFromLibrary=false
from oa import *


x = OaFnMatrix.FromFloats([-2.0, -1.0, 0.0, 1.0, 2.0], [5, 1])
target = OaFnMatrix.FromFloats([-3.0, -1.0, 1.0, 3.0, 5.0], [5, 1])

OaFnAutograd.SetEnabled(True)
model = OaLinear(1, 1)
optimizer = OaSGD(model.Parameters(), Lr=0.05)
config = OaItTrainingConfig()
config.TotalSteps = 80
config.BatchSize = 5
training = OaItTraining(optimizer, config)

initial_loss = 0.0
while not training.IsDone():
	optimizer.ZeroGrad()
	tape = OaGradientTape()
	prediction = model.Forward(x)
	loss = OaFnLoss.Mse(prediction, target)
	tape.Backward(loss)
	del tape
	training.Next(loss)
	if training.Index() == 1:
		initial_loss = training.LiveLoss()

training.Finish()
prediction = model.Forward(x)
values = OaFnMatrix.CopyToHost(prediction)

assert training.LastLoss() < initial_loss * 0.01
assert max(abs(a - b) for a, b in zip(values, [-3, -1, 1, 3, 5])) < 0.1
print(f"loss: {initial_loss:.6f} -> {training.LastLoss():.6f}")
print(values)
