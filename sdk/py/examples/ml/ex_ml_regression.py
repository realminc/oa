# OA_DOC_BEGIN: ml-regression
import oa

engine = oa.Engine()

x = oa.Matrix.from_f32(engine, [5, 1], [-2.0, -1.0, 0.0, 1.0, 2.0])
target = oa.Matrix.from_f32(engine, [5, 1], [-3.0, -1.0, 1.0, 3.0, 5.0])

model = oa.ml.Linear(engine, 1, 1, seed=0)
optimizer = oa.ml.Sgd(model.parameters(), learning_rate=0.05)

initial_loss = 0.0
for step in range(80):
	optimizer.zero_grad()
	with oa.ml.GradientTape() as tape:
		prediction = model.forward(x)
		loss = oa.ml.loss.mse(prediction, target)
		tape.backward(loss)
	optimizer.step()
	if step == 0:
		initial_loss = oa.ml.metric.scalar_loss(loss)

final_loss = oa.ml.metric.scalar_loss(loss)
values = model.forward(x).read_f32()

assert final_loss < initial_loss * 0.01
assert max(abs(a - b) for a, b in zip(values, [-3, -1, 1, 3, 5])) < 0.1

print(f"loss: {initial_loss:.6f} -> {final_loss:.6f}")
print(values)
# OA_DOC_END: ml-regression
