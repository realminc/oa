# OA_DOC_BEGIN: core-matrix-add
import oa

engine = oa.Engine()

one = oa.matrix.ones(engine, [2, 3])
two = oa.matrix.full(engine, [2, 3], 2.0)
result = oa.matrix.add(one, two)

values = result.read_f32()
assert len(values) == 6
assert all(abs(v - 3.0) <= 1e-6 for v in values)

print("Matrix addition verified: every value is 3")
# OA_DOC_END: core-matrix-add
