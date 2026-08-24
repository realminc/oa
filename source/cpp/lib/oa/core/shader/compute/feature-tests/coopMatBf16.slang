using namespace linalg;

typealias TestA = CoopMat<BFloat16, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias TestB = CoopMat<BFloat16, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias TestC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

[shader("compute")]
[numthreads(32, 1, 1)]
void main() {
	TestA a;
	TestB b;
	TestC c;
	c.fill(0.0f);
	c = coopMatMulAdd<float, false>(a, b, c);
}
