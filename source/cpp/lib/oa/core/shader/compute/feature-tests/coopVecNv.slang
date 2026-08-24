typealias TestVec = CoopVec<float, 16>;

[shader("compute")]
[numthreads(32, 1, 1)]
void main() {
	TestVec v;
	(void)v;
}
