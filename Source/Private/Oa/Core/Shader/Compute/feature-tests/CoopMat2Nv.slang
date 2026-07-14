// Slang-native CoopMat2 probe. This should compile only once Slang exposes
// NV tensor layouts / tensor-addressed cooperative matrix loads.

[shader("compute")]
[numthreads(32, 1, 1)]
void main() {
	tensorLayoutNV<2> layout = createTensorLayoutNV(2);
	layout = setTensorLayoutDimensionNV(layout, 16, 16);
	(void)layout;
}
