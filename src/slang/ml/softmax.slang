// ML softmax operation

#include "../common/types.slang"
#include "../common/math.slang"

import attributes;

[kernel_name("softmax")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void mlSoftmaxGeneric(
	float[] input,
	float[] output,
	uint axis,
	uint3 dispatchThreadID : SV_DispatchThreadID
) {
	uint index = dispatchThreadID.x;
	// TODO: Implement softmax computation
	output[index] = input[index];
}

[kernel_name("layer_norm")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void mlLayerNormGeneric(
	float[] input,
	float[] output,
	float[] gamma,
	float[] beta,
	float epsilon,
	uint3 dispatchThreadID : SV_DispatchThreadID
) {
	uint index = dispatchThreadID.x;
	// TODO: Implement layer norm computation
	output[index] = input[index];
}
