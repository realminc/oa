// Vision resize operations

#include "../common/types.slang"
#include "../common/attributes.slang"

[kernel_name("resize")]
[domain("vision")]
[variant("nearest")]
[dtype("fp32")]
[status("experimental")]
[shader("compute")]
[numthreads(8, 8, 1)]
void visionResizeNearest(
	Texture2D<float4> input,
	RWTexture2D<float4> output,
	uint2 dispatchThreadID : SV_DispatchThreadID
) {
	uint2 outputSize;
	output.GetDimensions(outputSize.x, outputSize.y);

	if (dispatchThreadID.x >= outputSize.x || dispatchThreadID.y >= outputSize.y) {
		return;
	}

	float2 uv = float2(dispatchThreadID) / float2(outputSize);
	uint2 inputSize;
	input.GetDimensions(inputSize.x, inputSize.y);
	uint2 inputCoord = uint2(uv * float2(inputSize));
	output[dispatchThreadID] = input[inputCoord];
}
