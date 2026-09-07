// Matrix arithmetic operations

#include "../common/types.slang"
#include "../common/attributes.slang"

[kernel_name("add")]
[domain("matrix")]
[variant("generic")]
[dtype("fp32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void matrixAddGeneric(
	float4[] a,
	float4[] b,
	float4[] out,
	uint3 dispatchThreadID : SV_DispatchThreadID
) {
	uint index = dispatchThreadID.x;
	out[index] = a[index] + b[index];
}

[kernel_name("mul")]
[domain("matrix")]
[variant("generic")]
[dtype("fp32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void matrixMulGeneric(
	float4[] a,
	float4[] b,
	float4[] out,
	uint3 dispatchThreadID : SV_DispatchThreadID
) {
	uint index = dispatchThreadID.x;
	out[index] = a[index] * b[index];
}
