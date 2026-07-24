// UnlitTextured.vert.slang — Minimal vertex shader for textured quad rendering.
//
// Inputs:  position (vec3), uv (vec2)
// Output:  clip-space position + UV to fragment stage
// Push:    MVP 4x4 matrix

struct VertexInput {
	float3 position : POSITION;
	float2 uv       : TEXCOORD0;
};

struct VertexOutput {
	float4 sv_position : SV_Position;
	float2 uv          : TEXCOORD0;
};

struct PushConstants {
	row_major float4x4 mvp;
	float4 tint;
};

[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
VertexOutput main(VertexInput input) {
	VertexOutput output;
	output.sv_position = mul(float4(input.position, 1.0), pc.mvp);
	output.uv = input.uv;
	return output;
}
