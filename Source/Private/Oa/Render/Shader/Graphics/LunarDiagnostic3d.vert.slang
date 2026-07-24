// Tutorial-local Lunar Lander diagnostic vertex path. Geometry is already in
// world space; the only transform here is the canonical row-vector camera VP.

struct VertexInput {
	float3 position : POSITION;
	float3 normal   : NORMAL;
	float4 color    : COLOR0;
};

struct VertexOutput {
	float4 sv_position : SV_Position;
	float4 color       : COLOR0;
};

struct PushConstants {
	// Explicit qualifier preserves the Vlm row-vector push-constant ABI through
	// Slang's SPIR-V matrix-layout lowering.
	row_major float4x4 view_projection;
	// XYZ points from the surface toward the light; W is ambient intensity.
	float4 light_direction_ambient;
};

[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
VertexOutput main(VertexInput input) {
	VertexOutput output;
	output.sv_position = mul(float4(input.position, 1.0), pc.view_projection);
	float3 normal = normalize(input.normal);
	float3 toLight = normalize(pc.light_direction_ambient.xyz);
	float ambient = saturate(pc.light_direction_ambient.w);
	float diffuse = max(dot(normal, toLight), 0.0);
	float lighting = ambient + (1.0 - ambient) * diffuse;
	output.color = float4(input.color.rgb * lighting, input.color.a);
	return output;
}
