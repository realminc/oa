// Surface vertex shader — Tier 3 PBR material.
//
// Forwards world-space position, world-space normal, UV (transformed), and
// vertex color to the fragment stage. MVP and model matrices occupy the
// vertex push block; normals use the model matrix's upper 3x3.
//
// Push layout (128 bytes):
//   offset   0 : row_major float4x4  model_view_projection    (64 bytes)
//   offset  64 : row_major float4x4  model_matrix             (64 bytes)
//
// Normal transform is exact for uniform scale. Inverse-transpose handling
// for non-uniform scale remains Planned.
//
// Provenance: OA Rust baseline blob e25cc48d50b9f10b6c802a2d0d66ce26c905e12b.
// Adaptation: symbol naming and comments only; shader algorithm is preserved.
// The previously claimed C++ donor is unavailable; see oaShading.md provenance.

#include "material_common.slang"

struct SurfacePush {
	row_major float4x4 model_view_projection;
	row_major float4x4 model_matrix;
};

[[vk::push_constant]] SurfacePush pc;

[shader("vertex")]
SurfaceVertexOutput main(MeshVertexInput input) {
	SurfaceVertexOutput output;
	output.sv_position    = mul(float4(input.position, 1.0), pc.model_view_projection);

	// Transform position to world space for lighting.
	float4 world_pos4     = mul(float4(input.position, 1.0), pc.model_matrix);
	output.world_position = world_pos4.xyz / world_pos4.w;

	// Transform normal with the upper-3×3 of the model matrix, then
	// re-normalize in the fragment stage.  Correct for uniform scale.
	float3x3 m3 = (float3x3)pc.model_matrix;
	output.world_normal = mul(input.normal, m3);

	output.uv    = input.uv;
	output.color = input.color;
	return output;
}
