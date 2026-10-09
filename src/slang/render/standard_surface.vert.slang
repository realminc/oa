// Standard Surface vertex shader — Tier 3 PBR material.
//
// Forwards world-space position, world-space normal, UV (transformed), and
// vertex color to the fragment stage.  The model matrix is embedded in the
// MVP push constant; the inverse-transpose normal matrix is pre-computed on
// the CPU and passed as separate push fields to avoid per-vertex inversion.
//
// Push layout (128 bytes):
//   offset   0 : row_major float4x4  model_view_projection    (64 bytes)
//   offset  64 : row_major float4x4  model_matrix             (64 bytes)
//
// Normal transform: the fragment shader re-derives the normal from the model
// matrix's upper-3x3 using the pre-normalized input.  For uniform-scale
// geometry this is exact.  A separate normal matrix push field will be added
// when non-uniform scale is common.
//
// Donor: oacpp/source/cpp/lib/oa/render/shader/graphics/standardSurface.vert.slang

#include "material_common.slang"

struct StandardSurfacePush {
	row_major float4x4 model_view_projection;
	row_major float4x4 model_matrix;
};

[[vk::push_constant]] StandardSurfacePush pc;

[shader("vertex")]
StandardVertexOutput main(MeshVertexInput input) {
	StandardVertexOutput output;
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
