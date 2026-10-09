// Flat-color fragment shader — Tier 1 material, UI and debug geometry.
//
// Passes the interpolated color straight to SV_Target0.  No lighting,
// no texture sampling.
//
// Donor: oacpp/source/cpp/lib/oa/render/shader/graphics/flatColor.frag.slang
// Adaptation: none structurally; color carried from vertex stage.

#include "material_common.slang"

[shader("fragment")]
float4 main(SimpleVertexOutput input) : SV_Target0 {
	return input.color;
}
