// Shared types and helpers used by all OA render material shaders.
//
// Donor: oacpp/source/cpp/lib/oa/render/shader/graphics/
// Adaptation: extracted from per-shader duplicates into a single include;
// UV transform and vertex layout match the MeshVertex CPU-side stride of
// 48 bytes (position=12, normal=12, uv=8, color=16).

#ifndef OA_MATERIAL_COMMON_SLANG
#define OA_MATERIAL_COMMON_SLANG

// ── MeshVertex layout ──────────────────────────────────────────────────────────
// Matches src/rs/render/scene.rs MeshVertex and renderer.rs VERTEX_STRIDE=48.
// location 0 : float3 position  (offset  0)
// location 1 : float3 normal    (offset 12)
// location 2 : float2 uv        (offset 24)
// location 3 : float4 color     (offset 32)

struct MeshVertexInput {
	float3 position : POSITION;
	float3 normal   : NORMAL;
	float2 uv       : TEXCOORD0;
	float4 color    : COLOR0;
};

// ── Shared vertex output ───────────────────────────────────────────────────────

struct StandardVertexOutput {
	float4 sv_position : SV_Position;
	float3 world_position : TEXCOORD0;
	float3 world_normal   : TEXCOORD1;
	float2 uv             : TEXCOORD2;
	float4 color          : COLOR0;
};

struct SimpleVertexOutput {
	float4 sv_position : SV_Position;
	float2 uv          : TEXCOORD0;
	float4 color       : COLOR0;
};

// ── UV transform ───────────────────────────────────────────────────────────────

float2 apply_uv_transform(float2 uv, float2 scale, float2 offset) {
	return uv * scale + offset;
}

// ── Tone / linear helpers ──────────────────────────────────────────────────────

// Clamp each channel to [0,1] without discarding — used for validated output.
float4 saturate4(float4 v) { return saturate(v); }

#endif // OA_MATERIAL_COMMON_SLANG
