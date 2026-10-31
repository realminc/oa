// Surface fragment shader — Tier 3 PBR material.
//
// Evaluates OA's metallic/roughness PBR material with:
//   - GGX microfacet specular (NDF + Smith G + Schlick F)
//   - Energy-conserving Lambertian diffuse
//   - Per-material scalar parameters from a uniform buffer (set 1, binding 3)
//   - Optional base-color, ORM, and normal map textures (descriptor set 1)
//   - Scene punctual lights from a uniform buffer (descriptor set 0)
//   - IBL stubs (returns 0 until cube maps are bound)
//
// Texture branches require shader compile-time defines. The current build
// leaves them disabled and evaluates scalar defaults:
//   FEAT_BASE_TEXTURE    — sample base_color_texture and multiply
//   FEAT_ORM_TEXTURE     — sample orm_texture (R=AO, G=rough, B=metal)
//   FEAT_NORMAL_TEXTURE  — sample normal_texture (BC5 RG, reconstruct Z)
//
// Descriptor set 0, binding 0 : ConstantBuffer<SceneLights>     (scene UBO)
// Descriptor set 1, binding 0 : Sampler2D   base_color_texture
// Descriptor set 1, binding 1 : Sampler2D   orm_texture
// Descriptor set 1, binding 2 : Sampler2D   normal_texture
//
// Push layout (128 bytes, matches surface.vert.slang):
//   offset   0 : row_major float4x4  model_view_projection  (64 bytes)
//   offset  64 : row_major float4x4  model_matrix           (64 bytes)
//
// The fragment stage has no push block; material scalars use binding 3.
//
// Provenance: OA Rust baseline blob aa362a019f705c6467909426fff32e5010682b59.
// Adaptation: symbol naming and comments only; shader algorithm is preserved.
// The previously claimed C++ donor is unavailable; see oaShading.md provenance.

#include "material_common.slang"
#include "brdf.slang"
#include "lights.slang"

// ── Material parameters uniform buffer ───────────────────────────────────────
// Descriptor set 1, binding 3.  Holds per-draw material scalars.
struct MaterialParams {
	float3 base_color;        float  metalness;
	float  roughness;         float  normal_scale;
	float  specular_ior;      float  emission_scale;
	float3 specular_tint;     float  opacity;
	float3 emission_color;    float  _pad0;
	float2 uv_scale;          float2 uv_offset;
};

// ── Descriptor bindings ───────────────────────────────────────────────────────

[[vk::binding(0, 0)]] ConstantBuffer<SceneLights>  scene_lights;

[[vk::binding(0, 1)]] Sampler2D  base_color_texture;
[[vk::binding(1, 1)]] Sampler2D  orm_texture;
[[vk::binding(2, 1)]] Sampler2D  normal_texture;
[[vk::binding(3, 1)]] ConstantBuffer<MaterialParams> material;

// ── Fragment entry point ──────────────────────────────────────────────────────

[shader("fragment")]
float4 main(SurfaceVertexOutput input) : SV_Target0 {
	// ── UV transform ─────────────────────────────────────────────────────────
	float2 uv = apply_uv_transform(input.uv, material.uv_scale, material.uv_offset);

	// ── Base color ────────────────────────────────────────────────────────────
	float3 base_color = material.base_color * input.color.rgb;
#ifdef FEAT_BASE_TEXTURE
	float4 base_texel = base_color_texture.Sample(uv);
	base_color *= base_texel.rgb;
	float base_alpha  = base_texel.a;
#else
	float base_alpha  = 1.0;
#endif

	// ── ORM (occlusion / roughness / metalness) ────────────────────────────
	float ao        = 1.0;
	float roughness = material.roughness;
	float metalness = material.metalness;
#ifdef FEAT_ORM_TEXTURE
	float3 orm = orm_texture.Sample(uv).rgb;
	ao         = orm.r;
	roughness  = orm.g;
	metalness  = orm.b;
#endif

	// ── Normal map ────────────────────────────────────────────────────────────
	float3 N = normalize(input.world_normal);
#ifdef FEAT_NORMAL_TEXTURE
	// BC5 SNORM: RG stores XY of tangent-space normal; reconstruct Z.
	float2 rg  = normal_texture.Sample(uv).rg * 2.0 - 1.0;
	float  nz  = sqrt(max(0.0, 1.0 - dot(rg, rg)));
	float3 tbn_normal = float3(rg * material.normal_scale, nz);
	// TODO: rotate into world space via TBN when tangent stream is added.
	// For now tbn_normal overrides N when tangents are unavailable.
	N = normalize(tbn_normal);
#endif

	// ── Derived BRDF inputs ───────────────────────────────────────────────────
	float  alpha    = roughness * roughness; // perceptual → linear roughness
	float3 f0_diel  = f0_from_ior(material.specular_ior) * material.specular_tint;
	float3 f0       = f0_blend(base_color, f0_diel, metalness);
	float3 V        = normalize(-input.world_position); // approximate; exact with camera_pos uniform
	float  n_dot_v  = saturate(dot(N, V));

	// ── Lighting ──────────────────────────────────────────────────────────────
	float3 radiance = evaluate_all_lights(
		scene_lights,
		input.world_position, N, V,
		base_color, f0, alpha, metalness
	);

	// Ambient (flat IBL stub).
	float3 ambient_diffuse  = ibl_diffuse(N);
	float3 ambient_specular = ibl_specular(reflect(-V, N), roughness, f0, n_dot_v);
	float3 ambient = (ambient_diffuse * base_color + ambient_specular)
									+ scene_lights.ambient_color * scene_lights.ambient_scale * base_color;
	radiance += ambient * ao;

	// Emission.
	radiance += material.emission_color * material.emission_scale;

	// ── Output ────────────────────────────────────────────────────────────────
	float alpha_out = base_alpha * material.opacity * input.color.a;
	return float4(radiance, alpha_out);
}
