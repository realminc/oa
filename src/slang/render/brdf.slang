// GGX microfacet BRDF functions — shared by all Standard Surface shader
// variants.
//
// Follows the Filament / OpenPBR formulation for numerical stability and exact
// correspondence with the Arnold Standard Surface specification.  All functions
// operate in world space with normalised input vectors.
//
// Reference:
//   - Filament PBR white-paper (Google, 2018)
//   - Autodesk Standard Surface specification (2020)
//   - "Real Shading in Unreal Engine 4" (Epic Games, SIGGRAPH 2013)
//
// Donor: oacpp/source/cpp/lib/oa/render/shader/graphics/brdf.slang
// Adaptation: IBL split-sum left as a stub returning 0 until the prefiltered
// cube and BRDF LUT are available; all analytic terms are exact donor ports.

#ifndef OA_BRDF_SLANG
#define OA_BRDF_SLANG

// ── Constants ─────────────────────────────────────────────────────────────────

static const float PI = 3.14159265358979323846;
static const float INV_PI = 0.31830988618379067154;
static const float EPSILON = 1e-5;

// ── IOR → F0 conversion ───────────────────────────────────────────────────────

// Schlick F0 from index of refraction.
float3 f0_from_ior(float ior) {
	float f = (ior - 1.0) / (ior + 1.0);
	return float3(f * f, f * f, f * f);
}

// Per-channel metalness blend: dielectrics use IOR-derived F0; metals use
// base color as F0 directly.
float3 f0_blend(float3 base_color, float3 f0_dielectric, float metalness) {
	return lerp(f0_dielectric, base_color, metalness);
}

// ── Schlick Fresnel ───────────────────────────────────────────────────────────

// F(v,h) = F0 + (1 - F0) * (1 - v·h)^5
float3 fresnel_schlick(float3 f0, float v_dot_h) {
	float t = 1.0 - saturate(v_dot_h);
	float t2 = t * t;
	float t5 = t2 * t2 * t;
	return f0 + (1.0 - f0) * t5;
}

// ── GGX NDF (Trowbridge-Reitz) ────────────────────────────────────────────────

// α = roughness²  (perceptual → linear roughness)
// D(h) = α² / (π * ((n·h)² * (α² - 1) + 1)²)
float ggx_ndf(float n_dot_h, float alpha) {
	float a2 = alpha * alpha;
	float d  = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0;
	return a2 / max(PI * d * d, EPSILON);
}

// ── Smith GGX Geometry (height-correlated) ────────────────────────────────────

// G1(x) = 2(n·x) / ((n·x) + sqrt(α² + (1-α²)(n·x)²))
float ggx_g1(float n_dot_x, float alpha) {
	float a2 = alpha * alpha;
	float d  = n_dot_x + sqrt(a2 + (1.0 - a2) * n_dot_x * n_dot_x);
	return (2.0 * n_dot_x) / max(d, EPSILON);
}

// G(v,l) = G1(v) * G1(l)
float ggx_geometry(float n_dot_v, float n_dot_l, float alpha) {
	return ggx_g1(n_dot_v, alpha) * ggx_g1(n_dot_l, alpha);
}

// ── Cook-Torrance specular BRDF ───────────────────────────────────────────────

// f_spec = D * G * F / (4 * n·v * n·l)
// Returns the specular radiance contribution; multiply by n·l * L_i at the
// call site.
float3 ggx_specular(
	float3 f0,
	float  n_dot_v,
	float  n_dot_l,
	float  n_dot_h,
	float  v_dot_h,
	float  alpha
) {
	float  D = ggx_ndf(n_dot_h, alpha);
	float  G = ggx_geometry(n_dot_v, n_dot_l, alpha);
	float3 F = fresnel_schlick(f0, v_dot_h);
	float  denom = max(4.0 * n_dot_v * n_dot_l, EPSILON);
	return (D * G * F) / denom;
}

// ── Energy-conserving Lambertian diffuse ──────────────────────────────────────

// Weighted by (1 - F) * (1 - metalness) so the total reflected energy ≤ 1.
// diffuse = (1 - F) * (1 - metalness) * base_color / π
float3 lambertian_diffuse(float3 base_color, float3 f0, float v_dot_h, float metalness) {
	float3 F   = fresnel_schlick(f0, v_dot_h);
	float3 kd  = (1.0 - F) * (1.0 - metalness);
	return kd * base_color * INV_PI;
}

// ── Combined direct-light BRDF evaluation ─────────────────────────────────────

// Returns direct reflected radiance for one light sample (caller provides
// n·l * L_i * light_color).  Adds specular and diffuse lobes.
float3 brdf_direct(
	float3 base_color,
	float3 f0,
	float  n_dot_v,
	float  n_dot_l,
	float  n_dot_h,
	float  v_dot_h,
	float  alpha,
	float  metalness
) {
	float3 spec = ggx_specular(f0, n_dot_v, n_dot_l, n_dot_h, v_dot_h, alpha);
	float3 diff = lambertian_diffuse(base_color, f0, v_dot_h, metalness);
	return (spec + diff) * n_dot_l;
}

// ── IBL stub ──────────────────────────────────────────────────────────────────
// Returns zero until the prefiltered cube and BRDF LUT are bound.
// Replaced in a later pass when IBL resources are available.

float3 ibl_specular(float3 r, float roughness, float3 f0, float n_dot_v) {
	// Planned: sample prefiltered_env_cube + brdf_lut
	return float3(0.0, 0.0, 0.0);
}

float3 ibl_diffuse(float3 n) {
	// Planned: sample irradiance_cube or evaluate SH9
	return float3(0.0, 0.0, 0.0);
}

#endif // OA_BRDF_SLANG
