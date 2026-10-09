// Punctual light types and evaluation loop — shared by Standard Surface.
//
// Supports up to MAX_DIRECTIONAL_LIGHTS directional lights and
// MAX_POINT_LIGHTS point/spot lights stored in a uniform buffer bound at
// descriptor set 0, binding 0.
//
// Donor: oacpp/source/cpp/lib/oa/render/shader/graphics/lights.slang
// Adaptation: spot angle support stubbed (cone_angle = 0 → omnidirectional);
// shadow maps not yet connected (always unshadowed).

#ifndef OA_LIGHTS_SLANG
#define OA_LIGHTS_SLANG

static const int MAX_DIRECTIONAL_LIGHTS = 8;
static const int MAX_POINT_LIGHTS       = 4;

// ── Light structures (std140 layout) ─────────────────────────────────────────

struct DirectionalLight {
	float3 direction;     // world-space, unit vector pointing FROM surface TOWARD source
	float  intensity;     // linear, lux-equivalent
	float3 color;         // linear RGB
	float  _pad;
};

struct PointLight {
	float3 position;      // world-space
	float  intensity;     // linear, lumens
	float3 color;         // linear RGB
	float  radius;        // smooth falloff window; 0 = no falloff
};

struct SceneLights {
	DirectionalLight directional[MAX_DIRECTIONAL_LIGHTS];
	PointLight       point[MAX_POINT_LIGHTS];
	int              directional_count;
	int              point_count;
	float            exposure;       // linear exposure multiplier (1.0 = neutral)
	float            ambient_scale;  // flat ambient intensity when IBL is off
	float3           ambient_color;  // flat ambient color
	int              use_ibl;        // 1 = use IBL, 0 = use flat ambient
};

// ── Distance attenuation ──────────────────────────────────────────────────────

// Smooth window falloff matching Unreal / Filament:
//   atten = (1 - (d/r)^4)^2 / (d^2 + 1)
// Falls to 0 at d == r; 1 at d == 0.
float point_attenuation(float dist_sq, float radius) {
	if (radius <= 0.0) { return 1.0 / (dist_sq + 1.0); }
	float r2 = radius * radius;
	float t  = dist_sq / (r2 * r2);       // (d/r)^4
	float w  = saturate(1.0 - t);
	return (w * w) / (dist_sq + 1.0);
}

// ── Directional light contribution ───────────────────────────────────────────

float3 evaluate_directional(
	DirectionalLight light,
	float3           normal,           // world normal (unit)
	float3           view_dir,         // unit vector from surface toward camera
	float3           base_color,
	float3           f0,
	float            alpha,
	float            metalness
) {
	float3 l = normalize(light.direction);
	float  n_dot_l = saturate(dot(normal, l));
	if (n_dot_l <= 0.0) return float3(0.0, 0.0, 0.0);

	float3 h       = normalize(view_dir + l);
	float  n_dot_v = saturate(dot(normal, view_dir));
	float  n_dot_h = saturate(dot(normal, h));
	float  v_dot_h = saturate(dot(view_dir, h));

	// brdf_direct already multiplies by n_dot_l
	float3 brdf = brdf_direct(base_color, f0, n_dot_v, n_dot_l, n_dot_h, v_dot_h, alpha, metalness);
	return brdf * light.color * light.intensity;
}

// ── Point light contribution ──────────────────────────────────────────────────

float3 evaluate_point(
	PointLight light,
	float3     surface_pos,
	float3     normal,
	float3     view_dir,
	float3     base_color,
	float3     f0,
	float      alpha,
	float      metalness
) {
	float3 delta  = light.position - surface_pos;
	float  dist_sq = dot(delta, delta);
	float3 l       = delta / sqrt(max(dist_sq, 1e-8));
	float  n_dot_l = saturate(dot(normal, l));
	if (n_dot_l <= 0.0) return float3(0.0, 0.0, 0.0);

	float  atten   = point_attenuation(dist_sq, light.radius);
	float3 h       = normalize(view_dir + l);
	float  n_dot_v = saturate(dot(normal, view_dir));
	float  n_dot_h = saturate(dot(normal, h));
	float  v_dot_h = saturate(dot(view_dir, h));

	float3 brdf = brdf_direct(base_color, f0, n_dot_v, n_dot_l, n_dot_h, v_dot_h, alpha, metalness);
	return brdf * light.color * light.intensity * atten;
}

// ── Full scene light accumulation loop ───────────────────────────────────────

float3 evaluate_all_lights(
	SceneLights lights,
	float3      surface_pos,
	float3      normal,
	float3      view_dir,
	float3      base_color,
	float3      f0,
	float       alpha,
	float       metalness
) {
	float3 radiance = float3(0.0, 0.0, 0.0);

	int num_dir = clamp(lights.directional_count, 0, MAX_DIRECTIONAL_LIGHTS);
	for (int i = 0; i < num_dir; ++i)	{
		radiance += evaluate_directional(
			lights.directional[i], normal, view_dir,
			base_color, f0, alpha, metalness
		);
	}

	int num_pt = clamp(lights.point_count, 0, MAX_POINT_LIGHTS);
	for (int j = 0; j < num_pt; ++j) {
		radiance += evaluate_point(
			lights.point[j], surface_pos, normal, view_dir,
			base_color, f0, alpha, metalness
		);
	}

	return radiance * lights.exposure;
}

#endif // OA_LIGHTS_SLANG
