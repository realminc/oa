// OaScene — Scene graph for 3D rendering.
//
// Represents a 3D scene with:
// - Scene objects (mesh + material + transform)
// - Lights (directional, point, spot, area)
// - Camera
// - Hierarchical transforms (scene graph)
//
// Design:
// - Scene objects with mesh, material, transform
// - Lights for PBR rendering
// - Camera for view
// - Bounding volume hierarchy (BVH) for culling (future)
//
// Usage:
//   OaScene scene;
//   OaSceneObject obj;
//   obj.Mesh = &mesh;
//   obj.Material = &material;
//   obj.Transform = VlmMat4::Identity();
//   scene.AddObject(obj);
//
//   OaDirectionalLight light;
//   light.Direction = {0.0f, -1.0f, 0.0f};
//   light.Intensity = 1.0f;
//   scene.AddLight(light);

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Vlm.h>
#include <Oa/Render/Mesh.h>
#include <Oa/Ui/Camera.h>
#include <Oa/Render/Material.h>

// Light types
enum class OaLightType : OaU8 {
	Directional,  // Directional light (sun)
	Point,        // Point light (omnidirectional)
	Spot,         // Spot light (cone)
	Area,         // Area light (rectangle)
};

// Base light
struct OaLight {
	OaLightType Type = OaLightType::Point;
	VlmVec3      Color = {1.0f, 1.0f, 1.0f};
	OaF32       Intensity = 1.0f;
	bool        CastShadows = false;
};

// Directional light
struct OaDirectionalLight : public OaLight {
	VlmVec3 Direction = {0.0f, -1.0f, 0.0f};
};

// Point light
struct OaPointLight : public OaLight {
	VlmVec3 Position = {0.0f, 0.0f, 0.0f};
	OaF32 Range = 10.0f;  // Light range
};

// Spot light
struct OaSpotLight : public OaPointLight {
	VlmVec3 Direction = {0.0f, -1.0f, 0.0f};
	OaF32 InnerConeAngle = 30.0f;  // Degrees
	OaF32 OuterConeAngle = 45.0f;  // Degrees
};

// Area light
struct OaAreaLight : public OaLight {
	VlmVec3 Position = {0.0f, 0.0f, 0.0f};
	VlmVec3 Direction = {0.0f, -1.0f, 0.0f};
	VlmVec2 Size = {1.0f, 1.0f};  // Width, height
};

// Scene object
struct OaSceneObject {
	OaMesh*     Mesh = nullptr;
	OaMaterial* Material = nullptr;
	VlmMat4      Transform = VlmMat4::Identity();
	OaString    Name;
	bool        Visible = true;
	bool        CastShadows = true;
	bool        ReceiveShadows = true;
};

// Scene
class OaScene {
public:
	OaScene() = default;
	~OaScene() = default;

	// Objects
	void AddObject(const OaSceneObject& InObject);
	void RemoveObject(OaU32 InIndex);
	void ClearObjects();

	[[nodiscard]] OaU32 GetObjectCount() const noexcept;
	[[nodiscard]] const OaSceneObject& GetObject(OaU32 InIndex) const;
	[[nodiscard]] OaSceneObject& GetObject(OaU32 InIndex);

	// Lights
	void AddDirectionalLight(const OaDirectionalLight& InLight);
	void AddPointLight(const OaPointLight& InLight);
	void AddSpotLight(const OaSpotLight& InLight);
	void AddAreaLight(const OaAreaLight& InLight);
	void ClearLights();

	[[nodiscard]] OaU32 GetDirectionalLightCount() const noexcept;
	[[nodiscard]] OaU32 GetPointLightCount() const noexcept;
	[[nodiscard]] OaU32 GetSpotLightCount() const noexcept;
	[[nodiscard]] OaU32 GetAreaLightCount() const noexcept;

	[[nodiscard]] const OaDirectionalLight& GetDirectionalLight(OaU32 InIndex) const;
	[[nodiscard]] const OaPointLight& GetPointLight(OaU32 InIndex) const;
	[[nodiscard]] const OaSpotLight& GetSpotLight(OaU32 InIndex) const;
	[[nodiscard]] const OaAreaLight& GetAreaLight(OaU32 InIndex) const;

	// Camera
	void SetCamera(const OaCamera& InCamera);
	[[nodiscard]] const OaCamera& GetCamera() const noexcept;
	[[nodiscard]] OaCamera& GetCamera();

	// Background
	void SetBackgroundColor(const VlmVec4& InColor);
	[[nodiscard]] const VlmVec4& GetBackgroundColor() const noexcept;

private:
	OaVec<OaSceneObject>      Objects_;
	OaVec<OaDirectionalLight> DirectionalLights_;
	OaVec<OaPointLight>        PointLights_;
	OaVec<OaSpotLight>         SpotLights_;
	OaVec<OaAreaLight>         AreaLights_;
	OaCamera                   Camera_;
	VlmVec4                     BackgroundColor_ = {0.1f, 0.1f, 0.1f, 1.0f};
};
