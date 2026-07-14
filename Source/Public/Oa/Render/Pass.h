// OaPass — Render pass and subgraph templates.
//
// Represents a render pass as a subgraph template over dynamic rendering.
// Passes can be composed like ML layers using the >> operator.
//
// Design (from UnifiedExecutionArchitecture.md §7):
// - Named subgraph template
// - Record() method to record draw calls into OaContext
// - Composition via >> operator (pipe)
// - OaPipeline to compose multiple passes
//
// Example passes:
// - OaShadowPass(scene, light) → OaTexture (depth)
// - OaGBufferPass(scene, camera) → OaTexture×4
// - OaLightingPass(gbuffer, lights, shadowMaps) → OaTexture (HDR)
// - OaPostPass(hdr) → OaTexture (LDR)
// - OaBlitToSwapchain(ldr, swapchain)
//
// Usage:
//   OaShadowPass shadow{lights};
//   OaGBufferPass gbuf;
//   OaLightingPass light;
//   OaPostPass post;
//   OaPresentPass present;
//
//   OaPipeline forward;
//   forward.Add(shadow); forward.Add(gbuf); forward.Add(light);
//   forward.Add(post); forward.Add(present);
//
//   forward.Run(ctx, sceneView);

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Render/Scene.h>
#include <Oa/Render/Camera.h>

// Forward declarations
class OaContext;
struct OaTexture;
struct OaSwapchain;

// Pass inputs
struct OaPassInputs {
	const OaScene*  Scene  = nullptr;
	const OaCamera* Camera = nullptr;
	OaTexture*      PreviousFrame = nullptr;  // For TAA, motion vectors, etc.
	// Additional inputs can be added per pass type
};

// Pass outputs
struct OaPassOutputs {
	OaTexture* ColorOutput = nullptr;
	OaTexture* DepthOutput = nullptr;
	// Additional outputs can be added per pass type (e.g., GBuffer textures)
};

// Base render pass
class OaPass {
public:
	virtual ~OaPass() = default;

	// Record the pass into the context
	virtual void Record(
		OaContext& ctx,
		const OaPassInputs& InInputs,
		OaPassOutputs& InOutputs) = 0;

	// Composition (pipe operator)
	OaPass& operator>>(OaPass& InNext);

	// Pass name (for debugging)
	void SetName(const OaString& InName);
	[[nodiscard]] const OaString& GetName() const noexcept;

protected:
	OaString Name_;
};

// Render pipeline (composes multiple passes)
class OaPipeline {
public:
	OaPipeline() = default;
	~OaPipeline() = default;

	void Add(OaPass& InPass);
	void Clear();

	// Run the entire pipeline
	void Run(
		OaContext& ctx,
		const OaPassInputs& InInputs);

	[[nodiscard]] OaU32 GetPassCount() const noexcept;

private:
	OaVec<OaPass*> Passes_;
};

// Example: Shadow pass
class OaShadowPass : public OaPass {
public:
	explicit OaShadowPass(const OaVec<OaDirectionalLight>& InLights);

	void Record(
		OaContext& ctx,
		const OaPassInputs& InInputs,
		OaPassOutputs& InOutputs) override;

private:
	OaVec<OaDirectionalLight> Lights_;
};

// Example: GBuffer pass
class OaGBufferPass : public OaPass {
public:
	void Record(
		OaContext& ctx,
		const OaPassInputs& InInputs,
		OaPassOutputs& InOutputs) override;
};

// Example: Lighting pass
class OaLightingPass : public OaPass {
public:
	void Record(
		OaContext& ctx,
		const OaPassInputs& InInputs,
		OaPassOutputs& InOutputs) override;
};

// Example: Post-processing pass
class OaPostPass : public OaPass {
public:
	void Record(
		OaContext& ctx,
		const OaPassInputs& InInputs,
		OaPassOutputs& InOutputs) override;
};

// Example: Present pass (wraps Acquire + Blit + Present)
class OaPresentPass : public OaPass {
public:
	explicit OaPresentPass(OaSwapchain& InSwapchain);

	void Record(
		OaContext& ctx,
		const OaPassInputs& InInputs,
		OaPassOutputs& InOutputs) override;

private:
	OaSwapchain* Swapchain_;
};
