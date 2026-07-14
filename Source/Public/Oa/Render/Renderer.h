// OaRenderer — Unified rendering system.
//
// Manages GPU resources (mesh buffers, pipeline cache) and records draw calls.
// Wraps OaFnRender functional API and provides 2D canvas rendering.
//
// Usage (3D scene with meshes):
//   OaRenderer renderer(ctx);
//   renderer.LoadShader("UnlitTextured.vert", "UnlitTextured.frag");
//   for (auto& obj : scene.Objects) {
//       renderer.Draw(obj.Mesh, obj.Material, camera, w, h);
//   }
//
// Usage (2D canvas for UI overlays):
//   OaRenderer::CanvasRenderer canvas;
//   canvas.Init(engine, width, height);
//   canvas.DrawImage(imageDraw);
//   canvas.Record(cmdBuf, destImage);

#pragma once

#include <Oa/Core/Vlm.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Render/FnRender.h>
#include <Oa/Render/Mesh.h>
#include <Oa/Render/Material.h>
#include <Oa/Ui/Camera.h>
#include <Oa/Runtime/OaVk.h>

class OaGraphicsEngine;

// ─── 3D Mesh Renderer ─────────────────────────────────────────────────────────

class OaRenderer {
public:
	explicit OaRenderer(OaContext& InCtx);
	~OaRenderer();

	// Load shaders by SPIR-V registry name (must be embedded).
	// Lazily creates pipeline on first draw.
	void LoadShader(const char* InVertName, const char* InFragName);

	// Upload mesh data to GPU. Returns handle for drawing.
	// Idempotent: same mesh data returns same handle.
	[[nodiscard]] OaFnRender::OaGpuMesh UploadMesh(const OaMeshData& InMesh);

	// Release a GPU mesh.
	void ReleaseMesh(OaFnRender::OaGpuMesh& InMesh);

	// Release all GPU meshes.
	void ReleaseAllMeshes();

	// Draw a mesh with material + camera.
	// Calculates MVP = Projection * View * Model, uploads if needed, records draw.
	void Draw(
		const OaMeshData&    InMesh,
		const OaMaterialData& InMaterial,
		const OaCameraState& InCamera,
		OaF32                InViewportW,
		OaF32                InViewportH
	);

	// One-shot: mesh data + material + camera. Uploads mesh if not cached.
	void Draw(
		const OaMesh&        InMesh,
		const OaMaterial&    InMaterial,
		const OaCamera&      InCamera
	);

	// Set the texture bindless index for subsequent draws.
	void SetTextureIdx(OaI32 InIdx) { TextureIdx_ = InIdx; }
	[[nodiscard]] OaI32 GetTextureIdx() const noexcept { return TextureIdx_; }

	// Direct access to underlying pipeline (for advanced use)
	[[nodiscard]] bool HasPipeline() const noexcept { return Pipeline_.Valid; }

private:
	OaContext&              Ctx_;
	OaFnRender::OaGpuPipeline Pipeline_;
	OaI32                   TextureIdx_ = -1;
	const char*             VertShaderName_ = nullptr;
	const char*             FragShaderName_ = nullptr;

	// Simple mesh cache (mesh data pointer → GPU mesh)
	// Production: use hash or UUID
	OaVec<OaFnRender::OaGpuMesh> MeshCache_;
};

// ─── 2D Canvas Renderer (UI overlays) ─────────────────────────────────────────
// Camera-driven textured mesh rasterization for images, rects, glyphs.
// Coordinate convention:
// - world/camera space is right-handed with +Y up;
// - Vulkan clip depth is [0, 1];
// - the renderer uses a negative-height viewport to map +Y to screen up;
// - image UVs are top-origin (V=0 is the source image top).

struct OaCanvasImageDraw {
	VkImageView SourceView = VK_NULL_HANDLE;
	VkImageLayout SourceLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	OaU32 SourceWidth = 0;
	OaU32 SourceHeight = 0;
	OaCamera Camera;
	VlmMat4 Model = VlmMat4::Identity();
	VlmVec4 Tint = {1.0F, 1.0F, 1.0F, 1.0F};
};

struct OaCanvasRectInstanceDraw {
	OaU32 InstanceBufferIndex = UINT32_MAX;
	OaU32 InstanceCount = 0;
	OaCamera Camera;
	VlmMat4 Model = VlmMat4::Identity();
	VlmVec4 Color = {0.188F, 0.820F, 0.345F, 1.0F};
	OaF32 ThicknessPixels = 3.0F;
};

struct OaCanvasGlyphInstanceDraw {
	OaU32 InstanceBufferIndex = UINT32_MAX;
	OaU32 InstanceCount = 0;
	OaU32 AtlasBufferIndex = UINT32_MAX;
	OaU32 AtlasWidth = 0;
	OaU32 AtlasHeight = 0;
	OaU32 ReferenceWidth = 0;
	OaU32 ReferenceHeight = 0;
	OaF32 AtlasPxRange = 4.0F;
	OaCamera Camera;
	VlmMat4 Model = VlmMat4::Identity();
};

class OaCanvasRenderer {
public:
	struct Impl;

	OaCanvasRenderer() = default;
	OaCanvasRenderer(const OaCanvasRenderer&) = delete;
	OaCanvasRenderer& operator=(const OaCanvasRenderer&) = delete;
	OaCanvasRenderer(OaCanvasRenderer&&) noexcept;
	OaCanvasRenderer& operator=(OaCanvasRenderer&&) noexcept;
	~OaCanvasRenderer();

	[[nodiscard]] OaStatus Init(
		OaGraphicsEngine& InEngine,
		OaU32 InTargetWidth,
		OaU32 InTargetHeight);
	void SetTarget(
		OaU32 InTargetWidth,
		OaU32 InTargetHeight);
	void Destroy();

	void BeginFrame();
	void DrawImage(const OaCanvasImageDraw& InDraw);
	void DrawRectInstances(const OaCanvasRectInstanceDraw& InDraw);
	void DrawGlyphInstances(const OaCanvasGlyphInstanceDraw& InDraw);
	[[nodiscard]] bool HasDraws() const noexcept;
	void Record(VkCommandBuffer InCommandBuffer, VkImage InDestinationImage);

private:
	OaUniquePtr<Impl> Impl_;
};
