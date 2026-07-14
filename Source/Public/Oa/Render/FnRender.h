// OaFnRender — Functional rendering primitives.
//
// Stateless bridge between OaMesh/OaMaterial/OaCamera and Vulkan command recording.
// For the OOP wrapper, see <Oa/Render/Renderer.h>.
//
// Design:
// - No pipeline cache here — just raw "bind this, draw that" primitives.
// - Buffer upload is separate (mesh data → GPU vertex/index buffers).
// - Pipeline creation is separate (SPIR-V + vertex format → VkPipeline).
//
// Usage (high-level, via OaRenderer):
//   OaRenderer renderer(ctx);
//   renderer.Draw(mesh, material, camera, viewportW, viewportH);
//
// Usage (low-level, functional):
//   OaGpuMesh gpuMesh = OaFnRender::UploadMesh(ctx, meshData);
//   OaGpuPipeline pipeline = OaFnRender::CreatePipeline(ctx, vsSpv, fsSpv, vertexFmt);
//   OaFnRender::BindPipeline(cmd, pipeline);
//   OaFnRender::BindTexture(cmd, set, textureIdx);
//   OaFnRender::PushMvp(cmd, mvpMatrix);
//   OaFnRender::DrawIndexed(cmd, gpuMesh, indexCount);

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Vlm.h>
#include <Oa/Render/FnMesh.h>
#include <Oa/Render/FnMaterial.h>
#include <Oa/Ui/FnCamera.h>

// Forward declarations
class OaContext;
struct OaTexture;

namespace OaFnRender {

// ─── GPU Mesh handle ─────────────────────────────────────────────────────
// Uploaded vertex/index buffers. Opaque to callers.
struct OaGpuMesh {
	void* VertexBuffer = nullptr;  // VkBuffer (position, normal, uv, color interleaved)
	void* IndexBuffer  = nullptr;  // VkBuffer (uint32 indices)
	OaU32 VertexCount  = 0;
	OaU32 IndexCount   = 0;
	bool  Valid        = false;
};

// ─── GPU Pipeline handle ─────────────────────────────────────────────────
// Compiled graphics pipeline. Opaque to callers.
struct OaGpuPipeline {
	void* Handle    = nullptr;  // VkPipeline
	void* Layout    = nullptr;  // VkPipelineLayout
	bool  Valid     = false;
};

// ─── Upload ────────────────────────────────────────────────────────────────

// Upload mesh data to GPU buffers. Returns handle for drawing.
// Internally creates VkBuffer with DEVICE_LOCAL + staging upload.
[[nodiscard]] OaGpuMesh UploadMesh(OaContext& InCtx, const OaMeshData& InMesh);

// Release GPU mesh buffers.
void DestroyMesh(OaContext& InCtx, OaGpuMesh& InMesh);

// ─── Pipeline ──────────────────────────────────────────────────────────────

// Create a graphics pipeline from vertex/fragment SPIR-V.
// Vertex format is derived from OaMeshVertex layout.
// Pipeline layout uses:
//   set=0, binding=2  → sampled texture array (bindless)
//   set=0, binding=3  → sampler array
//   push constants    → MVP matrix (vert) + texture_idx + tint (frag)
[[nodiscard]] OaGpuPipeline CreatePipeline(
	OaContext& InCtx,
	const OaU8* InVertSpv,
	OaU32       InVertSpvSize,
	const OaU8* InFragSpv,
	OaU32       InFragSpvSize,
	OaBlendMode InBlendMode = OaBlendMode::Opaque
);

// Destroy a graphics pipeline.
void DestroyPipeline(OaContext& InCtx, OaGpuPipeline& InPipeline);

// ─── Drawing ─────────────────────────────────────────────────────────────

// Bind pipeline + descriptor set for a single texture draw.
// textureBindlessIdx is the index into the bindless sampled image array.
void BindTexturePipeline(
	OaContext&         InCtx,
	const OaGpuPipeline& InPipeline,
	OaI32              InTextureBindlessIdx
);

// Push MVP matrix (vertex shader push constant).
void PushMvp(OaContext& InCtx, const VlmMat4& InMvp);

// Push fragment constants (texture_idx + tint).
void PushFragmentConstants(
	OaContext& InCtx,
	OaI32      InTextureIdx,
	const VlmVec4& InTint
);

// Record indexed draw call.
void DrawIndexed(OaContext& InCtx, const OaGpuMesh& InMesh);

// Record non-indexed draw call.
void DrawArrays(OaContext& InCtx, const OaGpuMesh& InMesh);

// ─── High-level convenience ──────────────────────────────────────────────

// One-shot: upload mesh, create pipeline if needed, bind, push constants, draw.
// This is what OaRenderer::Draw() delegates to.
void DrawMesh(
	OaContext&         InCtx,
	const OaMeshData&    InMesh,
	const OaMaterialData& InMaterial,
	const OaCameraState& InCamera,
	OaF32                InViewportW,
	OaF32                InViewportH,
	OaI32                InTextureBindlessIdx
);

} // namespace OaFnRender
