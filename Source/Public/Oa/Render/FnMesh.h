// OaFnMesh — Functional mesh operations.
//
// Stateless mesh math and geometry generation. Operates on OaMeshData POD.
// For the OOP wrapper, see <Oa/Render/Mesh.h>.
//
// Usage:
//   OaMeshData mesh = OaFnMesh::CreateQuad(1920.0f, 1080.0f);
//   OaFnMesh::ComputeBounds(mesh);

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Vlm.h>

// Mesh primitive topology
enum class OaMeshTopology : OaU8 {
	TriangleList,   // Separate triangles
	TriangleStrip,  // Triangle strip
	TriangleFan,    // Triangle fan
	LineList,       // Separate lines
	LineStrip,      // Line strip
	PointList,      // Points
};

// Axis-aligned bounding box
struct OaAabb {
	VlmVec3 Min = {0.0f, 0.0f, 0.0f};
	VlmVec3 Max = {0.0f, 0.0f, 0.0f};

	[[nodiscard]] VlmVec3 Center() const noexcept { return Vlm::Scale(Vlm::Add(Min, Max), 0.5f); }
	[[nodiscard]] VlmVec3 Extent() const noexcept { return Vlm::Sub(Max, Min); }
	[[nodiscard]] OaF32  Volume() const noexcept { VlmVec3 e = Extent(); return e.X * e.Y * e.Z; }
	[[nodiscard]] bool Contains(const VlmVec3& InPoint) const noexcept {
		return InPoint.X >= Min.X && InPoint.X <= Max.X &&
		       InPoint.Y >= Min.Y && InPoint.Y <= Max.Y &&
		       InPoint.Z >= Min.Z && InPoint.Z <= Max.Z;
	}
};

// Per-vertex data (CPU-side POD, GPU upload is separate)
struct OaMeshVertex {
	VlmVec3 Position = {0.0f, 0.0f, 0.0f};
	VlmVec3 Normal   = {0.0f, 0.0f, 1.0f};
	VlmVec2 Uv       = {0.0f, 0.0f};
	VlmVec4 Color    = {1.0f, 1.0f, 1.0f, 1.0f};
};

// Mesh data (CPU-side POD, can be uploaded to GPU buffers)
struct OaMeshData {
	OaVec<OaMeshVertex> Vertices;
	OaVec<OaU32>        Indices;
	OaMeshTopology      Topology = OaMeshTopology::TriangleList;
	OaAabb              Bounds;
	bool                BoundsDirty = true;
};

namespace OaFnMesh {

// ─── Geometry generation ─────────────────────────────────────────────────

// Create a centered quad (2 triangles, 4 vertices) with top-origin image UVs.
// UV V=0 is the source image top and world +Y is the quad top.
// Default is unit quad [-0.5, 0.5] in XY plane at Z=0.
// For image/video viewing, pass the pixel dimensions (e.g. 1920, 1080).
[[nodiscard]] OaMeshData CreateQuad(OaF32 InWidth = 1.0f, OaF32 InHeight = 1.0f);

// Create a fullscreen NDC quad [-1, 1] with top-origin image UVs.
// Used for post-processing passes.
[[nodiscard]] OaMeshData CreateFullscreenQuad();

// Create a 3D cube (12 triangles, 24 vertices).
[[nodiscard]] OaMeshData CreateCube(OaF32 InSize = 1.0f);

// ─── Bounds ──────────────────────────────────────────────────────────────

void ComputeBounds(OaMeshData& InMesh);

// ─── Queries ─────────────────────────────────────────────────────────────

[[nodiscard]] OaU32 GetVertexCount(const OaMeshData& InMesh) noexcept;
[[nodiscard]] OaU32 GetIndexCount(const OaMeshData& InMesh) noexcept;
[[nodiscard]] bool HasIndices(const OaMeshData& InMesh) noexcept;

// ─── Transforms ──────────────────────────────────────────────────────────

// Apply a 4x4 matrix to all vertex positions in-place.
void Transform(OaMeshData& InMesh, const VlmMat4& InMatrix);

// Translate all vertices in-place.
void Translate(OaMeshData& InMesh, const VlmVec3& InOffset);

// Scale all vertices in-place (from center or origin).
void Scale(OaMeshData& InMesh, const VlmVec3& InScale);

// Flip UVs vertically for explicitly bottom-origin source content.
void FlipUvsY(OaMeshData& InMesh);

} // namespace OaFnMesh
