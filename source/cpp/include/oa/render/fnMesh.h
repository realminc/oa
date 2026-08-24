// oa::FnMesh — functional mesh operations.
//
// Stateless mesh math and geometry generation. Operates on oa::MeshData.
//
// usage:
//   oa::MeshData mesh = oa::FnMesh::createQuad(1920.0f, 1080.0f);
//   oa::FnMesh::computeBounds(mesh);

#pragma once

#include <oa/core/types.h>
#include <oa/core/vlm.h>

namespace oa {

// axis-aligned bounding box
struct Aabb {
	oa::vlm::Vec3 min = {0.0f, 0.0f, 0.0f};
	oa::vlm::Vec3 max = {0.0f, 0.0f, 0.0f};

	[[nodiscard]] oa::vlm::Vec3 center() const noexcept {
		return oa::vlm::scale(oa::vlm::add(min, max), 0.5f);
	}
	[[nodiscard]] oa::vlm::Vec3 extent() const noexcept { return oa::vlm::sub(max, min); }
	[[nodiscard]] oa::F32 volume() const noexcept {
		const oa::vlm::Vec3 e = extent();
		return e.x * e.y * e.z;
	}
	[[nodiscard]] bool contains(const oa::vlm::Vec3& inPoint) const noexcept {
		return inPoint.x >= min.x && inPoint.x <= max.x &&
		       inPoint.y >= min.y && inPoint.y <= max.y &&
		       inPoint.z >= min.z && inPoint.z <= max.z;
	}
};

// Per-vertex data (CPU-side POD, GPU upload is separate)
struct MeshVertex {
	oa::vlm::Vec3 position = {0.0f, 0.0f, 0.0f};
	oa::vlm::Vec3 normal   = {0.0f, 0.0f, 1.0f};
	oa::vlm::Vec2 uv       = {0.0f, 0.0f};
	oa::vlm::Vec4 color    = {1.0f, 1.0f, 1.0f, 1.0f};
};

// Mesh data (CPU-side POD, can be uploaded to GPU buffers)
struct MeshData {
	oa::Vec<MeshVertex> vertices;
	oa::Vec<oa::U32>        indices;
	Aabb                bounds;
	bool                boundsDirty = true;
};

namespace FnMesh {

// ─── Geometry generation ─────────────────────────────────────────────────

// Create a centered quad (2 triangles, 4 vertices) with top-origin image UVs.
// UV V=0 is the source image top and world +Y is the quad top.
// Default is unit quad [-0.5, 0.5] in XY plane at Z=0.
// For image/video viewing, pass the pixel dimensions (e.g. 1920, 1080).
[[nodiscard]] oa::MeshData createQuad(oa::F32 inWidth = 1.0f, oa::F32 inHeight = 1.0f);

// Create a fullscreen NDC quad [-1, 1] with top-origin image UVs.
// Used for post-processing passes.
[[nodiscard]] oa::MeshData createFullscreenQuad();

// Create a 3D cube (12 triangles, 24 vertices).
[[nodiscard]] oa::MeshData createCube(oa::F32 inSize = 1.0f);

// ─── bounds ──────────────────────────────────────────────────────────────

void computeBounds(oa::MeshData& inMesh);

// ─── Queries ─────────────────────────────────────────────────────────────

[[nodiscard]] oa::U32 getVertexCount(const oa::MeshData& inMesh) noexcept;
[[nodiscard]] oa::U32 getIndexCount(const oa::MeshData& inMesh) noexcept;
[[nodiscard]] bool hasIndices(const oa::MeshData& inMesh) noexcept;

// ─── Transforms ──────────────────────────────────────────────────────────

// apply a 4x4 matrix to all vertex positions in-place.
void transform(oa::MeshData& inMesh, const oa::vlm::Mat4& inMatrix);

// translate all vertices in-place.
void translate(oa::MeshData& inMesh, const oa::vlm::Vec3& inOffset);

// scale all vertices in-place (from center or origin).
void scale(oa::MeshData& inMesh, const oa::vlm::Vec3& inScale);

// Flip UVs vertically for explicitly bottom-origin source content.
void flipUvsY(oa::MeshData& inMesh);

} // namespace FnMesh

} // namespace oa
