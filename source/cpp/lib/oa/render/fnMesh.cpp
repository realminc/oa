// oa::FnMesh implementation

#include <oa/render/fnMesh.h>
#include <oa/core/std/algo.h>

namespace oa {

namespace FnMesh {

// ─── Geometry generation ─────────────────────────────────────────────────

oa::MeshData createQuad(oa::F32 inWidth, oa::F32 inHeight) {
	oa::MeshData mesh;

	oa::F32 halfW = inWidth * 0.5f;
	oa::F32 halfH = inHeight * 0.5f;

	// 4 vertices: positions, normals, uvs
	mesh.vertices.resize(4);

	// top-origin image UVs: world +Y is the quad top and samples V=0.

	// bottom-left
	mesh.vertices[0].position = {-halfW, -halfH, 0.0f};
	mesh.vertices[0].normal   = {0.0f, 0.0f, 1.0f};
	mesh.vertices[0].uv       = {0.0f, 1.0f};

	// bottom-right
	mesh.vertices[1].position = { halfW, -halfH, 0.0f};
	mesh.vertices[1].normal   = {0.0f, 0.0f, 1.0f};
	mesh.vertices[1].uv       = {1.0f, 1.0f};

	// top-right
	mesh.vertices[2].position = { halfW,  halfH, 0.0f};
	mesh.vertices[2].normal   = {0.0f, 0.0f, 1.0f};
	mesh.vertices[2].uv       = {1.0f, 0.0f};

	// top-left
	mesh.vertices[3].position = {-halfW,  halfH, 0.0f};
	mesh.vertices[3].normal   = {0.0f, 0.0f, 1.0f};
	mesh.vertices[3].uv       = {0.0f, 0.0f};

	// 2 triangles = 6 indices
	mesh.indices = {0, 1, 2, 2, 3, 0};

	computeBounds(mesh);
	return mesh;
}

oa::MeshData createFullscreenQuad() {
	oa::MeshData mesh;

	mesh.vertices.resize(4);

	// NDC corners [-1, 1] with top-origin image UVs.
	mesh.vertices[0].position = {-1.0f, -1.0f, 0.0f};
	mesh.vertices[0].normal   = {0.0f, 0.0f, 1.0f};
	mesh.vertices[0].uv       = {0.0f, 1.0f};

	mesh.vertices[1].position = { 1.0f, -1.0f, 0.0f};
	mesh.vertices[1].normal   = {0.0f, 0.0f, 1.0f};
	mesh.vertices[1].uv       = {1.0f, 1.0f};

	mesh.vertices[2].position = { 1.0f,  1.0f, 0.0f};
	mesh.vertices[2].normal   = {0.0f, 0.0f, 1.0f};
	mesh.vertices[2].uv       = {1.0f, 0.0f};

	mesh.vertices[3].position = {-1.0f,  1.0f, 0.0f};
	mesh.vertices[3].normal   = {0.0f, 0.0f, 1.0f};
	mesh.vertices[3].uv       = {0.0f, 0.0f};

	mesh.indices = {0, 1, 2, 2, 3, 0};

	computeBounds(mesh);
	return mesh;
}

oa::MeshData createCube(oa::F32 inSize) {
	oa::MeshData mesh;

	oa::F32 h = inSize * 0.5f;

	// 24 vertices (4 per face, 6 faces)
	mesh.vertices.resize(24);

	// Helper to set face vertices
	auto setFace = [&](oa::Usize baseIdx, const oa::vlm::Vec3& normal,
		const oa::vlm::Vec3& p0, const oa::vlm::Vec3& p1,
		const oa::vlm::Vec3& p2, const oa::vlm::Vec3& p3,
		const oa::vlm::Vec2& uv0, const oa::vlm::Vec2& uv1,
		const oa::vlm::Vec2& uv2, const oa::vlm::Vec2& uv3) {
		const oa::vlm::Vec3 positions[4] = {p0, p1, p2, p3};
		const oa::vlm::Vec2 uvs[4] = {uv0, uv1, uv2, uv3};
		for (oa::Usize index = 0U; index < 4U; ++index) {
			oa::MeshVertex& vertex = mesh.vertices[baseIdx + index];
			vertex.position = positions[index];
			vertex.normal = normal;
			vertex.uv = uvs[index];
		}
	};

	// front face (+Z)
	setFace(0,  {0, 0, 1},
	        {-h, -h,  h}, { h, -h,  h}, { h,  h,  h}, {-h,  h,  h},
	        {0, 0}, {1, 0}, {1, 1}, {0, 1});

	// back face (-Z)
	setFace(4,  {0, 0, -1},
	        { h, -h, -h}, {-h, -h, -h}, {-h,  h, -h}, { h,  h, -h},
	        {0, 0}, {1, 0}, {1, 1}, {0, 1});

	// Right face (+X)
	setFace(8,  {1, 0, 0},
	        { h, -h,  h}, { h, -h, -h}, { h,  h, -h}, { h,  h,  h},
	        {0, 0}, {1, 0}, {1, 1}, {0, 1});

	// Left face (-X)
	setFace(12, {-1, 0, 0},
	        {-h, -h, -h}, {-h, -h,  h}, {-h,  h,  h}, {-h,  h, -h},
	        {0, 0}, {1, 0}, {1, 1}, {0, 1});

	// top face (+Y)
	setFace(16, {0, 1, 0},
	        {-h,  h,  h}, { h,  h,  h}, { h,  h, -h}, {-h,  h, -h},
	        {0, 0}, {1, 0}, {1, 1}, {0, 1});

	// bottom face (-Y)
	setFace(20, {0, -1, 0},
	        {-h, -h, -h}, { h, -h, -h}, { h, -h,  h}, {-h, -h,  h},
	        {0, 0}, {1, 0}, {1, 1}, {0, 1});

	// 12 triangles = 36 indices
	mesh.indices.resize(36);
	oa::Usize index = 0U;
	for (oa::U32 face = 0U; face < 6U; ++face) {
		const oa::U32 base = face * 4U;
		mesh.indices[index++] = base + 0U;
		mesh.indices[index++] = base + 1U;
		mesh.indices[index++] = base + 2U;
		mesh.indices[index++] = base + 2U;
		mesh.indices[index++] = base + 3U;
		mesh.indices[index++] = base + 0U;
	}

	computeBounds(mesh);
	return mesh;
}

// ─── bounds ──────────────────────────────────────────────────────────────

void computeBounds(oa::MeshData& inMesh) {
	if (inMesh.vertices.empty()) {
		inMesh.bounds = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
		inMesh.boundsDirty = false;
		return;
	}

	oa::vlm::Vec3 minP = inMesh.vertices[0].position;
	oa::vlm::Vec3 maxP = inMesh.vertices[0].position;

	for (const auto& v : inMesh.vertices) {
		minP.x = oa::min(minP.x, v.position.x);
		minP.y = oa::min(minP.y, v.position.y);
		minP.z = oa::min(minP.z, v.position.z);
		maxP.x = oa::max(maxP.x, v.position.x);
		maxP.y = oa::max(maxP.y, v.position.y);
		maxP.z = oa::max(maxP.z, v.position.z);
	}

	inMesh.bounds.min = minP;
	inMesh.bounds.max = maxP;
	inMesh.boundsDirty = false;
}

// ─── Queries ─────────────────────────────────────────────────────────────

oa::U32 getVertexCount(const oa::MeshData& inMesh) noexcept {
	return static_cast<oa::U32>(inMesh.vertices.size());
}

oa::U32 getIndexCount(const oa::MeshData& inMesh) noexcept {
	return static_cast<oa::U32>(inMesh.indices.size());
}

bool hasIndices(const oa::MeshData& inMesh) noexcept {
	return !inMesh.indices.empty();
}

// ─── Transforms ──────────────────────────────────────────────────────────

oa::Status transform(oa::MeshData& inMesh, const oa::vlm::Mat4& inMatrix) {
	oa::vlm::Mat3 normalMatrix{};
	if (not oa::vlm::tryNormalMatrix(inMatrix, normalMatrix)) {
		return oa::Status::invalidArgument(
			"FnMesh::transform requires a finite invertible affine transform");
	}
	for (const auto& vertex : inMesh.vertices) {
		const oa::vlm::Vec3 position = oa::vlm::transformPoint(
			vertex.position, inMatrix);
		oa::vlm::Vec3 normal{};
		if (not position.isFinite()
			or not oa::vlm::tryTransformNormal(
				vertex.normal, normalMatrix, normal)) {
			return oa::Status::invalidArgument(
				"FnMesh::transform produced a non-finite position or normal");
		}
	}
	for (auto& vertex : inMesh.vertices) {
		vertex.position = oa::vlm::transformPoint(vertex.position, inMatrix);
		const bool normalValid = oa::vlm::tryTransformNormal(
			vertex.normal, normalMatrix, vertex.normal);
		(void)normalValid;
		OA_ASSERT(normalValid);
	}
	inMesh.boundsDirty = true;
	return oa::Status::ok();
}

void translate(oa::MeshData& inMesh, const oa::vlm::Vec3& inOffset) {
	for (auto& v : inMesh.vertices) {
		v.position.x += inOffset.x;
		v.position.y += inOffset.y;
		v.position.z += inOffset.z;
	}
	inMesh.boundsDirty = true;
}

oa::Status scale(oa::MeshData& inMesh, const oa::vlm::Vec3& inScale) {
	return transform(inMesh, oa::vlm::scaleMatrix(inScale));
}

void flipUvsY(oa::MeshData& inMesh) {
	for (auto& v : inMesh.vertices) {
		v.uv.y = 1.0f - v.uv.y;
	}
}

} // namespace FnMesh

} // namespace oa
