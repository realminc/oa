// OaFnMesh implementation

#include <Oa/Render/FnMesh.h>
#include <cmath>

namespace OaFnMesh {

// ─── Geometry generation ─────────────────────────────────────────────────

OaMeshData CreateQuad(OaF32 InWidth, OaF32 InHeight) {
	OaMeshData mesh;
	mesh.Topology = OaMeshTopology::TriangleList;

	OaF32 halfW = InWidth * 0.5f;
	OaF32 halfH = InHeight * 0.5f;

	// 4 vertices: positions, normals, uvs
	mesh.Vertices.resize(4);

	// Top-origin image UVs: world +Y is the quad top and samples V=0.

	// Bottom-left
	mesh.Vertices[0].Position = {-halfW, -halfH, 0.0f};
	mesh.Vertices[0].Normal   = {0.0f, 0.0f, 1.0f};
	mesh.Vertices[0].Uv       = {0.0f, 1.0f};

	// Bottom-right
	mesh.Vertices[1].Position = { halfW, -halfH, 0.0f};
	mesh.Vertices[1].Normal   = {0.0f, 0.0f, 1.0f};
	mesh.Vertices[1].Uv       = {1.0f, 1.0f};

	// Top-right
	mesh.Vertices[2].Position = { halfW,  halfH, 0.0f};
	mesh.Vertices[2].Normal   = {0.0f, 0.0f, 1.0f};
	mesh.Vertices[2].Uv       = {1.0f, 0.0f};

	// Top-left
	mesh.Vertices[3].Position = {-halfW,  halfH, 0.0f};
	mesh.Vertices[3].Normal   = {0.0f, 0.0f, 1.0f};
	mesh.Vertices[3].Uv       = {0.0f, 0.0f};

	// 2 triangles = 6 indices
	mesh.Indices = {0, 1, 2, 2, 3, 0};

	ComputeBounds(mesh);
	return mesh;
}

OaMeshData CreateFullscreenQuad() {
	OaMeshData mesh;
	mesh.Topology = OaMeshTopology::TriangleList;

	mesh.Vertices.resize(4);

	// NDC corners [-1, 1] with top-origin image UVs.
	mesh.Vertices[0].Position = {-1.0f, -1.0f, 0.0f};
	mesh.Vertices[0].Normal   = {0.0f, 0.0f, 1.0f};
	mesh.Vertices[0].Uv       = {0.0f, 1.0f};

	mesh.Vertices[1].Position = { 1.0f, -1.0f, 0.0f};
	mesh.Vertices[1].Normal   = {0.0f, 0.0f, 1.0f};
	mesh.Vertices[1].Uv       = {1.0f, 1.0f};

	mesh.Vertices[2].Position = { 1.0f,  1.0f, 0.0f};
	mesh.Vertices[2].Normal   = {0.0f, 0.0f, 1.0f};
	mesh.Vertices[2].Uv       = {1.0f, 0.0f};

	mesh.Vertices[3].Position = {-1.0f,  1.0f, 0.0f};
	mesh.Vertices[3].Normal   = {0.0f, 0.0f, 1.0f};
	mesh.Vertices[3].Uv       = {0.0f, 0.0f};

	mesh.Indices = {0, 1, 2, 2, 3, 0};

	ComputeBounds(mesh);
	return mesh;
}

OaMeshData CreateCube(OaF32 InSize) {
	OaMeshData mesh;
	mesh.Topology = OaMeshTopology::TriangleList;

	OaF32 h = InSize * 0.5f;

	// 24 vertices (4 per face, 6 faces)
	mesh.Vertices.resize(24);

	// Helper to set face vertices
	auto SetFace = [&](OaI32 BaseIdx, const VlmVec3& N,
	                  const VlmVec3& P0, const VlmVec3& P1, const VlmVec3& P2, const VlmVec3& P3,
	                  const VlmVec2& U0, const VlmVec2& U1, const VlmVec2& U2, const VlmVec2& U3) {
		mesh.Vertices[BaseIdx + 0].Position = P0; mesh.Vertices[BaseIdx + 0].Normal = N; mesh.Vertices[BaseIdx + 0].Uv = U0;
		mesh.Vertices[BaseIdx + 1].Position = P1; mesh.Vertices[BaseIdx + 1].Normal = N; mesh.Vertices[BaseIdx + 1].Uv = U1;
		mesh.Vertices[BaseIdx + 2].Position = P2; mesh.Vertices[BaseIdx + 2].Normal = N; mesh.Vertices[BaseIdx + 2].Uv = U2;
		mesh.Vertices[BaseIdx + 3].Position = P3; mesh.Vertices[BaseIdx + 3].Normal = N; mesh.Vertices[BaseIdx + 3].Uv = U3;
	};

	// Front face (+Z)
	SetFace(0,  {0, 0, 1},
	        {-h, -h,  h}, { h, -h,  h}, { h,  h,  h}, {-h,  h,  h},
	        {0, 0}, {1, 0}, {1, 1}, {0, 1});

	// Back face (-Z)
	SetFace(4,  {0, 0, -1},
	        { h, -h, -h}, {-h, -h, -h}, {-h,  h, -h}, { h,  h, -h},
	        {0, 0}, {1, 0}, {1, 1}, {0, 1});

	// Right face (+X)
	SetFace(8,  {1, 0, 0},
	        { h, -h,  h}, { h, -h, -h}, { h,  h, -h}, { h,  h,  h},
	        {0, 0}, {1, 0}, {1, 1}, {0, 1});

	// Left face (-X)
	SetFace(12, {-1, 0, 0},
	        {-h, -h, -h}, {-h, -h,  h}, {-h,  h,  h}, {-h,  h, -h},
	        {0, 0}, {1, 0}, {1, 1}, {0, 1});

	// Top face (+Y)
	SetFace(16, {0, 1, 0},
	        {-h,  h,  h}, { h,  h,  h}, { h,  h, -h}, {-h,  h, -h},
	        {0, 0}, {1, 0}, {1, 1}, {0, 1});

	// Bottom face (-Y)
	SetFace(20, {0, -1, 0},
	        {-h, -h, -h}, { h, -h, -h}, { h, -h,  h}, {-h, -h,  h},
	        {0, 0}, {1, 0}, {1, 1}, {0, 1});

	// 12 triangles = 36 indices
	mesh.Indices.resize(36);
	OaI32 idx = 0;
	for (OaI32 face = 0; face < 6; ++face) {
		OaU32 base = face * 4;
		mesh.Indices[idx++] = base + 0;
		mesh.Indices[idx++] = base + 1;
		mesh.Indices[idx++] = base + 2;
		mesh.Indices[idx++] = base + 2;
		mesh.Indices[idx++] = base + 3;
		mesh.Indices[idx++] = base + 0;
	}

	ComputeBounds(mesh);
	return mesh;
}

// ─── Bounds ──────────────────────────────────────────────────────────────

void ComputeBounds(OaMeshData& InMesh) {
	if (InMesh.Vertices.empty()) {
		InMesh.Bounds = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
		InMesh.BoundsDirty = false;
		return;
	}

	VlmVec3 minP = InMesh.Vertices[0].Position;
	VlmVec3 maxP = InMesh.Vertices[0].Position;

	for (const auto& v : InMesh.Vertices) {
		minP.X = std::min(minP.X, v.Position.X);
		minP.Y = std::min(minP.Y, v.Position.Y);
		minP.Z = std::min(minP.Z, v.Position.Z);
		maxP.X = std::max(maxP.X, v.Position.X);
		maxP.Y = std::max(maxP.Y, v.Position.Y);
		maxP.Z = std::max(maxP.Z, v.Position.Z);
	}

	InMesh.Bounds.Min = minP;
	InMesh.Bounds.Max = maxP;
	InMesh.BoundsDirty = false;
}

// ─── Queries ─────────────────────────────────────────────────────────────

OaU32 GetVertexCount(const OaMeshData& InMesh) noexcept {
	return static_cast<OaU32>(InMesh.Vertices.size());
}

OaU32 GetIndexCount(const OaMeshData& InMesh) noexcept {
	return static_cast<OaU32>(InMesh.Indices.size());
}

bool HasIndices(const OaMeshData& InMesh) noexcept {
	return !InMesh.Indices.empty();
}

// ─── Transforms ──────────────────────────────────────────────────────────

void Transform(OaMeshData& InMesh, const VlmMat4& InMatrix) {
	for (auto& v : InMesh.Vertices) {
		VlmVec3 p = v.Position;
		VlmVec3 t = {
			InMatrix.M[0][0] * p.X + InMatrix.M[1][0] * p.Y + InMatrix.M[2][0] * p.Z + InMatrix.M[3][0],
			InMatrix.M[0][1] * p.X + InMatrix.M[1][1] * p.Y + InMatrix.M[2][1] * p.Z + InMatrix.M[3][1],
			InMatrix.M[0][2] * p.X + InMatrix.M[1][2] * p.Y + InMatrix.M[2][2] * p.Z + InMatrix.M[3][2]
		};
		v.Position = t;
	}
	InMesh.BoundsDirty = true;
}

void Translate(OaMeshData& InMesh, const VlmVec3& InOffset) {
	for (auto& v : InMesh.Vertices) {
		v.Position.X += InOffset.X;
		v.Position.Y += InOffset.Y;
		v.Position.Z += InOffset.Z;
	}
	InMesh.BoundsDirty = true;
}

void Scale(OaMeshData& InMesh, const VlmVec3& InScale) {
	for (auto& v : InMesh.Vertices) {
		v.Position.X *= InScale.X;
		v.Position.Y *= InScale.Y;
		v.Position.Z *= InScale.Z;
	}
	InMesh.BoundsDirty = true;
}

void FlipUvsY(OaMeshData& InMesh) {
	for (auto& v : InMesh.Vertices) {
		v.Uv.Y = 1.0f - v.Uv.Y;
	}
}

} // namespace OaFnMesh
