// OaMesh — OOP wrapper for 3D mesh data.
//
// Wraps OaFnMesh functional API. All methods delegate to OaFnMesh.
// For the functional namespace, see <Oa/Render/FnMesh.h>.
//
// Usage (image/video viewer):
//   OaMesh mesh = OaMesh::CreateQuad(1920.0f, 1080.0f);  // pixel-sized quad
//   (top-origin image UVs; viewport owns Vulkan framebuffer Y mapping)
//
// Usage (3D scene):
//   OaMesh mesh = OaMesh::CreateCube(1.0f);
//   mesh.Transform(camera.GetViewProjectionMatrix());

#pragma once

#include <Oa/Render/FnMesh.h>

// Forward declaration
class OaMaterial;

class OaMesh {
public:
	OaMesh() = default;
	~OaMesh() = default;

	// Geometry factory methods
	[[nodiscard]] static OaMesh CreateQuad(OaF32 InWidth = 1.0f, OaF32 InHeight = 1.0f) {
		OaMesh m;
		m.Data_ = OaFnMesh::CreateQuad(InWidth, InHeight);
		return m;
	}

	[[nodiscard]] static OaMesh CreateFullscreenQuad() {
		OaMesh m;
		m.Data_ = OaFnMesh::CreateFullscreenQuad();
		return m;
	}

	[[nodiscard]] static OaMesh CreateCube(OaF32 InSize = 1.0f) {
		OaMesh m;
		m.Data_ = OaFnMesh::CreateCube(InSize);
		return m;
	}

	// Bounds
	void ComputeBounds() { OaFnMesh::ComputeBounds(Data_); }
	[[nodiscard]] const OaAabb& GetBounds() const noexcept { return Data_.Bounds; }

	// Queries
	[[nodiscard]] OaU32 GetVertexCount() const noexcept { return OaFnMesh::GetVertexCount(Data_); }
	[[nodiscard]] OaU32 GetIndexCount() const noexcept { return OaFnMesh::GetIndexCount(Data_); }
	[[nodiscard]] bool HasIndices() const noexcept { return OaFnMesh::HasIndices(Data_); }

	// Transforms
	void Transform(const VlmMat4& InMatrix) { OaFnMesh::Transform(Data_, InMatrix); }
	void Translate(const VlmVec3& InOffset) { OaFnMesh::Translate(Data_, InOffset); }
	void Scale(const VlmVec3& InScale) { OaFnMesh::Scale(Data_, InScale); }
	void FlipUvsY() { OaFnMesh::FlipUvsY(Data_); }

	// Topology
	void SetTopology(OaMeshTopology InTopology) { Data_.Topology = InTopology; }
	[[nodiscard]] OaMeshTopology GetTopology() const noexcept { return Data_.Topology; }

	// Material
	void SetMaterial(OaMaterial* InMaterial) { Material_ = InMaterial; }
	[[nodiscard]] OaMaterial* GetMaterial() const noexcept { return Material_; }

	// Direct data access
	[[nodiscard]] OaMeshData& GetData() noexcept { return Data_; }
	[[nodiscard]] const OaMeshData& GetData() const noexcept { return Data_; }

private:
	OaMeshData     Data_;
	OaMaterial*    Material_ = nullptr;
};
