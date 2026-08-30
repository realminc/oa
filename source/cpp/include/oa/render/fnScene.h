// oa::FnScene — stateless semantic-scene operations.

#pragma once

#include <oa/core/status.h>
#include <oa/render/scene.h>

namespace oa {
namespace FnScene {

struct CompileConfig {
	oa::U32 maxMeshCount = oa::Limits<oa::U32>::max();
	oa::U32 maxNodeCount = oa::Limits<oa::U32>::max();
	oa::U32 maxVertexCount = oa::Limits<oa::U32>::max();
	oa::U32 maxIndexCount = oa::Limits<oa::U32>::max();
};

// Validation is order-independent. Parent and mesh references use stable IDs;
// missing references, duplicates, cycles, malformed meshes, non-finite
// transforms, singular normal transforms, and configured capacity overflow
// fail explicitly.
[[nodiscard]] oa::Status validate(
	const oa::Scene& inScene,
	const CompileConfig& inConfig = {});

// Compiles the semantic snapshot into the current renderer's indexed triangle
// snapshot. Node order determines deterministic draw/append order, while parent
// declaration order is unrestricted.
[[nodiscard]] oa::Result<oa::MeshData> compile(
	const oa::Scene& inScene,
	const CompileConfig& inConfig = {});

[[nodiscard]] oa::SceneNode* findNode(
	oa::Scene& inScene,
	oa::SceneNodeId inId) noexcept;
[[nodiscard]] const oa::SceneNode* findNode(
	const oa::Scene& inScene,
	oa::SceneNodeId inId) noexcept;

} // namespace FnScene
} // namespace oa
