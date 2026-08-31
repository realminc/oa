// oa::Scene — semantic CPU scene snapshot.
//
// Scene owns immutable mesh values and stable node identities. Nodes form a
// transform hierarchy and instance meshes without exposing renderer storage,
// Vulkan handles, passes, queues, or upload policy.

#pragma once

#include <oa/core/types.h>
#include <oa/core/transform.h>
#include <oa/core/vlm.h>
#include <oa/render/fnMesh.h>

namespace oa {

class SceneMeshId {
public:
	constexpr SceneMeshId() noexcept = default;
	explicit constexpr SceneMeshId(oa::U64 inValue) noexcept : value_(inValue) {}

	[[nodiscard]] constexpr oa::U64 value() const noexcept { return value_; }
	[[nodiscard]] constexpr bool isValid() const noexcept { return value_ != 0U; }
	[[nodiscard]] constexpr bool operator==(const SceneMeshId&) const noexcept = default;

private:
	oa::U64 value_ = 0U;
};

class SceneNodeId {
public:
	constexpr SceneNodeId() noexcept = default;
	explicit constexpr SceneNodeId(oa::U64 inValue) noexcept : value_(inValue) {}

	[[nodiscard]] constexpr oa::U64 value() const noexcept { return value_; }
	[[nodiscard]] constexpr bool isValid() const noexcept { return value_ != 0U; }
	[[nodiscard]] constexpr bool operator==(const SceneNodeId&) const noexcept = default;

private:
	oa::U64 value_ = 0U;
};

struct SceneMesh {
	SceneMeshId id;
	oa::String name;
	oa::MeshData data;
};

struct SceneNode {
	SceneNodeId id;
	oa::String name;
	SceneNodeId parent;
	SceneMeshId mesh;
	oa::Transform localTransform;
	bool visible = true;
};

struct Scene {
	oa::Vector<SceneMesh> meshes;
	oa::Vector<SceneNode> nodes;
};

} // namespace oa
