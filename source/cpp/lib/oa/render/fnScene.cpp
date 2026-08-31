#include <oa/render/fnScene.h>

#include <oa/core/fnTransform.h>
#include <oa/core/std/hashMap.h>

namespace {

enum class ResolveState : oa::U8 {
	Unresolved,
	Resolving,
	Resolved,
};

struct ResolvedScene {
	oa::HashMap<oa::U64, oa::Usize> meshIndices;
	oa::HashMap<oa::U64, oa::Usize> nodeIndices;
	oa::Vector<oa::vlm::Mat4> worldTransforms;
	oa::Vector<bool> visible;
};

[[nodiscard]] bool isFinite(const oa::MeshVertex& inVertex) noexcept {
	return inVertex.position.isFinite()
		and inVertex.normal.isFinite()
		and inVertex.uv.isFinite()
		and inVertex.color.isFinite();
}

[[nodiscard]] oa::Status validateMesh(const oa::SceneMesh& inMesh) {
	if (not inMesh.id.isValid()) {
		return oa::Status::invalidArgument("oa::Scene mesh identity zero is reserved");
	}
	if (inMesh.data.vertices.empty() or inMesh.data.indices.empty()
		or inMesh.data.indices.size() % 3U != 0U) {
		return oa::Status::invalidArgument(
			"oa::Scene meshes must be non-empty indexed triangle lists");
	}
	for (const oa::MeshVertex& vertex : inMesh.data.vertices) {
		if (not isFinite(vertex)) {
			return oa::Status::invalidArgument(
				"oa::Scene mesh contains non-finite vertex data");
		}
	}
	for (oa::U32 index : inMesh.data.indices) {
		if (index >= inMesh.data.vertices.size()) {
			return oa::Status::invalidArgument(
				"oa::Scene mesh index is outside its vertex array");
		}
	}
	return oa::Status::ok();
}

[[nodiscard]] oa::Status resolve(
	const oa::Scene& inScene,
	const oa::FnScene::CompileConfig& inConfig,
	ResolvedScene& outResolved) {
	if (inScene.meshes.size() > inConfig.maxMeshCount
		or inScene.nodes.size() > inConfig.maxNodeCount) {
		return oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::Scene exceeds the configured mesh or node capacity");
	}

	outResolved.meshIndices.reserve(inScene.meshes.size());
	for (oa::Usize index = 0U; index < inScene.meshes.size(); ++index) {
		const oa::SceneMesh& mesh = inScene.meshes[index];
		OA_RETURN_IF_ERROR(validateMesh(mesh));
		if (not outResolved.meshIndices.emplace(mesh.id.value(), index).second) {
			return oa::Status::invalidArgument(
				"oa::Scene mesh identities must be unique");
		}
	}

	outResolved.nodeIndices.reserve(inScene.nodes.size());
	for (oa::Usize index = 0U; index < inScene.nodes.size(); ++index) {
		const oa::SceneNode& node = inScene.nodes[index];
		if (not node.id.isValid()) {
			return oa::Status::invalidArgument(
				"oa::Scene node identity zero is reserved");
		}
		OA_RETURN_IF_ERROR(oa::FnTransform::validate(node.localTransform));
		if (not outResolved.nodeIndices.emplace(node.id.value(), index).second) {
			return oa::Status::invalidArgument(
				"oa::Scene node identities must be unique");
		}
		if (node.mesh.isValid()
			and not outResolved.meshIndices.contains(node.mesh.value())) {
			return oa::Status::invalidArgument(
				"oa::Scene node references a missing mesh identity");
		}
	}

	outResolved.worldTransforms.resize(inScene.nodes.size());
	outResolved.visible.resize(inScene.nodes.size());
	oa::Vector<ResolveState> states(inScene.nodes.size());
	oa::Vector<oa::Usize> path;
	path.reserve(inScene.nodes.size());

	for (oa::Usize start = 0U; start < inScene.nodes.size(); ++start) {
		if (states[start] == ResolveState::Resolved) continue;
		path.clear();
		oa::Usize current = start;
		for (;;) {
			if (states[current] == ResolveState::Resolved) break;
			if (states[current] == ResolveState::Resolving) {
				return oa::Status::invalidArgument(
					"oa::Scene node hierarchy contains a cycle");
			}
			states[current] = ResolveState::Resolving;
			path.pushBack(current);
			const oa::SceneNodeId parent = inScene.nodes[current].parent;
			if (not parent.isValid()) break;
			const auto parentIterator = outResolved.nodeIndices.find(parent.value());
			if (parentIterator == outResolved.nodeIndices.end()) {
				return oa::Status::invalidArgument(
					"oa::Scene node references a missing parent identity");
			}
			current = parentIterator->second;
		}

		while (not path.empty()) {
			const oa::Usize index = path.back();
			path.popBack();
			const oa::SceneNode& node = inScene.nodes[index];
			const oa::vlm::Mat4 localTransform =
				oa::FnTransform::getMatrix(node.localTransform);
			if (node.parent.isValid()) {
				const oa::Usize parentIndex =
					outResolved.nodeIndices.at(node.parent.value());
				if (states[parentIndex] != ResolveState::Resolved) {
					return oa::Status::error(
						oa::StatusCode::Internal,
						"oa::Scene hierarchy resolution order is inconsistent");
				}
				outResolved.worldTransforms[index] = oa::vlm::matrixMul(
					localTransform,
					outResolved.worldTransforms[parentIndex]);
				outResolved.visible[index] =
					node.visible and outResolved.visible[parentIndex];
			} else {
				outResolved.worldTransforms[index] = localTransform;
				outResolved.visible[index] = node.visible;
			}
			if (not outResolved.worldTransforms[index].isFinite()) {
				return oa::Status::error(
					oa::StatusCode::OutOfRange,
					"oa::Scene world transform overflowed finite FP32");
			}
			states[index] = ResolveState::Resolved;
		}
	}

	return oa::Status::ok();
}

} // namespace

oa::Status oa::FnScene::validate(
	const oa::Scene& inScene,
	const CompileConfig& inConfig) {
	ResolvedScene resolved;
	OA_RETURN_IF_ERROR(resolve(inScene, inConfig, resolved));
	oa::U64 vertexCount = 0U;
	oa::U64 indexCount = 0U;
	for (oa::Usize index = 0U; index < inScene.nodes.size(); ++index) {
		const oa::SceneNode& node = inScene.nodes[index];
		if (not resolved.visible[index] or not node.mesh.isValid()) continue;
		const oa::MeshData& mesh =
			inScene.meshes[resolved.meshIndices.at(node.mesh.value())].data;
		vertexCount += mesh.vertices.size();
		indexCount += mesh.indices.size();
		if (vertexCount > inConfig.maxVertexCount
			or indexCount > inConfig.maxIndexCount) {
			return oa::Status::error(
				oa::StatusCode::ResourceExhausted,
				"oa::Scene compiled geometry exceeds configured capacity");
		}
		if (vertexCount > oa::Limits<oa::U32>::max()) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				"oa::Scene compiled vertex indices exceed U32 representation");
		}
		oa::vlm::Mat3 normalMatrix{};
		if (not oa::vlm::tryNormalMatrix(
				resolved.worldTransforms[index], normalMatrix)) {
			return oa::Status::invalidArgument(
				"oa::Scene visible mesh transform must admit finite normal transformation");
		}
		for (const oa::MeshVertex& vertex : mesh.vertices) {
			oa::vlm::Vec3 transformedNormal{};
			if (not oa::vlm::tryTransformNormal(
					vertex.normal, normalMatrix, transformedNormal)) {
				return oa::Status::invalidArgument(
					"oa::Scene visible mesh normals must be non-zero and transformable");
			}
		}
	}
	return oa::Status::ok();
}

oa::Result<oa::MeshData> oa::FnScene::compile(
	const oa::Scene& inScene,
	const CompileConfig& inConfig) {
	ResolvedScene resolved;
	OA_RETURN_IF_ERROR(resolve(inScene, inConfig, resolved));

	oa::U64 vertexCount = 0U;
	oa::U64 indexCount = 0U;
	for (oa::Usize index = 0U; index < inScene.nodes.size(); ++index) {
		const oa::SceneNode& node = inScene.nodes[index];
		if (not resolved.visible[index] or not node.mesh.isValid()) continue;
		const oa::MeshData& mesh =
			inScene.meshes[resolved.meshIndices.at(node.mesh.value())].data;
		vertexCount += mesh.vertices.size();
		indexCount += mesh.indices.size();
		if (vertexCount > inConfig.maxVertexCount
			or indexCount > inConfig.maxIndexCount) {
			return oa::Status::error(
				oa::StatusCode::ResourceExhausted,
				"oa::Scene compiled geometry exceeds configured capacity");
		}
		if (vertexCount > oa::Limits<oa::U32>::max()) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				"oa::Scene compiled vertex indices exceed U32 representation");
		}
	}
	if (vertexCount == 0U or indexCount == 0U) {
		return oa::Status::invalidArgument(
			"oa::Scene has no visible indexed geometry");
	}

	oa::MeshData output;
	output.vertices.reserve(static_cast<oa::Usize>(vertexCount));
	output.indices.reserve(static_cast<oa::Usize>(indexCount));
	for (oa::Usize nodeIndex = 0U; nodeIndex < inScene.nodes.size(); ++nodeIndex) {
		const oa::SceneNode& node = inScene.nodes[nodeIndex];
		if (not resolved.visible[nodeIndex] or not node.mesh.isValid()) continue;
		const oa::MeshData& mesh =
			inScene.meshes[resolved.meshIndices.at(node.mesh.value())].data;
		const oa::vlm::Mat4& transform = resolved.worldTransforms[nodeIndex];
		oa::vlm::Mat3 normalMatrix{};
		if (not oa::vlm::tryNormalMatrix(transform, normalMatrix)) {
			return oa::Status::invalidArgument(
				"oa::Scene visible mesh transform must admit finite normal transformation");
		}
		const oa::U32 base = static_cast<oa::U32>(output.vertices.size());
		for (const oa::MeshVertex& source : mesh.vertices) {
			oa::MeshVertex vertex = source;
			vertex.position = oa::vlm::transformPoint(source.position, transform);
			if (not vertex.position.isFinite()
				or not oa::vlm::tryTransformNormal(
					source.normal, normalMatrix, vertex.normal)) {
				return oa::Status::error(
					oa::StatusCode::OutOfRange,
					"oa::Scene instance transform produced invalid vertex data");
			}
			output.vertices.pushBack(vertex);
		}
		for (oa::U32 source : mesh.indices) output.indices.pushBack(base + source);
	}
	oa::FnMesh::computeBounds(output);
	return output;
}

oa::SceneNode* oa::FnScene::findNode(
	oa::Scene& inScene,
	oa::SceneNodeId inId) noexcept {
	if (not inId.isValid()) return nullptr;
	for (oa::SceneNode& node : inScene.nodes) {
		if (node.id == inId) return &node;
	}
	return nullptr;
}

const oa::SceneNode* oa::FnScene::findNode(
	const oa::Scene& inScene,
	oa::SceneNodeId inId) noexcept {
	if (not inId.isValid()) return nullptr;
	for (const oa::SceneNode& node : inScene.nodes) {
		if (node.id == inId) return &node;
	}
	return nullptr;
}
