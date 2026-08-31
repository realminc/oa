#include "lunarLander3dRender.h"

#include <oa/render/fnMesh.h>


namespace {

constexpr oa::U32 MaxTerrainCellsPerAxis = 32U;
constexpr oa::SceneMeshId TerrainMeshId{1U};
constexpr oa::SceneMeshId HullMeshId{2U};
constexpr oa::SceneMeshId LegMeshId{3U};
constexpr oa::SceneMeshId FootMeshId{4U};
constexpr oa::SceneNodeId TerrainNodeId{1U};
constexpr oa::SceneNodeId LanderRootNodeId{2U};
constexpr oa::U64 FirstLanderPartNodeId = 100U;

[[nodiscard]] bool isFinite(const oa::vlm::Vec3& inVector) noexcept {
	return oa::isFinite(inVector.x)
		and oa::isFinite(inVector.y)
		and oa::isFinite(inVector.z);
}

[[nodiscard]] bool tryNarrowFinite(
	double inValue,
	oa::F32& outValue) noexcept {
	if (not oa::isFinite(inValue)
		or inValue > static_cast<double>(oa::Limits<oa::F32>::max())
		or inValue < -static_cast<double>(oa::Limits<oa::F32>::max())) {
		return false;
	}
	const oa::F32 converted = static_cast<oa::F32>(inValue);
	if (not oa::isFinite(converted)
		or (inValue != 0.0 and converted == 0.0F)
		or (converted != 0.0F && oa::abs(converted) < oa::Limits<oa::F32>::min())) {
		return false;
	}
	outValue = converted;
	return true;
}

[[nodiscard]] bool tryToVlm(
	const oa::vlm::DVec3& inValue,
	oa::vlm::Vec3& outValue) noexcept {
	return tryNarrowFinite(inValue.x, outValue.x)
		and tryNarrowFinite(inValue.y, outValue.y)
		and tryNarrowFinite(inValue.z, outValue.z);
}

[[nodiscard]] oa::vlm::Vec3 add(
	const oa::vlm::Vec3& inLeft,
	const oa::vlm::Vec3& inRight) noexcept {
	return {
		inLeft.x + inRight.x,
		inLeft.y + inRight.y,
		inLeft.z + inRight.z,
	};
}

[[nodiscard]] oa::vlm::Vec3 sub(
	const oa::vlm::Vec3& inLeft,
	const oa::vlm::Vec3& inRight) noexcept {
	return {
		inLeft.x - inRight.x,
		inLeft.y - inRight.y,
		inLeft.z - inRight.z,
	};
}

[[nodiscard]] oa::vlm::Vec3 scale(
	const oa::vlm::Vec3& inValue,
	oa::F32 inScale) noexcept {
	return {
		inValue.x * inScale,
		inValue.y * inScale,
		inValue.z * inScale,
	};
}

[[nodiscard]] oa::vlm::Vec3 cross(
	const oa::vlm::Vec3& inLeft,
	const oa::vlm::Vec3& inRight) noexcept {
	return {
		inLeft.y * inRight.z - inLeft.z * inRight.y,
		inLeft.z * inRight.x - inLeft.x * inRight.z,
		inLeft.x * inRight.y - inLeft.y * inRight.x,
	};
}

[[nodiscard]] oa::vlm::Vec3 normalize(const oa::vlm::Vec3& inValue) noexcept {
	const oa::F32 maximumComponent = oa::max(
		oa::abs(inValue.x), oa::max(oa::abs(inValue.y), oa::abs(inValue.z)));
	if (not oa::isFinite(maximumComponent)
		or maximumComponent <= 1.0e-6F) {
		return {0.0F, 1.0F, 0.0F};
	}
	const oa::vlm::Vec3 scaled = scale(inValue, 1.0F / maximumComponent);
	const oa::F32 lengthSquared =
		scaled.x * scaled.x + scaled.y * scaled.y + scaled.z * scaled.z;
	if (not oa::isFinite(lengthSquared) or lengthSquared <= 0.0F) {
		return {0.0F, 1.0F, 0.0F};
	}
	return scale(scaled, 1.0F / oa::sqrt(lengthSquared));
}

[[nodiscard]] bool tryToVlm(
	const oa::vlm::DQuat& inValue,
	oa::vlm::Quat& outValue) noexcept {
	if (not tryNarrowFinite(inValue.x, outValue.x)
		or not tryNarrowFinite(inValue.y, outValue.y)
		or not tryNarrowFinite(inValue.z, outValue.z)
		or not tryNarrowFinite(inValue.w, outValue.w)) {
		return false;
	}
	outValue = outValue.normalized();
	return outValue.isFinite();
}

[[nodiscard]] oa::MeshData coloredCube(const oa::vlm::Vec4& inColor) {
	oa::MeshData mesh = oa::FnMesh::createCube(2.0F);
	for (oa::MeshVertex& vertex : mesh.vertices) vertex.color = inColor;
	return mesh;
}

void appendPartNode(
	oa::Scene& inScene,
	oa::U64& inOutNextNodeId,
	oa::StringView inName,
	oa::SceneMeshId inMesh,
	const oa::vlm::Vec3& inCenter,
	const oa::vlm::Vec3& inHalfExtent) {
	inScene.nodes.pushBack({
		oa::SceneNodeId{inOutNextNodeId++},
		oa::String(inName),
		LanderRootNodeId,
		inMesh,
		oa::Transform{
			inCenter, oa::vlm::Quat::identity(), inHalfExtent},
		true,
	});
}

[[nodiscard]] oa::Status buildTerrainMesh(
	const oa::LunarTerrain& inTerrain,
	oa::MeshData& outMesh) {
	const auto& config = inTerrain.config();
	const oa::U32 verticesX = config.cellsX_ + 1U;
	const oa::U32 verticesZ = config.cellsZ_ + 1U;
	const oa::U64 vertexCount =
		static_cast<oa::U64>(verticesX) * static_cast<oa::U64>(verticesZ);
	if (vertexCount != inTerrain.heights().size()) {
		return oa::Status::invalidArgument(
			"LunarLander3d terrain height count does not match its grid");
	}

	oa::Vector<oa::vlm::Vec3> positions(static_cast<oa::Usize>(vertexCount));
	oa::Vector<oa::vlm::Vec3> normalSums(static_cast<oa::Usize>(vertexCount));
	for (oa::U32 z = 0U; z < verticesZ; ++z) {
		for (oa::U32 x = 0U; x < verticesX; ++x) {
			const oa::U32 index = z * verticesX + x;
			oa::vlm::Vec3 position{};
			if (not tryNarrowFinite(
					inTerrain.minX()
						+ static_cast<double>(x) * config.cellSize_,
					position.x)
				or not tryNarrowFinite(
					inTerrain.heights()[index], position.y)
				or not tryNarrowFinite(
					inTerrain.minZ()
						+ static_cast<double>(z) * config.cellSize_,
					position.z)) {
				return oa::Status::error(
					oa::StatusCode::OutOfRange,
					"LunarLander3d terrain is not representable as FP32 geometry");
			}
			positions[index] = position;
		}
	}

	outMesh.indices.reserve(
		static_cast<oa::Usize>(config.cellsX_) * config.cellsZ_ * 6U);
	for (oa::U32 z = 0U; z < config.cellsZ_; ++z) {
		for (oa::U32 x = 0U; x < config.cellsX_; ++x) {
			const oa::U32 v00 = z * verticesX + x;
			const oa::U32 v10 = v00 + 1U;
			const oa::U32 v01 = v00 + verticesX;
			const oa::U32 v11 = v01 + 1U;
			const oa::U32 triangles[6] = {
				v00, v11, v10, v00, v01, v11,
			};
			for (oa::U32 index : triangles) {
				outMesh.indices.pushBack(index);
			}
			for (oa::U32 triangle = 0U; triangle < 2U; ++triangle) {
				const oa::U32 a = triangles[triangle * 3U];
				const oa::U32 b = triangles[triangle * 3U + 1U];
				const oa::U32 c = triangles[triangle * 3U + 2U];
				const oa::vlm::Vec3 edgeAb = sub(positions[b], positions[a]);
				const oa::vlm::Vec3 edgeAc = sub(positions[c], positions[a]);
				const oa::vlm::Vec3 normal = cross(edgeAb, edgeAc);
				const oa::vlm::Vec3 sumA = add(normalSums[a], normal);
				const oa::vlm::Vec3 sumB = add(normalSums[b], normal);
				const oa::vlm::Vec3 sumC = add(normalSums[c], normal);
				if (not isFinite(edgeAb)
					or not isFinite(edgeAc)
					or not isFinite(normal)
					or not isFinite(sumA)
					or not isFinite(sumB)
					or not isFinite(sumC)) {
					return oa::Status::error(
						oa::StatusCode::OutOfRange,
						"LunarLander3d terrain normals overflow FP32 geometry");
				}
				normalSums[a] = sumA;
				normalSums[b] = sumB;
				normalSums[c] = sumC;
			}
		}
	}

	outMesh.vertices.reserve(static_cast<oa::Usize>(vertexCount)
		+ static_cast<oa::Usize>(config.cellsX_) * config.cellsZ_ * 6U);
	for (oa::U32 index = 0U; index < vertexCount; ++index) {
		const oa::F32 height = positions[index].y;
		const oa::F32 shade =
			oa::clamp(0.30F + height * 0.025F, 0.22F, 0.38F);
		const oa::vlm::Vec3 normal = normalize(normalSums[index]);
		if (not isFinite(normal)) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				"LunarLander3d terrain normal is not representable as FP32");
		}
		outMesh.vertices.pushBack({
			positions[index],
			normal,
			{},
			{shade, shade * 0.96F, shade * 0.90F, 1.0F},
		});
	}

	const oa::vlm::Vec4 padColor{0.96F, 0.64F, 0.08F, 1.0F};
	for (oa::U32 z = 0U; z < config.cellsZ_; ++z) {
		for (oa::U32 x = 0U; x < config.cellsX_; ++x) {
			const double centerX = inTerrain.minX()
				+ (static_cast<double>(x) + 0.5) * config.cellSize_;
			const double centerZ = inTerrain.minZ()
				+ (static_cast<double>(z) + 0.5) * config.cellSize_;
			if (not inTerrain.isOnPad(centerX, centerZ)) continue;
			const oa::U32 v00 = z * verticesX + x;
			const oa::U32 v10 = v00 + 1U;
			const oa::U32 v01 = v00 + verticesX;
			const oa::U32 v11 = v01 + 1U;
			const oa::U32 triangles[6] = {
				v00, v11, v10, v00, v01, v11,
			};
			for (oa::U32 triangle = 0U; triangle < 2U; ++triangle) {
				oa::vlm::Vec3 a = positions[triangles[triangle * 3U]];
				oa::vlm::Vec3 b = positions[triangles[triangle * 3U + 1U]];
				oa::vlm::Vec3 c = positions[triangles[triangle * 3U + 2U]];
				a.y += 0.025F;
				b.y += 0.025F;
				c.y += 0.025F;
				const oa::vlm::Vec3 normal =
					normalize(cross(sub(b, a), sub(c, a)));
				const oa::U32 base =
					static_cast<oa::U32>(outMesh.vertices.size());
				outMesh.vertices.pushBack({a, normal, {}, padColor});
				outMesh.vertices.pushBack({b, normal, {}, padColor});
				outMesh.vertices.pushBack({c, normal, {}, padColor});
				outMesh.indices.pushBack(base);
				outMesh.indices.pushBack(base + 1U);
				outMesh.indices.pushBack(base + 2U);
			}
		}
	}
	outMesh.boundsDirty = true;
	return oa::Status::ok();
}

} // namespace

class LunarLander3dRenderSession::Impl {
public:
	LunarLander3dRenderConfig renderConfig;
	oa::Scene scene;
	oa::UniquePtr<oa::Renderer> renderer;

	[[nodiscard]] oa::Status initialize(
		oa::Engine& inEngine,
		const oa::LunarLander3dConfig& inLanderConfig,
		const oa::LunarTerrain& inTerrain,
		const LunarLander3dRenderConfig& inRenderConfig) {
		if (not inTerrain.isValid()) {
			return oa::Status::invalidArgument(
				"LunarLander3d renderer requires a valid terrain snapshot");
		}
		if (not inLanderConfig.validationError().empty()) {
			return oa::Status::invalidArgument(
				"LunarLander3d renderer requires a valid lander configuration");
		}
		if (not (inLanderConfig.terrain_ == inTerrain.config())) {
			return oa::Status::invalidArgument(
				"LunarLander3d renderer terrain/config snapshots do not match");
		}
		if (inTerrain.config().cellsX_ == 0U
			or inTerrain.config().cellsZ_ == 0U
			or inTerrain.config().cellsX_ > MaxTerrainCellsPerAxis
			or inTerrain.config().cellsZ_ > MaxTerrainCellsPerAxis) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				"LunarLander3d scene supports at most 32x32 terrain cells");
		}
		auto supportRepresentable = [](
			const oa::LunarSupportSphere& inSupport) {
			oa::vlm::Vec3 offset{};
			oa::F32 radius = 0.0F;
			return tryToVlm(inSupport.bodyOffset_, offset)
				and tryNarrowFinite(inSupport.radius_, radius)
				and oa::isFinite(radius * 1.35F);
		};
		for (const oa::LunarSupportSphere& support
			: inLanderConfig.bodySupports_) {
			if (not supportRepresentable(support)) {
				return oa::Status::error(
					oa::StatusCode::OutOfRange,
					"LunarLander3d body support is not representable as FP32 geometry");
			}
		}
		for (const oa::LunarSupportSphere& support
			: inLanderConfig.footSupports_) {
			if (not supportRepresentable(support)) {
				return oa::Status::error(
					oa::StatusCode::OutOfRange,
					"LunarLander3d foot support is not representable as FP32 geometry");
			}
		}

		renderConfig = inRenderConfig;
		oa::MeshData terrain;
		OA_RETURN_IF_ERROR(buildTerrainMesh(inTerrain, terrain));
		scene.meshes.pushBack({TerrainMeshId, "terrain", oa::move(terrain)});
		scene.meshes.pushBack({
			HullMeshId, "lander hull cube",
			coloredCube({0.08F, 0.62F, 0.96F, 1.0F})});
		scene.meshes.pushBack({
			LegMeshId, "lander leg cube",
			coloredCube({0.24F, 0.42F, 0.58F, 1.0F})});
		scene.meshes.pushBack({
			FootMeshId, "lander foot cube",
			coloredCube({0.82F, 0.90F, 0.96F, 1.0F})});
		scene.nodes.pushBack({
			TerrainNodeId, "terrain", {}, TerrainMeshId,
			oa::Transform{}, true});
		scene.nodes.pushBack({
			LanderRootNodeId, "lander", {}, {},
			oa::Transform{}, true});

		oa::U64 nextNodeId = FirstLanderPartNodeId;
		for (const oa::LunarSupportSphere& support
			: inLanderConfig.bodySupports_) {
			oa::vlm::Vec3 offset{};
			oa::F32 radius = 0.0F;
			if (not tryToVlm(support.bodyOffset_, offset)
				or not tryNarrowFinite(support.radius_, radius)) {
				return oa::Status::error(
					oa::StatusCode::OutOfRange,
					"LunarLander3d body support is not representable as FP32 geometry");
			}
			appendPartNode(
				scene, nextNodeId, "hull", HullMeshId, offset,
				{radius * 0.72F, radius * 0.72F, radius * 0.72F});
		}
		for (const oa::LunarSupportSphere& support
			: inLanderConfig.footSupports_) {
			oa::vlm::Vec3 foot{};
			oa::F32 radius = 0.0F;
			if (not tryToVlm(support.bodyOffset_, foot)
				or not tryNarrowFinite(support.radius_, radius)) {
				return oa::Status::error(
					oa::StatusCode::OutOfRange,
					"LunarLander3d foot support is not representable as FP32 geometry");
			}
			const oa::F32 attachmentY = -0.20F;
			const oa::F32 legHalfHeight =
				oa::max(0.05F, oa::abs(attachmentY - foot.y) * 0.5F);
			appendPartNode(
				scene, nextNodeId, "leg", LegMeshId,
				{foot.x * 0.72F, (attachmentY + foot.y) * 0.5F,
					foot.z * 0.72F},
				{0.065F, legHalfHeight, 0.065F});
			appendPartNode(
				scene, nextNodeId, "foot", FootMeshId, foot,
				{radius * 1.35F, radius * 0.35F, radius * 1.35F});
		}

		auto initialSnapshot = oa::FnScene::compile(scene);
		if (initialSnapshot.isError()) return initialSnapshot.getStatus();

		oa::RendererConfig rendererConfig;
		rendererConfig.width_ = renderConfig.width_;
		rendererConfig.height_ = renderConfig.height_;
		rendererConfig.targetSlotCount_ = renderConfig.targetSlotCount_;
		rendererConfig.maxVertexCount_ = static_cast<oa::U32>(
			initialSnapshot->vertices.size());
		rendererConfig.maxIndexCount_ = static_cast<oa::U32>(
			initialSnapshot->indices.size());
		rendererConfig.sampleCount_ = renderConfig.sampleCount_;
		rendererConfig.clearColor_ = renderConfig.clearColor_;
		auto rendererResult = oa::Renderer::create(inEngine, rendererConfig);
		if (not rendererResult.isOk()) return rendererResult.getStatus();
		renderer = oa::move(*rendererResult);
		return oa::Status::ok();
	}

	[[nodiscard]] oa::Status buildFrame(
		const oa::LunarLander3dState& inState) {
		if (not inState.isFinite()) {
			return oa::Status::invalidArgument(
				"LunarLander3d renderer state snapshot must be finite");
		}
		oa::vlm::Vec3 position{};
		oa::vlm::Quat orientation{};
		if (not tryToVlm(inState.position_, position)
			or not tryToVlm(inState.orientation_, orientation)) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				"LunarLander3d pose is not representable as an FP32 scene transform");
		}
		oa::SceneNode* root = oa::FnScene::findNode(scene, LanderRootNodeId);
		if (root == nullptr) {
			return oa::Status::error(
				oa::StatusCode::Internal,
				"LunarLander3d scene lost its stable lander root identity");
		}
		root->localTransform.setPosition(position);
		root->localTransform.setRotation(orientation);
		root->localTransform.setScale({1.0F, 1.0F, 1.0F});
		return oa::Status::ok();
	}
};

LunarLander3dRenderSession::~LunarLander3dRenderSession() = default;

oa::Result<oa::UniquePtr<LunarLander3dRenderSession>>
LunarLander3dRenderSession::create(
	oa::Engine& inEngine,
	const oa::LunarLander3dConfig& inLanderConfig,
	const oa::LunarTerrain& inTerrain,
	const LunarLander3dRenderConfig& inRenderConfig) {
	oa::UniquePtr<LunarLander3dRenderSession> session(
		new LunarLander3dRenderSession());
	session->impl_ = oa::makeUnique<Impl>();
	const oa::Status status = session->impl_->initialize(
		inEngine,
		inLanderConfig,
		inTerrain,
		inRenderConfig);
	if (not status.isOk()) return status;
	return oa::move(session);
}

oa::Status LunarLander3dRenderSession::beginFrame(
	const oa::LunarLander3dState& inState,
	const oa::Camera& inCamera) {
	if (not impl_ or not impl_->renderer) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	OA_RETURN_IF_ERROR(impl_->buildFrame(inState));
	return impl_->renderer->beginFrame(impl_->scene, inCamera);
}

oa::Result<oa::RenderFrame> LunarLander3dRenderSession::submitFrame(
	oa::Span<const oa::Event> inDependencies) {
	if (not impl_ or not impl_->renderer) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	return impl_->renderer->submitFrame(inDependencies);
}

oa::Status LunarLander3dRenderSession::cancelFrame() {
	if (not impl_ or not impl_->renderer) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	return impl_->renderer->cancelFrame();
}

oa::Result<oa::RenderReadback>
LunarLander3dRenderSession::consumeReadback(
	const oa::RenderFrame& inFrame) {
	if (not impl_ or not impl_->renderer) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	return impl_->renderer->consumeReadback(inFrame);
}

oa::Status LunarLander3dRenderSession::markConsumed(
	const oa::RenderFrame& inFrame,
	const oa::Event& inConsumer) {
	if (not impl_ or not impl_->renderer) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	return impl_->renderer->markConsumed(inFrame, inConsumer);
}

oa::Status LunarLander3dRenderSession::abandonFrame(
	const oa::RenderFrame& inFrame) {
	if (not impl_ or not impl_->renderer) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	return impl_->renderer->abandonFrame(inFrame);
}

oa::Status LunarLander3dRenderSession::collect() {
	if (not impl_ or not impl_->renderer) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	return impl_->renderer->collect();
}

oa::Status LunarLander3dRenderSession::resize(
	oa::U32 inWidth,
	oa::U32 inHeight) {
	if (not impl_ or not impl_->renderer) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	const oa::Status status = impl_->renderer->resize(inWidth, inHeight);
	if (status.isOk()) {
		impl_->renderConfig.width_ = inWidth;
		impl_->renderConfig.height_ = inHeight;
	}
	return status;
}

oa::Status LunarLander3dRenderSession::close() {
	if (not impl_ or not impl_->renderer) return oa::Status::ok();
	const oa::Status status = impl_->renderer->close();
	if (status.isOk()) {
		impl_->renderer.reset();
	}
	return status;
}

oa::Camera LunarLander3dRenderSession::defaultCamera(
	oa::U32 inWidth,
	oa::U32 inHeight) {
	oa::Camera camera;
	const oa::F32 aspect = inHeight == 0U
		? 1.0F
		: static_cast<oa::F32>(inWidth) / static_cast<oa::F32>(inHeight);
	oa::FnCamera::initPerspective(
		camera,
		{16.0F, 12.0F, 16.0F},
		{0.0F, 2.0F, 0.0F},
		48.0F,
		aspect,
		0.1F,
		100.0F);
	return camera;
}
