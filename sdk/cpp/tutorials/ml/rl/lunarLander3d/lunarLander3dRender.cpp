#include "lunarLander3dRender.h"

#include <oa/render/fnMesh.h>


namespace {

constexpr oa::U32 MaxTerrainCellsPerAxis = 32U;
constexpr oa::U32 VerticesPerBox = 24U;
constexpr oa::U32 IndicesPerBox = 36U;

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

[[nodiscard]] bool bodyPointToWorld(
	const oa::LunarLander3dState& inState,
	const oa::vlm::Vec3& inBodyPoint,
	oa::vlm::Vec3& outWorldPoint) noexcept {
	const oa::vlm::DVec3 rotated = inState.orientation_.rotate({
		static_cast<double>(inBodyPoint.x),
		static_cast<double>(inBodyPoint.y),
		static_cast<double>(inBodyPoint.z),
	});
	return tryToVlm(inState.position_ + rotated, outWorldPoint);
}

[[nodiscard]] bool bodyDirectionToWorld(
	const oa::LunarLander3dState& inState,
	const oa::vlm::Vec3& inBodyDirection,
	oa::vlm::Vec3& outWorldDirection) noexcept {
	oa::vlm::Vec3 rotated{};
	if (not tryToVlm(inState.orientation_.rotate({
		static_cast<double>(inBodyDirection.x),
		static_cast<double>(inBodyDirection.y),
		static_cast<double>(inBodyDirection.z),
	}), rotated)) {
		return false;
	}
	outWorldDirection = normalize(rotated);
	return isFinite(outWorldDirection);
}

class LunarGeometryWriter {
public:
	explicit LunarGeometryWriter(oa::MeshData& inMesh) noexcept
		: mesh_(inMesh) {}

	[[nodiscard]] oa::Status appendBodyBox(
		const oa::LunarLander3dState& inState,
		const oa::vlm::Vec3& inCenter,
		const oa::vlm::Vec3& inHalfExtent,
		const oa::vlm::Vec4& inColor) {
		if (not isFinite(inCenter) or not isFinite(inHalfExtent)) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				"LunarLander3d body geometry is not representable as FP32");
		}
		static constexpr oa::Array<oa::Array<oa::U32, 4U>, 6U> faces{
			oa::Array<oa::U32, 4U>{1U, 5U, 7U, 3U},
			oa::Array<oa::U32, 4U>{4U, 0U, 2U, 6U},
			oa::Array<oa::U32, 4U>{2U, 3U, 7U, 6U},
			oa::Array<oa::U32, 4U>{4U, 5U, 1U, 0U},
			oa::Array<oa::U32, 4U>{5U, 4U, 6U, 7U},
			oa::Array<oa::U32, 4U>{0U, 1U, 3U, 2U},
		};
		static constexpr oa::Array<oa::vlm::Vec3, 6U> Normals{
			oa::vlm::Vec3{1.0F, 0.0F, 0.0F},
			oa::vlm::Vec3{-1.0F, 0.0F, 0.0F},
			oa::vlm::Vec3{0.0F, 1.0F, 0.0F},
			oa::vlm::Vec3{0.0F, -1.0F, 0.0F},
			oa::vlm::Vec3{0.0F, 0.0F, 1.0F},
			oa::vlm::Vec3{0.0F, 0.0F, -1.0F},
		};
		oa::Array<oa::vlm::Vec3, 8U> corners{};
		for (oa::U32 corner = 0U; corner < corners.size(); ++corner) {
			const oa::vlm::Vec3 offset{
				(corner & 1U) ? inHalfExtent.x : -inHalfExtent.x,
				(corner & 2U) ? inHalfExtent.y : -inHalfExtent.y,
				(corner & 4U) ? inHalfExtent.z : -inHalfExtent.z,
			};
			const oa::vlm::Vec3 bodyPoint = add(inCenter, offset);
			if (not isFinite(bodyPoint)
				or not bodyPointToWorld(
					inState, bodyPoint, corners[corner])) {
				return oa::Status::error(
					oa::StatusCode::OutOfRange,
					"LunarLander3d world geometry is not representable as FP32");
			}
		}
		for (oa::U32 face = 0U; face < faces.size(); ++face) {
			const oa::U32 base = static_cast<oa::U32>(mesh_.vertices.size());
			oa::vlm::Vec3 normal{};
			if (not bodyDirectionToWorld(inState, Normals[face], normal)) {
				return oa::Status::error(
					oa::StatusCode::OutOfRange,
					"LunarLander3d world normal is not representable as FP32");
			}
			for (oa::U32 corner : faces[face]) {
				mesh_.vertices.pushBack({
					corners[corner],
					normal,
					{},
					inColor,
				});
			}
			static constexpr oa::U32 FaceIndices[6] = {
				0U, 1U, 2U, 2U, 3U, 0U,
			};
			for (oa::U32 index : FaceIndices) {
				mesh_.indices.pushBack(base + index);
			}
		}
		return oa::Status::ok();
	}

private:
	oa::MeshData& mesh_;
};

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
	oa::LunarLander3dConfig LanderConfig;
	LunarLander3dRenderConfig renderConfig;
	oa::MeshData TerrainMesh;
	oa::MeshData frameMesh;
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

		LanderConfig = inLanderConfig;
		renderConfig = inRenderConfig;
		OA_RETURN_IF_ERROR(buildTerrainMesh(inTerrain, TerrainMesh));

		const oa::U64 boxCount =
			static_cast<oa::U64>(LanderConfig.bodySupports_.size())
			+ static_cast<oa::U64>(LanderConfig.footSupports_.size()) * 2U;
		const oa::U64 vertexCapacity =
			static_cast<oa::U64>(TerrainMesh.vertices.size())
			+ boxCount * VerticesPerBox;
		const oa::U64 indexCapacity =
			static_cast<oa::U64>(TerrainMesh.indices.size())
			+ boxCount * IndicesPerBox;
		if (vertexCapacity > oa::Limits<oa::U32>::max()
			or indexCapacity > oa::Limits<oa::U32>::max()) {
			return oa::Status::error(
				oa::StatusCode::OutOfRange,
				"LunarLander3d scene exceeds renderer capacity representation");
		}

		oa::RendererConfig rendererConfig;
		rendererConfig.width_ = renderConfig.width_;
		rendererConfig.height_ = renderConfig.height_;
		rendererConfig.targetSlotCount_ = renderConfig.targetSlotCount_;
		rendererConfig.maxVertexCount_ =
			static_cast<oa::U32>(vertexCapacity);
		rendererConfig.maxIndexCount_ =
			static_cast<oa::U32>(indexCapacity);
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
		frameMesh = TerrainMesh;
		LunarGeometryWriter writer(frameMesh);
		const oa::vlm::Vec4 hullColor{0.08F, 0.62F, 0.96F, 1.0F};
		const oa::vlm::Vec4 legColor{0.24F, 0.42F, 0.58F, 1.0F};
		const oa::vlm::Vec4 footColor{0.82F, 0.90F, 0.96F, 1.0F};
		for (const oa::LunarSupportSphere& support
			: LanderConfig.bodySupports_) {
			oa::vlm::Vec3 offset{};
			oa::F32 radius = 0.0F;
			if (not tryToVlm(support.bodyOffset_, offset)
				or not tryNarrowFinite(support.radius_, radius)) {
				return oa::Status::error(
					oa::StatusCode::OutOfRange,
					"LunarLander3d body support is not representable as FP32 geometry");
			}
			OA_RETURN_IF_ERROR(writer.appendBodyBox(
				inState,
				offset,
				{radius * 0.72F, radius * 0.72F, radius * 0.72F},
				hullColor));
		}
		for (const oa::LunarSupportSphere& support
			: LanderConfig.footSupports_) {
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
			const oa::vlm::Vec3 legCenter{
				foot.x * 0.72F,
				(attachmentY + foot.y) * 0.5F,
				foot.z * 0.72F,
			};
			OA_RETURN_IF_ERROR(writer.appendBodyBox(
				inState,
				legCenter,
				{0.065F, legHalfHeight, 0.065F},
				legColor));
			OA_RETURN_IF_ERROR(writer.appendBodyBox(
				inState,
				foot,
				{radius * 1.35F, radius * 0.35F, radius * 1.35F},
				footColor));
		}
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
	const oa::CameraState& inCamera) {
	if (not impl_ or not impl_->renderer) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	OA_RETURN_IF_ERROR(impl_->buildFrame(inState));
	return impl_->renderer->beginFrame(impl_->frameMesh, inCamera);
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

oa::CameraState LunarLander3dRenderSession::defaultCamera(
	oa::U32 inWidth,
	oa::U32 inHeight) {
	oa::CameraState camera;
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
