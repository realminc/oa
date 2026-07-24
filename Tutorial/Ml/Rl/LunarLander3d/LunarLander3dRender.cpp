#include "LunarLander3dRender.h"

#include <Oa/Core/Log.h>
#include <Oa/Render/FnMesh.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Engine/BorrowedServiceRetirement.h>
#include <Oa/Runtime/GraphicsStream.h>
#include <Oa/Runtime/OaVma.h>
#include <Oa/Runtime/ShaderProvider.h>
#include <Oa/Runtime/Stream.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

namespace {

constexpr VkFormat ColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat DepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr OaU32 MaxTerrainCellsPerAxis = 32U;
constexpr OaU32 MaxTargetSlots = 4U;
constexpr OaU32 LanderBoxCount = 11U;
constexpr OaU32 VerticesPerBox = 24U;
constexpr OaU32 IndicesPerBox = 36U;

struct LunarPushConstants {
	VlmMat4 ViewProjection;
	VlmVec4 LightDirectionAmbient;
};
static_assert(sizeof(LunarPushConstants) == 80U);

enum class LunarSlotState : OaU8 {
	Free,
	Recording,
	Submitted,
	Retired,
};

struct LunarTarget {
	VkImage ColorImage = VK_NULL_HANDLE;
	VkImageView ColorView = VK_NULL_HANDLE;
	OaVmaAllocation ColorAllocation = VK_NULL_HANDLE;
	VkImageLayout ColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImage DepthImage = VK_NULL_HANDLE;
	VkImageView DepthView = VK_NULL_HANDLE;
	OaVmaAllocation DepthAllocation = VK_NULL_HANDLE;
	VkImageLayout DepthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	OaVkBuffer ColorReadback;
	OaVkBuffer DepthReadback;
};

struct LunarSlot {
	LunarSlotState State = LunarSlotState::Free;
	OaU64 Generation = 0U;
	LunarTarget Target;
	OaVkBuffer VertexBuffer;
	OaVkBuffer IndexBuffer;
	OaU32 IndexCount = 0U;
	std::optional<OaGraphicsStreamLease> StreamLease;
	OaEvent Producer;
	OaEvent Consumer;
};

[[nodiscard]] OaU64 NextGeneration(OaU64 InGeneration) noexcept {
	++InGeneration;
	return InGeneration == 0U ? 1U : InGeneration;
}

[[nodiscard]] bool CheckedMultiply(
	OaU64 InA, OaU64 InB, OaU64& OutResult) noexcept {
	if (InA != 0U && InB > std::numeric_limits<OaU64>::max() / InA) {
		return false;
	}
	OutResult = InA * InB;
	return true;
}

[[nodiscard]] bool IsFinite(const VlmMat4& InMatrix) noexcept {
	for (OaU32 row = 0U; row < 4U; ++row) {
		for (OaU32 column = 0U; column < 4U; ++column) {
			if (not std::isfinite(InMatrix.M[row][column])) return false;
		}
	}
	return true;
}

[[nodiscard]] bool IsFinite(const VlmVec3& InVector) noexcept {
	return std::isfinite(InVector.X)
		and std::isfinite(InVector.Y)
		and std::isfinite(InVector.Z);
}

[[nodiscard]] bool TryNarrowFinite(double InValue, OaF32& OutValue) noexcept {
	if (not std::isfinite(InValue)
		or InValue > static_cast<double>(std::numeric_limits<OaF32>::max())
		or InValue < -static_cast<double>(std::numeric_limits<OaF32>::max())) {
		return false;
	}
	const OaF32 converted = static_cast<OaF32>(InValue);
	if (not std::isfinite(converted)
		or (InValue != 0.0 and converted == 0.0F)
		or (converted != 0.0F and std::fpclassify(converted) == FP_SUBNORMAL)) {
		return false;
	}
	OutValue = converted;
	return true;
}

[[nodiscard]] bool TryToVlm(
	const OaLunarVec3& InValue,
	VlmVec3& OutValue) noexcept {
	return TryNarrowFinite(InValue.ComponentX_, OutValue.X)
		and TryNarrowFinite(InValue.ComponentY_, OutValue.Y)
		and TryNarrowFinite(InValue.ComponentZ_, OutValue.Z);
}

[[nodiscard]] OaStatus ValidateTargetExtent(
	OaEngine& InEngine,
	const OaVkInstanceTable& InInstanceTable,
	OaU32 InWidth,
	OaU32 InHeight) {
	if (InWidth == 0U or InHeight == 0U) {
		return OaStatus::InvalidArgument(
			"LunarLander3d renderer target dimensions must be non-zero");
	}
	const VkPhysicalDevice physicalDevice =
		static_cast<VkPhysicalDevice>(InEngine.Device.PhysicalDevice);
	if (physicalDevice == VK_NULL_HANDLE
		or InInstanceTable.vkGetPhysicalDeviceProperties == nullptr
		or InInstanceTable.vkGetPhysicalDeviceImageFormatProperties == nullptr) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d renderer requires physical-device capability queries");
	}

	VkPhysicalDeviceProperties properties{};
	InInstanceTable.vkGetPhysicalDeviceProperties(physicalDevice, &properties);
	const auto& limits = properties.limits;
	if (InWidth > limits.maxImageDimension2D
		or InHeight > limits.maxImageDimension2D
		or InWidth > limits.maxFramebufferWidth
		or InHeight > limits.maxFramebufferHeight
		or InWidth > limits.maxViewportDimensions[0]
		or InHeight > limits.maxViewportDimensions[1]
		or static_cast<OaF32>(InWidth) > limits.viewportBoundsRange[1]
		or static_cast<OaF32>(InHeight) > limits.viewportBoundsRange[1]) {
		return OaStatus::Error(
			OaStatusCode::OutOfRange,
			"LunarLander3d renderer target exceeds queried image, framebuffer, or viewport limits");
	}
	OaU64 pixelCount = 0U;
	OaU64 readbackBytes = 0U;
	if (not CheckedMultiply(InWidth, InHeight, pixelCount)
		or not CheckedMultiply(pixelCount, 4U, readbackBytes)) {
		return OaStatus::Error(
			OaStatusCode::OutOfRange,
			"LunarLander3d renderer readback size overflows");
	}
	const OaU32 apiVersion = InEngine.Device.Info.Software.ApiVersionPacked;
	const bool core13 = VK_API_VERSION_MAJOR(apiVersion) > 1U
		or (VK_API_VERSION_MAJOR(apiVersion) == 1U
			and VK_API_VERSION_MINOR(apiVersion) >= 3U);
	if (core13) {
		if (InInstanceTable.vkGetPhysicalDeviceProperties2 == nullptr) {
			return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"LunarLander3d renderer requires Vulkan 1.3 property queries");
		}
		VkPhysicalDeviceMaintenance4Properties maintenance4{};
		maintenance4.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES;
		VkPhysicalDeviceProperties2 properties2{};
		properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		properties2.pNext = &maintenance4;
		InInstanceTable.vkGetPhysicalDeviceProperties2(
			physicalDevice, &properties2);
		if (readbackBytes > maintenance4.maxBufferSize) {
			return OaStatus::Error(
				OaStatusCode::OutOfRange,
				"LunarLander3d renderer readback buffer exceeds the queried maintenance4 limit");
		}
	}

	struct ImageRequirement {
		VkFormat Format;
		VkImageUsageFlags Usage;
	};
	const ImageRequirement requirements[2] = {
		{
			ColorFormat,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
				| VK_IMAGE_USAGE_TRANSFER_SRC_BIT
				| VK_IMAGE_USAGE_SAMPLED_BIT,
		},
		{
			DepthFormat,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
				| VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		},
	};
	for (const ImageRequirement& requirement : requirements) {
		VkImageFormatProperties imageProperties{};
		const VkResult result =
			InInstanceTable.vkGetPhysicalDeviceImageFormatProperties(
				physicalDevice,
				requirement.Format,
				VK_IMAGE_TYPE_2D,
				VK_IMAGE_TILING_OPTIMAL,
				requirement.Usage,
				0U,
				&imageProperties);
		if (result == VK_ERROR_FORMAT_NOT_SUPPORTED) {
			return OaStatus::Error(
				OaStatusCode::Unavailable,
				"LunarLander3d renderer attachment/readback image tuple is unsupported");
		}
		if (result != VK_SUCCESS) {
			return OaStatus::Error(
				OaStatusCode::VulkanError,
				"LunarLander3d renderer image-format capability query failed");
		}
		if (InWidth > imageProperties.maxExtent.width
			or InHeight > imageProperties.maxExtent.height
			or imageProperties.maxMipLevels < 1U
			or imageProperties.maxArrayLayers < 1U
			or (imageProperties.sampleCounts & VK_SAMPLE_COUNT_1_BIT) == 0U) {
			return OaStatus::Error(
				OaStatusCode::OutOfRange,
				"LunarLander3d renderer target exceeds the exact image tuple limits");
		}
	}
	return OaStatus::Ok();
}

[[nodiscard]] VlmVec3 Add(const VlmVec3& InA, const VlmVec3& InB) noexcept {
	return {InA.X + InB.X, InA.Y + InB.Y, InA.Z + InB.Z};
}

[[nodiscard]] VlmVec3 Sub(const VlmVec3& InA, const VlmVec3& InB) noexcept {
	return {InA.X - InB.X, InA.Y - InB.Y, InA.Z - InB.Z};
}

[[nodiscard]] VlmVec3 Scale(const VlmVec3& InValue, OaF32 InScale) noexcept {
	return {InValue.X * InScale, InValue.Y * InScale, InValue.Z * InScale};
}

[[nodiscard]] VlmVec3 Cross(const VlmVec3& InA, const VlmVec3& InB) noexcept {
	return {
		InA.Y * InB.Z - InA.Z * InB.Y,
		InA.Z * InB.X - InA.X * InB.Z,
		InA.X * InB.Y - InA.Y * InB.X,
	};
}

[[nodiscard]] VlmVec3 Normalize(const VlmVec3& InValue) noexcept {
	const OaF32 maximumComponent = std::max({
		std::abs(InValue.X), std::abs(InValue.Y), std::abs(InValue.Z)});
	if (not std::isfinite(maximumComponent)
		or maximumComponent <= 1.0e-6F) {
		return {0.0F, 1.0F, 0.0F};
	}
	const VlmVec3 scaled = Scale(InValue, 1.0F / maximumComponent);
	const OaF32 lengthSquared =
		scaled.X * scaled.X + scaled.Y * scaled.Y + scaled.Z * scaled.Z;
	if (not std::isfinite(lengthSquared) or lengthSquared <= 0.0F) {
		return {0.0F, 1.0F, 0.0F};
	}
	return Scale(scaled, 1.0F / std::sqrt(lengthSquared));
}

[[nodiscard]] bool BodyPointToWorld(
	const OaLunarLander3dState& InState,
	const VlmVec3& InBodyPoint,
	VlmVec3& OutWorldPoint) noexcept {
	const OaLunarVec3 rotated = InState.Orientation_.Rotate({
		static_cast<double>(InBodyPoint.X),
		static_cast<double>(InBodyPoint.Y),
		static_cast<double>(InBodyPoint.Z),
	});
	return TryToVlm(InState.Position_ + rotated, OutWorldPoint);
}

[[nodiscard]] bool BodyDirectionToWorld(
	const OaLunarLander3dState& InState,
	const VlmVec3& InBodyDirection,
	VlmVec3& OutWorldDirection) noexcept {
	VlmVec3 rotated{};
	if (not TryToVlm(InState.Orientation_.Rotate({
		static_cast<double>(InBodyDirection.X),
		static_cast<double>(InBodyDirection.Y),
		static_cast<double>(InBodyDirection.Z),
	}), rotated)) {
		return false;
	}
	OutWorldDirection = Normalize(rotated);
	return IsFinite(OutWorldDirection);
}

class GeometryWriter {
public:
	GeometryWriter(
		OaMeshVertex* InVertices,
		OaU32 InVertexCapacity,
		OaU32* InIndices,
		OaU32 InIndexCapacity) noexcept
		: Vertices_(InVertices)
		, VertexCapacity_(InVertexCapacity)
		, Indices_(InIndices)
		, IndexCapacity_(InIndexCapacity) {}

	[[nodiscard]] bool AppendVertex(const OaMeshVertex& InVertex) noexcept {
		if (VertexCount_ >= VertexCapacity_) return false;
		Vertices_[VertexCount_++] = InVertex;
		return true;
	}

	[[nodiscard]] bool AppendIndex(OaU32 InIndex) noexcept {
		if (IndexCount_ >= IndexCapacity_) return false;
		Indices_[IndexCount_++] = InIndex;
		return true;
	}

	[[nodiscard]] bool AppendCached(
		const std::vector<OaMeshVertex>& InVertices,
		const std::vector<OaU32>& InIndices) noexcept {
		if (InVertices.size() > VertexCapacity_
			or InIndices.size() > IndexCapacity_) return false;
		std::memcpy(
			Vertices_, InVertices.data(),
			InVertices.size() * sizeof(OaMeshVertex));
		std::memcpy(
			Indices_, InIndices.data(),
			InIndices.size() * sizeof(OaU32));
		VertexCount_ = static_cast<OaU32>(InVertices.size());
		IndexCount_ = static_cast<OaU32>(InIndices.size());
		return true;
	}

	[[nodiscard]] OaStatus AppendBodyBox(
		const OaLunarLander3dState& InState,
		const VlmVec3& InCenter,
		const VlmVec3& InHalfExtent,
		const VlmVec4& InColor) noexcept {
		if (not IsFinite(InCenter) or not IsFinite(InHalfExtent)) {
			return OaStatus::Error(
				OaStatusCode::OutOfRange,
				"LunarLander3d body geometry is not representable as FP32");
		}
		static constexpr std::array<std::array<OaU32, 4U>, 6U> Faces{{
			{{1U, 5U, 7U, 3U}}, // +X
			{{4U, 0U, 2U, 6U}}, // -X
			{{2U, 3U, 7U, 6U}}, // +Y
			{{4U, 5U, 1U, 0U}}, // -Y
			{{5U, 4U, 6U, 7U}}, // +Z
			{{0U, 1U, 3U, 2U}}, // -Z
		}};
		static constexpr std::array<VlmVec3, 6U> Normals{{
			{1.0F, 0.0F, 0.0F}, {-1.0F, 0.0F, 0.0F},
			{0.0F, 1.0F, 0.0F}, {0.0F, -1.0F, 0.0F},
			{0.0F, 0.0F, 1.0F}, {0.0F, 0.0F, -1.0F},
		}};
		std::array<VlmVec3, 8U> corners{};
		for (OaU32 corner = 0U; corner < 8U; ++corner) {
			const VlmVec3 offset{
				(corner & 1U) ? InHalfExtent.X : -InHalfExtent.X,
				(corner & 2U) ? InHalfExtent.Y : -InHalfExtent.Y,
				(corner & 4U) ? InHalfExtent.Z : -InHalfExtent.Z,
			};
			const VlmVec3 bodyPoint = Add(InCenter, offset);
			if (not IsFinite(bodyPoint)
				or not BodyPointToWorld(InState, bodyPoint, corners[corner])) {
				return OaStatus::Error(
					OaStatusCode::OutOfRange,
					"LunarLander3d world geometry is not representable as FP32");
			}
		}
		for (OaU32 face = 0U; face < Faces.size(); ++face) {
			const OaU32 base = VertexCount_;
			VlmVec3 normal{};
			if (not BodyDirectionToWorld(InState, Normals[face], normal)) {
				return OaStatus::Error(
					OaStatusCode::OutOfRange,
					"LunarLander3d world normal is not representable as FP32");
			}
			for (OaU32 corner : Faces[face]) {
				if (not AppendVertex({corners[corner], normal, {}, InColor})) {
					return OaStatus::Error(
						OaStatusCode::ResourceExhausted,
						"LunarLander3d lander geometry exceeds its bounded slot");
				}
			}
			static constexpr OaU32 FaceIndices[6] = {0U, 1U, 2U, 2U, 3U, 0U};
			for (OaU32 index : FaceIndices) {
				if (not AppendIndex(base + index)) {
					return OaStatus::Error(
						OaStatusCode::ResourceExhausted,
						"LunarLander3d lander indices exceed their bounded slot");
				}
			}
		}
		return OaStatus::Ok();
	}

	[[nodiscard]] OaU32 VertexCount() const noexcept { return VertexCount_; }
	[[nodiscard]] OaU32 IndexCount() const noexcept { return IndexCount_; }

private:
	OaMeshVertex* Vertices_ = nullptr;
	OaU32 VertexCapacity_ = 0U;
	OaU32* Indices_ = nullptr;
	OaU32 IndexCapacity_ = 0U;
	OaU32 VertexCount_ = 0U;
	OaU32 IndexCount_ = 0U;
};

[[nodiscard]] VkShaderModule CreateShaderModule(
	const OaVkDeviceTable& InDeviceTable,
	VkDevice InDevice,
	const OaSpvEntry& InSpirv) noexcept {
	VkShaderModuleCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	info.codeSize = InSpirv.Size;
	info.pCode = reinterpret_cast<const OaU32*>(InSpirv.Data);
	VkShaderModule module = VK_NULL_HANDLE;
	return InDeviceTable.vkCreateShaderModule(
		InDevice, &info, nullptr, &module) == VK_SUCCESS
		? module : VK_NULL_HANDLE;
}

[[nodiscard]] OaResult<OaVkBuffer> CreateMappedVertexOrIndexBuffer(
	OaEngine& InEngine,
	OaU64 InSize,
	VkBufferUsageFlags InUsage) {
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = InSize;
	bufferInfo.usage = InUsage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	OaVmaAllocationCreateInfo allocationInfo{};
	allocationInfo.usage = OA_VMA_MEMORY_USAGE_CPU_TO_GPU;
	allocationInfo.flags = OA_VMA_ALLOCATION_CREATE_MAPPED_BIT
		| OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

	VkBuffer buffer = VK_NULL_HANDLE;
	OaVmaAllocation allocation = VK_NULL_HANDLE;
	OaVmaAllocationInfo allocationResult{};
	if (OaVmaCreateBuffer(
			static_cast<OaVmaAllocator>(InEngine.Allocator.Allocator),
			&bufferInfo, &allocationInfo, &buffer, &allocation,
			&allocationResult) != VK_SUCCESS) {
		return OaStatus::Error(
			OaStatusCode::OutOfMemory,
			"LunarLander3d renderer mapped geometry allocation failed");
	}

	OaVkBuffer result;
	result.Buffer = buffer;
	result.Allocation = allocation;
	result.AllocatorIdentity = InEngine.Allocator.Allocator;
	result.Size = InSize;
	result.Capacity = InSize;
	result.MappedPtr = allocationResult.pMappedData;
	result.Placement = OaMemoryPlacement::HostUpload;
	return result;
}

void DestroyTarget(
	OaEngine& InEngine,
	const OaVkDeviceTable& InDeviceTable,
	LunarTarget& InOutTarget) noexcept {
	const VkDevice device = static_cast<VkDevice>(InEngine.Device.Device);
	if (InOutTarget.ColorView != VK_NULL_HANDLE) {
		InDeviceTable.vkDestroyImageView(
			device, InOutTarget.ColorView, nullptr);
	}
	if (InOutTarget.DepthView != VK_NULL_HANDLE) {
		InDeviceTable.vkDestroyImageView(
			device, InOutTarget.DepthView, nullptr);
	}
	if (InOutTarget.ColorImage != VK_NULL_HANDLE) {
		OaVmaDestroyImage(
			static_cast<OaVmaAllocator>(InEngine.Allocator.Allocator),
			InOutTarget.ColorImage, InOutTarget.ColorAllocation);
	}
	if (InOutTarget.DepthImage != VK_NULL_HANDLE) {
		OaVmaDestroyImage(
			static_cast<OaVmaAllocator>(InEngine.Allocator.Allocator),
			InOutTarget.DepthImage, InOutTarget.DepthAllocation);
	}
	InEngine.Allocator.Free(InOutTarget.ColorReadback);
	InEngine.Allocator.Free(InOutTarget.DepthReadback);
	InOutTarget = {};
}

[[nodiscard]] OaStatus CreateImage(
	OaEngine& InEngine,
	const OaVkDeviceTable& InDeviceTable,
	OaU32 InWidth,
	OaU32 InHeight,
	VkFormat InFormat,
	VkImageUsageFlags InUsage,
	VkImageAspectFlags InAspect,
	VkImage& OutImage,
	VkImageView& OutView,
	OaVmaAllocation& OutAllocation) {
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = InFormat;
	imageInfo.extent = {InWidth, InHeight, 1U};
	imageInfo.mipLevels = 1U;
	imageInfo.arrayLayers = 1U;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = InUsage;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	OaVmaAllocationCreateInfo allocationInfo{};
	allocationInfo.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;
	if (OaVmaCreateImage(
			static_cast<OaVmaAllocator>(InEngine.Allocator.Allocator),
			&imageInfo, &allocationInfo, &OutImage, &OutAllocation,
			nullptr) != VK_SUCCESS) {
		return OaStatus::Error(
			OaStatusCode::OutOfMemory,
			"LunarLander3d renderer target image allocation failed");
	}

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = OutImage;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = InFormat;
	viewInfo.subresourceRange.aspectMask = InAspect;
	viewInfo.subresourceRange.levelCount = 1U;
	viewInfo.subresourceRange.layerCount = 1U;
	if (InDeviceTable.vkCreateImageView(
			static_cast<VkDevice>(InEngine.Device.Device),
			&viewInfo, nullptr, &OutView) != VK_SUCCESS) {
		OaVmaDestroyImage(
			static_cast<OaVmaAllocator>(InEngine.Allocator.Allocator),
			OutImage, OutAllocation);
		OutImage = VK_NULL_HANDLE;
		OutAllocation = VK_NULL_HANDLE;
		return OaStatus::Error(
			OaStatusCode::VulkanError,
			"LunarLander3d renderer image-view creation failed");
	}
	return OaStatus::Ok();
}

[[nodiscard]] OaStatus CreateTarget(
	OaEngine& InEngine,
	const OaVkDeviceTable& InDeviceTable,
	OaU32 InWidth,
	OaU32 InHeight,
	LunarTarget& OutTarget) {
	OaU64 pixelCount = 0U;
	OaU64 colorBytes = 0U;
	OaU64 depthBytes = 0U;
	if (not CheckedMultiply(InWidth, InHeight, pixelCount)
		or not CheckedMultiply(pixelCount, 4U, colorBytes)
		or not CheckedMultiply(pixelCount, sizeof(OaF32), depthBytes)) {
		return OaStatus::InvalidArgument(
			"LunarLander3d renderer target size overflows");
	}

	OaStatus status = CreateImage(
		InEngine, InDeviceTable, InWidth, InHeight, ColorFormat,
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
			| VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT,
		OutTarget.ColorImage, OutTarget.ColorView,
		OutTarget.ColorAllocation);
	if (not status.IsOk()) return status;
	status = CreateImage(
		InEngine, InDeviceTable, InWidth, InHeight, DepthFormat,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
			| VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		VK_IMAGE_ASPECT_DEPTH_BIT,
		OutTarget.DepthImage, OutTarget.DepthView,
		OutTarget.DepthAllocation);
	if (not status.IsOk()) {
		DestroyTarget(InEngine, InDeviceTable, OutTarget);
		return status;
	}
	auto colorReadback = InEngine.Allocator.AllocHostReadback(colorBytes);
	if (not colorReadback.IsOk()) {
		DestroyTarget(InEngine, InDeviceTable, OutTarget);
		return colorReadback.GetStatus();
	}
	OutTarget.ColorReadback = OaStdMove(*colorReadback);
	auto depthReadback = InEngine.Allocator.AllocHostReadback(depthBytes);
	if (not depthReadback.IsOk()) {
		DestroyTarget(InEngine, InDeviceTable, OutTarget);
		return depthReadback.GetStatus();
	}
	OutTarget.DepthReadback = OaStdMove(*depthReadback);
	return OaStatus::Ok();
}

} // namespace

class OaLunarLander3dRenderSession::Impl {
public:
	OaEngine* Engine = nullptr;
	OaVkInstanceTable InstanceTable{};
	OaVkDeviceTable DeviceTable{};
	OaLunarLander3dConfig LanderConfig;
	OaLunarLander3dRenderConfig RenderConfig;
	std::vector<OaMeshVertex> TerrainVertices;
	std::vector<OaU32> TerrainIndices;
	std::vector<LunarSlot> Slots;
	OaU32 VertexCapacity = 0U;
	OaU32 IndexCapacity = 0U;
	OaU32 ActiveSlot = std::numeric_limits<OaU32>::max();
	OaU64 TargetGeneration = 1U;
	VkPipelineLayout PipelineLayout = VK_NULL_HANDLE;
	VkPipeline Pipeline = VK_NULL_HANDLE;
	bool Closed = false;

	[[nodiscard]] OaStatus Initialize(
		OaEngine& InEngine,
		const OaLunarLander3dConfig& InLanderConfig,
		const OaLunarTerrain& InTerrain,
		const OaLunarLander3dRenderConfig& InRenderConfig);
	[[nodiscard]] OaStatus ValidateCapabilities() const;
	[[nodiscard]] OaStatus BuildTerrainSnapshot(const OaLunarTerrain& InTerrain);
	[[nodiscard]] OaStatus CreatePipeline();
	[[nodiscard]] OaStatus CreateSlots();
	[[nodiscard]] OaStatus WriteFrameGeometry(
		LunarSlot& InSlot,
		const OaLunarLander3dState& InState);
	[[nodiscard]] OaStatus RecordFrame(
		LunarSlot& InSlot,
		const OaCameraState& InCamera);
	[[nodiscard]] OaStatus ValidateFrame(
		const OaLunarLander3dRenderFrame& InFrame,
		LunarSlotState InRequiredState,
		LunarSlot*& OutSlot);
	[[nodiscard]] OaStatus CollectRetired();
	void DestroyAll() noexcept;
	[[nodiscard]] bool PrepareNonWaitingRetirement() noexcept;
	[[nodiscard]] static OaStatus CompleteRetired(void* InPayload);
	static void ReleaseRetired(void* InPayload);
};

OaStatus OaLunarLander3dRenderSession::Impl::ValidateCapabilities() const {
	if (Engine == nullptr or not Engine->IsReady()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d renderer requires a ready engine");
	}
	if (not Engine->HasGraphics()
		or Engine->Device.Queues.GraphicsQueue == nullptr
		or Engine->Device.Queues.GraphicsQueueFamily
			== OaVkEnumerationIndexUnset) {
		return OaStatus::Error(
			OaStatusCode::Unavailable,
			"LunarLander3d renderer requires a graphics-capable engine");
	}
	if (DeviceTable.vkCmdBeginRendering == nullptr
		or DeviceTable.vkCmdEndRendering == nullptr
		or DeviceTable.vkCmdPipelineBarrier2 == nullptr
		or DeviceTable.vkCmdCopyImageToBuffer == nullptr
		or DeviceTable.vkCreateGraphicsPipelines == nullptr) {
		return OaStatus::Error(
			OaStatusCode::Unavailable,
			"LunarLander3d renderer requires dynamic rendering, synchronization2, and image copy commands");
	}

	OA_RETURN_IF_ERROR(ValidateTargetExtent(
		*Engine, InstanceTable, RenderConfig.Width_, RenderConfig.Height_));
	VkPhysicalDeviceProperties properties{};
	InstanceTable.vkGetPhysicalDeviceProperties(
		static_cast<VkPhysicalDevice>(Engine->Device.PhysicalDevice),
		&properties);
	if (properties.limits.maxPushConstantsSize < sizeof(LunarPushConstants)) {
		return OaStatus::Error(
			OaStatusCode::Unavailable,
			"LunarLander3d renderer requires 80 bytes of push constants");
	}

	return OaStatus::Ok();
}

OaStatus OaLunarLander3dRenderSession::Impl::BuildTerrainSnapshot(
	const OaLunarTerrain& InTerrain) {
	const auto& config = InTerrain.Config();
	const OaU32 verticesX = config.CellsX_ + 1U;
	const OaU32 verticesZ = config.CellsZ_ + 1U;
	const OaU64 vertexCount64 =
		static_cast<OaU64>(verticesX) * static_cast<OaU64>(verticesZ);
	if (vertexCount64 != InTerrain.Heights().size()) {
		return OaStatus::InvalidArgument(
			"LunarLander3d terrain height count does not match its grid");
	}

	std::vector<VlmVec3> positions(static_cast<std::size_t>(vertexCount64));
	std::vector<VlmVec3> normalSums(static_cast<std::size_t>(vertexCount64));
	for (OaU32 z = 0U; z < verticesZ; ++z) {
		for (OaU32 x = 0U; x < verticesX; ++x) {
			const OaU32 index = z * verticesX + x;
			VlmVec3 position{};
			if (not TryNarrowFinite(
					InTerrain.MinX()
						+ static_cast<double>(x) * config.CellSize_,
					position.X)
				or not TryNarrowFinite(InTerrain.Heights()[index], position.Y)
				or not TryNarrowFinite(
					InTerrain.MinZ()
						+ static_cast<double>(z) * config.CellSize_,
					position.Z)) {
				return OaStatus::Error(
					OaStatusCode::OutOfRange,
					"LunarLander3d terrain is not representable as FP32 geometry");
			}
			positions[index] = position;
		}
	}

	TerrainIndices.reserve(
		static_cast<std::size_t>(config.CellsX_) * config.CellsZ_ * 6U);
	for (OaU32 z = 0U; z < config.CellsZ_; ++z) {
		for (OaU32 x = 0U; x < config.CellsX_; ++x) {
			const OaU32 v00 = z * verticesX + x;
			const OaU32 v10 = v00 + 1U;
			const OaU32 v01 = v00 + verticesX;
			const OaU32 v11 = v01 + 1U;
			const OaU32 triangles[6] = {v00, v11, v10, v00, v01, v11};
			for (OaU32 index : triangles) TerrainIndices.push_back(index);
			for (OaU32 triangle = 0U; triangle < 2U; ++triangle) {
				const OaU32 a = triangles[triangle * 3U];
				const OaU32 b = triangles[triangle * 3U + 1U];
				const OaU32 c = triangles[triangle * 3U + 2U];
				const VlmVec3 edgeAB = Sub(positions[b], positions[a]);
				const VlmVec3 edgeAC = Sub(positions[c], positions[a]);
				const VlmVec3 normal = Cross(edgeAB, edgeAC);
				const VlmVec3 sumA = Add(normalSums[a], normal);
				const VlmVec3 sumB = Add(normalSums[b], normal);
				const VlmVec3 sumC = Add(normalSums[c], normal);
				if (not IsFinite(edgeAB) or not IsFinite(edgeAC)
					or not IsFinite(normal) or not IsFinite(sumA)
					or not IsFinite(sumB) or not IsFinite(sumC)) {
					return OaStatus::Error(
						OaStatusCode::OutOfRange,
						"LunarLander3d terrain normals overflow FP32 geometry");
				}
				normalSums[a] = sumA;
				normalSums[b] = sumB;
				normalSums[c] = sumC;
			}
		}
	}

	TerrainVertices.reserve(static_cast<std::size_t>(vertexCount64)
		+ static_cast<std::size_t>(config.CellsX_) * config.CellsZ_ * 6U);
	for (OaU32 index = 0U; index < vertexCount64; ++index) {
		const OaF32 height = positions[index].Y;
		const OaF32 shade = std::clamp(0.30F + height * 0.025F, 0.22F, 0.38F);
		const VlmVec3 normal = Normalize(normalSums[index]);
		if (not IsFinite(normal)) {
			return OaStatus::Error(
				OaStatusCode::OutOfRange,
				"LunarLander3d terrain normal is not representable as FP32");
		}
		TerrainVertices.push_back({
			positions[index], normal, {},
			{shade, shade * 0.96F, shade * 0.90F, 1.0F},
		});
	}

	// Conform the visible pad overlay to the actual height samples. This is
	// deliberately not keyed from the manifest: custom heightfields with equal
	// manifests are legal in the scalar environment.
	const VlmVec4 padColor{0.96F, 0.64F, 0.08F, 1.0F};
	for (OaU32 z = 0U; z < config.CellsZ_; ++z) {
		for (OaU32 x = 0U; x < config.CellsX_; ++x) {
			const double centerX = InTerrain.MinX()
				+ (static_cast<double>(x) + 0.5) * config.CellSize_;
			const double centerZ = InTerrain.MinZ()
				+ (static_cast<double>(z) + 0.5) * config.CellSize_;
			if (not InTerrain.IsOnPad(centerX, centerZ)) continue;
			const OaU32 v00 = z * verticesX + x;
			const OaU32 v10 = v00 + 1U;
			const OaU32 v01 = v00 + verticesX;
			const OaU32 v11 = v01 + 1U;
			const OaU32 triangles[6] = {v00, v11, v10, v00, v01, v11};
			for (OaU32 triangle = 0U; triangle < 2U; ++triangle) {
				VlmVec3 a = positions[triangles[triangle * 3U]];
				VlmVec3 b = positions[triangles[triangle * 3U + 1U]];
				VlmVec3 c = positions[triangles[triangle * 3U + 2U]];
				a.Y += 0.025F;
				b.Y += 0.025F;
				c.Y += 0.025F;
				const VlmVec3 normal = Normalize(Cross(Sub(b, a), Sub(c, a)));
				const OaU32 base = static_cast<OaU32>(TerrainVertices.size());
				TerrainVertices.push_back({a, normal, {}, padColor});
				TerrainVertices.push_back({b, normal, {}, padColor});
				TerrainVertices.push_back({c, normal, {}, padColor});
				TerrainIndices.push_back(base);
				TerrainIndices.push_back(base + 1U);
				TerrainIndices.push_back(base + 2U);
			}
		}
	}
	return OaStatus::Ok();
}

OaStatus OaLunarLander3dRenderSession::Impl::CreatePipeline() {
	const OaSpvEntry* vertexSpirv =
		OaShaderProviderFind("LunarDiagnostic3d.vert");
	const OaSpvEntry* fragmentSpirv =
		OaShaderProviderFind("LunarDiagnostic3d.frag");
	if (vertexSpirv == nullptr or fragmentSpirv == nullptr) {
		return OaStatus::Error(
			OaStatusCode::NotFound,
			"LunarLander3d renderer diagnostic shaders are unavailable");
	}
	const VkDevice device = static_cast<VkDevice>(Engine->Device.Device);
	VkShaderModule vertexModule =
		CreateShaderModule(DeviceTable, device, *vertexSpirv);
	VkShaderModule fragmentModule =
		CreateShaderModule(DeviceTable, device, *fragmentSpirv);
	if (vertexModule == VK_NULL_HANDLE or fragmentModule == VK_NULL_HANDLE) {
		if (vertexModule != VK_NULL_HANDLE) {
			DeviceTable.vkDestroyShaderModule(
				device, vertexModule, nullptr);
		}
		if (fragmentModule != VK_NULL_HANDLE) {
			DeviceTable.vkDestroyShaderModule(
				device, fragmentModule, nullptr);
		}
		return OaStatus::Error(
			OaStatusCode::PipelineError,
			"LunarLander3d renderer shader-module creation failed");
	}

	VkPushConstantRange pushRange{};
	pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	pushRange.offset = 0U;
	pushRange.size = sizeof(LunarPushConstants);
	VkPipelineLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.pushConstantRangeCount = 1U;
	layoutInfo.pPushConstantRanges = &pushRange;
	if (DeviceTable.vkCreatePipelineLayout(
			device, &layoutInfo, nullptr, &PipelineLayout) != VK_SUCCESS) {
		DeviceTable.vkDestroyShaderModule(device, vertexModule, nullptr);
		DeviceTable.vkDestroyShaderModule(device, fragmentModule, nullptr);
		return OaStatus::Error(
			OaStatusCode::PipelineError,
			"LunarLander3d renderer pipeline-layout creation failed");
	}

	VkPipelineShaderStageCreateInfo stages[2]{};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vertexModule;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = fragmentModule;
	stages[1].pName = "main";

	VkVertexInputBindingDescription binding{};
	binding.binding = 0U;
	binding.stride = sizeof(OaMeshVertex);
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	VkVertexInputAttributeDescription attributes[3]{};
	attributes[0] = {
		0U, 0U, VK_FORMAT_R32G32B32_SFLOAT,
		static_cast<OaU32>(offsetof(OaMeshVertex, Position))};
	attributes[1] = {
		1U, 0U, VK_FORMAT_R32G32B32_SFLOAT,
		static_cast<OaU32>(offsetof(OaMeshVertex, Normal))};
	attributes[2] = {
		2U, 0U, VK_FORMAT_R32G32B32A32_SFLOAT,
		static_cast<OaU32>(offsetof(OaMeshVertex, Color))};
	VkPipelineVertexInputStateCreateInfo vertexInput{};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1U;
	vertexInput.pVertexBindingDescriptions = &binding;
	vertexInput.vertexAttributeDescriptionCount = 3U;
	vertexInput.pVertexAttributeDescriptions = attributes;

	VkPipelineInputAssemblyStateCreateInfo assembly{};
	assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkPipelineViewportStateCreateInfo viewport{};
	viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport.viewportCount = 1U;
	viewport.scissorCount = 1U;
	VkPipelineRasterizationStateCreateInfo raster{};
	raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.cullMode = VK_CULL_MODE_NONE;
	raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	raster.lineWidth = 1.0F;
	VkPipelineMultisampleStateCreateInfo multisample{};
	multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencil.minDepthBounds = 0.0F;
	depthStencil.maxDepthBounds = 1.0F;
	VkPipelineColorBlendAttachmentState blendAttachment{};
	blendAttachment.colorWriteMask =
		VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
		| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	VkPipelineColorBlendStateCreateInfo blend{};
	blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blend.attachmentCount = 1U;
	blend.pAttachments = &blendAttachment;
	const VkDynamicState dynamicStates[2] = {
		VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamic{};
	dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic.dynamicStateCount = 2U;
	dynamic.pDynamicStates = dynamicStates;

	VkPipelineRenderingCreateInfo rendering{};
	rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
	rendering.colorAttachmentCount = 1U;
	rendering.pColorAttachmentFormats = &ColorFormat;
	rendering.depthAttachmentFormat = DepthFormat;
	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.pNext = &rendering;
	pipelineInfo.stageCount = 2U;
	pipelineInfo.pStages = stages;
	pipelineInfo.pVertexInputState = &vertexInput;
	pipelineInfo.pInputAssemblyState = &assembly;
	pipelineInfo.pViewportState = &viewport;
	pipelineInfo.pRasterizationState = &raster;
	pipelineInfo.pMultisampleState = &multisample;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pColorBlendState = &blend;
	pipelineInfo.pDynamicState = &dynamic;
	pipelineInfo.layout = PipelineLayout;
	const VkResult pipelineResult = DeviceTable.vkCreateGraphicsPipelines(
		device, VK_NULL_HANDLE, 1U, &pipelineInfo, nullptr, &Pipeline);
	DeviceTable.vkDestroyShaderModule(device, vertexModule, nullptr);
	DeviceTable.vkDestroyShaderModule(device, fragmentModule, nullptr);
	if (pipelineResult != VK_SUCCESS) {
		return OaStatus::Error(
			OaStatusCode::PipelineError,
			"LunarLander3d renderer graphics-pipeline creation failed");
	}
	return OaStatus::Ok();
}

OaStatus OaLunarLander3dRenderSession::Impl::CreateSlots() {
	VertexCapacity = static_cast<OaU32>(TerrainVertices.size())
		+ LanderBoxCount * VerticesPerBox;
	IndexCapacity = static_cast<OaU32>(TerrainIndices.size())
		+ LanderBoxCount * IndicesPerBox;
	const OaU64 vertexBytes =
		static_cast<OaU64>(VertexCapacity) * sizeof(OaMeshVertex);
	const OaU64 indexBytes =
		static_cast<OaU64>(IndexCapacity) * sizeof(OaU32);

	Slots.reserve(RenderConfig.TargetSlotCount_);
	for (OaU32 index = 0U; index < RenderConfig.TargetSlotCount_; ++index) {
		Slots.emplace_back();
		LunarSlot& slot = Slots.back();
		auto vertices = CreateMappedVertexOrIndexBuffer(
			*Engine, vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		if (not vertices.IsOk()) return vertices.GetStatus();
		slot.VertexBuffer = OaStdMove(*vertices);
		auto indices = CreateMappedVertexOrIndexBuffer(
			*Engine, indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		if (not indices.IsOk()) return indices.GetStatus();
		slot.IndexBuffer = OaStdMove(*indices);
		const OaStatus targetStatus = CreateTarget(
			*Engine, DeviceTable, RenderConfig.Width_, RenderConfig.Height_,
			slot.Target);
		if (not targetStatus.IsOk()) return targetStatus;
	}
	return OaStatus::Ok();
}

OaStatus OaLunarLander3dRenderSession::Impl::Initialize(
	OaEngine& InEngine,
	const OaLunarLander3dConfig& InLanderConfig,
	const OaLunarTerrain& InTerrain,
	const OaLunarLander3dRenderConfig& InRenderConfig) {
	Engine = &InEngine;
	LanderConfig = InLanderConfig;
	RenderConfig = InRenderConfig;
	if (not InTerrain.IsValid()) {
		return OaStatus::InvalidArgument(
			"LunarLander3d renderer requires a valid terrain snapshot");
	}
	if (not InLanderConfig.ValidationError().empty()) {
		return OaStatus::InvalidArgument(
			"LunarLander3d renderer requires a valid lander configuration");
	}
	auto supportRepresentable = [](const OaLunarSupportSphere& InSupport) {
		VlmVec3 offset{};
		OaF32 radius = 0.0F;
		return TryToVlm(InSupport.BodyOffset_, offset)
			and TryNarrowFinite(InSupport.Radius_, radius)
			and std::isfinite(radius * 1.35F);
	};
	for (const OaLunarSupportSphere& support : InLanderConfig.BodySupports_) {
		if (not supportRepresentable(support)) {
			return OaStatus::Error(
				OaStatusCode::OutOfRange,
				"LunarLander3d body support is not representable as FP32 geometry");
		}
	}
	for (const OaLunarSupportSphere& support : InLanderConfig.FootSupports_) {
		if (not supportRepresentable(support)) {
			return OaStatus::Error(
				OaStatusCode::OutOfRange,
				"LunarLander3d foot support is not representable as FP32 geometry");
		}
	}
	if (not (InLanderConfig.Terrain_ == InTerrain.Config())) {
		return OaStatus::InvalidArgument(
			"LunarLander3d renderer terrain/config snapshots do not match");
	}
	if (RenderConfig.Width_ == 0U or RenderConfig.Height_ == 0U) {
		return OaStatus::InvalidArgument(
			"LunarLander3d renderer target dimensions must be non-zero");
	}
	if (RenderConfig.TargetSlotCount_ == 0U
		or RenderConfig.TargetSlotCount_ > MaxTargetSlots) {
		return OaStatus::InvalidArgument(
			"LunarLander3d renderer target slot count must be in [1, 4]");
	}
	if (InTerrain.Config().CellsX_ == 0U
		or InTerrain.Config().CellsZ_ == 0U
		or InTerrain.Config().CellsX_ > MaxTerrainCellsPerAxis
		or InTerrain.Config().CellsZ_ > MaxTerrainCellsPerAxis) {
		return OaStatus::Error(
			OaStatusCode::OutOfRange,
			"LunarLander3d A0 renderer supports at most 32x32 terrain cells");
	}
	if (not std::isfinite(RenderConfig.ClearColor_.X)
		or not std::isfinite(RenderConfig.ClearColor_.Y)
		or not std::isfinite(RenderConfig.ClearColor_.Z)
		or not std::isfinite(RenderConfig.ClearColor_.W)) {
		return OaStatus::InvalidArgument(
			"LunarLander3d renderer clear color must be finite");
	}
	if (not InEngine.IsReady()
		or InEngine.Device.Instance == nullptr
		or InEngine.Device.Device == nullptr) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d renderer requires a ready engine");
	}
	OaVkLoadInstanceTable(
		&InstanceTable,
		static_cast<VkInstance>(InEngine.Device.Instance));
	OaVkLoadDeviceTable(
		&DeviceTable,
		static_cast<VkDevice>(InEngine.Device.Device));
	OA_RETURN_IF_ERROR(ValidateCapabilities());
	OA_RETURN_IF_ERROR(BuildTerrainSnapshot(InTerrain));
	OA_RETURN_IF_ERROR(CreatePipeline());
	return CreateSlots();
}

OaStatus OaLunarLander3dRenderSession::Impl::WriteFrameGeometry(
	LunarSlot& InSlot,
	const OaLunarLander3dState& InState) {
	if (not InState.IsFinite()) {
		return OaStatus::InvalidArgument(
			"LunarLander3d renderer state snapshot must be finite");
	}
	GeometryWriter writer(
		static_cast<OaMeshVertex*>(InSlot.VertexBuffer.MappedPtr),
		VertexCapacity,
		static_cast<OaU32*>(InSlot.IndexBuffer.MappedPtr),
		IndexCapacity);
	if (not writer.AppendCached(TerrainVertices, TerrainIndices)) {
		return OaStatus::Error(
			OaStatusCode::Internal,
			"LunarLander3d cached terrain exceeds its mapped slot");
	}

	const VlmVec4 hullColor{0.08F, 0.62F, 0.96F, 1.0F};
	const VlmVec4 legColor{0.24F, 0.42F, 0.58F, 1.0F};
	const VlmVec4 footColor{0.82F, 0.90F, 0.96F, 1.0F};
	for (const OaLunarSupportSphere& support : LanderConfig.BodySupports_) {
		VlmVec3 offset{};
		OaF32 radius = 0.0F;
		if (not TryToVlm(support.BodyOffset_, offset)
			or not TryNarrowFinite(support.Radius_, radius)) {
			return OaStatus::Error(
				OaStatusCode::OutOfRange,
				"LunarLander3d body support is not representable as FP32 geometry");
		}
		OA_RETURN_IF_ERROR(writer.AppendBodyBox(
			InState, offset,
			{radius * 0.72F, radius * 0.72F, radius * 0.72F},
			hullColor));
	}
	for (const OaLunarSupportSphere& support : LanderConfig.FootSupports_) {
		VlmVec3 foot{};
		OaF32 radius = 0.0F;
		if (not TryToVlm(support.BodyOffset_, foot)
			or not TryNarrowFinite(support.Radius_, radius)) {
			return OaStatus::Error(
				OaStatusCode::OutOfRange,
				"LunarLander3d foot support is not representable as FP32 geometry");
		}
		const OaF32 attachmentY = -0.20F;
		const OaF32 legHalfHeight =
			std::max(0.05F, std::abs(attachmentY - foot.Y) * 0.5F);
		const VlmVec3 legCenter{
			foot.X * 0.72F,
			(attachmentY + foot.Y) * 0.5F,
			foot.Z * 0.72F,
		};
		OA_RETURN_IF_ERROR(writer.AppendBodyBox(
			InState, legCenter, {0.065F, legHalfHeight, 0.065F},
			legColor));
		OA_RETURN_IF_ERROR(writer.AppendBodyBox(
			InState, foot,
			{radius * 1.35F, radius * 0.35F, radius * 1.35F},
			footColor));
	}
	InSlot.IndexCount = writer.IndexCount();
	const OaU64 vertexBytes =
		static_cast<OaU64>(writer.VertexCount()) * sizeof(OaMeshVertex);
	const OaU64 indexBytes =
		static_cast<OaU64>(writer.IndexCount()) * sizeof(OaU32);
	if (not Engine->Allocator.FlushHostBuffer(
			InSlot.VertexBuffer, 0U, vertexBytes)
		or not Engine->Allocator.FlushHostBuffer(
			InSlot.IndexBuffer, 0U, indexBytes)) {
		return OaStatus::Error(
			OaStatusCode::VulkanError,
			"LunarLander3d mapped geometry flush failed");
	}
	return OaStatus::Ok();
}

OaStatus OaLunarLander3dRenderSession::Impl::RecordFrame(
	LunarSlot& InSlot,
	const OaCameraState& InCamera) {
	if (not IsFinite(OaFnCamera::GetViewProjectionMatrix(InCamera))) {
		return OaStatus::InvalidArgument(
			"LunarLander3d renderer camera matrix must be finite");
	}
	if (not InSlot.StreamLease.has_value()) {
		return OaStatus::Error(
			OaStatusCode::Internal,
			"LunarLander3d renderer has no active graphics lease");
	}
	OaVkStream* stream = InSlot.StreamLease->GetStream();
	if (stream == nullptr or stream->CommandBuffer == nullptr) {
		return OaStatus::Error(
			OaStatusCode::Internal,
			"LunarLander3d renderer graphics encoder is unavailable");
	}
	const VkCommandBuffer commandBuffer =
		static_cast<VkCommandBuffer>(stream->CommandBuffer);

	VkMemoryBarrier2 hostToGeometry{};
	hostToGeometry.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
	hostToGeometry.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
	hostToGeometry.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
	hostToGeometry.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
	hostToGeometry.dstAccessMask =
		VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;

	VkImageMemoryBarrier2 toAttachments[2]{};
	toAttachments[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	toAttachments[0].srcStageMask =
		InSlot.Target.ColorLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_2_NONE;
	toAttachments[0].srcAccessMask =
		InSlot.Target.ColorLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE;
	toAttachments[0].dstStageMask =
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
	toAttachments[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
	toAttachments[0].oldLayout = InSlot.Target.ColorLayout;
	toAttachments[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	toAttachments[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toAttachments[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toAttachments[0].image = InSlot.Target.ColorImage;
	toAttachments[0].subresourceRange = {
		VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
	toAttachments[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	toAttachments[1].srcStageMask =
		InSlot.Target.DepthLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
			? VK_PIPELINE_STAGE_2_COPY_BIT : VK_PIPELINE_STAGE_2_NONE;
	toAttachments[1].srcAccessMask =
		InSlot.Target.DepthLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
			? VK_ACCESS_2_TRANSFER_READ_BIT : VK_ACCESS_2_NONE;
	toAttachments[1].dstStageMask =
		VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
		| VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
	toAttachments[1].dstAccessMask =
		VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
		| VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	toAttachments[1].oldLayout = InSlot.Target.DepthLayout;
	toAttachments[1].newLayout =
		VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	toAttachments[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toAttachments[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toAttachments[1].image = InSlot.Target.DepthImage;
	toAttachments[1].subresourceRange = {
		VK_IMAGE_ASPECT_DEPTH_BIT, 0U, 1U, 0U, 1U};
	VkDependencyInfo attachmentDependency{};
	attachmentDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	attachmentDependency.memoryBarrierCount = 1U;
	attachmentDependency.pMemoryBarriers = &hostToGeometry;
	attachmentDependency.imageMemoryBarrierCount = 2U;
	attachmentDependency.pImageMemoryBarriers = toAttachments;
	DeviceTable.vkCmdPipelineBarrier2(
		commandBuffer, &attachmentDependency);

	VkClearValue colorClear{};
	colorClear.color.float32[0] = RenderConfig.ClearColor_.X;
	colorClear.color.float32[1] = RenderConfig.ClearColor_.Y;
	colorClear.color.float32[2] = RenderConfig.ClearColor_.Z;
	colorClear.color.float32[3] = RenderConfig.ClearColor_.W;
	VkClearValue depthClear{};
	depthClear.depthStencil = {1.0F, 0U};
	VkRenderingAttachmentInfo colorAttachment{};
	colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	colorAttachment.imageView = InSlot.Target.ColorView;
	colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.clearValue = colorClear;
	VkRenderingAttachmentInfo depthAttachment{};
	depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	depthAttachment.imageView = InSlot.Target.DepthView;
	depthAttachment.imageLayout =
		VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depthAttachment.clearValue = depthClear;
	VkRenderingInfo rendering{};
	rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
	rendering.renderArea = {{0, 0}, {RenderConfig.Width_, RenderConfig.Height_}};
	rendering.layerCount = 1U;
	rendering.colorAttachmentCount = 1U;
	rendering.pColorAttachments = &colorAttachment;
	rendering.pDepthAttachment = &depthAttachment;
	DeviceTable.vkCmdBeginRendering(commandBuffer, &rendering);

	const VkViewport viewport{
		0.0F,
		static_cast<OaF32>(RenderConfig.Height_),
		static_cast<OaF32>(RenderConfig.Width_),
		-static_cast<OaF32>(RenderConfig.Height_),
		0.0F,
		1.0F,
	};
	const VkRect2D scissor{{0, 0}, {RenderConfig.Width_, RenderConfig.Height_}};
	DeviceTable.vkCmdSetViewport(commandBuffer, 0U, 1U, &viewport);
	DeviceTable.vkCmdSetScissor(commandBuffer, 0U, 1U, &scissor);
	DeviceTable.vkCmdBindPipeline(
		commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
	const VkBuffer vertexBuffer =
		static_cast<VkBuffer>(InSlot.VertexBuffer.Buffer);
	const VkDeviceSize vertexOffset = 0U;
	DeviceTable.vkCmdBindVertexBuffers(
		commandBuffer, 0U, 1U, &vertexBuffer, &vertexOffset);
	DeviceTable.vkCmdBindIndexBuffer(
		commandBuffer,
		static_cast<VkBuffer>(InSlot.IndexBuffer.Buffer),
		0U, VK_INDEX_TYPE_UINT32);
	const LunarPushConstants push{
		OaFnCamera::GetViewProjectionMatrix(InCamera),
		{0.35F, 0.82F, 0.45F, 0.24F},
	};
	DeviceTable.vkCmdPushConstants(
		commandBuffer, PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
		0U, sizeof(push), &push);
	DeviceTable.vkCmdDrawIndexed(
		commandBuffer, InSlot.IndexCount, 1U, 0U, 0, 0U);
	DeviceTable.vkCmdEndRendering(commandBuffer);

	VkImageMemoryBarrier2 toCopies[2]{};
	toCopies[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	toCopies[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
	toCopies[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
	toCopies[0].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
	toCopies[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	toCopies[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	toCopies[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	toCopies[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toCopies[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toCopies[0].image = InSlot.Target.ColorImage;
	toCopies[0].subresourceRange = {
		VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
	toCopies[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	toCopies[1].srcStageMask =
		VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
		| VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
	toCopies[1].srcAccessMask =
		VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	toCopies[1].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
	toCopies[1].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	toCopies[1].oldLayout =
		VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	toCopies[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	toCopies[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toCopies[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toCopies[1].image = InSlot.Target.DepthImage;
	toCopies[1].subresourceRange = {
		VK_IMAGE_ASPECT_DEPTH_BIT, 0U, 1U, 0U, 1U};
	VkBufferMemoryBarrier2 readbackReuse[2]{};
	const bool hasPriorReadback =
		InSlot.Target.DepthLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	if (hasPriorReadback) {
		readbackReuse[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
		readbackReuse[0].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
		readbackReuse[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		readbackReuse[0].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
		readbackReuse[0].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
		readbackReuse[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		readbackReuse[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		readbackReuse[0].buffer =
			static_cast<VkBuffer>(InSlot.Target.ColorReadback.Buffer);
		readbackReuse[0].offset = 0U;
		readbackReuse[0].size = VK_WHOLE_SIZE;
		readbackReuse[1] = readbackReuse[0];
		readbackReuse[1].buffer =
			static_cast<VkBuffer>(InSlot.Target.DepthReadback.Buffer);
	}
	VkDependencyInfo copyDependency{};
	copyDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	copyDependency.bufferMemoryBarrierCount = hasPriorReadback ? 2U : 0U;
	copyDependency.pBufferMemoryBarriers =
		hasPriorReadback ? readbackReuse : nullptr;
	copyDependency.imageMemoryBarrierCount = 2U;
	copyDependency.pImageMemoryBarriers = toCopies;
	DeviceTable.vkCmdPipelineBarrier2(commandBuffer, &copyDependency);

	VkBufferImageCopy colorCopy{};
	colorCopy.imageSubresource = {
		VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
	colorCopy.imageExtent = {RenderConfig.Width_, RenderConfig.Height_, 1U};
	DeviceTable.vkCmdCopyImageToBuffer(
		commandBuffer, InSlot.Target.ColorImage,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		static_cast<VkBuffer>(InSlot.Target.ColorReadback.Buffer),
		1U, &colorCopy);
	VkBufferImageCopy depthCopy{};
	depthCopy.imageSubresource = {
		VK_IMAGE_ASPECT_DEPTH_BIT, 0U, 0U, 1U};
	depthCopy.imageExtent = {RenderConfig.Width_, RenderConfig.Height_, 1U};
	DeviceTable.vkCmdCopyImageToBuffer(
		commandBuffer, InSlot.Target.DepthImage,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		static_cast<VkBuffer>(InSlot.Target.DepthReadback.Buffer),
		1U, &depthCopy);

	VkBufferMemoryBarrier2 toHost[2]{};
	toHost[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
	toHost[0].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
	toHost[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
	toHost[0].dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
	toHost[0].dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
	toHost[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toHost[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toHost[0].buffer =
		static_cast<VkBuffer>(InSlot.Target.ColorReadback.Buffer);
	toHost[0].offset = 0U;
	toHost[0].size = VK_WHOLE_SIZE;
	toHost[1] = toHost[0];
	toHost[1].buffer =
		static_cast<VkBuffer>(InSlot.Target.DepthReadback.Buffer);
	VkImageMemoryBarrier2 colorForSampling{};
	colorForSampling.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	colorForSampling.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
	colorForSampling.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
	colorForSampling.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	colorForSampling.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
	colorForSampling.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	colorForSampling.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	colorForSampling.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	colorForSampling.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	colorForSampling.image = InSlot.Target.ColorImage;
	colorForSampling.subresourceRange = {
		VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
	VkDependencyInfo hostDependency{};
	hostDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	hostDependency.bufferMemoryBarrierCount = 2U;
	hostDependency.pBufferMemoryBarriers = toHost;
	hostDependency.imageMemoryBarrierCount = 1U;
	hostDependency.pImageMemoryBarriers = &colorForSampling;
	DeviceTable.vkCmdPipelineBarrier2(commandBuffer, &hostDependency);
	return OaStatus::Ok();
}

OaStatus OaLunarLander3dRenderSession::Impl::ValidateFrame(
	const OaLunarLander3dRenderFrame& InFrame,
	LunarSlotState InRequiredState,
	LunarSlot*& OutSlot) {
	OutSlot = nullptr;
	if (Closed) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	if (InFrame.TargetGeneration_ != TargetGeneration
		or InFrame.Slot_ >= Slots.size()
		or InFrame.Width_ != RenderConfig.Width_
		or InFrame.Height_ != RenderConfig.Height_) {
		return OaStatus::InvalidArgument(
			"LunarLander3d render frame has stale or forged target metadata");
	}
	LunarSlot& slot = Slots[InFrame.Slot_];
	if (slot.Generation != InFrame.SlotGeneration_
		or slot.State != InRequiredState
		or InFrame.Image_ != slot.Target.ColorImage
		or InFrame.ImageView_ != slot.Target.ColorView
		or InFrame.ImageLayout_ != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		or not Engine->OwnsEvent(InFrame.Producer_)
		or not slot.Producer.IsSameCompletion(InFrame.Producer_)) {
		return OaStatus::InvalidArgument(
			"LunarLander3d render frame is stale or not the exact live completion");
	}
	OutSlot = &slot;
	return OaStatus::Ok();
}

OaStatus OaLunarLander3dRenderSession::Impl::CollectRetired() {
	if (Closed) return OaStatus::Ok();
	for (LunarSlot& slot : Slots) {
		if (slot.State != LunarSlotState::Retired) continue;
		if (not slot.Producer.IsValid()) {
			return OaStatus::Error(
				OaStatusCode::Internal,
				"LunarLander3d retired target lost its producer event");
		}
		if (not slot.Producer.IsComplete()
			or (slot.Consumer.IsValid() and not slot.Consumer.IsComplete())) {
			continue;
		}
		// The counter query above is only a non-blocking observation. Complete
		// the exact semaphore wait dependency before making mapped geometry and
		// readback buffers host-reusable. Because the observed value is already
		// reached, this wait does not stall for outstanding device work.
		OA_RETURN_IF_ERROR(slot.Producer.Wait());
		if (slot.Consumer.IsValid()) {
			OA_RETURN_IF_ERROR(slot.Consumer.Wait());
		}
		slot.Producer = {};
		slot.Consumer = {};
		slot.State = LunarSlotState::Free;
	}
	return OaStatus::Ok();
}

void OaLunarLander3dRenderSession::Impl::DestroyAll() noexcept {
	if (Engine == nullptr or Engine->Device.Device == nullptr) return;
	for (LunarSlot& slot : Slots) {
		DestroyTarget(*Engine, DeviceTable, slot.Target);
		Engine->Allocator.Free(slot.VertexBuffer);
		Engine->Allocator.Free(slot.IndexBuffer);
		slot.StreamLease.reset();
		slot.Producer = {};
		slot.Consumer = {};
		slot.State = LunarSlotState::Free;
	}
	Slots.clear();
	const VkDevice device = static_cast<VkDevice>(Engine->Device.Device);
	if (Pipeline != VK_NULL_HANDLE) {
		DeviceTable.vkDestroyPipeline(device, Pipeline, nullptr);
		Pipeline = VK_NULL_HANDLE;
	}
	if (PipelineLayout != VK_NULL_HANDLE) {
		DeviceTable.vkDestroyPipelineLayout(
			device, PipelineLayout, nullptr);
		PipelineLayout = VK_NULL_HANDLE;
	}
}

bool OaLunarLander3dRenderSession::Impl::PrepareNonWaitingRetirement() noexcept {
	if (Closed) return false;
	bool hasSubmission = false;
	if (ActiveSlot < Slots.size()) {
		LunarSlot& slot = Slots[ActiveSlot];
		if (slot.StreamLease.has_value()) {
			const OaStatus status = slot.StreamLease->Cancel();
			if (not status.IsOk()) {
				(void)slot.StreamLease->Close();
			}
			slot.StreamLease.reset();
		}
		slot.State = LunarSlotState::Free;
		slot.Producer = {};
		slot.Consumer = {};
	}
	ActiveSlot = std::numeric_limits<OaU32>::max();
	for (LunarSlot& slot : Slots) {
		if (slot.State == LunarSlotState::Submitted) {
			if (slot.StreamLease.has_value()) {
				(void)slot.StreamLease->Close();
				slot.StreamLease.reset();
			}
			slot.State = LunarSlotState::Retired;
			hasSubmission = true;
		} else if (slot.State == LunarSlotState::Retired) {
			hasSubmission = true;
		}
	}
	return hasSubmission;
}

OaStatus OaLunarLander3dRenderSession::Impl::CompleteRetired(
	void* InPayload) {
	auto* impl = static_cast<Impl*>(InPayload);
	if (impl == nullptr) return OaStatus::Ok();
	OaStatus firstError = OaStatus::Ok();
	for (LunarSlot& slot : impl->Slots) {
		if (slot.State != LunarSlotState::Retired) continue;
		const OaStatus producerStatus = slot.Producer.Wait();
		OaStatus consumerStatus = OaStatus::Ok();
		if (producerStatus.IsOk() and slot.Consumer.IsValid()) {
			consumerStatus = slot.Consumer.Wait();
		}
		if (not producerStatus.IsOk() and firstError.IsOk()) {
			firstError = producerStatus;
		} else if (not consumerStatus.IsOk() and firstError.IsOk()) {
			firstError = consumerStatus;
		}
		if (producerStatus.IsOk() and consumerStatus.IsOk()) {
			slot.Producer = {};
			slot.Consumer = {};
			slot.State = LunarSlotState::Free;
		}
	}
	if (firstError.IsOk()) {
		impl->DestroyAll();
		impl->Closed = true;
	}
	return firstError;
}

void OaLunarLander3dRenderSession::Impl::ReleaseRetired(
	void* InPayload) {
	OaUniquePtr<Impl> impl(static_cast<Impl*>(InPayload));
	if (impl and not impl->Closed) {
		// CompleteRetired is the only callback allowed to destroy device objects.
		// Reaching release while still open means the engine violated its exact-wait
		// retirement contract; deleting this host shell intentionally leaves the
		// Vulkan objects for device teardown rather than freeing potentially-live work.
		OA_LOG_ERROR(
			OaLogComponent::Core,
			"LunarLander3d retired renderer released without successful completion; preserving live Vulkan resources");
	}
}

OaLunarLander3dRenderSession::~OaLunarLander3dRenderSession() {
	if (not Impl_ or Impl_->Closed) {
		Impl_.Reset();
		return;
	}
	OaEngine* engine = Impl_->Engine;
	const bool hasSubmission = Impl_->PrepareNonWaitingRetirement();
	if (not hasSubmission and engine != nullptr and engine->IsReady()) {
		OA_LOG_ERROR(
			OaLogComponent::Core,
			"LunarLander3d render session destroyed without Close; performing non-waiting cleanup");
		Impl_->DestroyAll();
		Impl_->Closed = true;
		Impl_.Reset();
		return;
	}
	if (engine != nullptr and engine->IsReady()) {
		OA_LOG_ERROR(
			OaLogComponent::Core,
			"LunarLander3d render session abandoned; exact frame retirement transferred to OaEngine");
		OaBorrowedServiceRetirement::Retire(
			*engine,
			Impl_.Release(),
			&Impl::CompleteRetired,
			&Impl::ReleaseRetired);
		return;
	}
	// Contract violation: a borrowed session must not outlive its engine. Do
	// not call Vulkan through a destroyed owner or wait from the destructor.
	OA_LOG_ERROR(
		OaLogComponent::Core,
		"LunarLander3d render session outlived OaEngine; Vulkan resources cannot be retired safely");
	(void)Impl_.Release();
}

OaResult<OaUniquePtr<OaLunarLander3dRenderSession>>
OaLunarLander3dRenderSession::Create(
	OaEngine& InEngine,
	const OaLunarLander3dConfig& InLanderConfig,
	const OaLunarTerrain& InTerrain,
	const OaLunarLander3dRenderConfig& InRenderConfig) {
	OaUniquePtr<OaLunarLander3dRenderSession> session(
		new OaLunarLander3dRenderSession());
	session->Impl_ = OaMakeUniquePtr<Impl>();
	const OaStatus status = session->Impl_->Initialize(
		InEngine, InLanderConfig, InTerrain, InRenderConfig);
	if (not status.IsOk()) {
		session->Impl_->DestroyAll();
		session->Impl_->Closed = true;
		return status;
	}
	return OaStdMove(session);
}

OaStatus OaLunarLander3dRenderSession::BeginFrame(
	const OaLunarLander3dState& InState,
	const OaCameraState& InCamera) {
	if (not Impl_ or Impl_->Closed) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	if (Impl_->ActiveSlot < Impl_->Slots.size()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d renderer already has an active recording");
	}
	OA_RETURN_IF_ERROR(Impl_->CollectRetired());
	OaU32 slotIndex = static_cast<OaU32>(Impl_->Slots.size());
	for (OaU32 index = 0U; index < Impl_->Slots.size(); ++index) {
		if (Impl_->Slots[index].State == LunarSlotState::Free) {
			slotIndex = index;
			break;
		}
	}
	if (slotIndex == Impl_->Slots.size()) {
		return OaStatus::Error(
			OaStatusCode::ResourceExhausted,
			"LunarLander3d target ring is full; consume or abandon an older frame");
	}

	LunarSlot& slot = Impl_->Slots[slotIndex];
	slot.Generation = NextGeneration(slot.Generation);
	slot.State = LunarSlotState::Recording;
	slot.Producer = {};
	slot.Consumer = {};
	auto lease = OaGraphicsStreamLease::Acquire(*Impl_->Engine);
	if (not lease.IsOk()) {
		slot.State = LunarSlotState::Free;
		return lease.GetStatus();
	}
	slot.StreamLease.emplace(OaStdMove(*lease));
	OaStatus status = Impl_->WriteFrameGeometry(slot, InState);
	if (status.IsOk()) status = Impl_->RecordFrame(slot, InCamera);
	if (not status.IsOk()) {
		const OaStatus cancelStatus = slot.StreamLease->Cancel();
		if (not cancelStatus.IsOk()) (void)slot.StreamLease->Close();
		slot.StreamLease.reset();
		slot.State = LunarSlotState::Free;
		return status;
	}
	Impl_->ActiveSlot = slotIndex;
	return OaStatus::Ok();
}

OaResult<OaLunarLander3dRenderFrame>
OaLunarLander3dRenderSession::SubmitFrame(
	OaSpan<const OaEvent> InDependencies) {
	if (not Impl_ or Impl_->Closed
		or Impl_->ActiveSlot >= Impl_->Slots.size()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d renderer has no active recording to submit");
	}
	LunarSlot& slot = Impl_->Slots[Impl_->ActiveSlot];
	if (slot.State != LunarSlotState::Recording
		or not slot.StreamLease.has_value()) {
		return OaStatus::Error(
			OaStatusCode::Internal,
			"LunarLander3d active target lost its graphics recording");
	}
	auto submission = slot.StreamLease->Submit(InDependencies);
	if (not submission.IsOk()) {
		// Dependency provenance failures occur before queue submission, so the
		// exact recording stays active and CancelFrame() remains valid. A queue
		// submission failure is reset by the engine and is closed here.
		const OaStatusCode code = submission.GetStatus().GetCode();
		if (code != OaStatusCode::InvalidArgument
			and code != OaStatusCode::FailedPrecondition) {
			(void)slot.StreamLease->Close();
			slot.StreamLease.reset();
			slot.State = LunarSlotState::Free;
			Impl_->ActiveSlot = std::numeric_limits<OaU32>::max();
		}
		return submission.GetStatus();
	}
	slot.Producer = *submission;
	slot.Consumer = {};
	slot.Target.ColorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	slot.Target.DepthLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	slot.State = LunarSlotState::Submitted;
	const OaU32 submittedSlot = Impl_->ActiveSlot;
	Impl_->ActiveSlot = std::numeric_limits<OaU32>::max();
	return OaLunarLander3dRenderFrame{
		submittedSlot,
		slot.Generation,
		Impl_->TargetGeneration,
		Impl_->RenderConfig.Width_,
		Impl_->RenderConfig.Height_,
		slot.Target.ColorImage,
		slot.Target.ColorView,
		slot.Target.ColorLayout,
		slot.Producer,
	};
}

OaStatus OaLunarLander3dRenderSession::CancelFrame() {
	if (not Impl_ or Impl_->Closed
		or Impl_->ActiveSlot >= Impl_->Slots.size()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d renderer has no active recording to cancel");
	}
	LunarSlot& slot = Impl_->Slots[Impl_->ActiveSlot];
	OaStatus status = slot.StreamLease.has_value()
		? slot.StreamLease->Cancel()
		: OaStatus::Error(
			OaStatusCode::Internal,
			"LunarLander3d active target has no graphics lease");
	if (not status.IsOk() and slot.StreamLease.has_value()) {
		(void)slot.StreamLease->Close();
	}
	slot.StreamLease.reset();
	slot.State = LunarSlotState::Free;
	slot.Producer = {};
	slot.Consumer = {};
	Impl_->ActiveSlot = std::numeric_limits<OaU32>::max();
	return status;
}

OaResult<OaLunarLander3dReadback>
OaLunarLander3dRenderSession::ConsumeReadback(
	const OaLunarLander3dRenderFrame& InFrame) {
	if (not Impl_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	LunarSlot* slot = nullptr;
	const OaStatus validation = Impl_->ValidateFrame(
		InFrame, LunarSlotState::Submitted, slot);
	if (not validation.IsOk()) return validation;

	const OaStatus waitStatus = InFrame.Producer_.Wait();
	if (not waitStatus.IsOk()) return waitStatus;
	OaU64 pixelCount = 0U;
	OaU64 colorBytes = 0U;
	OaU64 depthBytes = 0U;
	if (not CheckedMultiply(
			Impl_->RenderConfig.Width_, Impl_->RenderConfig.Height_, pixelCount)
		or not CheckedMultiply(pixelCount, 4U, colorBytes)
		or not CheckedMultiply(pixelCount, sizeof(OaF32), depthBytes)) {
		return OaStatus::Error(
			OaStatusCode::Internal,
			"LunarLander3d validated target extent overflowed readback size");
	}
	if (not Impl_->Engine->Allocator.InvalidateHostBuffer(
			slot->Target.ColorReadback, 0U, colorBytes)
		or not Impl_->Engine->Allocator.InvalidateHostBuffer(
			slot->Target.DepthReadback, 0U, depthBytes)) {
		return OaStatus::Error(
			OaStatusCode::VulkanError,
			"LunarLander3d target readback invalidate failed");
	}

	OaLunarLander3dReadback readback;
	readback.Width_ = Impl_->RenderConfig.Width_;
	readback.Height_ = Impl_->RenderConfig.Height_;
	readback.ColorRgba8_.resize(static_cast<std::size_t>(colorBytes));
	readback.Depth32_.resize(static_cast<std::size_t>(pixelCount));
	std::memcpy(
		readback.ColorRgba8_.data(),
		slot->Target.ColorReadback.MappedPtr,
		static_cast<std::size_t>(colorBytes));
	std::memcpy(
		readback.Depth32_.data(),
		slot->Target.DepthReadback.MappedPtr,
		static_cast<std::size_t>(depthBytes));

	if (not slot->StreamLease.has_value()) {
		return OaStatus::Error(
			OaStatusCode::Internal,
			"LunarLander3d submitted target lost its graphics lease");
	}
	const OaStatus recycleStatus =
		slot->StreamLease->Recycle(InFrame.Producer_);
	if (not recycleStatus.IsOk()) return recycleStatus;
	slot->StreamLease.reset();
	slot->Producer = {};
	slot->Consumer = {};
	slot->State = LunarSlotState::Free;
	return readback;
}

OaStatus OaLunarLander3dRenderSession::MarkConsumed(
	const OaLunarLander3dRenderFrame& InFrame,
	const OaEvent& InConsumer) {
	if (not Impl_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	LunarSlot* slot = nullptr;
	OA_RETURN_IF_ERROR(Impl_->ValidateFrame(
		InFrame, LunarSlotState::Submitted, slot));
	if (not Impl_->Engine->OwnsEvent(InConsumer)
		or not InConsumer.HasQueueFamily()) {
		return OaStatus::InvalidArgument(
			"LunarLander3d consumer must be an exact event from this engine");
	}
	if (InConsumer.QueueFamily()
		!= Impl_->Engine->Device.Queues.GraphicsQueueFamily) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d cross-family consumption requires an explicit image ownership transfer");
	}
	if (InConsumer.IsSameCompletion(slot->Producer)) {
		return OaStatus::InvalidArgument(
			"LunarLander3d consumer completion cannot alias its producer completion");
	}
	const OaVkTimelineSemaphore* producerSemaphore = slot->Producer.Semaphore();
	const OaVkTimelineSemaphore* consumerSemaphore = InConsumer.Semaphore();
	if (producerSemaphore != nullptr and consumerSemaphore != nullptr
		and producerSemaphore->Semaphore == consumerSemaphore->Semaphore
		and InConsumer.Value() <= slot->Producer.Value()) {
		return OaStatus::InvalidArgument(
			"LunarLander3d consumer completion must follow its producer on a shared timeline");
	}

	OaStatus closeStatus = OaStatus::Ok();
	if (slot->StreamLease.has_value()) {
		closeStatus = slot->StreamLease->Close();
		slot->StreamLease.reset();
	}
	// Register before returning even if graphics-stream retirement reports a
	// failure. A failed registration path must never make a sampled target
	// reusable while the exact consumer event is still outstanding.
	slot->Consumer = InConsumer;
	slot->State = LunarSlotState::Retired;
	return closeStatus;
}

OaStatus OaLunarLander3dRenderSession::AbandonFrame(
	const OaLunarLander3dRenderFrame& InFrame) {
	if (not Impl_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	LunarSlot* slot = nullptr;
	OA_RETURN_IF_ERROR(Impl_->ValidateFrame(
		InFrame, LunarSlotState::Submitted, slot));
	OaStatus closeStatus = OaStatus::Ok();
	if (slot->StreamLease.has_value()) {
		closeStatus = slot->StreamLease->Close();
		slot->StreamLease.reset();
	}
	// Abandon is strictly non-waiting even if the producer already appears
	// complete. Collect() performs the exact, already-satisfied semaphore wait
	// before this target's mapped buffers become reusable by the host.
	slot->Consumer = {};
	slot->State = LunarSlotState::Retired;
	return closeStatus;
}

OaStatus OaLunarLander3dRenderSession::Collect() {
	if (not Impl_ or Impl_->Closed) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	return Impl_->CollectRetired();
}

OaStatus OaLunarLander3dRenderSession::Resize(
	OaU32 InWidth, OaU32 InHeight) {
	if (not Impl_ or Impl_->Closed) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d renderer session is closed");
	}
	if (InWidth == 0U or InHeight == 0U) {
		return OaStatus::InvalidArgument(
			"LunarLander3d renderer target dimensions must be non-zero");
	}
	OA_RETURN_IF_ERROR(Impl_->CollectRetired());
	if (Impl_->ActiveSlot < Impl_->Slots.size()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"LunarLander3d resize requires cancelling the active recording");
	}
	for (const LunarSlot& slot : Impl_->Slots) {
		if (slot.State != LunarSlotState::Free) {
			return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"LunarLander3d resize requires every old-generation frame to be consumed or retired");
		}
	}
	OA_RETURN_IF_ERROR(ValidateTargetExtent(
		*Impl_->Engine, Impl_->InstanceTable, InWidth, InHeight));
	if (InWidth == Impl_->RenderConfig.Width_
		and InHeight == Impl_->RenderConfig.Height_) {
		return OaStatus::Ok();
	}

	// Allocate a full replacement generation first. A Busy or allocation-failure
	// path does not mutate old resources, layouts, dimensions, or generations.
	std::vector<LunarTarget> replacements(Impl_->Slots.size());
	for (std::size_t index = 0U; index < replacements.size(); ++index) {
		const OaStatus status = CreateTarget(
			*Impl_->Engine, Impl_->DeviceTable,
			InWidth, InHeight, replacements[index]);
		if (not status.IsOk()) {
			for (LunarTarget& target : replacements) {
				DestroyTarget(
					*Impl_->Engine, Impl_->DeviceTable, target);
			}
			return status;
		}
	}
	for (std::size_t index = 0U; index < Impl_->Slots.size(); ++index) {
		DestroyTarget(
			*Impl_->Engine, Impl_->DeviceTable,
			Impl_->Slots[index].Target);
		Impl_->Slots[index].Target = replacements[index];
		replacements[index] = {};
		Impl_->Slots[index].Generation =
			NextGeneration(Impl_->Slots[index].Generation);
	}
	Impl_->RenderConfig.Width_ = InWidth;
	Impl_->RenderConfig.Height_ = InHeight;
	Impl_->TargetGeneration = NextGeneration(Impl_->TargetGeneration);
	return OaStatus::Ok();
}

OaStatus OaLunarLander3dRenderSession::Close() {
	if (not Impl_ or Impl_->Closed) return OaStatus::Ok();
	if (Impl_->ActiveSlot < Impl_->Slots.size()) {
		const OaStatus cancelStatus = CancelFrame();
		if (not cancelStatus.IsOk()) return cancelStatus;
	}
	for (LunarSlot& slot : Impl_->Slots) {
		if (slot.State == LunarSlotState::Submitted) {
			const OaStatus waitStatus = slot.Producer.Wait();
			if (not waitStatus.IsOk()) return waitStatus;
			if (not slot.StreamLease.has_value()) {
				return OaStatus::Error(
					OaStatusCode::Internal,
					"LunarLander3d submitted target lost its graphics lease during Close");
			}
			const OaStatus releaseStatus =
				slot.StreamLease->Recycle(slot.Producer);
			if (not releaseStatus.IsOk()) return releaseStatus;
			slot.StreamLease.reset();
		} else if (slot.State == LunarSlotState::Retired) {
			const OaStatus waitStatus = slot.Producer.Wait();
			if (not waitStatus.IsOk()) return waitStatus;
			if (slot.Consumer.IsValid()) {
				const OaStatus consumerStatus = slot.Consumer.Wait();
				if (not consumerStatus.IsOk()) return consumerStatus;
			}
		}
		slot.Producer = {};
		slot.Consumer = {};
		slot.State = LunarSlotState::Free;
	}
	Impl_->DestroyAll();
	Impl_->Closed = true;
	return OaStatus::Ok();
}

OaCameraState OaLunarLander3dRenderSession::DefaultCamera(
	OaU32 InWidth, OaU32 InHeight) {
	OaCameraState camera;
	const OaF32 aspect = InHeight == 0U
		? 1.0F
		: static_cast<OaF32>(InWidth) / static_cast<OaF32>(InHeight);
	OaFnCamera::InitPerspective(
		camera,
		{16.0F, 12.0F, 16.0F},
		{0.0F, 2.0F, 0.0F},
		48.0F,
		aspect,
		0.1F,
		100.0F);
	return camera;
}
