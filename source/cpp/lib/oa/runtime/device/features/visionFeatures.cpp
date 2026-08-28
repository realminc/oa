// OA vulkan Vision features Module
// Handles vision/video-specific features:
// - Video decode queue (H.264, H.265, AV1)
// - Video encode queue (H.264, H.265)
// - YCbCr sampler conversion

#include "../featureModule.h"
#include <oa/core/log.h>
#include <oa/runtime/init.h>
#include <string.h>


class VisionFeatures : public oavk::FeatureModule {
public:
	oa::StringView name() const override {
		return "Vision";
	}

	void probeExtensions(
		const oa::Vector<VkExtensionProperties>& inAvailableExtensions,
		oavk::PhysicalExtensionProbe& outProbe
	) override {
		for (const auto& ext : inAvailableExtensions) {
			if (strcmp(ext.extensionName, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) == 0) {
				outProbe.khrVideoQueue = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME) == 0) {
				outProbe.khrVideoDecodeQueue = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME) == 0) {
				outProbe.khrVideoDecodeH264 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME) == 0) {
				outProbe.khrVideoDecodeH265 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME) == 0) {
				outProbe.khrVideoDecodeAV1 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME) == 0) {
				outProbe.khrVideoDecodeVP9 = true;
				vp9ExtensionPresent_ = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME) == 0) {
				outProbe.khrVideoEncodeQueue = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME) == 0) {
				outProbe.khrVideoEncodeH264 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME) == 0) {
				outProbe.khrVideoEncodeH265 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME) == 0) {
				outProbe.khrVideoEncodeAV1 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME) == 0) {
				outProbe.khrSamplerYcbcr = true;
			}
		}
	}

	void queryFeatures(
		const VklInstanceTable& inDispatch,
		VkPhysicalDevice inPhysicalDevice,
		oavk::DeviceFeatureBundle& outBundle
	) override {
		// Most video decode codecs (H.264/H.265/AV1) are gated purely by their
		// extension. VP9 decode is the exception: VK_KHR_video_decode_vp9 adds a
		// VkPhysicalDeviceVideoDecodeVP9FeaturesKHR::videoDecodeVP9 bit that must
		// be queried and explicitly enabled, or vkCreateVideoSessionKHR rejects a
		// VP9 profile and the decoder produces undefined output.
#if defined(VK_KHR_video_decode_vp9)
		if (vp9ExtensionPresent_) {
			VkPhysicalDeviceVideoDecodeVP9FeaturesKHR vp9 = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR,
			};
			VkPhysicalDeviceFeatures2 f2 = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
			};
			f2.pNext = &vp9;
			inDispatch.vkGetPhysicalDeviceFeatures2(inPhysicalDevice, &f2);
			outBundle.hasVideoDecodeVp9 = vp9.videoDecodeVP9 == VK_TRUE;
		}
#endif
	}

	void buildFeatureChain(
		oavk::DeviceFeatureBundle& inOutBundle
	) override {
		// YCbCr conversion is part of vulkan 1.1 core features (Core module).
		// VP9 decode requires its feature bit enabled and chained into device
		// creation (see queryFeatures).
#if defined(VK_KHR_video_decode_vp9)
		if (inOutBundle.hasVideoDecodeVp9) {
			inOutBundle.decodeVp9Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR;
			inOutBundle.decodeVp9Features.videoDecodeVP9 = VK_TRUE;
			inOutBundle.decodeVp9Features.pNext = inOutBundle.features13.pNext;
			inOutBundle.features13.pNext = &inOutBundle.decodeVp9Features;
		}
#endif
	}

	void collectExtensions(
		const oavk::PhysicalExtensionProbe& inProbe,
		const oavk::DeviceFeatureBundle& inBundle,
		oa::Vector<const char*>& outExtensions
	) override {
		// Video queue (base for all video operations)
		if (inProbe.khrVideoQueue) {
			outExtensions.pushBack(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);

			// Video Decode
			if (inProbe.khrVideoDecodeQueue) {
				outExtensions.pushBack(VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME);

				// Decode codecs
				if (inProbe.khrVideoDecodeH264) {
					outExtensions.pushBack(VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME);
				}
				if (inProbe.khrVideoDecodeH265) {
					outExtensions.pushBack(VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME);
				}
				if (inProbe.khrVideoDecodeAV1) {
					outExtensions.pushBack(VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME);
				}
				if (inProbe.khrVideoDecodeVP9 && inBundle.hasVideoDecodeVp9) {
					outExtensions.pushBack(VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME);
				}
			}

			// Video Encode
			if (inProbe.khrVideoEncodeQueue) {
				outExtensions.pushBack(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME);

				// Encode codecs
				if (inProbe.khrVideoEncodeH264) {
					outExtensions.pushBack(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME);
				}
				if (inProbe.khrVideoEncodeH265) {
					outExtensions.pushBack(VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME);
				}
				if (inProbe.khrVideoEncodeAV1) {
					outExtensions.pushBack(VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME);
				}
			}
		}

		// YCbCr sampler conversion (useful for video texture sampling)
		if (inProbe.khrSamplerYcbcr) {
			outExtensions.pushBack(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
		}
	}

	oa::Vector<oa::StringView> dependencies() const override {
		return {"Core"};
	}

private:
	// Remembered from probeExtensions so queryFeatures only queries the VP9
	// feature struct when the extension is actually present on this device.
	bool vp9ExtensionPresent_ = false;
};


oa::UniquePtr<oavk::FeatureModule> oavk::createVisionFeatures() {
	return oa::makeUnique<VisionFeatures>();
}
