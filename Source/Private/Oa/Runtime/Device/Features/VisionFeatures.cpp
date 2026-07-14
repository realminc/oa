// OA Vulkan Vision Features Module
// Handles vision/video-specific features:
// - Video decode queue (H.264, H.265, AV1)
// - Video encode queue (H.264, H.265)
// - YCbCr sampler conversion

#include "../FeatureModule.h"
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Init.h>
#include <cstring>


class OaVkVisionFeatures : public OaVkFeatureModule {
public:
	OaStringView Name() const override {
		return "Vision";
	}

	void ProbeExtensions(
		const OaVec<VkExtensionProperties>& InAvailableExtensions,
		OaVkPhysExtProbe& OutProbe
	) override {
		for (const auto& ext : InAvailableExtensions) {
			if (strcmp(ext.extensionName, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) == 0) {
				OutProbe.KhrVideoQueue = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME) == 0) {
				OutProbe.KhrVideoDecodeQueue = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME) == 0) {
				OutProbe.KhrVideoDecodeH264 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME) == 0) {
				OutProbe.KhrVideoDecodeH265 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME) == 0) {
				OutProbe.KhrVideoDecodeAV1 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME) == 0) {
				OutProbe.KhrVideoDecodeVP9 = true;
				Vp9ExtensionPresent_ = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME) == 0) {
				OutProbe.KhrVideoEncodeQueue = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME) == 0) {
				OutProbe.KhrVideoEncodeH264 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME) == 0) {
				OutProbe.KhrVideoEncodeH265 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME) == 0) {
				OutProbe.KhrVideoEncodeAV1 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME) == 0) {
				OutProbe.KhrSamplerYcbcr = true;
			}
		}
	}

	void QueryFeatures(
		VkPhysicalDevice InPhysicalDevice,
		OaVkDeviceFeatureBundle& OutBundle
	) override {
		// Most video decode codecs (H.264/H.265/AV1) are gated purely by their
		// extension. VP9 decode is the exception: VK_KHR_video_decode_vp9 adds a
		// VkPhysicalDeviceVideoDecodeVP9FeaturesKHR::videoDecodeVP9 bit that must
		// be queried and explicitly enabled, or vkCreateVideoSessionKHR rejects a
		// VP9 profile and the decoder produces undefined output.
#if defined(VK_KHR_video_decode_vp9)
		if (Vp9ExtensionPresent_) {
			VkPhysicalDeviceVideoDecodeVP9FeaturesKHR vp9 = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR,
			};
			VkPhysicalDeviceFeatures2 f2 = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
			};
			f2.pNext = &vp9;
			vkGetPhysicalDeviceFeatures2(InPhysicalDevice, &f2);
			OutBundle.HasVideoDecodeVp9 = vp9.videoDecodeVP9 == VK_TRUE;
		}
#endif
	}

	void BuildFeatureChain(
		OaVkDeviceFeatureBundle& InOutBundle
	) override {
		// YCbCr conversion is part of Vulkan 1.1 core features (Core module).
		// VP9 decode requires its feature bit enabled and chained into device
		// creation (see QueryFeatures).
#if defined(VK_KHR_video_decode_vp9)
		if (InOutBundle.HasVideoDecodeVp9) {
			InOutBundle.DecodeVp9Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR;
			InOutBundle.DecodeVp9Features.videoDecodeVP9 = VK_TRUE;
			InOutBundle.DecodeVp9Features.pNext = InOutBundle.Features13.pNext;
			InOutBundle.Features13.pNext = &InOutBundle.DecodeVp9Features;
		}
#endif
	}

	void CollectExtensions(
		const OaVkPhysExtProbe& InProbe,
		const OaVkDeviceFeatureBundle& InBundle,
		OaVec<const char*>& OutExtensions
	) override {
		// Video Queue (base for all video operations)
		if (InProbe.KhrVideoQueue) {
			OutExtensions.PushBack(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);

			// Video Decode
			if (InProbe.KhrVideoDecodeQueue) {
				OutExtensions.PushBack(VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME);

				// Decode codecs
				if (InProbe.KhrVideoDecodeH264) {
					OutExtensions.PushBack(VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME);
				}
				if (InProbe.KhrVideoDecodeH265) {
					OutExtensions.PushBack(VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME);
				}
				if (InProbe.KhrVideoDecodeAV1) {
					OutExtensions.PushBack(VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME);
				}
				if (InProbe.KhrVideoDecodeVP9 && InBundle.HasVideoDecodeVp9) {
					OutExtensions.PushBack(VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME);
				}
			}

			// Video Encode
			if (InProbe.KhrVideoEncodeQueue) {
				OutExtensions.PushBack(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME);

				// Encode codecs
				if (InProbe.KhrVideoEncodeH264) {
					OutExtensions.PushBack(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME);
				}
				if (InProbe.KhrVideoEncodeH265) {
					OutExtensions.PushBack(VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME);
				}
				if (InProbe.KhrVideoEncodeAV1) {
					OutExtensions.PushBack(VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME);
				}
			}
		}

		// YCbCr sampler conversion (useful for video texture sampling)
		if (InProbe.KhrSamplerYcbcr) {
			OutExtensions.PushBack(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
		}
	}

	OaVec<OaStringView> Dependencies() const override {
		return {"Core"};
	}

private:
	// Remembered from ProbeExtensions so QueryFeatures only queries the VP9
	// feature struct when the extension is actually present on this device.
	bool Vp9ExtensionPresent_ = false;
};


OaUniquePtr<OaVkFeatureModule> OaVkCreateVisionFeatures() {
	return OaMakeUniquePtr<OaVkVisionFeatures>();
}
