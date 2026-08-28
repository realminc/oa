// OA vulkan ML features Module
// Handles ML-specific features:
// - Cooperative Matrix (KHR + NV fallback)
// - BFloat16 support
// - Integer Dot product (INT8 quantization)
// - Device generated commands (phase 2b DGC)

#include "../featureModule.h"
#include <oa/core/envFlag.h>
#include <oa/core/log.h>
#include <oa/runtime/init.h>
#include <string.h>


class MlFeatures : public oavk::FeatureModule {
public:
	oa::StringView name() const override {
		return "ML";
	}

	void probeExtensions(
		const oa::Vector<VkExtensionProperties>& inAvailableExtensions,
		oavk::PhysicalExtensionProbe& outProbe
	) override {
		for (const auto& ext : inAvailableExtensions) {
			if (strcmp(ext.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
				outProbe.khrCooperativeMatrix = true;
			}
			else if (strcmp(ext.extensionName, "VK_NV_cooperative_matrix") == 0) {
				outProbe.nvCooperativeMatrix = true;
			}
			else if (strcmp(ext.extensionName, oavk::ExtNvCooperativeVector) == 0) {
				outProbe.nvCooperativeVector = true;
			}
			else if (strcmp(ext.extensionName, oavk::ExtNvCooperativeMatrix2) == 0) {
				outProbe.nvCooperativeMatrix2 = true;
			}
			else if (strcmp(ext.extensionName, oavk::ExtNvCooperativeMatrixDecodeVector) == 0) {
				outProbe.nvCooperativeMatrixDecodeVector = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) == 0) {
				outProbe.khrShaderBfloat16 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_MAINTENANCE_5_EXTENSION_NAME) == 0) {
				outProbe.khrMaintenance5 = true;
			}
			else if (strcmp(ext.extensionName, VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME) == 0) {
				outProbe.extDeviceGeneratedCommands = true;
			}
		}
	}

	void queryFeatures(
		const VklInstanceTable& inDispatch,
		VkPhysicalDevice inPhysicalDevice,
		oavk::DeviceFeatureBundle& outBundle
	) override {
		// query Cooperative Matrix features
		if (outBundle.supported13.pNext == nullptr) {
			outBundle.supportedCoopMat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
			outBundle.supported13.pNext = &outBundle.supportedCoopMat;
		} else {
			// Chain after existing features
			void* lastNext = &outBundle.supported13;
			while (static_cast<VkBaseOutStructure*>(lastNext)->pNext != nullptr) {
				lastNext = static_cast<VkBaseOutStructure*>(lastNext)->pNext;
			}
			outBundle.supportedCoopMat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
			static_cast<VkBaseOutStructure*>(lastNext)->pNext = 
				reinterpret_cast<VkBaseOutStructure*>(&outBundle.supportedCoopMat);
		}

		// query BFloat16 features
		outBundle.supportedBf16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
		outBundle.supportedBf16.pNext = outBundle.supportedCoopMat.pNext;
		outBundle.supportedCoopMat.pNext = &outBundle.supportedBf16;

		// query NVIDIA cooperativeVector features.
		outBundle.supportedCoopVec.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV;
		outBundle.supportedCoopVec.pNext = outBundle.supportedBf16.pNext;
		outBundle.supportedBf16.pNext = &outBundle.supportedCoopVec;

#if defined(VK_NV_cooperative_matrix2)
		outBundle.supportedCoopMat2.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
		outBundle.supportedCoopMat2.pNext = outBundle.supportedCoopVec.pNext;
		outBundle.supportedCoopVec.pNext = &outBundle.supportedCoopMat2;
#endif

#if defined(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME)
		// Newer vulkan headers expose the decode-vector feature struct. Older
		// SDKs still get extension probing below, but cannot enable this SPIR-V
		// capability until the headers are updated.
		outBundle.supportedCoopMatDecodeVector.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV;
		outBundle.supportedCoopMatDecodeVector.pNext = outBundle.supportedBf16.pNext;
		outBundle.supportedBf16.pNext = &outBundle.supportedCoopMatDecodeVector;
#endif

		// Re-query with extended chain
		inDispatch.vkGetPhysicalDeviceFeatures2(
			inPhysicalDevice, &outBundle.supportedFeatures2);

		// Check Integer Dot product (part of vulkan 1.3)
		outBundle.hasIntDotProduct = outBundle.supported13.shaderIntegerDotProduct == VK_TRUE;

		// Check cooperativeVector. Feature presence alone is not enough; it must
		// also support compute-stage shaders.
		outBundle.hasCoopVector = outBundle.supportedCoopVec.cooperativeVector == VK_TRUE;
		if (outBundle.hasCoopVector) {
			VkPhysicalDeviceCooperativeVectorPropertiesNV coopVecProps{};
			coopVecProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_PROPERTIES_NV;

			VkPhysicalDeviceProperties2 physProps{};
			physProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
			physProps.pNext = &coopVecProps;
			inDispatch.vkGetPhysicalDeviceProperties2(inPhysicalDevice, &physProps);

			outBundle.hasCoopVector =
				(coopVecProps.cooperativeVectorSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
		}

#if defined(VK_NV_cooperative_matrix2)
		outBundle.hasCoopMatrix2 =
			outBundle.supportedCoopMat2.cooperativeMatrixWorkgroupScope == VK_TRUE &&
			outBundle.supportedCoopMat2.cooperativeMatrixFlexibleDimensions == VK_TRUE &&
			outBundle.supportedCoopMat2.cooperativeMatrixTensorAddressing == VK_TRUE;
#else
		outBundle.hasCoopMatrix2 = false;
#endif

#if defined(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME)
		outBundle.hasCoopMatrixDecodeVector =
			outBundle.supportedCoopMatDecodeVector.cooperativeMatrixDecodeVector == VK_TRUE;
#else
		outBundle.hasCoopMatrixDecodeVector = false;
#endif

		// Check Device generated commands
		if (outBundle.supported13.pNext) {
			VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT dgcFeatures{};
			dgcFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT;
			
			void* lastNext = &outBundle.supported13;
			while (static_cast<VkBaseOutStructure*>(lastNext)->pNext != nullptr) {
				lastNext = static_cast<VkBaseOutStructure*>(lastNext)->pNext;
			}
			static_cast<VkBaseOutStructure*>(lastNext)->pNext = 
				reinterpret_cast<VkBaseOutStructure*>(&dgcFeatures);
			
			inDispatch.vkGetPhysicalDeviceFeatures2(
				inPhysicalDevice, &outBundle.supportedFeatures2);
			outBundle.hasDeviceGeneratedCommands = dgcFeatures.deviceGeneratedCommands == VK_TRUE;
		}

		// Refine Cooperative Matrix capability (enumerate shapes)
		refineCooperativeMatrixCapability(inDispatch, inPhysicalDevice, outBundle);
	}

	void buildFeatureChain(
		oavk::DeviceFeatureBundle& inOutBundle
	) override {
		// Enable Cooperative Matrix if available
		if (inOutBundle.hasCoopMatrix && inOutBundle.supportedCoopMat.cooperativeMatrix) {
			inOutBundle.coopMatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
			inOutBundle.coopMatFeatures.cooperativeMatrix = VK_TRUE;
			inOutBundle.coopMatFeatures.pNext = inOutBundle.features13.pNext;
			inOutBundle.features13.pNext = &inOutBundle.coopMatFeatures;
		}

		// Enable BFloat16 if available
		if (inOutBundle.supportedBf16.shaderBFloat16Type) {
			inOutBundle.enableBf16Feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
			inOutBundle.enableBf16Feat.shaderBFloat16Type = inOutBundle.supportedBf16.shaderBFloat16Type;
			inOutBundle.enableBf16Feat.shaderBFloat16DotProduct = inOutBundle.supportedBf16.shaderBFloat16DotProduct;
			inOutBundle.enableBf16Feat.shaderBFloat16CooperativeMatrix = inOutBundle.supportedBf16.shaderBFloat16CooperativeMatrix;
			inOutBundle.enableBf16Feat.pNext = inOutBundle.features13.pNext;
			inOutBundle.features13.pNext = &inOutBundle.enableBf16Feat;
			inOutBundle.wantEnableBf16Ext = true;
		}

		// Enable Integer Dot product (vulkan 1.3 feature)
		if (inOutBundle.hasIntDotProduct) {
			inOutBundle.features13.shaderIntegerDotProduct = VK_TRUE;
		}

		// Enable NVIDIA cooperativeVector when available. Routing remains
		// vendor/shape gated, so this only makes the kernel legal to create.
		if (inOutBundle.hasCoopVector) {
			inOutBundle.coopVecFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV;
			inOutBundle.coopVecFeatures.cooperativeVector = VK_TRUE;
			inOutBundle.coopVecFeatures.cooperativeVectorTraining =
				inOutBundle.supportedCoopVec.cooperativeVectorTraining;
			inOutBundle.coopVecFeatures.pNext = inOutBundle.features13.pNext;
			inOutBundle.features13.pNext = &inOutBundle.coopVecFeatures;
		}

#if defined(VK_NV_cooperative_matrix2)
		if (inOutBundle.hasCoopMatrix2 && !oa::EnvFlag::isSet("OA_DISABLE_COOPMAT2")) {
			inOutBundle.coopMat2Features.sType =
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
			inOutBundle.coopMat2Features.cooperativeMatrixWorkgroupScope =
				inOutBundle.supportedCoopMat2.cooperativeMatrixWorkgroupScope;
			inOutBundle.coopMat2Features.cooperativeMatrixFlexibleDimensions =
				inOutBundle.supportedCoopMat2.cooperativeMatrixFlexibleDimensions;
			inOutBundle.coopMat2Features.cooperativeMatrixReductions =
				inOutBundle.supportedCoopMat2.cooperativeMatrixReductions;
			inOutBundle.coopMat2Features.cooperativeMatrixConversions =
				inOutBundle.supportedCoopMat2.cooperativeMatrixConversions;
			inOutBundle.coopMat2Features.cooperativeMatrixPerElementOperations =
				inOutBundle.supportedCoopMat2.cooperativeMatrixPerElementOperations;
			inOutBundle.coopMat2Features.cooperativeMatrixTensorAddressing =
				inOutBundle.supportedCoopMat2.cooperativeMatrixTensorAddressing;
			inOutBundle.coopMat2Features.cooperativeMatrixBlockLoads =
				inOutBundle.supportedCoopMat2.cooperativeMatrixBlockLoads;
			inOutBundle.coopMat2Features.pNext = inOutBundle.features13.pNext;
			inOutBundle.features13.pNext = &inOutBundle.coopMat2Features;
		}
#endif

#if defined(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME)
		if (inOutBundle.hasCoopMatrixDecodeVector) {
			inOutBundle.coopMatDecodeVectorFeatures.sType =
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV;
			inOutBundle.coopMatDecodeVectorFeatures.cooperativeMatrixDecodeVector = VK_TRUE;
			inOutBundle.coopMatDecodeVectorFeatures.pNext = inOutBundle.features13.pNext;
			inOutBundle.features13.pNext = &inOutBundle.coopMatDecodeVectorFeatures;
		}
#endif

		// Enable Device generated commands if available
		if (inOutBundle.hasDeviceGeneratedCommands) {
			inOutBundle.deviceGenFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT;
			inOutBundle.deviceGenFeatures.deviceGeneratedCommands = VK_TRUE;
			inOutBundle.deviceGenFeatures.pNext = inOutBundle.features13.pNext;
			inOutBundle.features13.pNext = &inOutBundle.deviceGenFeatures;
		}
	}

	void collectExtensions(
		const oavk::PhysicalExtensionProbe& inProbe,
		const oavk::DeviceFeatureBundle& inBundle,
		oa::Vector<const char*>& outExtensions
	) override {
		// Cooperative Matrix
		if (inBundle.hasCoopMatrix && inProbe.khrCooperativeMatrix) {
			outExtensions.pushBack(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
		}

		// BFloat16
		if (inBundle.wantEnableBf16Ext && inProbe.khrShaderBfloat16) {
			outExtensions.pushBack(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
		}

		// NVIDIA cooperativeVector / CooperativeMatrix2 support.
		if (inBundle.hasCoopVector && inProbe.nvCooperativeVector) {
			outExtensions.pushBack(oavk::ExtNvCooperativeVector);
		}
#if defined(VK_NV_cooperative_matrix2)
		if (inBundle.hasCoopMatrix2 && inProbe.nvCooperativeMatrix2
			&& !oa::EnvFlag::isSet("OA_DISABLE_COOPMAT2"))
		{
			outExtensions.pushBack(oavk::ExtNvCooperativeMatrix2);
		}
#endif
#if defined(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME)
		if (inBundle.hasCoopMatrixDecodeVector && inBundle.hasCoopMatrix2
			&& inProbe.nvCooperativeMatrix2 && inProbe.nvCooperativeMatrixDecodeVector
			&& !oa::EnvFlag::isSet("OA_DISABLE_COOPMAT2"))
		{
			outExtensions.pushBack(oavk::ExtNvCooperativeMatrixDecodeVector);
		}
#endif

		// Device generated commands (requires Maintenance5)
		if (inBundle.hasDeviceGeneratedCommands && inProbe.extDeviceGeneratedCommands) {
			if (inProbe.khrMaintenance5) {
				outExtensions.pushBack(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
			}
			outExtensions.pushBack(VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME);
		}
	}

	oa::Vector<oa::StringView> dependencies() const override {
		return {"Core"};
	}

private:
	void refineCooperativeMatrixCapability(
		const VklInstanceTable& inDispatch,
		VkPhysicalDevice inPhys,
		oavk::DeviceFeatureBundle& inOutBundle
	) {
		bool& hasCoopMatrix = inOutBundle.hasCoopMatrix;
		hasCoopMatrix = inOutBundle.supportedCoopMat.cooperativeMatrix == VK_TRUE;

		if (!hasCoopMatrix) {
			return;
		}

		if (!inDispatch.vkGetPhysicalDeviceProperties2
			|| !inDispatch.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
			hasCoopMatrix = false;
			return;
		}

		// Check if compute stage is supported
		VkPhysicalDeviceCooperativeMatrixPropertiesKHR physCoopMatProps{};
		physCoopMatProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
		
		VkPhysicalDeviceProperties2 physPropsCoop{};
		physPropsCoop.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		physPropsCoop.pNext = &physCoopMatProps;
		
		inDispatch.vkGetPhysicalDeviceProperties2(inPhys, &physPropsCoop);

		if ((physCoopMatProps.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
			hasCoopMatrix = false;
			return;
		}

		// Enumerate cooperative matrix shapes
		oa::U32 coopCount = 0;
		VkResult coopEnum = inDispatch.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
			inPhys, &coopCount, nullptr);
		if (coopEnum != VK_SUCCESS || coopCount == 0) {
			hasCoopMatrix = false;
			return;
		}

		oa::Vector<VkCooperativeMatrixPropertiesKHR> coopProps(coopCount);
		for (oa::U32 i = 0; i < coopCount; ++i) {
			coopProps[i] = {};
			coopProps[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
		}

		coopEnum = inDispatch.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
			inPhys, &coopCount, coopProps.data());
		if (coopEnum != VK_SUCCESS) {
			hasCoopMatrix = false;
			return;
		}

		// parse shapes and populate discoveredCoopMatShapes
		auto& shapes = inOutBundle.discoveredCoopMatShapes;
		shapes.totalShapesEnumerated = coopCount;

		auto captureFirst = [](oavk::CoopMatShape& s, const VkCooperativeMatrixPropertiesKHR& inProp) {
			if (s.available) return;
			s.m = inProp.MSize;
			s.n = inProp.NSize;
			s.k = inProp.KSize;
			s.scope = inProp.scope;
			s.available = true;
		};

		bool foundUsableCoop16 = false;
		for (oa::U32 i = 0; i < coopCount; ++i) {
			const VkCooperativeMatrixPropertiesKHR& p = coopProps[i];
			const bool isSubgroup = (p.scope == VK_SCOPE_SUBGROUP_KHR);
			const bool isWorkgroup = (p.scope == VK_SCOPE_WORKGROUP_KHR);
			if (!isSubgroup and !isWorkgroup) continue;

			const bool abFp16 = (p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && 
			                     p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR);
			const bool abBf16 = (p.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR && 
			                     p.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR);
			const bool abInt8 = (p.AType == VK_COMPONENT_TYPE_SINT8_KHR && 
			                     p.BType == VK_COMPONENT_TYPE_SINT8_KHR);
			const bool cFp32 = (p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && 
			                    p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR);
			const bool cFp16 = (p.CType == VK_COMPONENT_TYPE_FLOAT16_KHR && 
			                    p.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR);
			const bool cInt32 = (p.CType == VK_COMPONENT_TYPE_SINT32_KHR && 
			                     p.ResultType == VK_COMPONENT_TYPE_SINT32_KHR);

			if (abFp16 && cFp32) {
				captureFirst(shapes.fp16AccFp32, p);
				if (isSubgroup && p.MSize == 16 && p.NSize == 16 && p.KSize == 16) {
					shapes.has16x16x16_Fp32Acc = true;
					foundUsableCoop16 = true;
				}
			} else if (abFp16 && cFp16) {
				captureFirst(shapes.fp16AccFp16, p);
				if (isSubgroup && p.MSize == 16 && p.NSize == 16 && p.KSize == 16) {
					shapes.has16x16x16_Fp16Acc = true;
					foundUsableCoop16 = true;
				}
			} else if (abBf16 && cFp32) {
				captureFirst(isSubgroup ? shapes.bf16AccFp32 : shapes.bf16AccFp32Workgroup, p);
				if (isSubgroup && p.MSize == 16 && p.NSize == 16 && p.KSize == 16) {
					shapes.has16x16x16_Fp32Acc = true;
					foundUsableCoop16 = true;
				}
			} else if (abInt8 && cInt32) {
				captureFirst(shapes.int8AccInt32, p);
			}
		}

		// Until kernels are spec-const-driven, gate on 16x16x16 + FP32-acc
		if (!foundUsableCoop16) {
			hasCoopMatrix = false;
		}

		// CoopMat2 NV — record feature + property fields when the device
		// advertises them. R2 of the gemm-router rewrite reads these out of
		// oavk::CoopMatShapes when building the cap mask.
#if defined(VK_NV_cooperative_matrix2)
		if (inOutBundle.hasCoopMatrix2) {
			auto& s = inOutBundle.discoveredCoopMatShapes;
			s.coopMat2Supported          = true;
			s.coopMat2WorkgroupScope     =
				inOutBundle.supportedCoopMat2.cooperativeMatrixWorkgroupScope == VK_TRUE;
			s.coopMat2FlexibleDimensions =
				inOutBundle.supportedCoopMat2.cooperativeMatrixFlexibleDimensions == VK_TRUE;
			s.coopMat2Reductions         =
				inOutBundle.supportedCoopMat2.cooperativeMatrixReductions == VK_TRUE;
			s.coopMat2PerElementOps      =
				inOutBundle.supportedCoopMat2.cooperativeMatrixPerElementOperations == VK_TRUE;
			s.coopMat2TensorAddressing   =
				inOutBundle.supportedCoopMat2.cooperativeMatrixTensorAddressing == VK_TRUE;
			s.coopMat2BlockLoads         =
				inOutBundle.supportedCoopMat2.cooperativeMatrixBlockLoads == VK_TRUE;

			VkPhysicalDeviceCooperativeMatrix2PropertiesNV cm2Props{};
			cm2Props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES_NV;

			VkPhysicalDeviceProperties2 physProps2{};
			physProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
			physProps2.pNext = &cm2Props;
			inDispatch.vkGetPhysicalDeviceProperties2(inPhys, &physProps2);

			s.coopMat2WorkgroupMaxWgSize    = cm2Props.cooperativeMatrixWorkgroupScopeMaxWorkgroupSize;
			s.coopMat2FlexibleDimMax        = cm2Props.cooperativeMatrixFlexibleDimensionsMaxDimension;
			s.coopMat2WorkgroupReservedSmem = cm2Props.cooperativeMatrixWorkgroupScopeReservedSharedMemory;
		}
#endif
	}
};


oa::UniquePtr<oavk::FeatureModule> oavk::createMlFeatures() {
	return oa::makeUnique<MlFeatures>();
}
