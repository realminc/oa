// OA Vulkan ML Features Module
// Handles ML-specific features:
// - Cooperative Matrix (KHR + NV fallback)
// - BFloat16 support
// - Integer Dot Product (INT8 quantization)
// - Device Generated Commands (Phase 2b DGC)

#include "../FeatureModule.h"
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Init.h>
#include <cstring>


class OaVkMlFeatures : public OaVkFeatureModule {
public:
	OaStringView Name() const override {
		return "ML";
	}

	void ProbeExtensions(
		const OaVec<VkExtensionProperties>& InAvailableExtensions,
		OaVkPhysExtProbe& OutProbe
	) override {
		for (const auto& ext : InAvailableExtensions) {
			if (strcmp(ext.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
				OutProbe.KhrCooperativeMatrix = true;
			}
			else if (strcmp(ext.extensionName, "VK_NV_cooperative_matrix") == 0) {
				OutProbe.NvCooperativeMatrix = true;
			}
			else if (strcmp(ext.extensionName, OaVkExtNvCooperativeVector) == 0) {
				OutProbe.NvCooperativeVector = true;
			}
			else if (strcmp(ext.extensionName, OaVkExtNvCooperativeMatrix2) == 0) {
				OutProbe.NvCooperativeMatrix2 = true;
			}
			else if (strcmp(ext.extensionName, OaVkExtNvCooperativeMatrixDecodeVector) == 0) {
				OutProbe.NvCooperativeMatrixDecodeVector = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) == 0) {
				OutProbe.KhrShaderBfloat16 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_MAINTENANCE_5_EXTENSION_NAME) == 0) {
				OutProbe.KhrMaintenance5 = true;
			}
			else if (strcmp(ext.extensionName, VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME) == 0) {
				OutProbe.ExtDeviceGeneratedCommands = true;
			}
		}
	}

	void QueryFeatures(
		VkPhysicalDevice InPhysicalDevice,
		OaVkDeviceFeatureBundle& OutBundle
	) override {
		// Query Cooperative Matrix features
		if (OutBundle.Supported13.pNext == nullptr) {
			OutBundle.SupportedCoopMat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
			OutBundle.Supported13.pNext = &OutBundle.SupportedCoopMat;
		} else {
			// Chain after existing features
			void* lastNext = &OutBundle.Supported13;
			while (static_cast<VkBaseOutStructure*>(lastNext)->pNext != nullptr) {
				lastNext = static_cast<VkBaseOutStructure*>(lastNext)->pNext;
			}
			OutBundle.SupportedCoopMat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
			static_cast<VkBaseOutStructure*>(lastNext)->pNext = 
				reinterpret_cast<VkBaseOutStructure*>(&OutBundle.SupportedCoopMat);
		}

		// Query BFloat16 features
		OutBundle.SupportedBf16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
		OutBundle.SupportedBf16.pNext = OutBundle.SupportedCoopMat.pNext;
		OutBundle.SupportedCoopMat.pNext = &OutBundle.SupportedBf16;

		// Query NVIDIA CooperativeVector features.
		OutBundle.SupportedCoopVec.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV;
		OutBundle.SupportedCoopVec.pNext = OutBundle.SupportedBf16.pNext;
		OutBundle.SupportedBf16.pNext = &OutBundle.SupportedCoopVec;

#if defined(VK_NV_cooperative_matrix2)
		OutBundle.SupportedCoopMat2.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
		OutBundle.SupportedCoopMat2.pNext = OutBundle.SupportedCoopVec.pNext;
		OutBundle.SupportedCoopVec.pNext = &OutBundle.SupportedCoopMat2;
#endif

#if defined(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME)
		// Newer Vulkan headers expose the decode-vector feature struct. Older
		// SDKs still get extension probing below, but cannot enable this SPIR-V
		// capability until the headers are updated.
		OutBundle.SupportedCoopMatDecodeVector.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV;
		OutBundle.SupportedCoopMatDecodeVector.pNext = OutBundle.SupportedBf16.pNext;
		OutBundle.SupportedBf16.pNext = &OutBundle.SupportedCoopMatDecodeVector;
#endif

		// Re-query with extended chain
		vkGetPhysicalDeviceFeatures2(InPhysicalDevice, &OutBundle.SupportedFeatures2);

		// Check Integer Dot Product (part of Vulkan 1.3)
		OutBundle.HasIntDotProduct = OutBundle.Supported13.shaderIntegerDotProduct == VK_TRUE;

		// Check CooperativeVector. Feature presence alone is not enough; it must
		// also support compute-stage shaders.
		OutBundle.HasCoopVector = OutBundle.SupportedCoopVec.cooperativeVector == VK_TRUE;
		if (OutBundle.HasCoopVector) {
			VkPhysicalDeviceCooperativeVectorPropertiesNV coopVecProps{};
			coopVecProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_PROPERTIES_NV;

			VkPhysicalDeviceProperties2 physProps{};
			physProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
			physProps.pNext = &coopVecProps;
			vkGetPhysicalDeviceProperties2(InPhysicalDevice, &physProps);

			OutBundle.HasCoopVector =
				(coopVecProps.cooperativeVectorSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
		}

#if defined(VK_NV_cooperative_matrix2)
		OutBundle.HasCoopMatrix2 =
			OutBundle.SupportedCoopMat2.cooperativeMatrixWorkgroupScope == VK_TRUE &&
			OutBundle.SupportedCoopMat2.cooperativeMatrixFlexibleDimensions == VK_TRUE &&
			OutBundle.SupportedCoopMat2.cooperativeMatrixTensorAddressing == VK_TRUE;
#else
		OutBundle.HasCoopMatrix2 = false;
#endif

#if defined(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME)
		OutBundle.HasCoopMatrixDecodeVector =
			OutBundle.SupportedCoopMatDecodeVector.cooperativeMatrixDecodeVector == VK_TRUE;
#else
		OutBundle.HasCoopMatrixDecodeVector = false;
#endif

		// Check Device Generated Commands
		if (OutBundle.Supported13.pNext) {
			VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT dgcFeatures{};
			dgcFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT;
			
			void* lastNext = &OutBundle.Supported13;
			while (static_cast<VkBaseOutStructure*>(lastNext)->pNext != nullptr) {
				lastNext = static_cast<VkBaseOutStructure*>(lastNext)->pNext;
			}
			static_cast<VkBaseOutStructure*>(lastNext)->pNext = 
				reinterpret_cast<VkBaseOutStructure*>(&dgcFeatures);
			
			vkGetPhysicalDeviceFeatures2(InPhysicalDevice, &OutBundle.SupportedFeatures2);
			OutBundle.HasDeviceGeneratedCommands = dgcFeatures.deviceGeneratedCommands == VK_TRUE;
		}

		// Refine Cooperative Matrix capability (enumerate shapes)
		RefineCooperativeMatrixCapability(InPhysicalDevice, OutBundle);
	}

	void BuildFeatureChain(
		OaVkDeviceFeatureBundle& InOutBundle
	) override {
		// Enable Cooperative Matrix if available
		if (InOutBundle.HasCoopMatrix && InOutBundle.SupportedCoopMat.cooperativeMatrix) {
			InOutBundle.CoopMatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
			InOutBundle.CoopMatFeatures.cooperativeMatrix = VK_TRUE;
			InOutBundle.CoopMatFeatures.pNext = InOutBundle.Features13.pNext;
			InOutBundle.Features13.pNext = &InOutBundle.CoopMatFeatures;
		}

		// Enable BFloat16 if available
		if (InOutBundle.SupportedBf16.shaderBFloat16Type) {
			InOutBundle.EnableBf16Feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
			InOutBundle.EnableBf16Feat.shaderBFloat16Type = InOutBundle.SupportedBf16.shaderBFloat16Type;
			InOutBundle.EnableBf16Feat.shaderBFloat16DotProduct = InOutBundle.SupportedBf16.shaderBFloat16DotProduct;
			InOutBundle.EnableBf16Feat.shaderBFloat16CooperativeMatrix = InOutBundle.SupportedBf16.shaderBFloat16CooperativeMatrix;
			InOutBundle.EnableBf16Feat.pNext = InOutBundle.Features13.pNext;
			InOutBundle.Features13.pNext = &InOutBundle.EnableBf16Feat;
			InOutBundle.WantEnableBf16Ext = true;
		}

		// Enable Integer Dot Product (Vulkan 1.3 feature)
		if (InOutBundle.HasIntDotProduct) {
			InOutBundle.Features13.shaderIntegerDotProduct = VK_TRUE;
		}

		// Enable NVIDIA CooperativeVector when available. Routing remains
		// vendor/shape gated, so this only makes the kernel legal to create.
		if (InOutBundle.HasCoopVector) {
			InOutBundle.CoopVecFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV;
			InOutBundle.CoopVecFeatures.cooperativeVector = VK_TRUE;
			InOutBundle.CoopVecFeatures.cooperativeVectorTraining =
				InOutBundle.SupportedCoopVec.cooperativeVectorTraining;
			InOutBundle.CoopVecFeatures.pNext = InOutBundle.Features13.pNext;
			InOutBundle.Features13.pNext = &InOutBundle.CoopVecFeatures;
		}

#if defined(VK_NV_cooperative_matrix2)
		if (InOutBundle.HasCoopMatrix2 && !OaEnvFlag::IsSet("OA_DISABLE_COOPMAT2")) {
			InOutBundle.CoopMat2Features.sType =
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
			InOutBundle.CoopMat2Features.cooperativeMatrixWorkgroupScope =
				InOutBundle.SupportedCoopMat2.cooperativeMatrixWorkgroupScope;
			InOutBundle.CoopMat2Features.cooperativeMatrixFlexibleDimensions =
				InOutBundle.SupportedCoopMat2.cooperativeMatrixFlexibleDimensions;
			InOutBundle.CoopMat2Features.cooperativeMatrixReductions =
				InOutBundle.SupportedCoopMat2.cooperativeMatrixReductions;
			InOutBundle.CoopMat2Features.cooperativeMatrixConversions =
				InOutBundle.SupportedCoopMat2.cooperativeMatrixConversions;
			InOutBundle.CoopMat2Features.cooperativeMatrixPerElementOperations =
				InOutBundle.SupportedCoopMat2.cooperativeMatrixPerElementOperations;
			InOutBundle.CoopMat2Features.cooperativeMatrixTensorAddressing =
				InOutBundle.SupportedCoopMat2.cooperativeMatrixTensorAddressing;
			InOutBundle.CoopMat2Features.cooperativeMatrixBlockLoads =
				InOutBundle.SupportedCoopMat2.cooperativeMatrixBlockLoads;
			InOutBundle.CoopMat2Features.pNext = InOutBundle.Features13.pNext;
			InOutBundle.Features13.pNext = &InOutBundle.CoopMat2Features;
		}
#endif

#if defined(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME)
		if (InOutBundle.HasCoopMatrixDecodeVector) {
			InOutBundle.CoopMatDecodeVectorFeatures.sType =
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV;
			InOutBundle.CoopMatDecodeVectorFeatures.cooperativeMatrixDecodeVector = VK_TRUE;
			InOutBundle.CoopMatDecodeVectorFeatures.pNext = InOutBundle.Features13.pNext;
			InOutBundle.Features13.pNext = &InOutBundle.CoopMatDecodeVectorFeatures;
		}
#endif

		// Enable Device Generated Commands if available
		if (InOutBundle.HasDeviceGeneratedCommands) {
			InOutBundle.DeviceGenFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT;
			InOutBundle.DeviceGenFeatures.deviceGeneratedCommands = VK_TRUE;
			InOutBundle.DeviceGenFeatures.pNext = InOutBundle.Features13.pNext;
			InOutBundle.Features13.pNext = &InOutBundle.DeviceGenFeatures;
		}
	}

	void CollectExtensions(
		const OaVkPhysExtProbe& InProbe,
		const OaVkDeviceFeatureBundle& InBundle,
		OaVec<const char*>& OutExtensions
	) override {
		// Cooperative Matrix
		if (InBundle.HasCoopMatrix && InProbe.KhrCooperativeMatrix) {
			OutExtensions.PushBack(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
		}

		// BFloat16
		if (InBundle.WantEnableBf16Ext && InProbe.KhrShaderBfloat16) {
			OutExtensions.PushBack(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
		}

		// NVIDIA CooperativeVector / CooperativeMatrix2 support.
		if (InBundle.HasCoopVector && InProbe.NvCooperativeVector) {
			OutExtensions.PushBack(OaVkExtNvCooperativeVector);
		}
#if defined(VK_NV_cooperative_matrix2)
		if (InBundle.HasCoopMatrix2 && InProbe.NvCooperativeMatrix2
			&& !OaEnvFlag::IsSet("OA_DISABLE_COOPMAT2"))
		{
			OutExtensions.PushBack(OaVkExtNvCooperativeMatrix2);
		}
#endif
#if defined(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME)
		if (InBundle.HasCoopMatrixDecodeVector && InBundle.HasCoopMatrix2
			&& InProbe.NvCooperativeMatrix2 && InProbe.NvCooperativeMatrixDecodeVector
			&& !OaEnvFlag::IsSet("OA_DISABLE_COOPMAT2"))
		{
			OutExtensions.PushBack(OaVkExtNvCooperativeMatrixDecodeVector);
		}
#endif

		// Device Generated Commands (requires Maintenance5)
		if (InBundle.HasDeviceGeneratedCommands && InProbe.ExtDeviceGeneratedCommands) {
			if (InProbe.KhrMaintenance5) {
				OutExtensions.PushBack(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
			}
			OutExtensions.PushBack(VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME);
		}
	}

	OaVec<OaStringView> Dependencies() const override {
		return {"Core"};
	}

private:
	void RefineCooperativeMatrixCapability(
		VkPhysicalDevice InPhys,
		OaVkDeviceFeatureBundle& InOutBundle
	) {
		bool& hasCoopMatrix = InOutBundle.HasCoopMatrix;
		hasCoopMatrix = InOutBundle.SupportedCoopMat.cooperativeMatrix == VK_TRUE;

		if (!hasCoopMatrix) {
			return;
		}

		if (!vkGetPhysicalDeviceProperties2 || !vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
			hasCoopMatrix = false;
			return;
		}

		// Check if compute stage is supported
		VkPhysicalDeviceCooperativeMatrixPropertiesKHR physCoopMatProps{};
		physCoopMatProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
		
		VkPhysicalDeviceProperties2 physPropsCoop{};
		physPropsCoop.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		physPropsCoop.pNext = &physCoopMatProps;
		
		vkGetPhysicalDeviceProperties2(InPhys, &physPropsCoop);

		if ((physCoopMatProps.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
			hasCoopMatrix = false;
			return;
		}

		// Enumerate cooperative matrix shapes
		OaU32 coopCount = 0;
		VkResult coopEnum = vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(InPhys, &coopCount, nullptr);
		if (coopEnum != VK_SUCCESS || coopCount == 0) {
			hasCoopMatrix = false;
			return;
		}

		OaVec<VkCooperativeMatrixPropertiesKHR> coopProps(coopCount);
		for (OaU32 i = 0; i < coopCount; ++i) {
			coopProps[i] = {};
			coopProps[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
		}

		coopEnum = vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(InPhys, &coopCount, coopProps.Data());
		if (coopEnum != VK_SUCCESS) {
			hasCoopMatrix = false;
			return;
		}

		// Parse shapes and populate DiscoveredCoopMatShapes
		auto& shapes = InOutBundle.DiscoveredCoopMatShapes;
		shapes.TotalShapesEnumerated = coopCount;

		auto captureFirst = [](OaVkCoopMatShape& s, const VkCooperativeMatrixPropertiesKHR& InProp) {
			if (s.Available) return;
			s.M = InProp.MSize;
			s.N = InProp.NSize;
			s.K = InProp.KSize;
			s.Scope = InProp.scope;
			s.Available = true;
		};

		bool foundUsableCoop16 = false;
		for (OaU32 i = 0; i < coopCount; ++i) {
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
				captureFirst(shapes.Fp16AccFp32, p);
				if (isSubgroup && p.MSize == 16 && p.NSize == 16 && p.KSize == 16) {
					shapes.Has16x16x16_Fp32Acc = true;
					foundUsableCoop16 = true;
				}
			} else if (abFp16 && cFp16) {
				captureFirst(shapes.Fp16AccFp16, p);
				if (isSubgroup && p.MSize == 16 && p.NSize == 16 && p.KSize == 16) {
					shapes.Has16x16x16_Fp16Acc = true;
					foundUsableCoop16 = true;
				}
			} else if (abBf16 && cFp32) {
				captureFirst(isSubgroup ? shapes.Bf16AccFp32 : shapes.Bf16AccFp32Workgroup, p);
				if (isSubgroup && p.MSize == 16 && p.NSize == 16 && p.KSize == 16) {
					shapes.Has16x16x16_Fp32Acc = true;
					foundUsableCoop16 = true;
				}
			} else if (abInt8 && cInt32) {
				captureFirst(shapes.Int8AccInt32, p);
			}
		}

		// Until kernels are spec-const-driven, gate on 16x16x16 + FP32-acc
		if (!foundUsableCoop16) {
			hasCoopMatrix = false;
		}

		// CoopMat2 NV — record feature + property fields when the device
		// advertises them. R2 of the gemm-router rewrite reads these out of
		// OaVkCoopMatShapes when building the cap mask.
#if defined(VK_NV_cooperative_matrix2)
		if (InOutBundle.HasCoopMatrix2) {
			auto& s = InOutBundle.DiscoveredCoopMatShapes;
			s.CoopMat2Supported          = true;
			s.CoopMat2WorkgroupScope     =
				InOutBundle.SupportedCoopMat2.cooperativeMatrixWorkgroupScope == VK_TRUE;
			s.CoopMat2FlexibleDimensions =
				InOutBundle.SupportedCoopMat2.cooperativeMatrixFlexibleDimensions == VK_TRUE;
			s.CoopMat2Reductions         =
				InOutBundle.SupportedCoopMat2.cooperativeMatrixReductions == VK_TRUE;
			s.CoopMat2PerElementOps      =
				InOutBundle.SupportedCoopMat2.cooperativeMatrixPerElementOperations == VK_TRUE;
			s.CoopMat2TensorAddressing   =
				InOutBundle.SupportedCoopMat2.cooperativeMatrixTensorAddressing == VK_TRUE;
			s.CoopMat2BlockLoads         =
				InOutBundle.SupportedCoopMat2.cooperativeMatrixBlockLoads == VK_TRUE;

			VkPhysicalDeviceCooperativeMatrix2PropertiesNV cm2Props{};
			cm2Props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES_NV;

			VkPhysicalDeviceProperties2 physProps2{};
			physProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
			physProps2.pNext = &cm2Props;
			vkGetPhysicalDeviceProperties2(InPhys, &physProps2);

			s.CoopMat2WorkgroupMaxWgSize    = cm2Props.cooperativeMatrixWorkgroupScopeMaxWorkgroupSize;
			s.CoopMat2FlexibleDimMax        = cm2Props.cooperativeMatrixFlexibleDimensionsMaxDimension;
			s.CoopMat2WorkgroupReservedSmem = cm2Props.cooperativeMatrixWorkgroupScopeReservedSharedMemory;
		}
#endif
	}
};


OaUniquePtr<OaVkFeatureModule> OaVkCreateMlFeatures() {
	return OaMakeUniquePtr<OaVkMlFeatures>();
}
