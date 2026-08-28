#include "../../oaTest.h"

#include "oa/runtime/loader.h"
#include <vkl/vkl.h>

#include <cstdint>

namespace {

VKAPI_ATTR void VKAPI_CALL dispatchMarkerA() {}
VKAPI_ATTR void VKAPI_CALL dispatchMarkerB() {}

PFN_vkVoidFunction markerForHandle(const void* inHandle)
{
	return reinterpret_cast<std::uintptr_t>(inHandle) == 1U
		? &dispatchMarkerA
		: &dispatchMarkerB;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fakeGetInstanceProcAddr(
	VkInstance inInstance,
	const char*)
{
	return markerForHandle(inInstance);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL secondGetInstanceProcAddr(
	VkInstance inInstance,
	const char*)
{
	return markerForHandle(inInstance);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fakeGetDeviceProcAddrA(
	VkDevice,
	const char*)
{
	return &dispatchMarkerA;
}

TEST(VkDispatch, ProcessLoaderSelectionCannotChange) {
	vklInitCustom(&fakeGetInstanceProcAddr);
	EXPECT_TRUE(oavk::initializeLoader(&fakeGetInstanceProcAddr).isOk());
	EXPECT_TRUE(oavk::initializeLoader().isOk());

	const oa::Status changed =
		oavk::initializeLoader(&secondGetInstanceProcAddr);
	EXPECT_EQ(changed.getCode(), oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(vklGetInstanceProcAddr(), &fakeGetInstanceProcAddr);
	vklFinalize();
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fakeGetDeviceProcAddrB(
	VkDevice,
	const char*)
{
	return &dispatchMarkerB;
}

}  // namespace

TEST(VkDispatch, InstanceTablesRemainIndependent)
{
	vklInitCustom(&fakeGetInstanceProcAddr);
	const auto instanceA = reinterpret_cast<VkInstance>(std::uintptr_t{1U});
	const auto instanceB = reinterpret_cast<VkInstance>(std::uintptr_t{2U});
	VklInstanceTable tableA{};
	VklInstanceTable tableB{};

	vklLoadInstanceTable(&tableA, instanceA);
	const auto firstDestroy = tableA.vkDestroyInstance;
	ASSERT_NE(firstDestroy, nullptr);
	vklLoadInstanceTable(&tableB, instanceB);

	EXPECT_EQ(tableA.vkDestroyInstance, firstDestroy);
	EXPECT_NE(tableA.vkDestroyInstance, tableB.vkDestroyInstance);
	EXPECT_NE(tableA.vkGetPhysicalDeviceProperties,
		tableB.vkGetPhysicalDeviceProperties);
	vklFinalize();
}

TEST(VkDispatch, DeviceTablesRemainIndependent)
{
	VklInstanceTable instanceA{};
	VklInstanceTable instanceB{};
	instanceA.vkGetDeviceProcAddr = &fakeGetDeviceProcAddrA;
	instanceB.vkGetDeviceProcAddr = &fakeGetDeviceProcAddrB;
	const auto deviceA = reinterpret_cast<VkDevice>(std::uintptr_t{1U});
	const auto deviceB = reinterpret_cast<VkDevice>(std::uintptr_t{2U});
	VklDeviceTable tableA{};
	VklDeviceTable tableB{};

	vklLoadDeviceTable(&tableA, &instanceA, deviceA);
	const auto firstSubmit = tableA.vkQueueSubmit;
	ASSERT_NE(firstSubmit, nullptr);
	vklLoadDeviceTable(&tableB, &instanceB, deviceB);

	EXPECT_EQ(tableA.vkQueueSubmit, firstSubmit);
	EXPECT_NE(tableA.vkQueueSubmit, tableB.vkQueueSubmit);
	EXPECT_NE(tableA.vkDestroyDevice, tableB.vkDestroyDevice);
}
