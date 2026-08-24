#include <gtest/gtest.h>
#include "../oaTest.h"
#include "../oaTestMain.h"

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	testing::AddGlobalTestEnvironment(new VkTestEnvironment);
	return runAllTests();
}
