#include <gtest/gtest.h>
#include "../OaTest.h"

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	testing::AddGlobalTestEnvironment(new OaVkTestEnvironment);
	return RUN_ALL_TESTS();
}
