// OA_DOC_BEGIN: core-matrix-add
#include <oa/oa.h>

OA_MAIN("ExampleCoreMatrix") {
	auto one = oa::FnMatrix::ones({2, 3});

	auto two = oa::FnMatrix::full({2, 3}, 2.0F);

	auto sum = oa::FnMatrix::add(one, two);

	oa::Array<oa::F32, 6> values{};
	if (not oa::FnMatrix::copyToHost(sum, values.data(), sizeof(values)).isOk()) {
		return 1;
	}
	for (const oa::F32 value : values) {
		if (oa::abs(value - 3.0F) > 1e-06F) {
			return 1;
		}
	}

	if (not oa::print("Matrix addition verified: every value is 3").isOk()) {
		return 1;
	}
	return 0;
}
// OA_DOC_END: core-matrix-add
