#include <oa/core/simd.h>
#include <oa/core/vlm.h>

#include <xsimd/xsimd.hpp>

#if defined(OA_VLM_HAS_GLM)
	#include <glm/glm.hpp>
	#include <glm/gtc/quaternion.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

volatile oa::F32 gSink = 0.0F;

struct Options {
	oa::Usize items = 1U << 16U;
	oa::I32 samples = 15;
};

Options parseOptions(int inArgc, char** inArgv) {
	Options options;
	for (int index = 1; index < inArgc; ++index) {
		if (std::strcmp(inArgv[index], "--items") == 0 && index + 1 < inArgc) {
			options.items = static_cast<oa::Usize>(std::strtoull(inArgv[++index], nullptr, 10));
		} else if (std::strcmp(inArgv[index], "--samples") == 0 && index + 1 < inArgc) {
			options.samples = static_cast<oa::I32>(std::strtol(inArgv[++index], nullptr, 10));
		}
	}
	options.items = std::max<oa::Usize>(options.items, 1U);
	options.samples = std::max<oa::I32>(options.samples, 3);
	return options;
}

template <typename Fn>
void runCase(const char* inName, const Options& inOptions, Fn&& inFn) {
	for (oa::I32 warmup = 0; warmup < 3; ++warmup) gSink = inFn();
	std::vector<oa::F64> nanoseconds;
	nanoseconds.reserve(static_cast<oa::Usize>(inOptions.samples));
	for (oa::I32 sample = 0; sample < inOptions.samples; ++sample) {
		const auto begin = std::chrono::steady_clock::now();
		gSink = inFn();
		const auto end = std::chrono::steady_clock::now();
		const auto elapsed = std::chrono::duration<oa::F64, std::nano>(end - begin).count();
		nanoseconds.push_back(elapsed / static_cast<oa::F64>(inOptions.items));
	}
	std::sort(nanoseconds.begin(), nanoseconds.end());
	const oa::F64 median = nanoseconds[nanoseconds.size() / 2U];
	const oa::F64 minimum = nanoseconds.front();
	const oa::F64 maximum = nanoseconds.back();
	std::printf(
		"%-24s median=%9.3f ns/item min=%9.3f max=%9.3f checksum=%g\n",
		inName, median, minimum, maximum, static_cast<double>(gSink));
}

oa::F32 scalarDot(
	const std::vector<oa::F32>& inA,
	const std::vector<oa::F32>& inB
) {
	oa::F32 result = 0.0F;
	for (oa::Usize index = 0; index < inA.size(); ++index) {
		result += inA[index] * inB[index];
	}
	return result;
}

oa::F64 dotOracle(
	const std::vector<oa::F32>& inA,
	const std::vector<oa::F32>& inB
) {
	oa::F64 result = 0.0;
	for (oa::Usize index = 0; index < inA.size(); ++index) {
		result += static_cast<oa::F64>(inA[index])
			* static_cast<oa::F64>(inB[index]);
	}
	return result;
}

oa::F32 stockXsimdDot(
	const std::vector<oa::F32>& inA,
	const std::vector<oa::F32>& inB
) {
	using Batch = xsimd::batch<oa::F32>;
	constexpr oa::Usize lanes = Batch::size;
	Batch sum(0.0F);
	oa::Usize index = 0;
	for (; index + lanes <= inA.size(); index += lanes) {
		const Batch a = Batch::load_unaligned(inA.data() + index);
		const Batch b = Batch::load_unaligned(inB.data() + index);
		sum = xsimd::fma(a, b, sum);
	}
	oa::F32 result = xsimd::reduce_add(sum);
	for (; index < inA.size(); ++index) result += inA[index] * inB[index];
	return result;
}

} // namespace

int main(int argc, char** argv) {
	const Options options = parseOptions(argc, argv);
	std::vector<oa::vlm::Vec3> points(options.items);
	std::vector<oa::F32> a(options.items);
	std::vector<oa::F32> b(options.items);
	for (oa::Usize index = 0; index < options.items; ++index) {
		const oa::F32 value = static_cast<oa::F32>(index % 1021U) * (1.0F / 1021.0F);
		points[index] = {value, value * 0.5F - 0.25F, 1.0F - value * 0.75F};
		a[index] = value + 0.125F;
		b[index] = 1.25F - value * 0.5F;
	}

	const oa::vlm::Mat4 matrix = oa::vlm::composeTrs(
		{11.0F, -7.0F, 3.0F},
		oa::vlm::Quat::fromAxisAngle({1.0F, 2.0F, 3.0F}, 0.731F),
		{0.75F, 1.25F, 2.0F});
	const oa::vlm::Quat quaternion = oa::vlm::Quat::fromAxisAngle(
		{2.0F, -1.0F, 0.5F}, 1.117F);

	const oa::F64 expectedDot = dotOracle(a, b);
	const oa::F64 scalarDotResult = scalarDot(a, b);
	const oa::F64 stockXsimdDotResult = stockXsimdDot(a, b);
	const oa::F64 simdDotResult = oa::FnSimd::dotF32(
		a.data(), b.data(), static_cast<oa::I64>(options.items));
	const oa::F64 dotTolerance = std::max(1.0, std::abs(expectedDot)) * 5.0e-4;
	if (std::abs(scalarDotResult - expectedDot) > dotTolerance
		or std::abs(stockXsimdDotResult - expectedDot) > dotTolerance
		or std::abs(simdDotResult - expectedDot) > dotTolerance) {
		std::fprintf(stderr,
			"dot oracle mismatch: expected=%g scalar=%g stock_xsimd=%g fnsimd=%g\n",
			expectedDot, scalarDotResult, stockXsimdDotResult, simdDotResult);
		return 2;
	}

	std::printf(
		"OA VLM CPU benchmark: items=%zu samples=%d glm=%s\n",
		options.items, options.samples,
#if defined(OA_VLM_HAS_GLM)
		"GLM 1.0.1 reference enabled"
#else
		"disabled"
#endif
	);

	runCase("scalar dot", options, [&]() { return scalarDot(a, b); });
	runCase("stock xsimd dot", options, [&]() { return stockXsimdDot(a, b); });
	runCase("FnSimd dot", options, [&]() {
		return oa::FnSimd::dotF32(a.data(), b.data(), static_cast<oa::I64>(options.items));
	});
	runCase("vlm transformPoint", options, [&]() {
		oa::vlm::Vec3 sum{};
		for (const oa::vlm::Vec3& point : points) {
			sum += oa::vlm::transformPoint(point, matrix);
		}
		return sum.x + sum.y + sum.z;
	});
	runCase("vlm quaternion rotate", options, [&]() {
		oa::vlm::Vec3 sum{};
		for (const oa::vlm::Vec3& point : points) sum += quaternion.rotate(point);
		return sum.x + sum.y + sum.z;
	});

#if defined(OA_VLM_HAS_GLM)
	glm::mat4 glmMatrix(1.0F);
	for (oa::I32 row = 0; row < 4; ++row) {
		for (oa::I32 column = 0; column < 4; ++column) {
			glmMatrix[static_cast<glm::length_t>(column)][static_cast<glm::length_t>(row)]
				= matrix.m[row][column];
		}
	}
	const glm::quat glmQuaternion{
		quaternion.w, quaternion.x, quaternion.y, quaternion.z};
	for (oa::Usize index = 0; index < std::min<oa::Usize>(options.items, 1024U); ++index) {
		const oa::vlm::Vec3& point = points[index];
		const oa::vlm::Vec3 vlmPoint = oa::vlm::transformPoint(point, matrix);
		const glm::vec4 glmPoint = glm::vec4(point.x, point.y, point.z, 1.0F) * glmMatrix;
		if (not oa::vlm::approximatelyEqual(
			vlmPoint, oa::vlm::Vec3{glmPoint.x, glmPoint.y, glmPoint.z},
			2.0e-5F, 2.0e-5F)) {
			std::fprintf(stderr, "GLM point-transform oracle mismatch at %zu\n", index);
			return 3;
		}
		const oa::vlm::Vec3 vlmRotated = quaternion.rotate(point);
		const glm::vec3 glmRotated = glmQuaternion * glm::vec3(point.x, point.y, point.z);
		if (not oa::vlm::approximatelyEqual(
			vlmRotated, oa::vlm::Vec3{glmRotated.x, glmRotated.y, glmRotated.z},
			2.0e-5F, 2.0e-5F)) {
			std::fprintf(stderr, "GLM quaternion oracle mismatch at %zu\n", index);
			return 4;
		}
	}
	runCase("glm transform vec4", options, [&]() {
		glm::vec3 sum(0.0F);
		for (const oa::vlm::Vec3& point : points) {
			const glm::vec4 value = glm::vec4(point.x, point.y, point.z, 1.0F) * glmMatrix;
			sum += glm::vec3(value);
		}
		return sum.x + sum.y + sum.z;
	});
	runCase("glm quaternion rotate", options, [&]() {
		glm::vec3 sum(0.0F);
		for (const oa::vlm::Vec3& point : points) {
			sum += glmQuaternion * glm::vec3(point.x, point.y, point.z);
		}
		return sum.x + sum.y + sum.z;
	});
#endif

	return 0;
}
