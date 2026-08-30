#include <oa/core/vlm.h>

#if defined(OA_VLM_HAS_GLM)
	#define GLM_ENABLE_EXPERIMENTAL
	#include <glm/ext/matrix_clip_space.hpp>
	#include <glm/ext/matrix_projection.hpp>
	#include <glm/ext/matrix_transform.hpp>
	#include <glm/ext/quaternion_common.hpp>
	#include <glm/ext/quaternion_geometric.hpp>
	#include <glm/ext/quaternion_transform.hpp>
	#include <glm/glm.hpp>
	#include <glm/gtc/matrix_inverse.hpp>
	#include <glm/gtc/quaternion.hpp>
	#include <glm/gtx/matrix_decompose.hpp>
	#include <glm/gtx/projection.hpp>
	#include <glm/gtx/vector_angle.hpp>
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

volatile double gSink = 0.0;

enum class Precision {
	Float32,
	Float64,
	Both,
};

struct Options {
	oa::Usize items = 1U << 16U;
	oa::I32 warmups = 3;
	oa::I32 samples = 11;
	Precision precision = Precision::Both;
	std::string filter;
};

[[nodiscard]] bool parsePositive(const char* inValue, oa::Usize& outValue) {
	errno = 0;
	char* end = nullptr;
	const unsigned long long value = std::strtoull(inValue, &end, 10);
	if (errno == ERANGE or end == inValue or *end != '\0' or value == 0
		or value > std::numeric_limits<oa::Usize>::max()) {
		return false;
	}
	outValue = static_cast<oa::Usize>(value);
	return true;
}

[[nodiscard]] bool parsePositive(const char* inValue, oa::I32& outValue) {
	errno = 0;
	char* end = nullptr;
	const long value = std::strtol(inValue, &end, 10);
	if (errno == ERANGE or end == inValue or *end != '\0' or value <= 0
		or value > std::numeric_limits<oa::I32>::max()) {
		return false;
	}
	outValue = static_cast<oa::I32>(value);
	return true;
}

[[nodiscard]] bool parseOptions(
	int inArgc,
	char** inArgv,
	Options& outOptions) {
	for (int index = 1; index < inArgc; ++index) {
		const char* argument = inArgv[index];
		if (std::strcmp(argument, "--items") == 0 and index + 1 < inArgc) {
			if (not parsePositive(inArgv[++index], outOptions.items)) return false;
		} else if (std::strcmp(argument, "--warmups") == 0
			and index + 1 < inArgc) {
			if (not parsePositive(inArgv[++index], outOptions.warmups)) return false;
		} else if (std::strcmp(argument, "--samples") == 0
			and index + 1 < inArgc) {
			if (not parsePositive(inArgv[++index], outOptions.samples)) return false;
		} else if (std::strcmp(argument, "--filter") == 0
			and index + 1 < inArgc) {
			outOptions.filter = inArgv[++index];
		} else if (std::strcmp(argument, "--precision") == 0
			and index + 1 < inArgc) {
			const char* precision = inArgv[++index];
			if (std::strcmp(precision, "f32") == 0) {
				outOptions.precision = Precision::Float32;
			} else if (std::strcmp(precision, "f64") == 0) {
				outOptions.precision = Precision::Float64;
			} else if (std::strcmp(precision, "both") == 0) {
				outOptions.precision = Precision::Both;
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
	return outOptions.samples >= 3;
}

template <typename T>
[[nodiscard]] const char* precisionName() {
	return std::is_same_v<T, oa::F32> ? "f32" : "f64";
}

[[nodiscard]] double median(std::vector<double> inValues) {
	std::sort(inValues.begin(), inValues.end());
	return inValues[inValues.size() / 2U];
}

template <typename Fn>
[[nodiscard]] double measure(Fn&& inFunction, oa::Usize inItems) {
	const auto begin = Clock::now();
	const double checksum = inFunction();
	const auto end = Clock::now();
	gSink = checksum;
	return std::chrono::duration<double, std::nano>(end - begin).count()
		/ static_cast<double>(inItems);
}

template <typename T>
[[nodiscard]] bool checksumsAgree(double inOa, double inGlm) {
	if (not std::isfinite(inOa) or not std::isfinite(inGlm)) return false;
	const double scale = std::max({1.0, std::abs(inOa), std::abs(inGlm)});
	const double tolerance = std::is_same_v<T, oa::F32> ? 2.0e-4 : 2.0e-11;
	return std::abs(inOa - inGlm) <= scale * tolerance;
}

struct SuiteResult {
	oa::I32 cases = 0;
	oa::I32 faster = 0;
	oa::I32 parity = 0;
	oa::I32 slower = 0;
};

template <typename T, typename OaFn, typename GlmFn>
[[nodiscard]] bool runPeerCase(
	const char* inName,
	const char* inContract,
	const Options& inOptions,
	SuiteResult& inOutResult,
	OaFn&& inOa,
	GlmFn&& inGlm) {
	if (not inOptions.filter.empty()
		and std::strstr(inName, inOptions.filter.c_str()) == nullptr) {
		return true;
	}
	const double oaOracle = inOa();
	const double glmOracle = inGlm();
	if (not checksumsAgree<T>(oaOracle, glmOracle)) {
		std::fprintf(
			stderr,
			"oracle mismatch: precision=%s case=%s oa=%.17g glm=%.17g\n",
			precisionName<T>(), inName, oaOracle, glmOracle);
		return false;
	}
	for (oa::I32 warmup = 0; warmup < inOptions.warmups; ++warmup) {
		if ((warmup & 1) == 0) {
			gSink = inOa();
			gSink = inGlm();
		} else {
			gSink = inGlm();
			gSink = inOa();
		}
	}
	std::vector<double> oaTimes;
	std::vector<double> glmTimes;
	oaTimes.reserve(static_cast<oa::Usize>(inOptions.samples));
	glmTimes.reserve(static_cast<oa::Usize>(inOptions.samples));
	for (oa::I32 sample = 0; sample < inOptions.samples; ++sample) {
		if ((sample & 1) == 0) {
			oaTimes.push_back(measure(inOa, inOptions.items));
			glmTimes.push_back(measure(inGlm, inOptions.items));
		} else {
			glmTimes.push_back(measure(inGlm, inOptions.items));
			oaTimes.push_back(measure(inOa, inOptions.items));
		}
	}
	const double oaMedian = median(oaTimes);
	const double glmMedian = median(glmTimes);
	const double ratio = oaMedian / glmMedian;
	const double delta = (ratio - 1.0) * 100.0;
	++inOutResult.cases;
	if (ratio < 0.95) {
		++inOutResult.faster;
	} else if (ratio <= 1.03) {
		++inOutResult.parity;
	} else {
		++inOutResult.slower;
	}
	std::printf(
		"PAIR precision=%s case=%s contract=%s oa_ns=%.6f glm_ns=%.6f ratio=%.6f delta_pct=%+.3f oa_min=%.6f oa_max=%.6f glm_min=%.6f glm_max=%.6f checksum=%.9g\n",
		precisionName<T>(), inName, inContract,
		oaMedian, glmMedian, ratio, delta,
		*std::min_element(oaTimes.begin(), oaTimes.end()),
		*std::max_element(oaTimes.begin(), oaTimes.end()),
		*std::min_element(glmTimes.begin(), glmTimes.end()),
		*std::max_element(glmTimes.begin(), glmTimes.end()), oaOracle);
	return true;
}

#if defined(OA_VLM_HAS_GLM)

template <typename T>
using GlmVec2 = glm::vec<2, T, glm::defaultp>;
template <typename T>
using GlmVec3 = glm::vec<3, T, glm::defaultp>;
template <typename T>
using GlmVec4 = glm::vec<4, T, glm::defaultp>;
template <typename T>
using GlmQuat = glm::qua<T, glm::defaultp>;
template <typename T>
using GlmMat3 = glm::mat<3, 3, T, glm::defaultp>;
template <typename T>
using GlmMat4 = glm::mat<4, 4, T, glm::defaultp>;

template <typename T>
[[nodiscard]] GlmVec3<T> toGlm(const oa::vlm::detail::Vec3<T>& inValue) {
	return {inValue.x, inValue.y, inValue.z};
}
template <typename T>
[[nodiscard]] GlmQuat<T> toGlm(const oa::vlm::detail::Quat<T>& inValue) {
	return {inValue.w, inValue.x, inValue.y, inValue.z};
}
template <typename T>
[[nodiscard]] GlmMat3<T> toGlmColumn(const oa::vlm::detail::Mat3<T>& inValue) {
	GlmMat3<T> result(T(0));
	for (oa::I32 row = 0; row < 3; ++row) {
		for (oa::I32 column = 0; column < 3; ++column) {
			result[static_cast<glm::length_t>(column)]
				[static_cast<glm::length_t>(row)] = inValue.m[column][row];
		}
	}
	return result;
}
template <typename T>
[[nodiscard]] GlmMat4<T> toGlmColumn(const oa::vlm::detail::Mat4<T>& inValue) {
	GlmMat4<T> result(T(0));
	for (oa::I32 row = 0; row < 4; ++row) {
		for (oa::I32 column = 0; column < 4; ++column) {
			result[static_cast<glm::length_t>(column)]
				[static_cast<glm::length_t>(row)] = inValue.m[column][row];
		}
	}
	return result;
}

template <typename T>
[[nodiscard]] T checksum(const oa::vlm::detail::Vec2<T>& inValue) {
	return inValue.x + inValue.y * T(0.5);
}
template <typename T>
[[nodiscard]] T checksum(const GlmVec2<T>& inValue) {
	return inValue.x + inValue.y * T(0.5);
}
template <typename T>
[[nodiscard]] T checksum(const oa::vlm::detail::Vec3<T>& inValue) {
	return inValue.x + inValue.y * T(0.5) + inValue.z * T(0.25);
}
template <typename T>
[[nodiscard]] T checksum(const GlmVec3<T>& inValue) {
	return inValue.x + inValue.y * T(0.5) + inValue.z * T(0.25);
}
template <typename T>
[[nodiscard]] T checksum(const oa::vlm::detail::Vec4<T>& inValue) {
	return inValue.x + inValue.y * T(0.5)
		+ inValue.z * T(0.25) + inValue.w * T(0.125);
}
template <typename T>
[[nodiscard]] T checksum(const GlmVec4<T>& inValue) {
	return inValue.x + inValue.y * T(0.5)
		+ inValue.z * T(0.25) + inValue.w * T(0.125);
}
template <typename T>
[[nodiscard]] T checksum(const oa::vlm::detail::Quat<T>& inValue) {
	return inValue.x + inValue.y * T(0.5)
		+ inValue.z * T(0.25) + inValue.w * T(0.125);
}
template <typename T>
[[nodiscard]] T checksum(const GlmQuat<T>& inValue) {
	return inValue.x + inValue.y * T(0.5)
		+ inValue.z * T(0.25) + inValue.w * T(0.125);
}
template <typename T>
[[nodiscard]] T checksum(const oa::vlm::detail::Mat3<T>& inValue) {
	T result = T(0);
	for (oa::I32 row = 0; row < 3; ++row) {
		for (oa::I32 column = 0; column < 3; ++column) result += inValue.m[row][column];
	}
	return result;
}
template <typename T>
[[nodiscard]] T checksum(const GlmMat3<T>& inValue) {
	T result = T(0);
	for (glm::length_t column = 0; column < 3; ++column) {
		for (glm::length_t row = 0; row < 3; ++row) result += inValue[column][row];
	}
	return result;
}
template <typename T>
[[nodiscard]] T checksum(const oa::vlm::detail::Mat4<T>& inValue) {
	T result = T(0);
	for (oa::I32 row = 0; row < 4; ++row) {
		for (oa::I32 column = 0; column < 4; ++column) result += inValue.m[row][column];
	}
	return result;
}
template <typename T>
[[nodiscard]] T checksum(const GlmMat4<T>& inValue) {
	T result = T(0);
	for (glm::length_t column = 0; column < 4; ++column) {
		for (glm::length_t row = 0; row < 4; ++row) result += inValue[column][row];
	}
	return result;
}

template <typename T>
[[nodiscard]] bool runSuite(const Options& inOptions) {
	using Mat3 = oa::vlm::detail::Mat3<T>;
	using Mat4 = oa::vlm::detail::Mat4<T>;
	using Quat = oa::vlm::detail::Quat<T>;
	using Vec2 = oa::vlm::detail::Vec2<T>;
	using Vec3 = oa::vlm::detail::Vec3<T>;
	using Vec4 = oa::vlm::detail::Vec4<T>;
	using Viewport = oa::vlm::detail::Viewport<T>;

	std::vector<Vec2> a2(inOptions.items), b2(inOptions.items);
	std::vector<Vec3> a(inOptions.items), b(inOptions.items), normals(inOptions.items);
	std::vector<Vec4> a4(inOptions.items), b4(inOptions.items);
	std::vector<Quat> qa(inOptions.items), qb(inOptions.items);
	std::vector<Mat3> ma3(inOptions.items), mb3(inOptions.items);
	std::vector<Mat4> ma4(inOptions.items), mb4(inOptions.items);
	std::vector<GlmVec2<T>> ga2(inOptions.items), gb2(inOptions.items);
	std::vector<GlmVec3<T>> ga(inOptions.items), gb(inOptions.items), gnormals(inOptions.items);
	std::vector<GlmVec4<T>> ga4(inOptions.items), gb4(inOptions.items);
	std::vector<GlmQuat<T>> gqa(inOptions.items), gqb(inOptions.items);
	std::vector<GlmMat3<T>> gma3(inOptions.items), gmb3(inOptions.items);
	std::vector<GlmMat4<T>> gma4(inOptions.items), gmb4(inOptions.items);
	for (oa::Usize index = 0; index < inOptions.items; ++index) {
		const T value = static_cast<T>(index % 1021U) / T(1021);
		a2[index] = {value + T(0.125), value * T(0.5) - T(0.25)};
		b2[index] = {T(0.75) - value * T(0.25), value + T(0.375)};
		a[index] = {value + T(0.125), value * T(0.5) - T(0.25), T(1) - value * T(0.75)};
		b[index] = {T(0.75) - value * T(0.25), value + T(0.375), value * T(0.2) - T(0.4)};
		a4[index] = {a[index].x, a[index].y, a[index].z, value + T(0.5)};
		b4[index] = {b[index].x, b[index].y, b[index].z, T(1.25) - value};
		normals[index] = oa::vlm::normalize(Vec3{
			value + T(0.25), T(1.25) - value, value * T(0.3) + T(0.1)});
		qa[index] = oa::vlm::quaternionFromAxisAngle(
			Vec3{T(1), T(0.5) + value, T(0.25)}, T(0.1) + value * T(0.7));
		qb[index] = oa::vlm::quaternionFromAxisAngle(
			Vec3{T(0.25), T(1), T(0.75) - value * T(0.2)}, T(0.2) + value * T(0.5));
		ma4[index] = oa::vlm::composeTrs(
			Vec3{value * T(3), value * T(-2), value + T(0.5)}, qa[index],
			Vec3{T(0.75) + value, T(1.25), T(1.5) - value * T(0.25)});
		mb4[index] = oa::vlm::composeTrs(
			Vec3{T(-1) - value, value * T(0.5), T(2) - value}, qb[index],
			Vec3{T(1.1), T(0.8) + value * T(0.2), T(1.3)});
		ma3[index] = oa::vlm::linearPart(ma4[index]);
		mb3[index] = oa::vlm::linearPart(mb4[index]);
		ga2[index] = {a2[index].x, a2[index].y};
		gb2[index] = {b2[index].x, b2[index].y};
		ga[index] = toGlm(a[index]); gb[index] = toGlm(b[index]);
		ga4[index] = {a4[index].x, a4[index].y, a4[index].z, a4[index].w};
		gb4[index] = {b4[index].x, b4[index].y, b4[index].z, b4[index].w};
		gnormals[index] = toGlm(normals[index]);
		gqa[index] = toGlm(qa[index]); gqb[index] = toGlm(qb[index]);
		gma3[index] = toGlmColumn(ma3[index]); gmb3[index] = toGlmColumn(mb3[index]);
		gma4[index] = toGlmColumn(ma4[index]); gmb4[index] = toGlmColumn(mb4[index]);
	}

	const auto accumulateScalar = [](auto&& inOperation, oa::Usize inCount) {
		T total = T(0);
		for (oa::Usize index = 0; index < inCount; ++index) total += static_cast<T>(inOperation(index));
		return static_cast<double>(total);
	};
	const auto accumulateValue = [](auto&& inOperation, oa::Usize inCount) {
		T total = T(0);
		for (oa::Usize index = 0; index < inCount; ++index) total += checksum(inOperation(index));
		return static_cast<double>(total);
	};
	SuiteResult result{};
#define OA_PEER_CASE_CONTRACT(name, contract, oaExpression, glmExpression) \
	do { \
		if (not runPeerCase<T>(name, contract, inOptions, result, \
			[&]() { return (oaExpression); }, [&]() { return (glmExpression); })) return false; \
	} while (false)
#define OA_PEER_CASE(name, oaExpression, glmExpression) \
	OA_PEER_CASE_CONTRACT(name, "arithmetic", oaExpression, glmExpression)
#define OA_HARDENED_PEER_CASE(name, oaExpression, glmExpression) \
	OA_PEER_CASE_CONTRACT(name, "hardened", oaExpression, glmExpression)

	OA_PEER_CASE("vec2_add",
		accumulateValue([&](oa::Usize i) { return a2[i] + b2[i]; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return ga2[i] + gb2[i]; }, inOptions.items));
	OA_PEER_CASE("vec2_dot",
		accumulateScalar([&](oa::Usize i) { return oa::vlm::dot(a2[i], b2[i]); }, inOptions.items),
		accumulateScalar([&](oa::Usize i) { return glm::dot(ga2[i], gb2[i]); }, inOptions.items));
	OA_HARDENED_PEER_CASE("vec2_normalize",
		accumulateValue([&](oa::Usize i) { return oa::vlm::normalize(a2[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::normalize(ga2[i]); }, inOptions.items));
	OA_PEER_CASE("vec3_add",
		accumulateValue([&](oa::Usize i) { return a[i] + b[i]; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return ga[i] + gb[i]; }, inOptions.items));
	OA_PEER_CASE("vec3_component_mul",
		accumulateValue([&](oa::Usize i) { return oa::vlm::componentMul(a[i], b[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return ga[i] * gb[i]; }, inOptions.items));
	OA_PEER_CASE("vec3_dot",
		accumulateScalar([&](oa::Usize i) { return oa::vlm::dot(a[i], b[i]); }, inOptions.items),
		accumulateScalar([&](oa::Usize i) { return glm::dot(ga[i], gb[i]); }, inOptions.items));
	OA_HARDENED_PEER_CASE("vec3_length",
		accumulateScalar([&](oa::Usize i) { return oa::vlm::length(a[i]); }, inOptions.items),
		accumulateScalar([&](oa::Usize i) { return glm::length(ga[i]); }, inOptions.items));
	OA_HARDENED_PEER_CASE("vec3_distance",
		accumulateScalar([&](oa::Usize i) { return oa::vlm::distance(a[i], b[i]); }, inOptions.items),
		accumulateScalar([&](oa::Usize i) { return glm::distance(ga[i], gb[i]); }, inOptions.items));
	OA_PEER_CASE("vec3_cross",
		accumulateValue([&](oa::Usize i) { return oa::vlm::cross(a[i], b[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::cross(ga[i], gb[i]); }, inOptions.items));
	OA_HARDENED_PEER_CASE("vec3_normalize",
		accumulateValue([&](oa::Usize i) { return oa::vlm::normalize(a[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::normalize(ga[i]); }, inOptions.items));
	OA_PEER_CASE("vec3_reflect",
		accumulateValue([&](oa::Usize i) { return oa::vlm::reflect(a[i], normals[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::reflect(ga[i], gnormals[i]); }, inOptions.items));
	OA_PEER_CASE("vec3_lerp",
		accumulateValue([&](oa::Usize i) { return oa::vlm::lerp(a[i], b[i], T(0.37)); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::mix(ga[i], gb[i], T(0.37)); }, inOptions.items));
	OA_PEER_CASE("vec3_refract",
		accumulateValue([&](oa::Usize i) { return oa::vlm::refract(normals[i], normals[i], T(0.75)); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::refract(gnormals[i], gnormals[i], T(0.75)); }, inOptions.items));
	OA_HARDENED_PEER_CASE("vec3_project_checked",
		accumulateValue([&](oa::Usize i) { Vec3 output{}; return oa::vlm::tryProjectVector(a[i], b[i], output) ? output : Vec3{}; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::proj(ga[i], gb[i]); }, inOptions.items));
	OA_HARDENED_PEER_CASE("vec3_angle_checked",
		accumulateScalar([&](oa::Usize i) { T output = T(0); return oa::vlm::tryAngleBetween(a[i], b[i], output) ? output : T(0); }, inOptions.items),
		accumulateScalar([&](oa::Usize i) { return glm::angle(glm::normalize(ga[i]), glm::normalize(gb[i])); }, inOptions.items));
	OA_PEER_CASE("vec4_add",
		accumulateValue([&](oa::Usize i) { return a4[i] + b4[i]; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return ga4[i] + gb4[i]; }, inOptions.items));
	OA_PEER_CASE("vec4_dot",
		accumulateScalar([&](oa::Usize i) { return oa::vlm::dot(a4[i], b4[i]); }, inOptions.items),
		accumulateScalar([&](oa::Usize i) { return glm::dot(ga4[i], gb4[i]); }, inOptions.items));
	OA_HARDENED_PEER_CASE("vec4_normalize",
		accumulateValue([&](oa::Usize i) { return oa::vlm::normalize(a4[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::normalize(ga4[i]); }, inOptions.items));
	OA_PEER_CASE("quat_mul",
		accumulateValue([&](oa::Usize i) { return qa[i] * qb[i]; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return gqa[i] * gqb[i]; }, inOptions.items));
	OA_HARDENED_PEER_CASE("quat_inverse_checked",
		accumulateValue([&](oa::Usize i) { Quat output{}; return oa::vlm::tryInverse(qa[i], output) ? output : Quat{}; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::inverse(gqa[i]); }, inOptions.items));
	OA_HARDENED_PEER_CASE("quat_normalize",
		accumulateValue([&](oa::Usize i) { return qa[i].normalized(); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::normalize(gqa[i]); }, inOptions.items));
	OA_HARDENED_PEER_CASE("quat_rotate",
		accumulateValue([&](oa::Usize i) { return qa[i].rotate(a[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return gqa[i] * ga[i]; }, inOptions.items));
	OA_HARDENED_PEER_CASE("quat_nlerp",
		accumulateValue([&](oa::Usize i) { return oa::vlm::nlerp(qa[i], qb[i], T(0.37)); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::normalize(gqa[i] * T(0.63) + gqb[i] * T(0.37)); }, inOptions.items));
	OA_HARDENED_PEER_CASE("quat_slerp",
		accumulateValue([&](oa::Usize i) { return oa::vlm::slerp(qa[i], qb[i], T(0.37)); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::slerp(gqa[i], gqb[i], T(0.37)); }, inOptions.items));
	OA_HARDENED_PEER_CASE("quat_to_mat4",
		accumulateValue([&](oa::Usize i) { return oa::vlm::quaternionToMatrix(qa[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::mat4_cast(gqa[i]); }, inOptions.items));
	OA_PEER_CASE("mat3_vec3",
		accumulateValue([&](oa::Usize i) { return a[i] * ma3[i]; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return gma3[i] * ga[i]; }, inOptions.items));
	OA_PEER_CASE("mat3_add",
		accumulateValue([&](oa::Usize i) { return ma3[i] + mb3[i]; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return gma3[i] + gmb3[i]; }, inOptions.items));
	OA_PEER_CASE("mat3_transpose",
		accumulateValue([&](oa::Usize i) { return oa::vlm::transpose(ma3[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::transpose(gma3[i]); }, inOptions.items));
	OA_PEER_CASE("mat3_mul",
		accumulateValue([&](oa::Usize i) { return ma3[i] * mb3[i]; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return gmb3[i] * gma3[i]; }, inOptions.items));
	OA_PEER_CASE("mat3_determinant",
		accumulateScalar([&](oa::Usize i) { return oa::vlm::determinant(ma3[i]); }, inOptions.items),
		accumulateScalar([&](oa::Usize i) { return glm::determinant(gma3[i]); }, inOptions.items));
	OA_HARDENED_PEER_CASE("mat3_inverse_checked",
		accumulateValue([&](oa::Usize i) { Mat3 output{}; return oa::vlm::tryInverse(ma3[i], output) ? output : Mat3{}; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::inverse(gma3[i]); }, inOptions.items));
	OA_PEER_CASE("mat4_vec4",
		accumulateValue([&](oa::Usize i) { const auto value = oa::vlm::transform(oa::vlm::detail::Vec4<T>{a[i].x, a[i].y, a[i].z, T(1)}, ma4[i]); return Vec3{value.x, value.y, value.z}; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return GlmVec3<T>(gma4[i] * GlmVec4<T>(ga[i], T(1))); }, inOptions.items));
	OA_PEER_CASE("mat4_add",
		accumulateValue([&](oa::Usize i) { return ma4[i] + mb4[i]; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return gma4[i] + gmb4[i]; }, inOptions.items));
	OA_PEER_CASE("mat4_transpose",
		accumulateValue([&](oa::Usize i) { return oa::vlm::transpose(ma4[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::transpose(gma4[i]); }, inOptions.items));
	OA_PEER_CASE("mat4_mul",
		accumulateValue([&](oa::Usize i) { return ma4[i] * mb4[i]; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return gmb4[i] * gma4[i]; }, inOptions.items));
	OA_PEER_CASE("mat4_determinant",
		accumulateScalar([&](oa::Usize i) { return oa::vlm::determinant(ma4[i]); }, inOptions.items),
		accumulateScalar([&](oa::Usize i) { return glm::determinant(gma4[i]); }, inOptions.items));
	OA_HARDENED_PEER_CASE("mat4_inverse_checked",
		accumulateValue([&](oa::Usize i) { Mat4 output{}; return oa::vlm::tryInverse(ma4[i], output) ? output : Mat4{}; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::inverse(gma4[i]); }, inOptions.items));
	OA_PEER_CASE("transform_point",
		accumulateValue([&](oa::Usize i) { return oa::vlm::transformPoint(a[i], ma4[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return GlmVec3<T>(gma4[i] * GlmVec4<T>(ga[i], T(1))); }, inOptions.items));
	OA_PEER_CASE("transform_direction",
		accumulateValue([&](oa::Usize i) { return oa::vlm::transformDirection(a[i], ma4[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return GlmVec3<T>(gma4[i] * GlmVec4<T>(ga[i], T(0))); }, inOptions.items));
	OA_HARDENED_PEER_CASE("affine_inverse_checked",
		accumulateValue([&](oa::Usize i) { Mat4 output{}; return oa::vlm::tryAffineInverse(ma4[i], output) ? output : Mat4{}; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::affineInverse(gma4[i]); }, inOptions.items));
	OA_HARDENED_PEER_CASE("normal_matrix_checked",
		accumulateValue([&](oa::Usize i) { Mat3 output{}; return oa::vlm::tryNormalMatrix(ma4[i], output) ? output : Mat3{}; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::inverseTranspose(gma3[i]); }, inOptions.items));
	OA_PEER_CASE("compose_trs",
		accumulateValue([&](oa::Usize i) { return oa::vlm::composeTrs(a[i], qa[i], b[i] + Vec3{T(1.5), T(1.5), T(1.5)}); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::translate(GlmMat4<T>(T(1)), ga[i]) * glm::mat4_cast(gqa[i]) * glm::scale(GlmMat4<T>(T(1)), gb[i] + GlmVec3<T>(T(1.5))); }, inOptions.items));
	OA_PEER_CASE("translation_matrix",
		accumulateValue([&](oa::Usize i) { return oa::vlm::translation(a[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::translate(GlmMat4<T>(T(1)), ga[i]); }, inOptions.items));
	OA_PEER_CASE("scale_matrix",
		accumulateValue([&](oa::Usize i) { return oa::vlm::scaleMatrix(a[i]); }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::scale(GlmMat4<T>(T(1)), ga[i]); }, inOptions.items));
	OA_HARDENED_PEER_CASE("decompose_trs_checked",
		accumulateValue([&](oa::Usize i) { oa::vlm::detail::TrsDecomposition<T> output{}; return oa::vlm::tryDecomposeTrs(ma4[i], output) ? output.translation + output.scale : Vec3{}; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { GlmVec3<T> scaleValue{}, translationValue{}, skew{}; GlmQuat<T> rotation{}; GlmVec4<T> perspectiveValue{}; return glm::decompose(gma4[i], scaleValue, rotation, translationValue, skew, perspectiveValue) ? translationValue + scaleValue : GlmVec3<T>{}; }, inOptions.items));
	OA_HARDENED_PEER_CASE("look_at_checked",
		accumulateValue([&](oa::Usize i) { Mat4 output{}; return oa::vlm::tryLookAt(a[i] + Vec3{T(0), T(0), T(3)}, b[i], Vec3{T(0), T(1), T(0)}, output) ? output : Mat4{}; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::lookAtRH(ga[i] + GlmVec3<T>(T(0), T(0), T(3)), gb[i], GlmVec3<T>(T(0), T(1), T(0))); }, inOptions.items));
	OA_HARDENED_PEER_CASE("perspective_checked",
		accumulateValue([&](oa::Usize i) { Mat4 output{}; const T fov = T(55) + static_cast<T>(i % 17U) * T(0.25); return oa::vlm::tryPerspective(fov, T(16) / T(9), T(0.1), T(1000), output) ? output : Mat4{}; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { const T fov = T(55) + static_cast<T>(i % 17U) * T(0.25); return glm::perspectiveRH_ZO(glm::radians(fov), T(16) / T(9), T(0.1), T(1000)); }, inOptions.items));

	const Mat4 projection = oa::vlm::perspective(T(67), T(16) / T(9), T(0.125), T(4096));
	const GlmMat4<T> gProjection = toGlmColumn(projection);
	const Viewport viewport{T(13), T(17), T(1921), T(1081), T(0), T(1)};
	const glm::vec<4, T, glm::defaultp> gViewport{viewport.x, viewport.y, viewport.width, viewport.height};
	std::vector<Vec3> screen(inOptions.items);
	std::vector<GlmVec3<T>> gScreen(inOptions.items);
	for (oa::Usize index = 0; index < inOptions.items; ++index) {
		const Vec3 point{a[index].x, a[index].y, -a[index].z - T(1)};
		if (not oa::vlm::tryProjectToViewport(point, projection, viewport, screen[index])) return false;
		gScreen[index] = {
			screen[index].x,
			T(2) * viewport.y + viewport.height - screen[index].y,
			screen[index].z};
	}
	OA_HARDENED_PEER_CASE("project_point_checked",
		accumulateValue([&](oa::Usize i) { Vec3 output{}; return oa::vlm::tryProjectPoint(Vec3{a[i].x, a[i].y, -a[i].z - T(1)}, projection, output) ? output : Vec3{}; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { GlmVec4<T> output = gProjection * GlmVec4<T>(ga[i].x, ga[i].y, -ga[i].z - T(1), T(1)); return GlmVec3<T>(output) / output.w; }, inOptions.items));
	OA_HARDENED_PEER_CASE("viewport_project_checked",
		accumulateValue([&](oa::Usize i) { Vec3 output{}; return oa::vlm::tryProjectToViewport(Vec3{a[i].x, a[i].y, -a[i].z - T(1)}, projection, viewport, output) ? output : Vec3{}; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { GlmVec3<T> output = glm::projectZO(GlmVec3<T>(ga[i].x, ga[i].y, -ga[i].z - T(1)), GlmMat4<T>(T(1)), gProjection, gViewport); output.y = T(2) * viewport.y + viewport.height - output.y; return output; }, inOptions.items));
	OA_HARDENED_PEER_CASE("viewport_unproject_checked",
		accumulateValue([&](oa::Usize i) { Vec3 output{}; return oa::vlm::tryUnprojectFromViewport(screen[i], projection, viewport, output) ? output : Vec3{}; }, inOptions.items),
		accumulateValue([&](oa::Usize i) { return glm::unProjectZO(gScreen[i], GlmMat4<T>(T(1)), gProjection, gViewport); }, inOptions.items));

#undef OA_PEER_CASE
#undef OA_HARDENED_PEER_CASE
#undef OA_PEER_CASE_CONTRACT
	std::printf("SUMMARY precision=%s cases=%d faster=%d parity=%d slower=%d oracle=PASS\n",
		precisionName<T>(), result.cases, result.faster, result.parity, result.slower);
	return true;
}

#endif

} // namespace

int main(int argc, char** argv) {
	Options options{};
	if (not parseOptions(argc, argv, options)) {
		std::fprintf(stderr,
			"usage: %s [--items N] [--warmups N] [--samples N] [--precision f32|f64|both] [--filter substring]\n",
			argv[0]);
		return 2;
	}
#if not defined(OA_VLM_HAS_GLM)
	std::fprintf(stderr,
		"BenchVlm requires -DOA_VLM_GLM_ROOT=<GLM include root> for peer qualification\n");
	return 3;
#else
	std::printf(
		"OA VLM/GLM paired benchmark: items=%zu warmups=%d samples=%d glm=%d.%d.%d alternating_order=yes oracle=required\n",
		options.items, options.warmups, options.samples,
		GLM_VERSION_MAJOR, GLM_VERSION_MINOR, GLM_VERSION_PATCH);
	if ((options.precision == Precision::Float32 or options.precision == Precision::Both)
		and not runSuite<oa::F32>(options)) return 4;
	if ((options.precision == Precision::Float64 or options.precision == Precision::Both)
		and not runSuite<oa::F64>(options)) return 5;
	std::printf("BENCHMARK oracle=PASS\n");
	return 0;
#endif
}
