#include <oa/runtime/canary.h>
#include <oa/core/jsonWriter.h>

#include <oa/core/fnMatrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>
#include "engine/deviceAccess.h"

namespace {

oa::U64 hashBytes(const void* inData, oa::U64 inBytes) {
	const auto* bytes = static_cast<const oa::U8*>(inData);
	oa::U64 hash = 0xcbf29ce484222325ULL;
	for (oa::U64 i = 0; i < inBytes; ++i) {
		hash ^= bytes[i];
		hash *= 0x100000001b3ULL;
	}
	return hash;
}

template <typename T>
oa::Span<const oa::U8> bytes(const oa::Vector<T>& inValues) {
	return {
		reinterpret_cast<const oa::U8*>(inValues.data()),
		inValues.size() * sizeof(T),
	};
}

template <typename T>
oa::Status readback(
	oa::Engine& inEngine,
	const oa::Matrix& inMatrix,
	oa::Vector<T>& outValues)
{
	const oa::U64 bytes = static_cast<oa::U64>(outValues.size() * sizeof(T));
	if (bytes != static_cast<oa::U64>(inMatrix.byteSize())) {
		return oa::Status::invalidArgument("device canary readback size mismatch");
	}
	return oa::EngineResourceAccess::readbackBuffer(inEngine,
		oa::MatrixAccess::descriptor(inMatrix), inMatrix.byteOffset(),
		outValues.data(), bytes);
}

oa::DeviceCanaryCheck exactCheck(
	oa::StringView inName,
	const void* inExpected,
	const void* inActual,
	oa::U64 inBytes,
	oa::U32 inSamples)
{
	oa::DeviceCanaryCheck check;
	check.name = oa::String(inName);
	check.exact = true;
	check.sampleCount = inSamples;
	check.expectedHash = hashBytes(inExpected, inBytes);
	check.actualHash = hashBytes(inActual, inBytes);
	check.passed = check.expectedHash == check.actualHash
		and oa::memcmp(inExpected, inActual, static_cast<oa::Usize>(inBytes)) == 0;
	return check;
}

oa::DeviceCanaryCheck floatCheck(
	oa::StringView inName,
	const oa::Vector<oa::F32>& inExpected,
	const oa::Vector<oa::F32>& inActual,
	oa::F64 inTolerance)
{
	oa::DeviceCanaryCheck check;
	check.name = oa::String(inName);
	check.sampleCount = static_cast<oa::U32>(inExpected.size());
	check.tolerance = inTolerance;
	check.expectedHash = hashBytes(
		inExpected.data(), inExpected.size() * sizeof(oa::F32));
	check.actualHash = hashBytes(
		inActual.data(), inActual.size() * sizeof(oa::F32));
	check.passed = inExpected.size() == inActual.size();
	for (oa::Usize i = 0; i < oa::min(inExpected.size(), inActual.size()); ++i) {
		const oa::F64 error = oa::abs(
			static_cast<oa::F64>(inExpected[i]) - static_cast<oa::F64>(inActual[i]));
		check.maxAbsoluteError = oa::max(check.maxAbsoluteError, error);
		if (not oa::isFinite(inActual[i]) or error > inTolerance) {
			check.passed = false;
		}
	}
	return check;
}

void writeJsonString(oa::internal::JsonWriter& out, oa::StringView inValue) {
	out.writeString(inValue);
}

} // namespace

oa::Bool oa::DeviceCanaryReport::passed() const noexcept {
	if (checks.empty()) return false;
	for (const auto& check : checks) {
		if (not check.passed) return false;
	}
	return true;
}

oa::String oa::DeviceCanaryReport::debugReportJson() const {
	oa::internal::JsonWriter out;
	out << "{\n  \"schema\": \"oa.device_canary.v1\",\n"
		<< "  \"passed\": " << (passed() ? "true" : "false")
		<< ",\n  \"device\": {\"name\": ";
	writeJsonString(out, deviceName);
	out << ", \"vendor\": ";
	writeJsonString(out, vendorName);
	out << ", \"driver\": ";
	writeJsonString(out, driverName);
	out << ", \"driver_version\": ";
	writeJsonString(out, driverVersion);
	out << ", \"vulkan_api\": ";
	writeJsonString(out, apiVersion);
	out << "},\n  \"checks\": [";
	for (oa::U32 i = 0; i < checks.size(); ++i) {
		const auto& check = checks[i];
		out << (i == 0U ? "\n" : ",\n") << "    {\"name\": ";
		writeJsonString(out, check.name);
		out << ", \"passed\": " << (check.passed ? "true" : "false")
			<< ", \"exact\": " << (check.exact ? "true" : "false")
			<< ", \"samples\": " << check.sampleCount
			<< ", \"expected_hash\": ";
		out.writeHexId(check.expectedHash);
		out << ", \"actual_hash\": ";
		out.writeHexId(check.actualHash);
		out << ", \"max_absolute_error\": ";
		out.writeFloat(check.maxAbsoluteError);
		out << ", \"tolerance\": ";
		out.writeFloat(check.tolerance);
		out << "}";
	}
	if (not checks.empty()) out << '\n';
	out << "  ]\n}\n";
	return out.take();
}

oa::Status oa::DeviceCanary::run(
	oa::Engine& inEngine,
	oa::DeviceCanaryReport& outReport)
{
	outReport = {};
	if (not inEngine.isReady()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device canary requires a ready engine");
	}
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	if (context.nodeCount() != 0U) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"device canary requires an idle recording context");
	}

	const auto& info = oa::EngineDeviceAccess::get(inEngine).info;
	outReport.deviceName = info.hardware.deviceName;
	outReport.vendorName = info.hardware.vendorName;
	outReport.driverName = info.software.driverName;
	outReport.driverVersion = info.software.driverVersion;
	outReport.apiVersion = info.software.apiVersion;

	constexpr oa::U32 kVectorSize = 257U;
	oa::Vector<oa::U32> transportExpected(kVectorSize);
	oa::Vector<oa::F32> aValues(kVectorSize);
	oa::Vector<oa::F32> bValues(kVectorSize);
	for (oa::U32 i = 0; i < kVectorSize; ++i) {
		transportExpected[i] = 0x9e3779b9U * (i + 1U) ^ (i << 17U);
		aValues[i] = static_cast<oa::F32>(static_cast<oa::I32>(i % 31U) - 15)
			* 0.0625F;
		bValues[i] = static_cast<oa::F32>(static_cast<oa::I32>(i % 19U) - 9)
			* 0.03125F;
	}

	constexpr oa::U32 kM = 17U;
	constexpr oa::U32 kN = 11U;
	constexpr oa::U32 kK = 13U;
	oa::Vector<oa::F32> matrixA(kM * kK);
	oa::Vector<oa::F32> matrixB(kN * kK);
	for (oa::U32 i = 0; i < matrixA.size(); ++i) {
		matrixA[i] = static_cast<oa::F32>(static_cast<oa::I32>(i % 17U) - 8)
			* 0.03125F;
	}
	for (oa::U32 i = 0; i < matrixB.size(); ++i) {
		matrixB[i] = static_cast<oa::F32>(static_cast<oa::I32>(i % 13U) - 6)
			* 0.0625F;
	}

	constexpr oa::U32 kRows = 16U;
	constexpr oa::U32 kClasses = 7U;
	oa::Vector<oa::F32> logits(kRows * kClasses, -4.0F);
	oa::Vector<oa::U32> labels(kRows);
	oa::U32 expectedCorrect = 0;
	for (oa::U32 row = 0; row < kRows; ++row) {
		const oa::U32 winner = row % kClasses;
		logits[row * kClasses + winner] = 4.0F;
		labels[row] = row % 4U == 0U ? (winner + 1U) % kClasses : winner;
		if (labels[row] == winner) ++expectedCorrect;
	}

	oa::Matrix transport;
	oa::Matrix vectorOut;
	oa::Matrix vectorSum;
	oa::Matrix matrixOut;
	oa::Matrix correctCount;
	{
		oa::ExecutionSession::RecordingScope recording(context);
		transport = oa::FnMatrix::fromBytes(
			bytes(transportExpected), {kVectorSize}, oa::ScalarType::UInt32);
		auto a = oa::FnMatrix::fromBytes(bytes(aValues), {kVectorSize});
		auto b = oa::FnMatrix::fromBytes(bytes(bValues), {kVectorSize});
		vectorOut = oa::FnMatrix::mul(oa::FnMatrix::add(a, b), a);
		vectorSum = oa::FnMatrix::sum(vectorOut);
		auto matrixAM = oa::FnMatrix::fromBytes(bytes(matrixA), {kM, kK});
		auto matrixBM = oa::FnMatrix::fromBytes(bytes(matrixB), {kN, kK});
		matrixOut = oa::FnMatrix::matMulNt(
			matrixAM, matrixBM, oa::MatMulPrecision::Fp32);
		auto logitsM = oa::FnMatrix::fromBytes(bytes(logits), {kRows, kClasses});
		auto labelsM = oa::FnMatrix::fromBytes(
			bytes(labels), {kRows}, oa::ScalarType::UInt32);
		correctCount = oa::FnMatrix::categoricalAccuracyCount(logitsM, labelsM);
	}

	OA_RETURN_IF_ERROR(context.submitAndWait());

	oa::Vector<oa::U32> transportActual(kVectorSize);
	OA_RETURN_IF_ERROR(readback(inEngine, transport, transportActual));
	outReport.checks.pushBack(exactCheck(
		"host_device_roundtrip_u32",
		transportExpected.data(), transportActual.data(),
		transportExpected.size() * sizeof(oa::U32), kVectorSize));

	oa::Vector<oa::F32> vectorExpected(kVectorSize);
	oa::F32 sumExpected = 0.0F;
	for (oa::U32 i = 0; i < kVectorSize; ++i) {
		vectorExpected[i] = (aValues[i] + bValues[i]) * aValues[i];
		sumExpected += vectorExpected[i];
	}
	oa::Vector<oa::F32> vectorActual(kVectorSize);
	OA_RETURN_IF_ERROR(readback(inEngine, vectorOut, vectorActual));
	outReport.checks.pushBack(floatCheck(
		"fp32_elementwise_barrier_chain", vectorExpected, vectorActual, 1.0e-6));
	oa::Vector<oa::F32> sumActual(1);
	OA_RETURN_IF_ERROR(readback(inEngine, vectorSum, sumActual));
	outReport.checks.pushBack(floatCheck(
		"fp32_shared_reduction", {sumExpected}, sumActual, 2.0e-4));

	oa::Vector<oa::F32> matrixExpected(kM * kN, 0.0F);
	for (oa::U32 row = 0; row < kM; ++row) {
		for (oa::U32 col = 0; col < kN; ++col) {
			for (oa::U32 k = 0; k < kK; ++k) {
				matrixExpected[row * kN + col] +=
					matrixA[row * kK + k] * matrixB[col * kK + k];
			}
		}
	}
	oa::Vector<oa::F32> matrixActual(kM * kN);
	OA_RETURN_IF_ERROR(readback(inEngine, matrixOut, matrixActual));
	outReport.checks.pushBack(floatCheck(
		"fp32_matmul_irregular", matrixExpected, matrixActual, 2.0e-5));

	oa::Vector<oa::U32> countActual(1);
	OA_RETURN_IF_ERROR(readback(inEngine, correctCount, countActual));
	outReport.checks.pushBack(exactCheck(
		"uint32_accuracy_reduction",
		&expectedCorrect, countActual.data(), sizeof(oa::U32), 1U));

	if (not outReport.passed()) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"device canary produced a known-answer mismatch");
	}
	return oa::Status::ok();
}
