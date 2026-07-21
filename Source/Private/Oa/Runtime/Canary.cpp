#include <Oa/Runtime/Canary.h>

#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>

namespace {

OaU64 HashBytes(const void* InData, OaU64 InBytes) {
	const auto* bytes = static_cast<const OaU8*>(InData);
	OaU64 hash = 0xcbf29ce484222325ULL;
	for (OaU64 i = 0; i < InBytes; ++i) {
		hash ^= bytes[i];
		hash *= 0x100000001b3ULL;
	}
	return hash;
}

template <typename T>
OaSpan<const OaU8> Bytes(const std::vector<T>& InValues) {
	return {
		reinterpret_cast<const OaU8*>(InValues.data()),
		InValues.size() * sizeof(T),
	};
}

template <typename T>
OaStatus Readback(
	OaEngine& InEngine,
	const OaMatrix& InMatrix,
	std::vector<T>& OutValues)
{
	const OaU64 bytes = static_cast<OaU64>(OutValues.size() * sizeof(T));
	if (bytes != static_cast<OaU64>(InMatrix.ByteSize())) {
		return OaStatus::InvalidArgument("device canary readback size mismatch");
	}
	return InEngine.ReadbackBuffer(
		InMatrix.GetVkBuffer(), InMatrix.ByteOffset(), OutValues.data(), bytes);
}

OaDeviceCanaryCheck ExactCheck(
	OaStringView InName,
	const void* InExpected,
	const void* InActual,
	OaU64 InBytes,
	OaU32 InSamples)
{
	OaDeviceCanaryCheck check;
	check.Name = OaString(InName);
	check.Exact = true;
	check.SampleCount = InSamples;
	check.ExpectedHash = HashBytes(InExpected, InBytes);
	check.ActualHash = HashBytes(InActual, InBytes);
	check.Passed = check.ExpectedHash == check.ActualHash
		and std::memcmp(InExpected, InActual, static_cast<size_t>(InBytes)) == 0;
	return check;
}

OaDeviceCanaryCheck FloatCheck(
	OaStringView InName,
	const std::vector<OaF32>& InExpected,
	const std::vector<OaF32>& InActual,
	OaF64 InTolerance)
{
	OaDeviceCanaryCheck check;
	check.Name = OaString(InName);
	check.SampleCount = static_cast<OaU32>(InExpected.size());
	check.Tolerance = InTolerance;
	check.ExpectedHash = HashBytes(
		InExpected.data(), InExpected.size() * sizeof(OaF32));
	check.ActualHash = HashBytes(
		InActual.data(), InActual.size() * sizeof(OaF32));
	check.Passed = InExpected.size() == InActual.size();
	for (size_t i = 0; i < std::min(InExpected.size(), InActual.size()); ++i) {
		const OaF64 error = std::abs(
			static_cast<OaF64>(InExpected[i]) - static_cast<OaF64>(InActual[i]));
		check.MaxAbsoluteError = std::max(check.MaxAbsoluteError, error);
		if (not std::isfinite(InActual[i]) or error > InTolerance) {
			check.Passed = false;
		}
	}
	return check;
}

void WriteJsonString(std::ostringstream& Out, OaStringView InValue) {
	Out << '"';
	for (const char value : InValue) {
		switch (value) {
			case '"': Out << "\\\""; break;
			case '\\': Out << "\\\\"; break;
			case '\b': Out << "\\b"; break;
			case '\f': Out << "\\f"; break;
			case '\n': Out << "\\n"; break;
			case '\r': Out << "\\r"; break;
			case '\t': Out << "\\t"; break;
			default:
				if (static_cast<unsigned char>(value) < 0x20U) {
					Out << "\\u" << std::hex << std::setw(4)
						<< std::setfill('0')
						<< static_cast<unsigned>(
							static_cast<unsigned char>(value))
						<< std::dec << std::setfill(' ');
				} else {
					Out << value;
				}
		}
	}
	Out << '"';
}

} // namespace

OaBool OaDeviceCanaryReport::Passed() const noexcept {
	if (Checks.Empty()) return false;
	for (const auto& check : Checks) {
		if (not check.Passed) return false;
	}
	return true;
}

OaString OaDeviceCanaryReport::DebugReportJson() const {
	std::ostringstream out;
	out << "{\n  \"schema\": \"oa.device_canary.v1\",\n"
		<< "  \"passed\": " << (Passed() ? "true" : "false")
		<< ",\n  \"device\": {\"name\": ";
	WriteJsonString(out, DeviceName);
	out << ", \"vendor\": ";
	WriteJsonString(out, VendorName);
	out << ", \"driver\": ";
	WriteJsonString(out, DriverName);
	out << ", \"driver_version\": ";
	WriteJsonString(out, DriverVersion);
	out << ", \"vulkan_api\": ";
	WriteJsonString(out, ApiVersion);
	out << "},\n  \"checks\": [";
	for (OaU32 i = 0; i < Checks.Size(); ++i) {
		const auto& check = Checks[i];
		out << (i == 0U ? "\n" : ",\n") << "    {\"name\": ";
		WriteJsonString(out, check.Name);
		out << ", \"passed\": " << (check.Passed ? "true" : "false")
			<< ", \"exact\": " << (check.Exact ? "true" : "false")
			<< ", \"samples\": " << check.SampleCount
			<< ", \"expected_hash\": \"0x" << std::hex
			<< std::setw(16) << std::setfill('0') << check.ExpectedHash
			<< "\", \"actual_hash\": \"0x" << std::setw(16)
			<< check.ActualHash << std::dec << std::setfill(' ')
			<< "\", \"max_absolute_error\": " << std::setprecision(17)
			<< check.MaxAbsoluteError << ", \"tolerance\": "
			<< check.Tolerance << "}";
	}
	if (not Checks.Empty()) out << '\n';
	out << "  ]\n}\n";
	return OaString(out.str());
}

OaStatus OaDeviceCanary::Run(
	OaEngine& InEngine,
	OaDeviceCanaryReport& OutReport)
{
	OutReport = {};
	if (not InEngine.IsReady()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"device canary requires a ready engine");
	}
	auto& context = InEngine.GetContext();
	if (context.NodeCount() != 0U) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"device canary requires an idle recording context");
	}

	const auto& info = InEngine.Device.Info;
	OutReport.DeviceName = info.Hardware.DeviceName;
	OutReport.VendorName = info.Hardware.VendorName;
	OutReport.DriverName = info.Software.DriverName;
	OutReport.DriverVersion = info.Software.DriverVersion;
	OutReport.ApiVersion = info.Software.ApiVersion;

	constexpr OaU32 kVectorSize = 257U;
	std::vector<OaU32> transportExpected(kVectorSize);
	std::vector<OaF32> aValues(kVectorSize);
	std::vector<OaF32> bValues(kVectorSize);
	for (OaU32 i = 0; i < kVectorSize; ++i) {
		transportExpected[i] = 0x9e3779b9U * (i + 1U) ^ (i << 17U);
		aValues[i] = static_cast<OaF32>(static_cast<OaI32>(i % 31U) - 15)
			* 0.0625F;
		bValues[i] = static_cast<OaF32>(static_cast<OaI32>(i % 19U) - 9)
			* 0.03125F;
	}

	constexpr OaU32 kM = 17U;
	constexpr OaU32 kN = 11U;
	constexpr OaU32 kK = 13U;
	std::vector<OaF32> matrixA(kM * kK);
	std::vector<OaF32> matrixB(kN * kK);
	for (OaU32 i = 0; i < matrixA.size(); ++i) {
		matrixA[i] = static_cast<OaF32>(static_cast<OaI32>(i % 17U) - 8)
			* 0.03125F;
	}
	for (OaU32 i = 0; i < matrixB.size(); ++i) {
		matrixB[i] = static_cast<OaF32>(static_cast<OaI32>(i % 13U) - 6)
			* 0.0625F;
	}

	constexpr OaU32 kRows = 16U;
	constexpr OaU32 kClasses = 7U;
	std::vector<OaF32> logits(kRows * kClasses, -4.0F);
	std::vector<OaU32> labels(kRows);
	OaU32 expectedCorrect = 0;
	for (OaU32 row = 0; row < kRows; ++row) {
		const OaU32 winner = row % kClasses;
		logits[row * kClasses + winner] = 4.0F;
		labels[row] = row % 4U == 0U ? (winner + 1U) % kClasses : winner;
		if (labels[row] == winner) ++expectedCorrect;
	}

	OaMatrix transport;
	OaMatrix vectorOut;
	OaMatrix vectorSum;
	OaMatrix matrixOut;
	OaMatrix correctCount;
	{
		OaContext::RecordingScope recording(context);
		transport = OaFnMatrix::FromBytes(
			Bytes(transportExpected), {kVectorSize}, OaScalarType::UInt32);
		auto a = OaFnMatrix::FromBytes(Bytes(aValues), {kVectorSize});
		auto b = OaFnMatrix::FromBytes(Bytes(bValues), {kVectorSize});
		vectorOut = OaFnMatrix::Mul(OaFnMatrix::Add(a, b), a);
		vectorSum = OaFnMatrix::Sum(vectorOut);
		auto matrixAM = OaFnMatrix::FromBytes(Bytes(matrixA), {kM, kK});
		auto matrixBM = OaFnMatrix::FromBytes(Bytes(matrixB), {kN, kK});
		matrixOut = OaFnMatrix::MatMulNt(
			matrixAM, matrixBM, OaMatMulPrecision::Fp32);
		auto logitsM = OaFnMatrix::FromBytes(Bytes(logits), {kRows, kClasses});
		auto labelsM = OaFnMatrix::FromBytes(
			Bytes(labels), {kRows}, OaScalarType::UInt32);
		correctCount = OaFnMatrix::CategoricalAccuracyCount(logitsM, labelsM);
	}

	OA_RETURN_IF_ERROR(context.Execute());
	OA_RETURN_IF_ERROR(context.Sync());

	std::vector<OaU32> transportActual(kVectorSize);
	OA_RETURN_IF_ERROR(Readback(InEngine, transport, transportActual));
	OutReport.Checks.PushBack(ExactCheck(
		"host_device_roundtrip_u32",
		transportExpected.data(), transportActual.data(),
		transportExpected.size() * sizeof(OaU32), kVectorSize));

	std::vector<OaF32> vectorExpected(kVectorSize);
	OaF32 sumExpected = 0.0F;
	for (OaU32 i = 0; i < kVectorSize; ++i) {
		vectorExpected[i] = (aValues[i] + bValues[i]) * aValues[i];
		sumExpected += vectorExpected[i];
	}
	std::vector<OaF32> vectorActual(kVectorSize);
	OA_RETURN_IF_ERROR(Readback(InEngine, vectorOut, vectorActual));
	OutReport.Checks.PushBack(FloatCheck(
		"fp32_elementwise_barrier_chain", vectorExpected, vectorActual, 1.0e-6));
	std::vector<OaF32> sumActual(1);
	OA_RETURN_IF_ERROR(Readback(InEngine, vectorSum, sumActual));
	OutReport.Checks.PushBack(FloatCheck(
		"fp32_shared_reduction", {sumExpected}, sumActual, 2.0e-4));

	std::vector<OaF32> matrixExpected(kM * kN, 0.0F);
	for (OaU32 row = 0; row < kM; ++row) {
		for (OaU32 col = 0; col < kN; ++col) {
			for (OaU32 k = 0; k < kK; ++k) {
				matrixExpected[row * kN + col] +=
					matrixA[row * kK + k] * matrixB[col * kK + k];
			}
		}
	}
	std::vector<OaF32> matrixActual(kM * kN);
	OA_RETURN_IF_ERROR(Readback(InEngine, matrixOut, matrixActual));
	OutReport.Checks.PushBack(FloatCheck(
		"fp32_matmul_irregular", matrixExpected, matrixActual, 2.0e-5));

	std::vector<OaU32> countActual(1);
	OA_RETURN_IF_ERROR(Readback(InEngine, correctCount, countActual));
	OutReport.Checks.PushBack(ExactCheck(
		"uint32_accuracy_reduction",
		&expectedCorrect, countActual.data(), sizeof(OaU32), 1U));

	if (not OutReport.Passed()) {
		return OaStatus::Error(OaStatusCode::DataLoss,
			"device canary produced a known-answer mismatch");
	}
	return OaStatus::Ok();
}
