// OA cyber-range integrity checkpoint.
//
// This tutorial is deliberately incapable of touching the filesystem or
// network. It generates a deterministic in-memory fixture, hashes its chunks
// through the Vulkan SHAKE-256 path, reduces them to a Merkle root, and proves
// controlled tampering against the CPU oracle.

#include <Oa/Crypto.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {

constexpr OaU32 ChunkCount = 16;
constexpr OaU32 ChunkBytes = 256;
constexpr OaU32 TamperedChunk = 5;
constexpr OaU32 TamperedByte = 17;

std::vector<OaByte> MakeFixture() {
	std::vector<OaByte> fixture(ChunkCount * ChunkBytes);
	OaU32 state = 0x0A5EC123U;
	for (OaU32 chunk = 0; chunk < ChunkCount; ++chunk) {
		for (OaU32 byte = 0; byte < ChunkBytes; ++byte) {
			state = state * 1664525U + 1013904223U;
			fixture[chunk * ChunkBytes + byte] = static_cast<OaByte>(
				(state >> 24U) ^ (chunk * 29U) ^ byte);
		}
	}
	return fixture;
}

OaVec<OaHash> HashFixtureOnCpu(const std::vector<OaByte>& InFixture) {
	OaVec<OaHash> leaves;
	leaves.Reserve(ChunkCount);
	for (OaU32 chunk = 0; chunk < ChunkCount; ++chunk) {
		OaHash hash;
		OaShake256(
			InFixture.data() + chunk * ChunkBytes,
			ChunkBytes,
			hash.Bytes.data(),
			hash.Bytes.size());
		leaves.PushBack(hash);
	}
	return leaves;
}

OaStatus HashFixtureOnGpu(OaContext& InContext,	const std::vector<OaByte>& InFixture,	OaHash& OutRoot) {
	if (InFixture.size() != ChunkCount * ChunkBytes) {
		return OaStatus::InvalidArgument("cyber-range fixture has an invalid size");
	}

	OaMatrix messages = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(InFixture.data(), InFixture.size()),
		OaMatrixShape{ChunkCount, ChunkBytes},
		OaScalarType::UInt8);
	if (not messages.HasStorage()) {
		return OaStatus::Error(OaStatusCode::DataLoss,"cyber-range fixture upload failed");
	}

	OaMatrix leaves;
	OaMatrix root;
	{
		OaContext::RecordingScope recording(InContext);
		leaves = OaFnHash::Shake256(messages, OaHash::Size());
		root = OaFnHash::MerkleRoot(leaves);
	}
	if (not leaves.HasStorage() or not root.HasStorage()) {
		return OaStatus::Error(
			OaStatusCode::DataLoss,
			"cyber-range hash graph construction failed"
		);
	}

	auto submitted = InContext.Submit();
	if (not submitted.IsOk()) {
		return submitted.GetStatus();
	}
	OA_RETURN_IF_ERROR(InContext.Wait(submitted.GetValue()));

	std::array<OaByte, OaHash::Size()> rootBytes{};
	OA_RETURN_IF_ERROR(OaFnMatrix::CopyToHost(
		root,
		rootBytes.data(),
		rootBytes.size())
	);
	std::memcpy(OutRoot.Bytes.data(), rootBytes.data(), rootBytes.size());
	return OaStatus::Ok();
}

void PrintFailure(const char* InStage, const OaStatus& InStatus) {
	std::fprintf(stderr, "%s failed: %s\n",
		InStage, InStatus.ToString().CStr());
}

} // namespace

int main() {
	OaEngineConfig config;
	config.AppName = "TutorialCryptoCyberRangeIntegrity";
	config.Precision = OaPrecision::FP32;
	config.NumericMode = OaNumericMode::Deterministic;

	auto created = OaEngine::Create(config);
	if (not created.IsOk()) {
		std::fprintf(stderr, "[skip] Vulkan engine unavailable: %s\n",
			created.GetStatus().ToString().CStr());
		return 125;
	}
	auto engine = std::move(created).GetValue();
	auto& context = engine->GetContext();

	const auto baselineFixture = MakeFixture();
	auto tamperedFixture = baselineFixture;
	tamperedFixture[TamperedChunk * ChunkBytes + TamperedByte] ^= 0x80U;

	const auto baselineLeaves = HashFixtureOnCpu(baselineFixture);
	const auto tamperedLeaves = HashFixtureOnCpu(tamperedFixture);
	const OaHash baselineCpuRoot = OaMerkleRoot(baselineLeaves);
	const OaHash tamperedCpuRoot = OaMerkleRoot(tamperedLeaves);

	OaHash baselineGpuRoot;
	if (auto status = HashFixtureOnGpu(
			context, baselineFixture, baselineGpuRoot);
		not status.IsOk())
	{
		PrintFailure("baseline Vulkan integrity pass", status);
		return 2;
	}

	OaHash tamperedGpuRoot;
	if (auto status = HashFixtureOnGpu(
			context, tamperedFixture, tamperedGpuRoot);
		not status.IsOk())
	{
		PrintFailure("tampered Vulkan integrity pass", status);
		return 2;
	}

	const OaMerkleTree tree = OaBuildMerkleTree(baselineLeaves);
	auto proof = OaGetMerkleProof(tree, TamperedChunk);
	if (not proof.IsOk()) {
		PrintFailure("Merkle proof construction", proof.GetStatus());
		return 2;
	}

	const bool baselineParity = baselineGpuRoot == baselineCpuRoot;
	const bool tamperedParity = tamperedGpuRoot == tamperedCpuRoot;
	const bool rootChanged = baselineGpuRoot != tamperedGpuRoot;
	const bool originalProofValid = OaVerifyMerkleProof(
		baselineLeaves[TamperedChunk], proof.GetValue(), tree.Root);
	const bool tamperedProofRejected = not OaVerifyMerkleProof(
		tamperedLeaves[TamperedChunk], proof.GetValue(), tree.Root);
	const bool passed = baselineParity
		and tamperedParity
		and rootChanged
		and originalProofValid
		and tamperedProofRejected;

	std::printf("OA Linux cyber-range integrity checkpoint\n");
	std::printf("  scope: in-memory fixture only\n");
	std::printf("  device: %.*s\n",
		static_cast<int>(engine->DeviceName().Size()),
		engine->DeviceName().Data());
	std::printf("  chunks: %u x %u bytes\n", ChunkCount, ChunkBytes);
	std::printf("  baseline root: %s\n", baselineGpuRoot.ToHex().CStr());
	std::printf("  tampered root: %s\n", tamperedGpuRoot.ToHex().CStr());
	std::printf("  CPU/Vulkan parity: %s\n",
		baselineParity and tamperedParity ? "pass" : "FAIL");
	std::printf("  tamper detection: %s\n", rootChanged ? "pass" : "FAIL");
	std::printf("  proof rejection: %s\n",
		originalProofValid and tamperedProofRejected ? "pass" : "FAIL");
	std::printf("  result: %s\n", passed ? "PASS" : "FAIL");
	return passed ? 0 : 3;
}
