// OA cyber-range integrity checkpoint.
//
// This tutorial is deliberately incapable of touching the filesystem or
// network. It generates a deterministic in-memory fixture, hashes its chunks
// through the vulkan SHAKE-256 path, reduces them to a Merkle root, and proves
// controlled tampering against the CPU oracle.

#include <oa/crypto.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/engine.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {

constexpr oa::U32 kChunkCount = 16;
constexpr oa::U32 kChunkBytes = 256;
constexpr oa::U32 kTamperedChunk = 5;
constexpr oa::U32 kTamperedByte = 17;

std::vector<oa::Byte> makeFixture() {
	std::vector<oa::Byte> fixture(kChunkCount * kChunkBytes);
	oa::U32 state = 0x0A5EC123U;
	for (oa::U32 chunk = 0; chunk < kChunkCount; ++chunk) {
		for (oa::U32 byte = 0; byte < kChunkBytes; ++byte) {
			state = state * 1664525U + 1013904223U;
			fixture[chunk * kChunkBytes + byte] = static_cast<oa::Byte>(
				(state >> 24U) ^ (chunk * 29U) ^ byte);
		}
	}
	return fixture;
}

oa::Vec<oa::Hash> hashFixtureOnCpu(const std::vector<oa::Byte>& inFixture) {
	oa::Vec<oa::Hash> leaves;
	leaves.reserve(kChunkCount);
	for (oa::U32 chunk = 0; chunk < kChunkCount; ++chunk) {
		oa::Hash hash;
		oa::shake256(
			inFixture.data() + chunk * kChunkBytes,
			kChunkBytes,
			hash.bytes.data(),
			hash.bytes.size());
		leaves.pushBack(hash);
	}
	return leaves;
}

oa::Status hashFixtureOnGpu(
	oa::Engine& inEngine,
	const std::vector<oa::Byte>& inFixture,
	oa::Hash& outRoot)
{
	if (inFixture.size() != kChunkCount * kChunkBytes) {
		return oa::Status::invalidArgument("cyber-range fixture has an invalid size");
	}

	oa::Matrix messages = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(inFixture.data(), inFixture.size()),
		oa::MatrixShape{kChunkCount, kChunkBytes},
		oa::ScalarType::UInt8);
	if (not messages.hasStorage()) {
		return oa::Status::error(oa::StatusCode::DataLoss,"cyber-range fixture upload failed");
	}

	oa::Matrix leaves;
	oa::Matrix root;
	leaves = oa::FnHash::shake256(messages, oa::Hash::size());
	root = oa::FnHash::merkleRoot(leaves);
	if (not leaves.hasStorage() or not root.hasStorage()) {
		return oa::Status::error(
			oa::StatusCode::DataLoss,
			"cyber-range hash graph construction failed"
		);
	}

	auto submitted = inEngine.submit();
	if (not submitted.isOk()) {
		return submitted.getStatus();
	}
	OA_RETURN_IF_ERROR(inEngine.wait(submitted.getValue()));

	std::array<oa::Byte, oa::Hash::size()> rootBytes{};
	OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
		root,
		rootBytes.data(),
		rootBytes.size())
	);
	std::memcpy(outRoot.bytes.data(), rootBytes.data(), rootBytes.size());
	return oa::Status::ok();
}

void printFailure(const char* inStage, const oa::Status& inStatus) {
	std::fprintf(stderr, "%s failed: %s\n",
		inStage, inStatus.toString().cStr());
}

} // namespace

int main() {
	oa::EngineConfig config;
	config.appName = "TutorialCryptoCyberRangeIntegrity";
	config.precision = oa::Precision::FP32;
	config.numericMode = oa::NumericMode::Deterministic;

	auto created = oa::Engine::create(config);
	if (not created.isOk()) {
		std::fprintf(stderr, "[skip] vulkan engine unavailable: %s\n",
			created.getStatus().toString().cStr());
		return 125;
	}
	auto engine = std::move(created).getValue();

	const auto baselineFixture = makeFixture();
	auto tamperedFixture = baselineFixture;
	tamperedFixture[kTamperedChunk * kChunkBytes + kTamperedByte] ^= 0x80U;

	const auto baselineLeaves = hashFixtureOnCpu(baselineFixture);
	const auto tamperedLeaves = hashFixtureOnCpu(tamperedFixture);
	const oa::Hash baselineCpuRoot = oa::merkleRoot(baselineLeaves);
	const oa::Hash tamperedCpuRoot = oa::merkleRoot(tamperedLeaves);

	oa::Hash baselineGpuRoot;
	if (auto status = hashFixtureOnGpu(
			*engine, baselineFixture, baselineGpuRoot);
		not status.isOk())
	{
		printFailure("baseline vulkan integrity pass", status);
		return 2;
	}

	oa::Hash tamperedGpuRoot;
	if (auto status = hashFixtureOnGpu(
			*engine, tamperedFixture, tamperedGpuRoot);
		not status.isOk())
	{
		printFailure("tampered vulkan integrity pass", status);
		return 2;
	}

	const oa::MerkleTree tree = oa::buildMerkleTree(baselineLeaves);
	auto proof = oa::getMerkleProof(tree, kTamperedChunk);
	if (not proof.isOk()) {
		printFailure("Merkle proof construction", proof.getStatus());
		return 2;
	}

	const bool baselineParity = baselineGpuRoot == baselineCpuRoot;
	const bool tamperedParity = tamperedGpuRoot == tamperedCpuRoot;
	const bool rootChanged = baselineGpuRoot != tamperedGpuRoot;
	const bool originalProofValid = oa::verifyMerkleProof(
		baselineLeaves[kTamperedChunk], proof.getValue(), tree.root);
	const bool tamperedProofRejected = not oa::verifyMerkleProof(
		tamperedLeaves[kTamperedChunk], proof.getValue(), tree.root);
	const bool passed = baselineParity
		and tamperedParity
		and rootChanged
		and originalProofValid
		and tamperedProofRejected;

	std::printf("OA Linux cyber-range integrity checkpoint\n");
	std::printf("  scope: in-memory fixture only\n");
	std::printf("  device: %.*s\n",
		static_cast<int>(engine->deviceName().size()),
		engine->deviceName().data());
	std::printf("  chunks: %u x %u bytes\n", kChunkCount, kChunkBytes);
	std::printf("  baseline root: %s\n", baselineGpuRoot.toHex().cStr());
	std::printf("  tampered root: %s\n", tamperedGpuRoot.toHex().cStr());
	std::printf("  CPU/vulkan parity: %s\n",
		baselineParity and tamperedParity ? "pass" : "FAIL");
	std::printf("  tamper detection: %s\n", rootChanged ? "pass" : "FAIL");
	std::printf("  proof rejection: %s\n",
		originalProofValid and tamperedProofRejected ? "pass" : "FAIL");
	std::printf("  result: %s\n", passed ? "PASS" : "FAIL");
	return passed ? 0 : 3;
}
