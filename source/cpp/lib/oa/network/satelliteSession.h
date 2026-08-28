#pragma once

#include "satelliteProtocol.h"

#include <oa/network/tcp.h>
#include <oa/runtime/sync.h>

namespace oa {

class Engine;

using SatelliteMacFn = oa::Fn<oa::Status(
	oa::Span<const oa::Byte>,
	oa::Span<const oa::Byte>,
	oa::StringView,
	oa::Array<oa::Byte, 32>&)>;

// Application-owned named work is injected into the private transport session.
// Network validates object identity, bounds, version/hash preconditions and the
// exact completion event; the callback owns workload semantics and may depend on
// higher modules without creating a Network -> Ml dependency.
class SatelliteObjectView {
public:
	oa::U64 id = 0;
	oa::U64 version = 0;
	oa::ScalarType dtype = oa::ScalarType::UInt8;
	oa::Span<const oa::I64> shape;
	oa::Span<const oa::Byte> data;
	oa::Array<oa::Byte, 32> hash{};
};

class SatelliteNamedRequest {
public:
	oa::String operation;
	oa::Vector<SatelliteObjectView> inputs;
	oa::Span<const oa::Byte> arguments;
	oa::U64 expectedVersion = 0;
	oa::Array<oa::Byte, 32> expectedHash{};
};

class SatelliteNamedResult {
public:
	oa::U64 version = 0;
	oa::ScalarType dtype = oa::ScalarType::UInt8;
	oa::Vector<oa::I64> shape;
	oa::Vector<oa::Byte> data;
	oa::Vector<oa::Byte> profile;
	oa::Event completion;
};

using SatelliteNamedWorkFn = oa::Fn<oa::Result<SatelliteNamedResult>(
	oa::Engine&, const SatelliteNamedRequest&)>;

class SatelliteSecret {
public:
	SatelliteSecret() = default;
	~SatelliteSecret();
	SatelliteSecret(SatelliteSecret&& inOther) noexcept;
	SatelliteSecret& operator=(SatelliteSecret&& inOther) noexcept;
	SatelliteSecret(const SatelliteSecret&) = delete;
	SatelliteSecret& operator=(const SatelliteSecret&) = delete;

	[[nodiscard]] static oa::Result<SatelliteSecret> fromBytes(
		oa::Span<const oa::Byte> inBytes);
	[[nodiscard]] oa::Span<const oa::Byte> bytes() const noexcept;
	[[nodiscard]] oa::Bool isValid() const noexcept { return valid_; }

private:
	oa::Array<oa::Byte, 32> bytes_{};
	oa::Bool valid_ = false;
};

class SatelliteLimits {
public:
	oa::U32 maxPayloadBytes = 1024U * 1024U;
	oa::U64 maxResidentBytes = 256U * 1024U * 1024U;
	oa::U32 maxObjects = 256U;
	oa::U32 maxInflight = 1U;
};

class SatelliteSessionConfig {
public:
	SatelliteSessionConfig(
		SatelliteSecret inSecret,
		SatelliteMacFn inMac)
		: secret(oa::move(inSecret)), mac(oa::move(inMac)) {}

	SatelliteSecret secret;
	SatelliteMacFn mac;
	SatelliteLimits limits;
	oa::U32 ioTimeoutMs = 2000U;
	oa::String deviceName = "CPU oracle";
	// Empty disables application work. The session admits only names in this
	// bounded list; the handler is never invoked by poll, cancel, or disconnect.
	oa::Vector<oa::String> namedOperations;
	SatelliteNamedWorkFn namedWork;
};

class SatelliteProbe {
public:
	oa::U64 sessionEpoch = 0;
	oa::String deviceName;
	SatelliteLimits limits;
	oa::Array<oa::Byte, 32> buildHash{};
	oa::Array<oa::Byte, 32> schemaHash{};
};

// Private, synchronous, single-connection worker session. The caller owns any
// accept/progress thread and joins it explicitly; destruction only closes the
// socket held by a client and never drains or waits for application work.
class SatelliteServerSession {
public:
	explicit SatelliteServerSession(SatelliteSessionConfig inConfig)
		: config_(oa::move(inConfig)) {}
	SatelliteServerSession(oa::Engine& inEngine, SatelliteSessionConfig inConfig)
		: config_(oa::move(inConfig)), engine_(&inEngine) {}

	[[nodiscard]] oa::Status serve(TcpStream inStream);
	[[nodiscard]] oa::U64 lastSessionEpoch() const noexcept { return lastEpoch_; }
	[[nodiscard]] oa::Bool lastGpuEventWasOwned() const noexcept {
		return lastGpuEventWasOwned_;
	}
	[[nodiscard]] oa::Bool lastGpuEventCompleted() const noexcept {
		return lastGpuEventCompleted_;
	}
	[[nodiscard]] oa::U64 lastGpuEventValue() const noexcept {
		return lastGpuEventValue_;
	}

private:
	SatelliteSessionConfig config_;
	oa::Engine* engine_ = nullptr;
	oa::U64 lastEpoch_ = 0;
	oa::U64 lastGpuEventValue_ = 0;
	oa::Bool lastGpuEventWasOwned_ = false;
	oa::Bool lastGpuEventCompleted_ = false;
};

class SatelliteClientSession {
public:
	SatelliteClientSession() = default;
	~SatelliteClientSession() = default;
	SatelliteClientSession(SatelliteClientSession&&) noexcept = default;
	SatelliteClientSession& operator=(SatelliteClientSession&&) noexcept = default;
	SatelliteClientSession(const SatelliteClientSession&) = delete;
	SatelliteClientSession& operator=(const SatelliteClientSession&) = delete;

	[[nodiscard]] static oa::Result<SatelliteClientSession> connect(
		const oa::String& inHost,
		oa::U16 inPort,
		SatelliteSessionConfig inConfig);

	[[nodiscard]] const SatelliteProbe& probe() const noexcept { return probe_; }
	[[nodiscard]] oa::Status putF32(
		oa::U64 inObjectId,
		oa::Span<const oa::I64> inShape,
		oa::Span<const oa::F32> inValues);
	[[nodiscard]] oa::Status putF32Versioned(
		oa::U64 inObjectId,
		oa::U64 inVersion,
		oa::Span<const oa::I64> inShape,
		oa::Span<const oa::F32> inValues);
	[[nodiscard]] oa::Status putU32(
		oa::U64 inObjectId,
		oa::Span<const oa::I64> inShape,
		oa::Span<const oa::U32> inValues);
	[[nodiscard]] oa::Status putU32Versioned(
		oa::U64 inObjectId,
		oa::U64 inVersion,
		oa::Span<const oa::I64> inShape,
		oa::Span<const oa::U32> inValues);
	[[nodiscard]] oa::Status dropObject(oa::U64 inObjectId);
	[[nodiscard]] oa::Result<oa::U64> startCpuAdd(oa::U32 inA, oa::U32 inB);
	[[nodiscard]] oa::Result<oa::U64> startMatrixAddF32(
		oa::U64 inA,
		oa::U64 inB,
		oa::U64 inOutput);
	[[nodiscard]] oa::Result<oa::U64> startNamed(
		oa::StringView inOperation,
		oa::Span<const oa::U64> inInputs,
		oa::U64 inOutput,
		oa::Span<const oa::Byte> inArguments,
		oa::U64 inExpectedVersion,
		const oa::Array<oa::Byte, 32>& inExpectedHash);
	[[nodiscard]] oa::Result<SatelliteMessage> poll(oa::U64 inRequestId);
	[[nodiscard]] oa::Result<SatelliteMessage> wait(oa::U64 inRequestId);
	[[nodiscard]] oa::Result<SatelliteMessage> cancel(oa::U64 inRequestId);
	[[nodiscard]] oa::Result<SatelliteMessage> getResult(oa::U64 inRequestId);
	[[nodiscard]] oa::Status close();
	[[nodiscard]] oa::Status abort(oa::StringView inReason);

	[[nodiscard]] static oa::Result<oa::U32> readCpuAddResult(
		const SatelliteMessage& inMessage);
	[[nodiscard]] static oa::Result<oa::Vector<oa::F32>> readF32Result(
		const SatelliteMessage& inMessage);
	[[nodiscard]] static oa::Result<oa::Vector<oa::Byte>> readBytesResult(
		const SatelliteMessage& inMessage);
	[[nodiscard]] static oa::Result<oa::Vector<oa::Byte>> readProfile(
		const SatelliteMessage& inMessage);

private:
	[[nodiscard]] oa::Status putObject(
		oa::U64 inObjectId,
		oa::U64 inVersion,
		oa::ScalarType inDtype,
		oa::Span<const oa::I64> inShape,
		oa::Span<const oa::Byte> inData);
	[[nodiscard]] oa::Result<SatelliteMessage> exchange(
		SatelliteMessageType inType,
		oa::Vector<SatelliteField> inFields);

	TcpStream stream_;
	SatelliteProbe probe_;
	oa::U64 nextRequestId_ = 1U;
	oa::Bool open_ = false;
};

[[nodiscard]] oa::Array<oa::Byte, 32> satelliteBuildHash();
[[nodiscard]] oa::Array<oa::Byte, 32> satelliteSchemaHash();
[[nodiscard]] oa::Status satelliteValidateRequestEnvelope(
	const SatelliteMessage& inRequest,
	oa::U64 inSessionEpoch,
	oa::U64 inLastRequestId);

} // namespace oa
