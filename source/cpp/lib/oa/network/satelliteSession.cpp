#include "satelliteSession.h"

#include <oa/core/fnMatrix.h>
#include <oa/core/version.h>
#include <oa/network/tcpFramed.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>

#if defined(OA_PLATFORM_WINDOWS)
	#include <bcrypt.h>
#else
	#include <fcntl.h>
	#include <unistd.h>
#endif

namespace {

constexpr char kHelloCustomText[] = "oa-satellite-hello-v1";
constexpr char kReplyCustomText[] = "oa-satellite-reply-v1";
constexpr oa::StringView kHelloCustom(
	kHelloCustomText, sizeof(kHelloCustomText) - 1U);
constexpr oa::StringView kReplyCustom(
	kReplyCustomText, sizeof(kReplyCustomText) - 1U);

enum class PendingKind : oa::U8 {
	CpuAdd,
	MatrixAddF32,
	NamedWork,
};

struct PendingOp {
	oa::U64 requestId = 0;
	PendingKind kind = PendingKind::CpuAdd;
	oa::U32 a = 0;
	oa::U32 b = 0;
	oa::U64 inputA = 0;
	oa::U64 inputB = 0;
	oa::U64 output = 0;
	oa::String operation;
	oa::Vec<oa::U64> inputs;
	oa::Vec<oa::Byte> arguments;
	oa::U64 expectedVersion = 0;
	oa::Array<oa::Byte, 32> expectedHash{};
	oa::Bool complete = false;
	oa::StatusCode status = oa::StatusCode::Ok;
	oa::Vec<oa::Byte> result;
	oa::Vec<oa::Byte> profile;
};

struct SessionObject {
	oa::U64 id = 0;
	oa::U64 version = 0;
	oa::ScalarType dtype = oa::ScalarType::Float32;
	oa::Vec<oa::I64> shape;
	oa::Vec<oa::Byte> data;
	oa::Array<oa::Byte, 32> hash{};
};

void secureZero(void* inData, oa::Usize inBytes) noexcept {
	volatile oa::Byte* bytes = static_cast<volatile oa::Byte*>(inData);
	for (oa::Usize i = 0; i < inBytes; ++i) bytes[i] = 0U;
}

void appendU32Le(oa::Vec<oa::Byte>& out, oa::U32 inValue) {
	for (oa::U32 shift = 0; shift < 32U; shift += 8U) {
		out.pushBack(static_cast<oa::Byte>((inValue >> shift) & 0xffU));
	}
}

void appendU64Le(oa::Vec<oa::Byte>& out, oa::U64 inValue) {
	for (oa::U32 shift = 0; shift < 64U; shift += 8U) {
		out.pushBack(static_cast<oa::Byte>((inValue >> shift) & 0xffU));
	}
}

oa::Bool pendingUsesObject(const PendingOp& inPending, oa::U64 inObjectId) {
	if (inPending.output == inObjectId) return true;
	if (inPending.inputA == inObjectId or inPending.inputB == inObjectId) return true;
	for (const oa::U64 input : inPending.inputs) {
		if (input == inObjectId) return true;
	}
	return false;
}

oa::U32 readU32Le(const oa::Byte* inData) {
	oa::U32 value = 0;
	for (oa::U32 i = 0; i < 4U; ++i) {
		value |= static_cast<oa::U32>(inData[i]) << (i * 8U);
	}
	return value;
}

void appendF32Le(oa::Vec<oa::Byte>& out, oa::F32 inValue) {
	oa::U32 bits = 0;
	std::memcpy(&bits, &inValue, sizeof(bits));
	appendU32Le(out, bits);
}

oa::F32 readF32Le(const oa::Byte* inData) {
	const oa::U32 bits = readU32Le(inData);
	oa::F32 value = 0.0F;
	std::memcpy(&value, &bits, sizeof(value));
	return value;
}

oa::Array<oa::Byte, 32> hashBytes(oa::Span<const oa::Byte> inBytes) {
	return oa::SatelliteProtocol::stableDigest(inBytes);
}

oa::Array<oa::Byte, 32> hashString(oa::StringView inText) {
	return hashBytes(oa::Span<const oa::Byte>(
		reinterpret_cast<const oa::Byte*>(inText.data()), inText.size()));
}

oa::U64 totalObjectBytes(const oa::Vec<SessionObject>& inObjects) {
	oa::U64 total = 0U;
	for (const auto& object : inObjects) total += object.data.size();
	return total;
}

SessionObject* findObject(oa::Vec<SessionObject>& inObjects, oa::U64 inId) {
	for (auto& object : inObjects) {
		if (object.id == inId) return &object;
	}
	return nullptr;
}

oa::Result<oa::U64> checkedElementCount(oa::Span<const oa::I64> inShape) {
	if (inShape.empty() or inShape.size() > static_cast<oa::Usize>(OA_MAX_TENSOR_DIMS)) {
		return oa::Status::invalidArgument("satellite session: tensor rank is invalid");
	}
	oa::U64 count = 1U;
	for (const oa::I64 dimension : inShape) {
		if (dimension <= 0) {
			return oa::Status::invalidArgument(
				"satellite session: tensor dimensions must be positive");
		}
		const oa::U64 value = static_cast<oa::U64>(dimension);
		if (count > std::numeric_limits<oa::U64>::max() / value) {
			return oa::Status::error(oa::StatusCode::OutOfRange,
				"satellite session: tensor element count overflows");
		}
		count *= value;
	}
	return count;
}

oa::Result<oa::MatrixShape> matrixShapeFromSpan(oa::Span<const oa::I64> inShape) {
	auto checked = checkedElementCount(inShape);
	if (checked.isError()) return checked.getStatus();
	oa::MatrixShape shape;
	shape.rank = static_cast<oa::I32>(inShape.size());
	for (oa::Usize i = 0; i < inShape.size(); ++i) shape.dims[i] = inShape[i];
	return shape;
}

oa::Result<oa::Vec<oa::F32>> decodeF32(oa::Span<const oa::Byte> inData) {
	if (inData.size() % sizeof(oa::F32) != 0U) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: FP32 object has a partial element");
	}
	oa::Vec<oa::F32> values;
	values.reserve(inData.size() / sizeof(oa::F32));
	for (oa::Usize offset = 0; offset < inData.size(); offset += sizeof(oa::F32)) {
		values.pushBack(readF32Le(inData.data() + offset));
	}
	return values;
}

oa::Span<const oa::Byte> asBytes(const oa::Vec<oa::Byte>& inBytes) {
	return oa::Span<const oa::Byte>(inBytes.data(), inBytes.size());
}

oa::Result<oa::Array<oa::Byte, 32>> randomNonce() {
	oa::Array<oa::Byte, 32> nonce{};
#if defined(OA_PLATFORM_WINDOWS)
	const NTSTATUS status = bCryptGenRandom(
		nullptr, nonce.data(), static_cast<ULONG>(nonce.size()),
		BCRYPT_USE_SYSTEM_PREFERRED_RNG);
	if (status != 0) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"satellite session: operating-system random source failed");
	}
#else
	int fd = -1;
	do {
		fd = ::open("/dev/urandom", O_RDONLY | O_CLOEXEC);
	} while (fd < 0 and errno == EINTR);
	if (fd < 0) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"satellite session: operating-system random source is unavailable");
	}
	oa::Usize offset = 0U;
	while (offset < nonce.size()) {
		const auto count = ::read(fd, nonce.data() + offset, nonce.size() - offset);
		if (count < 0 and errno == EINTR) continue;
		if (count <= 0) {
			(void)::close(fd);
			secureZero(nonce.data(), nonce.size());
			return oa::Status::error(oa::StatusCode::Unavailable,
				"satellite session: operating-system random source failed");
		}
		offset += static_cast<oa::Usize>(count);
	}
	(void)::close(fd);
#endif
	return nonce;
}

oa::Result<oa::U64> nonZeroEpoch(oa::U64 inPrevious) {
	oa::U64 epoch = 0;
	do {
		epoch = 0;
		auto nonce = randomNonce();
		if (nonce.isError()) return nonce.getStatus();
		for (oa::U32 i = 0; i < 8U; ++i) {
			epoch |= static_cast<oa::U64>((*nonce)[i]) << (i * 8U);
		}
	} while (epoch == 0U or epoch == inPrevious);
	return epoch;
}

oa::Bool isKnownStatusCode(oa::U32 inCode) {
	return inCode <= 16U
		or (inCode >= 100U and inCode <= 104U)
		or (inCode >= 120U and inCode <= 125U)
		or (inCode >= 140U and inCode <= 144U)
		or (inCode >= 160U and inCode <= 164U)
		or (inCode >= 180U and inCode <= 183U)
		or (inCode >= 200U and inCode <= 203U);
}

oa::Result<oa::Array<oa::Byte, 32>> kmac(
	const oa::SatelliteMacFn& inMac,
	oa::Span<const oa::Byte> inSecret,
	oa::Span<const oa::Byte> inData,
	oa::StringView inCustom)
{
	if (inMac.empty()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite session: authentication primitive is not configured");
	}
	oa::Array<oa::Byte, 32> mac{};
	const auto status = inMac(inSecret, inData, inCustom, mac);
	if (status.isError()) return status;
	return mac;
}

oa::Bool constantEqual(oa::Span<const oa::Byte> inA, oa::Span<const oa::Byte> inB) {
	if (inA.size() != inB.size()) return false;
	oa::Byte difference = 0;
	for (oa::Usize i = 0; i < inA.size(); ++i) {
		difference |= static_cast<oa::Byte>(inA[i] ^ inB[i]);
	}
	return difference == 0U;
}

oa::Bool admitsNamedOperation(
	const oa::SatelliteSessionConfig& inConfig,
	oa::StringView inOperation)
{
	for (const auto& admitted : inConfig.namedOperations) {
		if (admitted == inOperation) return true;
	}
	return false;
}

const oa::SatelliteField* requiredField(
	const oa::SatelliteMessage& inMessage, oa::SatelliteFieldId inId)
{
	return oa::SatelliteProtocol::findField(inMessage, inId);
}

oa::Status writeMessage(oa::TcpStream& inStream, const oa::SatelliteMessage& inMessage) {
	auto encoded = oa::SatelliteProtocol::encode(inMessage);
	if (encoded.isError()) return encoded.getStatus();
	return oa::TcpFramed::writeMessage(inStream, oa::Span<const oa::Byte>(
		encoded->data(), encoded->size()));
}

oa::Result<oa::SatelliteMessage> readMessage(
	oa::TcpStream& inStream, oa::U32 inMaxPayloadBytes)
{
	if (inMaxPayloadBytes > oa::SatelliteProtocol::kMaxPayloadBytes) {
		return oa::Status::invalidArgument("satellite session: invalid payload limit");
	}
	oa::Vec<oa::Byte> frame;
	const auto status = oa::TcpFramed::readMessage(
		inStream, frame, oa::SatelliteProtocol::kHeaderBytes + inMaxPayloadBytes);
	if (status.isError()) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"satellite session: framed read failed");
	}
	return oa::SatelliteProtocol::decode(
		oa::Span<const oa::Byte>(frame.data(), frame.size()));
}

oa::SatelliteMessage errorResponse(
	const oa::SatelliteMessage& inRequest,
	const oa::Status& inStatus,
	oa::Bool inPoisoned)
{
	oa::SatelliteMessage response;
	response.type = oa::SatelliteMessageType::Error;
	response.flags = oa::SatelliteProtocol::kResponseFlag;
	response.requestId = inRequest.requestId;
	response.sessionEpoch = inRequest.sessionEpoch;
	response.fields.pushBack(oa::SatelliteField::u32(
		oa::SatelliteFieldId::ErrorStatusCode,
		static_cast<oa::U32>(inStatus.getCode())));
	const oa::StringView message = inStatus.getMessage().empty()
		? oa::statusCodeName(inStatus.getCode()) : oa::StringView(inStatus.getMessage());
	response.fields.pushBack(oa::SatelliteField::string(
		oa::SatelliteFieldId::ErrorMessage, message));
	response.fields.pushBack(oa::SatelliteField::u8(
		oa::SatelliteFieldId::ErrorPoisoned, inPoisoned ? 1U : 0U));
	return response;
}

oa::SatelliteMessage resultResponse(
	const oa::SatelliteMessage& inRequest,
	oa::U64 inTargetRequest,
	oa::Bool inComplete,
	oa::StatusCode inStatus,
	oa::Span<const oa::Byte> inBytes = {},
	oa::Span<const oa::Byte> inProfile = {})
{
	oa::SatelliteMessage response;
	response.type = oa::SatelliteMessageType::result;
	response.flags = oa::SatelliteProtocol::kResponseFlag;
	response.requestId = inRequest.requestId;
	response.sessionEpoch = inRequest.sessionEpoch;
	response.fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::ResultRequestId, inTargetRequest));
	response.fields.pushBack(oa::SatelliteField::u8(
		oa::SatelliteFieldId::ResultComplete, inComplete ? 1U : 0U));
	response.fields.pushBack(oa::SatelliteField::u32(
		oa::SatelliteFieldId::ResultStatusCode, static_cast<oa::U32>(inStatus)));
	if (not inBytes.empty()) {
		response.fields.pushBack(oa::SatelliteField::bytes(
			oa::SatelliteFieldId::ResultBytes, inBytes));
	}
	if (not inProfile.empty()) {
		response.fields.pushBack(oa::SatelliteField::bytes(
			oa::SatelliteFieldId::ResultProfile, inProfile));
	}
	return response;
}

oa::Status decodeError(const oa::SatelliteMessage& inMessage) {
	const auto* codeField = requiredField(
		inMessage, oa::SatelliteFieldId::ErrorStatusCode);
	const auto* messageField = requiredField(
		inMessage, oa::SatelliteFieldId::ErrorMessage);
	if (codeField == nullptr or messageField == nullptr) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: malformed error response");
	}
	auto code = oa::SatelliteProtocol::readU32(*codeField);
	auto message = oa::SatelliteProtocol::readString(*messageField);
	if (code.isError()) return code.getStatus();
	if (message.isError()) return message.getStatus();
	if (not isKnownStatusCode(*code)) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: invalid remote status code");
	}
	return oa::Status(static_cast<oa::StatusCode>(*code), oa::move(*message));
}

oa::SatelliteLimits readHelloLimits(const oa::SatelliteMessage& inHello) {
	oa::SatelliteLimits limits;
	limits.maxPayloadBytes = *oa::SatelliteProtocol::readU32(*requiredField(
		inHello, oa::SatelliteFieldId::HelloMaxPayloadBytes));
	limits.maxResidentBytes = *oa::SatelliteProtocol::readU64(*requiredField(
		inHello, oa::SatelliteFieldId::HelloMaxResidentBytes));
	limits.maxObjects = *oa::SatelliteProtocol::readU32(*requiredField(
		inHello, oa::SatelliteFieldId::HelloMaxObjects));
	limits.maxInflight = *oa::SatelliteProtocol::readU32(*requiredField(
		inHello, oa::SatelliteFieldId::HelloMaxInflight));
	return limits;
}

oa::Bool validLimits(const oa::SatelliteLimits& inLimits) {
	return inLimits.maxPayloadBytes > 0U
		and inLimits.maxPayloadBytes <= oa::SatelliteProtocol::kMaxPayloadBytes
		and inLimits.maxResidentBytes > 0U
		and inLimits.maxObjects > 0U
		and inLimits.maxInflight == 1U;
}

oa::SatelliteLimits negotiateLimits(
	const oa::SatelliteLimits& inClient, const oa::SatelliteLimits& inServer)
{
	oa::SatelliteLimits limits;
	limits.maxPayloadBytes = std::min(
		inClient.maxPayloadBytes, inServer.maxPayloadBytes);
	limits.maxResidentBytes = std::min(
		inClient.maxResidentBytes, inServer.maxResidentBytes);
	limits.maxObjects = std::min(inClient.maxObjects, inServer.maxObjects);
	limits.maxInflight = 1U;
	return limits;
}

oa::Status sendFailure(
	oa::TcpStream& inStream,
	const oa::SatelliteMessage& inRequest,
	oa::Status inStatus,
	oa::Bool inPoisoned = false)
{
	const auto write = writeMessage(
		inStream, errorResponse(inRequest, inStatus, inPoisoned));
	return write.isError() ? write : inStatus;
}

} // namespace

oa::SatelliteSecret::~SatelliteSecret() {
	secureZero(bytes_.data(), bytes_.size());
	valid_ = false;
}

oa::SatelliteSecret::SatelliteSecret(oa::SatelliteSecret&& inOther) noexcept
	: bytes_(inOther.bytes_), valid_(inOther.valid_)
{
	secureZero(inOther.bytes_.data(), inOther.bytes_.size());
	inOther.valid_ = false;
}

oa::SatelliteSecret& oa::SatelliteSecret::operator=(
	oa::SatelliteSecret&& inOther) noexcept
{
	if (this != &inOther) {
		secureZero(bytes_.data(), bytes_.size());
		bytes_ = inOther.bytes_;
		valid_ = inOther.valid_;
		secureZero(inOther.bytes_.data(), inOther.bytes_.size());
		inOther.valid_ = false;
	}
	return *this;
}

oa::Result<oa::SatelliteSecret> oa::SatelliteSecret::fromBytes(
	oa::Span<const oa::Byte> inBytes)
{
	if (inBytes.size() != 32U) {
		return oa::Status::invalidArgument(
			"satellite session: authentication secret must be exactly 32 bytes");
	}
	oa::SatelliteSecret secret;
	std::memcpy(secret.bytes_.data(), inBytes.data(), inBytes.size());
	secret.valid_ = true;
	return secret;
}

oa::Span<const oa::Byte> oa::SatelliteSecret::bytes() const noexcept {
	return oa::Span<const oa::Byte>(bytes_.data(), valid_ ? bytes_.size() : 0U);
}

oa::Array<oa::Byte, 32> oa::satelliteBuildHash() {
	return hashString(oa::version());
}

oa::Array<oa::Byte, 32> oa::satelliteSchemaHash() {
	static constexpr char schema[] =
#define OA_SATELLITE_MESSAGE(Name, Value) "M:" #Name ":" #Value ";"
#define OA_SATELLITE_FIELD(message, Name, Value, Kind, Required, MinBytes, MaxBytes) \
		"F:" #message ":" #Name ":" #Value ":" #Kind ":" #Required ":" \
		#MinBytes ":" #MaxBytes ";"
#include "satelliteProtocolSchema.inl"
#undef OA_SATELLITE_FIELD
#undef OA_SATELLITE_MESSAGE
		;
	return hashString(oa::StringView(schema, sizeof(schema) - 1U));
}

oa::Status oa::satelliteValidateRequestEnvelope(
	const oa::SatelliteMessage& inRequest,
	oa::U64 inSessionEpoch,
	oa::U64 inLastRequestId)
{
	if (inRequest.flags != 0U) {
		return oa::Status::error(oa::StatusCode::Aborted,
			"satellite session: a request cannot carry response flags");
	}
	if (inRequest.sessionEpoch != inSessionEpoch) {
		return oa::Status::error(oa::StatusCode::Aborted,
			"satellite session: stale session epoch");
	}
	if (inRequest.requestId <= inLastRequestId) {
		return oa::Status::error(oa::StatusCode::Aborted,
			"satellite session: request id is duplicate or non-monotonic");
	}
	return oa::Status::ok();
}

oa::Status oa::SatelliteServerSession::serve(oa::TcpStream inStream) {
	if (not config_.secret.isValid() or config_.mac.empty()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite session: server secret is not configured");
	}
	if (inStream.remoteAddr() != "127.0.0.1") {
		return oa::Status::error(oa::StatusCode::PermissionDenied,
			"satellite session: non-loopback peers require an authenticated encrypted channel");
	}
	OA_RETURN_IF_ERROR(inStream.setIoTimeout(config_.ioTimeoutMs));

	auto helloResult = readMessage(inStream, config_.limits.maxPayloadBytes);
	if (helloResult.isError()) return helloResult.getStatus();
	const auto& hello = *helloResult;
	if (hello.type != oa::SatelliteMessageType::Hello
		or hello.flags != 0U or hello.sessionEpoch != 0U)
	{
		return sendFailure(inStream, hello,
			oa::Status::error(oa::StatusCode::FailedPrecondition,
				"satellite session: HELLO must begin epoch zero"));
	}

	const oa::U16 protocolMin = *oa::SatelliteProtocol::readU16(*requiredField(
		hello, oa::SatelliteFieldId::HelloProtocolMin));
	const oa::U16 protocolMax = *oa::SatelliteProtocol::readU16(*requiredField(
		hello, oa::SatelliteFieldId::HelloProtocolMax));
	if (protocolMin > oa::SatelliteProtocol::kVersion
		or protocolMax < oa::SatelliteProtocol::kVersion)
	{
		return sendFailure(inStream, hello,
			oa::Status::error(oa::StatusCode::FailedPrecondition,
				"satellite session: no common protocol version"));
	}
	const auto requestedLimits = readHelloLimits(hello);
	if (not validLimits(requestedLimits) or not validLimits(config_.limits)) {
		return sendFailure(inStream, hello,
			oa::Status::invalidArgument("satellite session: invalid advertised limits"));
	}
	const auto limits = negotiateLimits(requestedLimits, config_.limits);

	const auto* clientNonceField = requiredField(
		hello, oa::SatelliteFieldId::HelloClientNonce);
	const auto* proofField = requiredField(
		hello, oa::SatelliteFieldId::HelloAuthProof);
	const auto* buildField = requiredField(
		hello, oa::SatelliteFieldId::HelloBuildHash);
	const auto* schemaField = requiredField(
		hello, oa::SatelliteFieldId::HelloSchemaHash);
	const auto buildHash = oa::satelliteBuildHash();
	const auto schemaHash = oa::satelliteSchemaHash();
	if (not constantEqual(asBytes(buildField->data), buildHash)
		or not constantEqual(asBytes(schemaField->data), schemaHash))
	{
		return sendFailure(inStream, hello,
			oa::Status::error(oa::StatusCode::FailedPrecondition,
				"satellite session: build or schema hash mismatch"));
	}
	auto expectedProof = kmac(
		config_.mac, config_.secret.bytes(),
		asBytes(clientNonceField->data), kHelloCustom);
	if (expectedProof.isError()) return expectedProof.getStatus();
	if (not constantEqual(asBytes(proofField->data), *expectedProof)) {
		return sendFailure(inStream, hello,
			oa::Status::error(oa::StatusCode::Unauthenticated,
				"satellite session: peer authentication failed"));
	}

	auto serverNonceResult = randomNonce();
	if (serverNonceResult.isError()) return serverNonceResult.getStatus();
	const auto& serverNonce = *serverNonceResult;
	auto epochResult = nonZeroEpoch(lastEpoch_);
	if (epochResult.isError()) return epochResult.getStatus();
	const oa::U64 epoch = *epochResult;
	lastEpoch_ = epoch;
	oa::Vec<oa::Byte> replyProofData;
	replyProofData.append(clientNonceField->data.data(), clientNonceField->data.size());
	replyProofData.append(serverNonce.data(), serverNonce.size());
	appendU64Le(replyProofData, epoch);
	auto replyProof = kmac(
		config_.mac, config_.secret.bytes(), asBytes(replyProofData), kReplyCustom);
	if (replyProof.isError()) return replyProof.getStatus();

	oa::SatelliteMessage reply;
	reply.type = oa::SatelliteMessageType::HelloReply;
	reply.flags = oa::SatelliteProtocol::kResponseFlag;
	reply.requestId = hello.requestId;
	reply.sessionEpoch = epoch;
	reply.fields.pushBack(oa::SatelliteField::u16(
		oa::SatelliteFieldId::HelloReplyProtocol, oa::SatelliteProtocol::kVersion));
	reply.fields.pushBack(oa::SatelliteField::bytes(
		oa::SatelliteFieldId::HelloReplyServerNonce, serverNonce));
	reply.fields.pushBack(oa::SatelliteField::bytes(
		oa::SatelliteFieldId::HelloReplyAuthProof, *replyProof));
	reply.fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::HelloReplySessionEpoch, epoch));
	reply.fields.pushBack(oa::SatelliteField::string(
		oa::SatelliteFieldId::HelloReplyDeviceName, config_.deviceName));
	reply.fields.pushBack(oa::SatelliteField::u32(
		oa::SatelliteFieldId::HelloReplyMaxPayloadBytes, limits.maxPayloadBytes));
	reply.fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::HelloReplyMaxResidentBytes, limits.maxResidentBytes));
	reply.fields.pushBack(oa::SatelliteField::u32(
		oa::SatelliteFieldId::HelloReplyMaxObjects, limits.maxObjects));
	reply.fields.pushBack(oa::SatelliteField::u32(
		oa::SatelliteFieldId::HelloReplyMaxInflight, limits.maxInflight));
	OA_RETURN_IF_ERROR(writeMessage(inStream, reply));

	oa::U64 lastRequestId = hello.requestId;
	oa::Vec<SessionObject> objects;
	oa::Optional<PendingOp> pending;
	while (true) {
		auto requestResult = readMessage(inStream, limits.maxPayloadBytes);
		if (requestResult.isError()) return requestResult.getStatus();
		const auto& request = *requestResult;
		const auto envelope = oa::satelliteValidateRequestEnvelope(
			request, epoch, lastRequestId);
		if (envelope.isError()) {
			return sendFailure(inStream, request,
				envelope, true);
		}
		lastRequestId = request.requestId;

		if (request.type == oa::SatelliteMessageType::PutObject) {
			if (objects.size() >= limits.maxObjects) {
				(void)writeMessage(inStream, errorResponse(request,
					oa::Status::error(oa::StatusCode::ResourceExhausted,
						"satellite session: object quota reached"), false));
				continue;
			}
			const oa::U64 objectId = *oa::SatelliteProtocol::readU64(*requiredField(
				request, oa::SatelliteFieldId::PutObjectId));
			const oa::U8 dtype = *oa::SatelliteProtocol::readU8(*requiredField(
				request, oa::SatelliteFieldId::PutObjectDtype));
			auto shape = oa::SatelliteProtocol::readI64Array(*requiredField(
				request, oa::SatelliteFieldId::PutObjectShape));
			const auto* data = requiredField(
				request, oa::SatelliteFieldId::PutObjectData);
			const auto* claimedHash = requiredField(
				request, oa::SatelliteFieldId::PutObjectContentHash);
			const oa::U64 objectVersion = *oa::SatelliteProtocol::readU64(*requiredField(
				request, oa::SatelliteFieldId::PutObjectVersion));
			if (objectId == 0U or findObject(objects, objectId) != nullptr) {
				(void)writeMessage(inStream, errorResponse(request,
					oa::Status::error(oa::StatusCode::AlreadyExists,
						"satellite session: object id is zero or already exists"), false));
				continue;
			}
			const auto scalarType = static_cast<oa::ScalarType>(dtype);
			const oa::Usize scalarBytes = oa::scalarSize(scalarType);
			if ((scalarType != oa::ScalarType::Float32
					and scalarType != oa::ScalarType::UInt32)
				or scalarBytes == 0U or shape.isError())
			{
				(void)writeMessage(inStream, errorResponse(request,
					oa::Status::error(oa::StatusCode::DtypeMismatch,
						"satellite session: named work admits canonical FP32/UInt32 objects"), false));
				continue;
			}
			auto elements = checkedElementCount(
				oa::Span<const oa::I64>(shape->data(), shape->size()));
			if (elements.isError()
				or *elements > std::numeric_limits<oa::U64>::max() / scalarBytes
				or *elements * scalarBytes != data->data.size())
			{
				(void)writeMessage(inStream, errorResponse(request,
					oa::Status::error(oa::StatusCode::ShapeMismatch,
						"satellite session: shape, dtype, and byte count disagree"), false));
				continue;
			}
			const auto actualHash = hashBytes(asBytes(data->data));
			if (not constantEqual(claimedHash->data.span(), actualHash)) {
				(void)writeMessage(inStream, errorResponse(request,
					oa::Status::error(oa::StatusCode::DataLoss,
						"satellite session: object content hash mismatch"), false));
				continue;
			}
			if (data->data.size() > limits.maxResidentBytes
				or totalObjectBytes(objects)
					> limits.maxResidentBytes - data->data.size())
			{
				(void)writeMessage(inStream, errorResponse(request,
					oa::Status::error(oa::StatusCode::ResourceExhausted,
						"satellite session: resident-byte quota reached"), false));
				continue;
			}
			SessionObject object;
			object.id = objectId;
			object.version = objectVersion;
			object.dtype = scalarType;
			object.shape = oa::move(*shape);
			object.data = data->data;
			object.hash = actualHash;
			objects.pushBack(oa::move(object));
			auto acknowledged = resultResponse(
				request, request.requestId, true, oa::StatusCode::Ok);
			acknowledged.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::ResultObjectId, objectId));
			acknowledged.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::ResultObjectVersion, objectVersion));
			acknowledged.fields.pushBack(oa::SatelliteField::bytes(
				oa::SatelliteFieldId::ResultObjectHash, actualHash));
			OA_RETURN_IF_ERROR(writeMessage(inStream, acknowledged));
			continue;
		}

		if (request.type == oa::SatelliteMessageType::dropObject) {
			const oa::U64 objectId = *oa::SatelliteProtocol::readU64(*requiredField(
				request, oa::SatelliteFieldId::DropObjectId));
			if (pending and not pending->complete
				and pendingUsesObject(*pending, objectId))
			{
				(void)writeMessage(inStream, errorResponse(request,
					oa::Status::error(oa::StatusCode::FailedPrecondition,
						"satellite session: object belongs to pending work"), false));
				continue;
			}
			oa::Bool erased = false;
			for (auto it = objects.begin(); it != objects.end(); ++it) {
				if (it->id == objectId) {
					objects.erase(it);
					erased = true;
					break;
				}
			}
			if (not erased) {
				(void)writeMessage(inStream, errorResponse(request,
					oa::Status::notFound("satellite session: object is not retained"), false));
				continue;
			}
			OA_RETURN_IF_ERROR(writeMessage(inStream, resultResponse(
				request, request.requestId, true, oa::StatusCode::Ok)));
			continue;
		}

		if (request.type == oa::SatelliteMessageType::ExecuteNamed) {
			if (pending and not pending->complete) {
				(void)writeMessage(inStream, errorResponse(request,
					oa::Status::error(oa::StatusCode::ResourceExhausted,
						"satellite session: in-flight quota reached"), false));
				continue;
			}
			const auto operation = oa::SatelliteProtocol::readString(*requiredField(
				request, oa::SatelliteFieldId::ExecuteOperation));
			const auto inputs = oa::SatelliteProtocol::readU64Array(*requiredField(
				request, oa::SatelliteFieldId::ExecuteInputObjectIds));
			const auto output = oa::SatelliteProtocol::readU64(*requiredField(
				request, oa::SatelliteFieldId::ExecuteOutputObjectId));
			const auto* arguments = oa::SatelliteProtocol::findField(
				request, oa::SatelliteFieldId::ExecuteArguments);
			const auto* expectedVersionField = oa::SatelliteProtocol::findField(
				request, oa::SatelliteFieldId::ExecuteExpectedVersion);
			const auto* expectedHashField = oa::SatelliteProtocol::findField(
				request, oa::SatelliteFieldId::ExecuteExpectedHash);
			if (operation.isError() or inputs.isError() or output.isError()) {
				(void)writeMessage(inStream, errorResponse(request,
					oa::Status::invalidArgument(
						"satellite session: malformed named operation"), false));
				continue;
			}
			PendingOp next;
			next.requestId = request.requestId;
			if (*operation == "u32-add-v1" and inputs->empty() and *output == 0U
				and arguments != nullptr and arguments->data.size() == 8U)
			{
				next.kind = PendingKind::CpuAdd;
				next.a = readU32Le(arguments->data.data());
				next.b = readU32Le(arguments->data.data() + 4U);
			} else if (*operation == "matrix-add-f32-v1" and inputs->size() == 2U
				and *output != 0U and arguments == nullptr and engine_ != nullptr)
			{
				const auto* a = findObject(objects, (*inputs)[0]);
				const auto* b = findObject(objects, (*inputs)[1]);
				if (objects.size() >= limits.maxObjects
					or a == nullptr or b == nullptr or a->dtype != oa::ScalarType::Float32
					or b->dtype != oa::ScalarType::Float32 or a->shape != b->shape
					or findObject(objects, *output) != nullptr
					or a->data.size() > limits.maxResidentBytes
					or totalObjectBytes(objects)
						> limits.maxResidentBytes - a->data.size())
				{
					(void)writeMessage(inStream, errorResponse(request,
						oa::Status::error(oa::StatusCode::FailedPrecondition,
							"satellite session: matrix inputs/output or quota are invalid"), false));
					continue;
				}
				next.kind = PendingKind::MatrixAddF32;
				next.inputA = (*inputs)[0];
				next.inputB = (*inputs)[1];
				next.output = *output;
			} else if (not config_.namedWork.empty()
				and admitsNamedOperation(config_, *operation) and not inputs->empty()
				and *output != 0U and engine_ != nullptr
				and expectedVersionField != nullptr and expectedHashField != nullptr)
			{
				const auto expectedVersion = oa::SatelliteProtocol::readU64(
					*expectedVersionField);
				const auto* first = findObject(objects, (*inputs)[0]);
				oa::Bool inputsExist = expectedVersion.isOk() and first != nullptr;
				for (const oa::U64 input : *inputs) {
					inputsExist = inputsExist and findObject(objects, input) != nullptr;
				}
				if (not inputsExist or objects.size() >= limits.maxObjects
					or findObject(objects, *output) != nullptr
					or expectedHashField->data.size() != 32U
					or *expectedVersion != first->version
					or not constantEqual(expectedHashField->data.span(), first->hash))
				{
					(void)writeMessage(inStream, errorResponse(request,
						oa::Status::error(oa::StatusCode::FailedPrecondition,
							"satellite session: named-work inputs, output, version, or hash are invalid"), false));
					continue;
				}
				next.kind = PendingKind::NamedWork;
				next.operation = oa::move(*operation);
				next.inputs = oa::move(*inputs);
				next.output = *output;
				next.expectedVersion = *expectedVersion;
				std::copy(expectedHashField->data.begin(), expectedHashField->data.end(),
					next.expectedHash.data());
				if (arguments != nullptr) next.arguments = arguments->data;
			} else {
				(void)writeMessage(inStream, errorResponse(request,
					oa::Status::invalidArgument(
						"satellite session: operation is unknown or unavailable"), false));
				continue;
			}
			pending = oa::move(next);
			OA_RETURN_IF_ERROR(writeMessage(inStream, resultResponse(
				request, request.requestId, false, oa::StatusCode::Ok)));
			continue;
		}

		if (request.type == oa::SatelliteMessageType::poll
			or request.type == oa::SatelliteMessageType::wait
			or request.type == oa::SatelliteMessageType::cancel
			or request.type == oa::SatelliteMessageType::getResult)
		{
			oa::SatelliteFieldId targetField = oa::SatelliteFieldId::PollRequestId;
			if (request.type == oa::SatelliteMessageType::wait) {
				targetField = oa::SatelliteFieldId::WaitRequestId;
			} else if (request.type == oa::SatelliteMessageType::cancel) {
				targetField = oa::SatelliteFieldId::CancelRequestId;
			} else if (request.type == oa::SatelliteMessageType::getResult) {
				targetField = oa::SatelliteFieldId::GetResultRequestId;
			}
			const oa::U64 target = *oa::SatelliteProtocol::readU64(*requiredField(
				request, targetField));
			if (not pending or pending->requestId != target) {
				(void)writeMessage(inStream, errorResponse(request,
					oa::Status::notFound("satellite session: request is not retained"), false));
				continue;
			}
			if (request.type == oa::SatelliteMessageType::cancel and not pending->complete) {
				pending->complete = true;
				pending->status = oa::StatusCode::Cancelled;
			}
			if (request.type == oa::SatelliteMessageType::wait and not pending->complete) {
				if (pending->kind == PendingKind::CpuAdd) {
					pending->result.clear();
					appendU32Le(pending->result, pending->a + pending->b);
					pending->complete = true;
				} else if (pending->kind == PendingKind::MatrixAddF32) {
					const auto* aObject = findObject(objects, pending->inputA);
					const auto* bObject = findObject(objects, pending->inputB);
					auto shape = matrixShapeFromSpan(oa::Span<const oa::I64>(
						aObject->shape.data(), aObject->shape.size()));
					auto aValues = decodeF32(asBytes(aObject->data));
					auto bValues = decodeF32(asBytes(bObject->data));
					if (shape.isError() or aValues.isError() or bValues.isError()) {
						return sendFailure(inStream, request,
							oa::Status::error(oa::StatusCode::DataLoss,
								"satellite session: retained matrix object is invalid"), true);
					}
					auto& context = oa::ExecutionSession::forEngine(*engine_);
					oa::ExecutionSession::RecordingScope recording(context);
					const auto a = oa::FnMatrix::fromBytes(
						oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(
							aValues->data()), aValues->size() * sizeof(oa::F32)),
						*shape, oa::ScalarType::Float32);
					const auto b = oa::FnMatrix::fromBytes(
						oa::Span<const oa::Byte>(reinterpret_cast<const oa::Byte*>(
							bValues->data()), bValues->size() * sizeof(oa::F32)),
						*shape, oa::ScalarType::Float32);
					const auto out = oa::FnMatrix::add(a, b);
					auto submitted = engine_->submit();
					if (submitted.isError()) {
						context.clear();
						(void)writeMessage(inStream, errorResponse(
							request, submitted.getStatus(), false));
						continue;
					}
					lastGpuEventWasOwned_ = engine_->ownsEvent(*submitted);
					lastGpuEventValue_ = submitted->value();
					const auto waitStatus = engine_->wait(*submitted);
					lastGpuEventCompleted_ = submitted->isComplete();
					if (waitStatus.isError() or not lastGpuEventWasOwned_
						or not lastGpuEventCompleted_)
					{
						const auto failure = waitStatus.isError() ? waitStatus
							: oa::Status::error(oa::StatusCode::Internal,
								"satellite session: exact GPU event provenance failed");
						return sendFailure(inStream, request, failure, true);
					}
					oa::Vec<oa::F32> host(aValues->size());
					const auto copied = oa::FnMatrix::copyToHost(
						out, host.data(), host.size() * sizeof(oa::F32));
					if (copied.isError()) {
						return sendFailure(inStream, request, copied, true);
					}
					SessionObject outputObject;
					outputObject.id = pending->output;
					outputObject.shape = aObject->shape;
					outputObject.data.reserve(host.size() * sizeof(oa::F32));
					for (const oa::F32 value : host) {
						appendF32Le(outputObject.data, value);
					}
					outputObject.hash = hashBytes(asBytes(outputObject.data));
					objects.pushBack(oa::move(outputObject));
					pending->complete = true;
				} else {
					oa::SatelliteNamedRequest namedRequest;
					namedRequest.operation = pending->operation;
					namedRequest.arguments = asBytes(pending->arguments);
					namedRequest.expectedVersion = pending->expectedVersion;
					namedRequest.expectedHash = pending->expectedHash;
					namedRequest.inputs.reserve(pending->inputs.size());
					for (const oa::U64 inputId : pending->inputs) {
						const auto* object = findObject(objects, inputId);
						if (object == nullptr) {
							return sendFailure(inStream, request,
								oa::Status::error(oa::StatusCode::DataLoss,
									"satellite session: retained named-work input disappeared"), true);
						}
						oa::SatelliteObjectView view;
						view.id = object->id;
						view.version = object->version;
						view.dtype = object->dtype;
						view.shape = oa::Span<const oa::I64>(object->shape.data(), object->shape.size());
						view.data = asBytes(object->data);
						view.hash = object->hash;
						namedRequest.inputs.pushBack(view);
					}
					auto namedResult = config_.namedWork(*engine_, namedRequest);
					if (namedResult.isError()) {
						return sendFailure(inStream, request, namedResult.getStatus(), true);
					}
					auto elements = checkedElementCount(oa::Span<const oa::I64>(
						namedResult->shape.data(), namedResult->shape.size()));
					const oa::Usize scalarBytes = oa::scalarSize(namedResult->dtype);
					const oa::Bool eventOwned = engine_->ownsEvent(namedResult->completion);
					const oa::Bool eventComplete = namedResult->completion.isComplete();
					constexpr oa::Usize kResultEnvelopeBytes = 55U;
					const oa::Bool resultFitsPayload =
						namedResult->data.size() <= limits.maxPayloadBytes
						and namedResult->profile.size()
							<= limits.maxPayloadBytes - namedResult->data.size()
						and kResultEnvelopeBytes
							<= limits.maxPayloadBytes - namedResult->data.size()
								- namedResult->profile.size();
					if (elements.isError() or scalarBytes == 0U
						or *elements > std::numeric_limits<oa::U64>::max() / scalarBytes
						or *elements * scalarBytes != namedResult->data.size()
						or not resultFitsPayload
						or namedResult->profile.size() > 4096U
						or namedResult->data.size() > limits.maxResidentBytes
						or totalObjectBytes(objects)
							> limits.maxResidentBytes - namedResult->data.size()
						or not namedResult->completion.isValid()
						or not eventOwned or not eventComplete)
					{
						return sendFailure(inStream, request,
							oa::Status::error(oa::StatusCode::DataLoss,
								"satellite session: named-work result contract is invalid"), true);
					}
					lastGpuEventWasOwned_ = eventOwned;
					lastGpuEventCompleted_ = eventComplete;
					lastGpuEventValue_ = namedResult->completion.value();
					SessionObject outputObject;
					outputObject.id = pending->output;
					outputObject.version = namedResult->version;
					outputObject.dtype = namedResult->dtype;
					outputObject.shape = oa::move(namedResult->shape);
					outputObject.data = oa::move(namedResult->data);
					outputObject.hash = hashBytes(asBytes(outputObject.data));
					pending->profile = oa::move(namedResult->profile);
					objects.pushBack(oa::move(outputObject));
					pending->complete = true;
				}
			}
			oa::Span<const oa::Byte> resultBytes;
			if (pending->complete and pending->status == oa::StatusCode::Ok) {
				if (pending->kind == PendingKind::CpuAdd) {
					resultBytes = asBytes(pending->result);
				} else if (const auto* result = findObject(objects, pending->output)) {
					resultBytes = asBytes(result->data);
				}
			}
			OA_RETURN_IF_ERROR(writeMessage(inStream, resultResponse(
				request, target, pending->complete, pending->status, resultBytes,
				asBytes(pending->profile))));
			continue;
		}

		if (request.type == oa::SatelliteMessageType::Close) {
			OA_RETURN_IF_ERROR(writeMessage(inStream, resultResponse(
				request, request.requestId, true, oa::StatusCode::Ok)));
			return oa::Status::ok();
		}
		if (request.type == oa::SatelliteMessageType::abort) {
			OA_RETURN_IF_ERROR(writeMessage(inStream, resultResponse(
				request, request.requestId, true, oa::StatusCode::Aborted)));
			return oa::Status::error(oa::StatusCode::Aborted,
				"satellite session: peer aborted the epoch");
		}

		(void)writeMessage(inStream, errorResponse(request,
			oa::Status::unimplemented(
				"satellite session: message is not implemented by the CPU oracle"), false));
	}
}

oa::Result<oa::SatelliteClientSession> oa::SatelliteClientSession::connect(
	const oa::String& inHost,
	oa::U16 inPort,
	oa::SatelliteSessionConfig inConfig)
{
	if (inHost != "127.0.0.1") {
		return oa::Status::error(oa::StatusCode::PermissionDenied,
			"satellite session: only explicit IPv4 loopback is admitted");
	}
	if (not inConfig.secret.isValid() or inConfig.mac.empty()
		or not validLimits(inConfig.limits)) {
		return oa::Status::invalidArgument(
			"satellite session: invalid client secret or limits");
	}
	auto connected = oa::TcpStream::connect(inHost, inPort);
	if (connected.isError()) return connected.getStatus();
	oa::SatelliteClientSession session;
	session.stream_ = oa::move(*connected);
	OA_RETURN_IF_ERROR(session.stream_.setIoTimeout(inConfig.ioTimeoutMs));

	auto clientNonceResult = randomNonce();
	if (clientNonceResult.isError()) return clientNonceResult.getStatus();
	const auto& clientNonce = *clientNonceResult;
	auto proof = kmac(
		inConfig.mac, inConfig.secret.bytes(), clientNonce, kHelloCustom);
	if (proof.isError()) return proof.getStatus();
	const auto buildHash = oa::satelliteBuildHash();
	const auto schemaHash = oa::satelliteSchemaHash();
	oa::SatelliteMessage hello;
	hello.type = oa::SatelliteMessageType::Hello;
	hello.requestId = session.nextRequestId_++;
	hello.sessionEpoch = 0U;
	hello.fields.pushBack(oa::SatelliteField::u16(
		oa::SatelliteFieldId::HelloProtocolMin, oa::SatelliteProtocol::kVersion));
	hello.fields.pushBack(oa::SatelliteField::u16(
		oa::SatelliteFieldId::HelloProtocolMax, oa::SatelliteProtocol::kVersion));
	hello.fields.pushBack(oa::SatelliteField::bytes(
		oa::SatelliteFieldId::HelloClientNonce, clientNonce));
	hello.fields.pushBack(oa::SatelliteField::bytes(
		oa::SatelliteFieldId::HelloAuthProof, *proof));
	hello.fields.pushBack(oa::SatelliteField::bytes(
		oa::SatelliteFieldId::HelloBuildHash, buildHash));
	hello.fields.pushBack(oa::SatelliteField::bytes(
		oa::SatelliteFieldId::HelloSchemaHash, schemaHash));
	hello.fields.pushBack(oa::SatelliteField::u32(
		oa::SatelliteFieldId::HelloMaxPayloadBytes, inConfig.limits.maxPayloadBytes));
	hello.fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::HelloMaxResidentBytes, inConfig.limits.maxResidentBytes));
	hello.fields.pushBack(oa::SatelliteField::u32(
		oa::SatelliteFieldId::HelloMaxObjects, inConfig.limits.maxObjects));
	hello.fields.pushBack(oa::SatelliteField::u32(
		oa::SatelliteFieldId::HelloMaxInflight, inConfig.limits.maxInflight));
	OA_RETURN_IF_ERROR(writeMessage(session.stream_, hello));
	auto replyResult = readMessage(session.stream_, inConfig.limits.maxPayloadBytes);
	if (replyResult.isError()) return replyResult.getStatus();
	const auto& reply = *replyResult;
	if (reply.type == oa::SatelliteMessageType::Error) {
		return decodeError(reply);
	}
	if (reply.type != oa::SatelliteMessageType::HelloReply
		or reply.flags != oa::SatelliteProtocol::kResponseFlag
		or reply.requestId != hello.requestId or reply.sessionEpoch == 0U)
	{
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: malformed HELLO_REPLY envelope");
	}
	const auto* serverNonce = requiredField(
		reply, oa::SatelliteFieldId::HelloReplyServerNonce);
	const auto* serverProof = requiredField(
		reply, oa::SatelliteFieldId::HelloReplyAuthProof);
	const oa::U64 epoch = *oa::SatelliteProtocol::readU64(*requiredField(
		reply, oa::SatelliteFieldId::HelloReplySessionEpoch));
	if (epoch != reply.sessionEpoch) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: reply epoch fields disagree");
	}
	oa::Vec<oa::Byte> replyProofData;
	replyProofData.append(clientNonce.data(), clientNonce.size());
	replyProofData.append(serverNonce->data.data(), serverNonce->data.size());
	appendU64Le(replyProofData, epoch);
	auto expectedReplyProof = kmac(
		inConfig.mac, inConfig.secret.bytes(),
		asBytes(replyProofData), kReplyCustom);
	if (expectedReplyProof.isError()) return expectedReplyProof.getStatus();
	if (not constantEqual(asBytes(serverProof->data), *expectedReplyProof)) {
		return oa::Status::error(oa::StatusCode::Unauthenticated,
			"satellite session: server authentication failed");
	}

	session.probe_.sessionEpoch = epoch;
	session.probe_.deviceName = *oa::SatelliteProtocol::readString(*requiredField(
		reply, oa::SatelliteFieldId::HelloReplyDeviceName));
	session.probe_.limits.maxPayloadBytes = *oa::SatelliteProtocol::readU32(*requiredField(
		reply, oa::SatelliteFieldId::HelloReplyMaxPayloadBytes));
	session.probe_.limits.maxResidentBytes = *oa::SatelliteProtocol::readU64(*requiredField(
		reply, oa::SatelliteFieldId::HelloReplyMaxResidentBytes));
	session.probe_.limits.maxObjects = *oa::SatelliteProtocol::readU32(*requiredField(
		reply, oa::SatelliteFieldId::HelloReplyMaxObjects));
	session.probe_.limits.maxInflight = *oa::SatelliteProtocol::readU32(*requiredField(
		reply, oa::SatelliteFieldId::HelloReplyMaxInflight));
	session.probe_.buildHash = buildHash;
	session.probe_.schemaHash = schemaHash;
	if (not validLimits(session.probe_.limits)
		or session.probe_.limits.maxPayloadBytes > inConfig.limits.maxPayloadBytes
		or session.probe_.limits.maxResidentBytes > inConfig.limits.maxResidentBytes
		or session.probe_.limits.maxObjects > inConfig.limits.maxObjects)
	{
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: server returned invalid negotiated limits");
	}
	session.open_ = true;
	return session;
}

oa::Status oa::SatelliteClientSession::putF32(
	oa::U64 inObjectId,
	oa::Span<const oa::I64> inShape,
	oa::Span<const oa::F32> inValues)
{
	return putF32Versioned(inObjectId, 0U, inShape, inValues);
}

oa::Status oa::SatelliteClientSession::putF32Versioned(
	oa::U64 inObjectId,
	oa::U64 inVersion,
	oa::Span<const oa::I64> inShape,
	oa::Span<const oa::F32> inValues)
{
	if (inObjectId == 0U) {
		return oa::Status::invalidArgument("satellite session: object id must be non-zero");
	}
	auto elements = checkedElementCount(inShape);
	if (elements.isError()) return elements.getStatus();
	if (*elements != inValues.size()) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"satellite session: shape and value count disagree");
	}
	oa::Vec<oa::Byte> data;
	data.reserve(inValues.size() * sizeof(oa::F32));
	for (const oa::F32 value : inValues) appendF32Le(data, value);
	return putObject(
		inObjectId, inVersion, oa::ScalarType::Float32, inShape, asBytes(data));
}

oa::Status oa::SatelliteClientSession::putU32(
	oa::U64 inObjectId,
	oa::Span<const oa::I64> inShape,
	oa::Span<const oa::U32> inValues)
{
	return putU32Versioned(inObjectId, 0U, inShape, inValues);
}

oa::Status oa::SatelliteClientSession::putU32Versioned(
	oa::U64 inObjectId,
	oa::U64 inVersion,
	oa::Span<const oa::I64> inShape,
	oa::Span<const oa::U32> inValues)
{
	if (inObjectId == 0U) {
		return oa::Status::invalidArgument("satellite session: object id must be non-zero");
	}
	auto elements = checkedElementCount(inShape);
	if (elements.isError()) return elements.getStatus();
	if (*elements != inValues.size()) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"satellite session: shape and value count disagree");
	}
	oa::Vec<oa::Byte> data;
	data.reserve(inValues.size() * sizeof(oa::U32));
	for (const oa::U32 value : inValues) appendU32Le(data, value);
	return putObject(
		inObjectId, inVersion, oa::ScalarType::UInt32, inShape, asBytes(data));
}

oa::Status oa::SatelliteClientSession::putObject(
	oa::U64 inObjectId,
	oa::U64 inVersion,
	oa::ScalarType inDtype,
	oa::Span<const oa::I64> inShape,
	oa::Span<const oa::Byte> inData)
{
	const auto hash = hashBytes(inData);
	oa::Vec<oa::SatelliteField> fields;
	fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::PutObjectId, inObjectId));
	fields.pushBack(oa::SatelliteField::u8(
		oa::SatelliteFieldId::PutObjectDtype,
		static_cast<oa::U8>(inDtype)));
	fields.pushBack(oa::SatelliteField::i64Array(
		oa::SatelliteFieldId::PutObjectShape, inShape));
	fields.pushBack(oa::SatelliteField::bytes(
		oa::SatelliteFieldId::PutObjectData, inData));
	fields.pushBack(oa::SatelliteField::bytes(
		oa::SatelliteFieldId::PutObjectContentHash, hash));
	fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::PutObjectVersion, inVersion));
	auto response = exchange(oa::SatelliteMessageType::PutObject, oa::move(fields));
	if (response.isError()) return response.getStatus();
	const auto* idField = oa::SatelliteProtocol::findField(
		*response, oa::SatelliteFieldId::ResultObjectId);
	const auto* versionField = oa::SatelliteProtocol::findField(
		*response, oa::SatelliteFieldId::ResultObjectVersion);
	const auto* hashField = oa::SatelliteProtocol::findField(
		*response, oa::SatelliteFieldId::ResultObjectHash);
	if (idField == nullptr or versionField == nullptr or hashField == nullptr) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: object acknowledgement is incomplete");
	}
	auto acknowledgedId = oa::SatelliteProtocol::readU64(*idField);
	auto acknowledgedVersion = oa::SatelliteProtocol::readU64(*versionField);
	if (acknowledgedId.isError()) return acknowledgedId.getStatus();
	if (acknowledgedVersion.isError()) return acknowledgedVersion.getStatus();
	if (*acknowledgedId != inObjectId or *acknowledgedVersion != inVersion
		or not constantEqual(hashField->data.span(), hash))
	{
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: object acknowledgement identity mismatch");
	}
	return oa::Status::ok();
}

oa::Status oa::SatelliteClientSession::dropObject(oa::U64 inObjectId) {
	oa::Vec<oa::SatelliteField> fields;
	fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::DropObjectId, inObjectId));
	auto response = exchange(oa::SatelliteMessageType::dropObject, oa::move(fields));
	return response.isOk() ? oa::Status::ok() : response.getStatus();
}

oa::Result<oa::SatelliteMessage> oa::SatelliteClientSession::exchange(
	oa::SatelliteMessageType inType,
	oa::Vec<oa::SatelliteField> inFields)
{
	if (not open_) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite session: client is closed");
	}
	oa::SatelliteMessage request;
	request.type = inType;
	request.requestId = nextRequestId_++;
	request.sessionEpoch = probe_.sessionEpoch;
	request.fields = oa::move(inFields);
	OA_RETURN_IF_ERROR(writeMessage(stream_, request));
	auto response = readMessage(stream_, probe_.limits.maxPayloadBytes);
	if (response.isError()) return response.getStatus();
	if (response->flags != oa::SatelliteProtocol::kResponseFlag
		or response->requestId != request.requestId
		or response->sessionEpoch != request.sessionEpoch)
	{
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: response envelope does not match request");
	}
	if (response->type == oa::SatelliteMessageType::Error) {
		return decodeError(*response);
	}
	if (response->type != oa::SatelliteMessageType::result) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: expected RESULT response");
	}
	return oa::move(*response);
}

oa::Result<oa::U64> oa::SatelliteClientSession::startCpuAdd(oa::U32 inA, oa::U32 inB) {
	oa::Vec<oa::Byte> arguments;
	appendU32Le(arguments, inA);
	appendU32Le(arguments, inB);
	oa::Vec<oa::SatelliteField> fields;
	fields.pushBack(oa::SatelliteField::string(
		oa::SatelliteFieldId::ExecuteOperation, "u32-add-v1"));
	fields.pushBack(oa::SatelliteField::u64Array(
		oa::SatelliteFieldId::ExecuteInputObjectIds, oa::Span<const oa::U64>()));
	fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::ExecuteOutputObjectId, 0U));
	fields.pushBack(oa::SatelliteField::bytes(
		oa::SatelliteFieldId::ExecuteArguments, asBytes(arguments)));
	const oa::U64 requestId = nextRequestId_;
	auto response = exchange(oa::SatelliteMessageType::ExecuteNamed, oa::move(fields));
	if (response.isError()) return response.getStatus();
	return requestId;
}

oa::Result<oa::U64> oa::SatelliteClientSession::startMatrixAddF32(
	oa::U64 inA,
	oa::U64 inB,
	oa::U64 inOutput)
{
	const oa::U64 inputs[] = {inA, inB};
	oa::Vec<oa::SatelliteField> fields;
	fields.pushBack(oa::SatelliteField::string(
		oa::SatelliteFieldId::ExecuteOperation, "matrix-add-f32-v1"));
	fields.pushBack(oa::SatelliteField::u64Array(
		oa::SatelliteFieldId::ExecuteInputObjectIds, inputs));
	fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::ExecuteOutputObjectId, inOutput));
	const oa::U64 requestId = nextRequestId_;
	auto response = exchange(oa::SatelliteMessageType::ExecuteNamed, oa::move(fields));
	if (response.isError()) return response.getStatus();
	return requestId;
}

oa::Result<oa::U64> oa::SatelliteClientSession::startNamed(
	oa::StringView inOperation,
	oa::Span<const oa::U64> inInputs,
	oa::U64 inOutput,
	oa::Span<const oa::Byte> inArguments,
	oa::U64 inExpectedVersion,
	const oa::Array<oa::Byte, 32>& inExpectedHash)
{
	if (inOperation.empty() or inInputs.empty() or inOutput == 0U) {
		return oa::Status::invalidArgument(
			"satellite session: named operation, inputs, or output is invalid");
	}
	for (const oa::U64 input : inInputs) {
		if (input == 0U) {
			return oa::Status::invalidArgument(
				"satellite session: named-operation input id must be non-zero");
		}
	}
	oa::Vec<oa::SatelliteField> fields;
	fields.pushBack(oa::SatelliteField::string(
		oa::SatelliteFieldId::ExecuteOperation, inOperation));
	fields.pushBack(oa::SatelliteField::u64Array(
		oa::SatelliteFieldId::ExecuteInputObjectIds, inInputs));
	fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::ExecuteOutputObjectId, inOutput));
	if (not inArguments.empty()) {
		fields.pushBack(oa::SatelliteField::bytes(
			oa::SatelliteFieldId::ExecuteArguments, inArguments));
	}
	fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::ExecuteExpectedVersion, inExpectedVersion));
	fields.pushBack(oa::SatelliteField::bytes(
		oa::SatelliteFieldId::ExecuteExpectedHash, inExpectedHash));
	const oa::U64 requestId = nextRequestId_;
	auto response = exchange(oa::SatelliteMessageType::ExecuteNamed, oa::move(fields));
	if (response.isError()) return response.getStatus();
	return requestId;
}

oa::Result<oa::SatelliteMessage> oa::SatelliteClientSession::poll(oa::U64 inRequestId) {
	oa::Vec<oa::SatelliteField> fields;
	fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::PollRequestId, inRequestId));
	return exchange(oa::SatelliteMessageType::poll, oa::move(fields));
}

oa::Result<oa::SatelliteMessage> oa::SatelliteClientSession::wait(oa::U64 inRequestId) {
	oa::Vec<oa::SatelliteField> fields;
	fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::WaitRequestId, inRequestId));
	return exchange(oa::SatelliteMessageType::wait, oa::move(fields));
}

oa::Result<oa::SatelliteMessage> oa::SatelliteClientSession::cancel(oa::U64 inRequestId) {
	oa::Vec<oa::SatelliteField> fields;
	fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::CancelRequestId, inRequestId));
	return exchange(oa::SatelliteMessageType::cancel, oa::move(fields));
}

oa::Result<oa::SatelliteMessage> oa::SatelliteClientSession::getResult(oa::U64 inRequestId) {
	oa::Vec<oa::SatelliteField> fields;
	fields.pushBack(oa::SatelliteField::u64(
		oa::SatelliteFieldId::GetResultRequestId, inRequestId));
	return exchange(oa::SatelliteMessageType::getResult, oa::move(fields));
}

oa::Status oa::SatelliteClientSession::close() {
	if (not open_) return oa::Status::ok();
	auto response = exchange(oa::SatelliteMessageType::Close, {});
	if (response.isError()) return response.getStatus();
	open_ = false;
	stream_.close();
	return oa::Status::ok();
}

oa::Status oa::SatelliteClientSession::abort(oa::StringView inReason) {
	if (not open_) return oa::Status::ok();
	oa::Vec<oa::SatelliteField> fields;
	fields.pushBack(oa::SatelliteField::string(
		oa::SatelliteFieldId::AbortReason,
		inReason.empty() ? oa::StringView("client abort") : inReason));
	auto response = exchange(oa::SatelliteMessageType::abort, oa::move(fields));
	open_ = false;
	stream_.close();
	return response.isError() ? response.getStatus() : oa::Status::ok();
}

oa::Result<oa::U32> oa::SatelliteClientSession::readCpuAddResult(
	const oa::SatelliteMessage& inMessage)
{
	const auto* completeField = requiredField(
		inMessage, oa::SatelliteFieldId::ResultComplete);
	const auto* statusField = requiredField(
		inMessage, oa::SatelliteFieldId::ResultStatusCode);
	const auto* bytesField = oa::SatelliteProtocol::findField(
		inMessage, oa::SatelliteFieldId::ResultBytes);
	if (completeField == nullptr or statusField == nullptr) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: malformed RESULT");
	}
	const oa::U8 complete = *oa::SatelliteProtocol::readU8(*completeField);
	const oa::U32 status = *oa::SatelliteProtocol::readU32(*statusField);
	if (not isKnownStatusCode(status)) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: invalid result status code");
	}
	if (complete == 0U) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite session: result is not complete");
	}
	if (status != static_cast<oa::U32>(oa::StatusCode::Ok)) {
		return oa::Status::error(static_cast<oa::StatusCode>(status),
			"satellite session: operation did not complete successfully");
	}
	if (bytesField == nullptr or bytesField->data.size() != 4U) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: CPU add result has the wrong size");
	}
	return readU32Le(bytesField->data.data());
}

oa::Result<oa::Vec<oa::F32>> oa::SatelliteClientSession::readF32Result(
	const oa::SatelliteMessage& inMessage)
{
	const auto* completeField = requiredField(
		inMessage, oa::SatelliteFieldId::ResultComplete);
	const auto* statusField = requiredField(
		inMessage, oa::SatelliteFieldId::ResultStatusCode);
	const auto* bytesField = oa::SatelliteProtocol::findField(
		inMessage, oa::SatelliteFieldId::ResultBytes);
	if (completeField == nullptr or statusField == nullptr) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: malformed RESULT");
	}
	const oa::U8 complete = *oa::SatelliteProtocol::readU8(*completeField);
	const oa::U32 status = *oa::SatelliteProtocol::readU32(*statusField);
	if (not isKnownStatusCode(status)) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: invalid result status code");
	}
	if (complete == 0U) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite session: result is not complete");
	}
	if (status != static_cast<oa::U32>(oa::StatusCode::Ok)) {
		return oa::Status::error(static_cast<oa::StatusCode>(status),
			"satellite session: operation did not complete successfully");
	}
	if (bytesField == nullptr) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: FP32 result is missing");
	}
	return decodeF32(asBytes(bytesField->data));
}

oa::Result<oa::Vec<oa::Byte>> oa::SatelliteClientSession::readBytesResult(
	const oa::SatelliteMessage& inMessage)
{
	const auto* completeField = requiredField(
		inMessage, oa::SatelliteFieldId::ResultComplete);
	const auto* statusField = requiredField(
		inMessage, oa::SatelliteFieldId::ResultStatusCode);
	const auto* bytesField = oa::SatelliteProtocol::findField(
		inMessage, oa::SatelliteFieldId::ResultBytes);
	if (completeField == nullptr or statusField == nullptr) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: malformed RESULT");
	}
	const oa::U8 complete = *oa::SatelliteProtocol::readU8(*completeField);
	const oa::U32 status = *oa::SatelliteProtocol::readU32(*statusField);
	if (not isKnownStatusCode(status)) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: invalid result status code");
	}
	if (complete == 0U) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"satellite session: result is not complete");
	}
	if (status != static_cast<oa::U32>(oa::StatusCode::Ok)) {
		return oa::Status::error(static_cast<oa::StatusCode>(status),
			"satellite session: operation did not complete successfully");
	}
	if (bytesField == nullptr) {
		return oa::Status::error(oa::StatusCode::DataLoss,
			"satellite session: byte result is missing");
	}
	return bytesField->data;
}

oa::Result<oa::Vec<oa::Byte>> oa::SatelliteClientSession::readProfile(
	const oa::SatelliteMessage& inMessage)
{
	const auto* profile = oa::SatelliteProtocol::findField(
		inMessage, oa::SatelliteFieldId::ResultProfile);
	if (profile == nullptr) return oa::Vec<oa::Byte>{};
	return profile->data;
}
