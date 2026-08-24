// OA CORE - status & result Types
//
// Error handling without exceptions. gRPC-compatible status codes.

#pragma once

#include <oa/core/types.h>

namespace oa {

// STATUS CODES

enum class StatusCode : oa::U8 {
	// standard (gRPC-compatible 0-16)
	Ok               = 0,
	Cancelled        = 1,
	Unknown          = 2,
	InvalidArgument  = 3,
	DeadlineExceeded = 4,
	NotFound         = 5,
	AlreadyExists    = 6,
	PermissionDenied = 7,
	ResourceExhausted = 8,
	FailedPrecondition = 9,
	Aborted          = 10,
	OutOfRange       = 11,
	Unimplemented    = 12,
	Internal         = 13,
	Unavailable      = 14,
	DataLoss         = 15,
	Unauthenticated  = 16,
	// GPU compute (100-119)
	VulkanError        = 100,
	DeviceNotFound     = 101,
	OutOfMemory        = 102,
	PipelineError      = 103,
	ShaderCompileError = 104,
	// crypto (120-139)
	InvalidSignature = 120,
	InvalidBlock     = 121,
	InvalidTransaction = 122,
	InsufficientFunds = 123,
	InsufficientMargin = 124,
	Slashed          = 125,
	// trading (140-159)
	OrderRejected    = 140,
	PositionNotFound = 141,
	MarketClosed     = 142,
	PriceLimitExceeded = 143,
	QuantityTooSmall = 144,
	// ML (160-179)
	ModelNotLoaded   = 160,
	ShapeMismatch    = 161,
	DtypeMismatch    = 162,
	GradientExplosion = 163,
	CheckpointCorrupt = 164,
	// network (180-199)
	ConnectionFailed = 180,
	Timeout          = 181,
	TlsError         = 182,
	DnsError         = 183,
	// file/IO (200-219)
	FileNotFound     = 200,
	FileCorrupt      = 201,
	PermissionError  = 202,
	DiskFull         = 203,
};

[[nodiscard]] constexpr StringView statusCodeName(StatusCode inCode) noexcept {
	switch (inCode) {
		case StatusCode::Ok:                return "OK";
		case StatusCode::Cancelled:         return "CANCELLED";
		case StatusCode::Unknown:           return "UNKNOWN";
		case StatusCode::InvalidArgument:   return "INVALID_ARGUMENT";
		case StatusCode::DeadlineExceeded:  return "DEADLINE_EXCEEDED";
		case StatusCode::NotFound:          return "NOT_FOUND";
		case StatusCode::AlreadyExists:     return "ALREADY_EXISTS";
		case StatusCode::PermissionDenied:  return "PERMISSION_DENIED";
		case StatusCode::ResourceExhausted: return "RESOURCE_EXHAUSTED";
		case StatusCode::FailedPrecondition: return "FAILED_PRECONDITION";
		case StatusCode::Aborted:           return "ABORTED";
		case StatusCode::OutOfRange:        return "OUT_OF_RANGE";
		case StatusCode::Unimplemented:     return "UNIMPLEMENTED";
		case StatusCode::Internal:          return "INTERNAL";
		case StatusCode::Unavailable:       return "UNAVAILABLE";
		case StatusCode::DataLoss:          return "DATA_LOSS";
		case StatusCode::Unauthenticated:   return "UNAUTHENTICATED";
		case StatusCode::VulkanError:       return "VULKAN_ERROR";
		case StatusCode::DeviceNotFound:    return "DEVICE_NOT_FOUND";
		case StatusCode::OutOfMemory:       return "OUT_OF_MEMORY";
		case StatusCode::PipelineError:     return "PIPELINE_ERROR";
		case StatusCode::ShaderCompileError: return "SHADER_COMPILE_ERROR";
		case StatusCode::InvalidSignature:  return "INVALID_SIGNATURE";
		case StatusCode::InvalidBlock:      return "INVALID_BLOCK";
		case StatusCode::InvalidTransaction: return "INVALID_TRANSACTION";
		case StatusCode::InsufficientFunds: return "INSUFFICIENT_FUNDS";
		case StatusCode::InsufficientMargin: return "INSUFFICIENT_MARGIN";
		case StatusCode::Slashed:           return "SLASHED";
		case StatusCode::OrderRejected:     return "ORDER_REJECTED";
		case StatusCode::PositionNotFound:  return "POSITION_NOT_FOUND";
		case StatusCode::MarketClosed:      return "MARKET_CLOSED";
		case StatusCode::PriceLimitExceeded: return "PRICE_LIMIT_EXCEEDED";
		case StatusCode::QuantityTooSmall:  return "QUANTITY_TOO_SMALL";
		case StatusCode::ModelNotLoaded:    return "MODEL_NOT_LOADED";
		case StatusCode::ShapeMismatch:     return "SHAPE_MISMATCH";
		case StatusCode::DtypeMismatch:     return "DTYPE_MISMATCH";
		case StatusCode::GradientExplosion: return "GRADIENT_EXPLOSION";
		case StatusCode::CheckpointCorrupt: return "CHECKPOINT_CORRUPT";
		case StatusCode::ConnectionFailed:  return "CONNECTION_FAILED";
		case StatusCode::Timeout:           return "TIMEOUT";
		case StatusCode::TlsError:          return "TLS_ERROR";
		case StatusCode::DnsError:          return "DNS_ERROR";
		case StatusCode::FileNotFound:      return "FILE_NOT_FOUND";
		case StatusCode::FileCorrupt:       return "FILE_CORRUPT";
		case StatusCode::PermissionError:   return "PERMISSION_ERROR";
		case StatusCode::DiskFull:          return "DISK_FULL";
		default:                              return "UNKNOWN";
	}
}

class Status {
public:
	// status class.

	// Constructors.
	Status() noexcept 
		: code_(StatusCode::Ok)
	{}
	Status(StatusCode inCode, String inMessage = "")
		: code_(inCode)
		, message_(std::move(inMessage))
	{}

	// Methods.
	[[nodiscard]] static Status ok() { return Status(); }
	[[nodiscard]] static Status error(String inMessage) {
		return Status(StatusCode::Internal, std::move(inMessage));
	}
	[[nodiscard]] static Status error(StatusCode inCode, String inMessage = "") {
		return Status(inCode, std::move(inMessage));
	}
	[[nodiscard]] static Status cancelled(String inMessage = "Operation cancelled") {
		return Status(StatusCode::Cancelled, std::move(inMessage));
	}
	[[nodiscard]] static Status invalidArgument(String inMessage) {
		return Status(StatusCode::InvalidArgument, std::move(inMessage));
	}
	[[nodiscard]] static Status notFound(String inMessage) {
		return Status(StatusCode::NotFound, std::move(inMessage));
	}
	[[nodiscard]] static Status unimplemented(String inMessage = "Not implemented") {
		return Status(StatusCode::Unimplemented, std::move(inMessage));
	}
	
	[[nodiscard]] bool isOk() const noexcept { return code_ == StatusCode::Ok; }
	[[nodiscard]] bool isError() const noexcept { return code_ != StatusCode::Ok; }
	[[nodiscard]] StatusCode getCode() const noexcept { return code_; }
	[[nodiscard]] const String& getMessage() const noexcept { return message_; }
	[[nodiscard]] StringView getCodeName() const noexcept { return statusCodeName(code_); }

	[[nodiscard]] String toString() const {
		String result(getCodeName());
		if (!message_.empty()) {
			result += ": "; result += message_; 
		}
		return result;
	}

	// Operators.
	explicit operator bool() const noexcept { return isOk(); }

private:
	// Data, class members.
	StatusCode code_;
	String message_;
};

template<typename T>
class Result {
public:
	// result class.

	// Constructors.
	Result(T inValue)
		: value_(std::move(inValue))
		, status_(Status::ok())
	{}
	Result(Status inStatus)
		: status_(std::move(inStatus))
	{
		assert(!status_.isOk() && "Result from status must be an error");
	}
	Result(Result&& inOther) noexcept
		: value_(std::move(inOther.value_))
		, status_(std::move(inOther.status_))
	{}

	// Methods.
	[[nodiscard]] bool isOk() const noexcept { return status_.isOk(); }
	[[nodiscard]] bool isError() const noexcept { return status_.isError(); }
	[[nodiscard]] const Status& getStatus() const noexcept { return status_; }

	[[nodiscard]] T& getValue() & { assert(isOk()); return *value_; }
	[[nodiscard]] const T& getValue() const & { assert(isOk()); return *value_; }
	[[nodiscard]] T&& getValue() && { assert(isOk()); return std::move(*value_); }

	[[nodiscard]] T valueOr(T inDefault) const & { return isOk() ? *value_ : std::move(inDefault); }
	[[nodiscard]] T valueOr(T inDefault) && { return isOk() ? std::move(*value_) : std::move(inDefault); }

	template<typename F>
	[[nodiscard]] auto map(F&& inFunc) && -> Result<decltype(inFunc(std::declval<T>()))> {
		using U = decltype(inFunc(std::declval<T>()));
		if (isOk()) return Result<U>(inFunc(std::move(*value_)));
		return Result<U>(std::move(status_));
	}

	template<typename F>
	[[nodiscard]] auto andThen(F&& inFunc) && -> decltype(inFunc(std::declval<T>())) {
		if (isOk()) return inFunc(std::move(*value_));
		return decltype(inFunc(std::declval<T>()))(std::move(status_));
	}

	// Operators.
	Result& operator=(Result&& inOther) noexcept {
		if (this != &inOther) {
			value_ = std::move(inOther.value_);
			status_ = std::move(inOther.status_);
		}
		return *this;
	}
	Result(const Result&) = delete;
	Result& operator=(const Result&) = delete;
	explicit operator bool() const noexcept { return isOk(); }
	[[nodiscard]] T& operator*() & { return getValue(); }
	[[nodiscard]] const T& operator*() const & { return getValue(); }
	[[nodiscard]] T&& operator*() && { return std::move(getValue()); }
	[[nodiscard]] T* operator->() { return &getValue(); }
	[[nodiscard]] const T* operator->() const { return &getValue(); }

private:
	// Data, class members.
	Optional<T> value_;
	Status status_;
};

} // namespace oa

// Macros.
#define OA_PP_GLUE_IMPL(a, b) a##b
#define OA_PP_GLUE(a, b) OA_PP_GLUE_IMPL(a, b)
#define OA_PP_LINE_VAR(prefix) OA_PP_GLUE(prefix, __LINE__)

#define OA_RETURN_IF_ERROR(expr) \
	do { auto _status = (expr); if (!_status.isOk()) return _status; } while (0)

#define OA_ASSIGN_OR_RETURN(lhs, expr) \
	auto OA_PP_LINE_VAR(_oa_ar_) = (expr); \
	if (!OA_PP_LINE_VAR(_oa_ar_).isOk()) return OA_PP_LINE_VAR(_oa_ar_).getStatus(); \
	lhs = std::move(OA_PP_LINE_VAR(_oa_ar_)).getValue()

#define OA_CHECK_OK(expr) \
	do { auto _status = (expr); assert(_status.isOk()); } while (0)
