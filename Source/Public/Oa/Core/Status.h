// OA CORE - Status & Result Types
//
// Error handling without exceptions. gRPC-compatible status codes.

#pragma once

#include <Oa/Core/Types.h>

// STATUS CODES

enum class OaStatusCode : OaU8 {
	// Standard (gRPC-compatible 0-16)
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
	// GPU Compute (100-119)
	VulkanError        = 100,
	DeviceNotFound     = 101,
	OutOfMemory        = 102,
	PipelineError      = 103,
	ShaderCompileError = 104,
	// Crypto (120-139)
	InvalidSignature = 120,
	InvalidBlock     = 121,
	InvalidTransaction = 122,
	InsufficientFunds = 123,
	InsufficientMargin = 124,
	Slashed          = 125,
	// Trading (140-159)
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
	// Network (180-199)
	ConnectionFailed = 180,
	Timeout          = 181,
	TlsError         = 182,
	DnsError         = 183,
	// File/IO (200-219)
	FileNotFound     = 200,
	FileCorrupt      = 201,
	PermissionError  = 202,
	DiskFull         = 203,
};

[[nodiscard]] constexpr OaStringView OaStatusCodeName(OaStatusCode InCode) noexcept {
	switch (InCode) {
		case OaStatusCode::Ok:                return "OK";
		case OaStatusCode::Cancelled:         return "CANCELLED";
		case OaStatusCode::Unknown:           return "UNKNOWN";
		case OaStatusCode::InvalidArgument:   return "INVALID_ARGUMENT";
		case OaStatusCode::DeadlineExceeded:  return "DEADLINE_EXCEEDED";
		case OaStatusCode::NotFound:          return "NOT_FOUND";
		case OaStatusCode::AlreadyExists:     return "ALREADY_EXISTS";
		case OaStatusCode::PermissionDenied:  return "PERMISSION_DENIED";
		case OaStatusCode::ResourceExhausted: return "RESOURCE_EXHAUSTED";
		case OaStatusCode::FailedPrecondition: return "FAILED_PRECONDITION";
		case OaStatusCode::Aborted:           return "ABORTED";
		case OaStatusCode::OutOfRange:        return "OUT_OF_RANGE";
		case OaStatusCode::Unimplemented:     return "UNIMPLEMENTED";
		case OaStatusCode::Internal:          return "INTERNAL";
		case OaStatusCode::Unavailable:       return "UNAVAILABLE";
		case OaStatusCode::DataLoss:          return "DATA_LOSS";
		case OaStatusCode::Unauthenticated:   return "UNAUTHENTICATED";
		case OaStatusCode::VulkanError:       return "VULKAN_ERROR";
		case OaStatusCode::DeviceNotFound:    return "DEVICE_NOT_FOUND";
		case OaStatusCode::OutOfMemory:       return "OUT_OF_MEMORY";
		case OaStatusCode::PipelineError:     return "PIPELINE_ERROR";
		case OaStatusCode::ShaderCompileError: return "SHADER_COMPILE_ERROR";
		case OaStatusCode::InvalidSignature:  return "INVALID_SIGNATURE";
		case OaStatusCode::InvalidBlock:      return "INVALID_BLOCK";
		case OaStatusCode::InvalidTransaction: return "INVALID_TRANSACTION";
		case OaStatusCode::InsufficientFunds: return "INSUFFICIENT_FUNDS";
		case OaStatusCode::InsufficientMargin: return "INSUFFICIENT_MARGIN";
		case OaStatusCode::Slashed:           return "SLASHED";
		case OaStatusCode::OrderRejected:     return "ORDER_REJECTED";
		case OaStatusCode::PositionNotFound:  return "POSITION_NOT_FOUND";
		case OaStatusCode::MarketClosed:      return "MARKET_CLOSED";
		case OaStatusCode::PriceLimitExceeded: return "PRICE_LIMIT_EXCEEDED";
		case OaStatusCode::QuantityTooSmall:  return "QUANTITY_TOO_SMALL";
		case OaStatusCode::ModelNotLoaded:    return "MODEL_NOT_LOADED";
		case OaStatusCode::ShapeMismatch:     return "SHAPE_MISMATCH";
		case OaStatusCode::DtypeMismatch:     return "DTYPE_MISMATCH";
		case OaStatusCode::GradientExplosion: return "GRADIENT_EXPLOSION";
		case OaStatusCode::CheckpointCorrupt: return "CHECKPOINT_CORRUPT";
		case OaStatusCode::ConnectionFailed:  return "CONNECTION_FAILED";
		case OaStatusCode::Timeout:           return "TIMEOUT";
		case OaStatusCode::TlsError:          return "TLS_ERROR";
		case OaStatusCode::DnsError:          return "DNS_ERROR";
		case OaStatusCode::FileNotFound:      return "FILE_NOT_FOUND";
		case OaStatusCode::FileCorrupt:       return "FILE_CORRUPT";
		case OaStatusCode::PermissionError:   return "PERMISSION_ERROR";
		case OaStatusCode::DiskFull:          return "DISK_FULL";
		default:                              return "UNKNOWN";
	}
}

class OaStatus {
public:
	// Status class.

	// Constructors.
	OaStatus() noexcept 
		: Code_(OaStatusCode::Ok)
	{}
	OaStatus(OaStatusCode InCode, OaString InMessage = "")
		: Code_(InCode)
		, Message_(OaStdMove(InMessage))
	{}

	// Methods.
	[[nodiscard]] static OaStatus Ok() { return OaStatus(); }
	[[nodiscard]] static OaStatus Error(OaString InMessage) {
		return OaStatus(OaStatusCode::Internal, OaStdMove(InMessage));
	}
	[[nodiscard]] static OaStatus Error(OaStatusCode InCode, OaString InMessage = "") {
		return OaStatus(InCode, OaStdMove(InMessage));
	}
	[[nodiscard]] static OaStatus Cancelled(OaString InMessage = "Operation cancelled") {
		return OaStatus(OaStatusCode::Cancelled, OaStdMove(InMessage));
	}
	[[nodiscard]] static OaStatus InvalidArgument(OaString InMessage) {
		return OaStatus(OaStatusCode::InvalidArgument, OaStdMove(InMessage));
	}
	[[nodiscard]] static OaStatus NotFound(OaString InMessage) {
		return OaStatus(OaStatusCode::NotFound, OaStdMove(InMessage));
	}
	[[nodiscard]] static OaStatus Unimplemented(OaString InMessage = "Not implemented") {
		return OaStatus(OaStatusCode::Unimplemented, OaStdMove(InMessage));
	}
	
	[[nodiscard]] bool IsOk() const noexcept { return Code_ == OaStatusCode::Ok; }
	[[nodiscard]] bool IsError() const noexcept { return Code_ != OaStatusCode::Ok; }
	[[nodiscard]] OaStatusCode GetCode() const noexcept { return Code_; }
	[[nodiscard]] const OaString& GetMessage() const noexcept { return Message_; }
	[[nodiscard]] OaStringView GetCodeName() const noexcept { return OaStatusCodeName(Code_); }

	[[nodiscard]] OaString ToString() const {
		OaString Result(GetCodeName());
		if (!Message_.empty()) {
			Result += ": "; Result += Message_; 
		}
		return Result;
	}

	// Operators.
	explicit operator bool() const noexcept { return IsOk(); }

private:
	// Data, class members.
	OaStatusCode Code_;
	OaString Message_;
};

template<typename T>
class OaResult {
public:
	// Result class.

	// Constructors.
	OaResult(T InValue)
		: Value_(std::move(InValue))
		, Status_(OaStatus::Ok())
	{}
	OaResult(OaStatus InStatus)
		: Status_(OaStdMove(InStatus))
	{
		assert(!Status_.IsOk() && "OaResult from status must be an error");
	}
	OaResult(OaResult&& InOther) noexcept
		: Value_(OaStdMove(InOther.Value_))
		, Status_(OaStdMove(InOther.Status_))
	{}

	// Methods.
	[[nodiscard]] bool IsOk() const noexcept { return Status_.IsOk(); }
	[[nodiscard]] bool IsError() const noexcept { return Status_.IsError(); }
	[[nodiscard]] const OaStatus& GetStatus() const noexcept { return Status_; }

	[[nodiscard]] T& GetValue() & { assert(IsOk()); return *Value_; }
	[[nodiscard]] const T& GetValue() const & { assert(IsOk()); return *Value_; }
	[[nodiscard]] T&& GetValue() && { assert(IsOk()); return std::move(*Value_); }

	[[nodiscard]] T ValueOr(T InDefault) const & { return IsOk() ? *Value_ : std::move(InDefault); }
	[[nodiscard]] T ValueOr(T InDefault) && { return IsOk() ? std::move(*Value_) : std::move(InDefault); }

	template<typename F>
	[[nodiscard]] auto Map(F&& InFunc) && -> OaResult<decltype(InFunc(std::declval<T>()))> {
		using U = decltype(InFunc(std::declval<T>()));
		if (IsOk()) return OaResult<U>(InFunc(std::move(*Value_)));
		return OaResult<U>(std::move(Status_));
	}

	template<typename F>
	[[nodiscard]] auto AndThen(F&& InFunc) && -> decltype(InFunc(std::declval<T>())) {
		if (IsOk()) return InFunc(std::move(*Value_));
		return decltype(InFunc(std::declval<T>()))(std::move(Status_));
	}

	// Operators.
	OaResult& operator=(OaResult&& InOther) noexcept {
		if (this != &InOther) {
			Value_ = OaStdMove(InOther.Value_);
			Status_ = OaStdMove(InOther.Status_);
		}
		return *this;
	}
	OaResult(const OaResult&) = delete;
	OaResult& operator=(const OaResult&) = delete;
	explicit operator bool() const noexcept { return IsOk(); }
	[[nodiscard]] T& operator*() & { return GetValue(); }
	[[nodiscard]] const T& operator*() const & { return GetValue(); }
	[[nodiscard]] T&& operator*() && { return std::move(GetValue()); }
	[[nodiscard]] T* operator->() { return &GetValue(); }
	[[nodiscard]] const T* operator->() const { return &GetValue(); }

private:
	// Data, class members.
	OaOpt<T> Value_;
	OaStatus Status_;
};

// Macros.
#define OA_PP_GLUE_IMPL(a, b) a##b
#define OA_PP_GLUE(a, b) OA_PP_GLUE_IMPL(a, b)
#define OA_PP_LINE_VAR(prefix) OA_PP_GLUE(prefix, __LINE__)

#define OA_RETURN_IF_ERROR(expr) \
	do { auto _status = (expr); if (!_status.IsOk()) return _status; } while (0)

#define OA_ASSIGN_OR_RETURN(lhs, expr) \
	auto OA_PP_LINE_VAR(_oa_ar_) = (expr); \
	if (!OA_PP_LINE_VAR(_oa_ar_).IsOk()) return OA_PP_LINE_VAR(_oa_ar_).GetStatus(); \
	lhs = std::move(OA_PP_LINE_VAR(_oa_ar_)).GetValue()

#define OA_CHECK_OK(expr) \
	do { auto _status = (expr); assert(_status.IsOk()); } while (0)
