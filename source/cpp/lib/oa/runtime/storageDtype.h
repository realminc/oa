#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

// Lower one semantic storage dtype to the specialization-constant ABI used by
// storage.slang. This is an internal lowering concern: public raw-dispatch
// callers name the storage type, while executable nodes and pipeline keys keep
// the compact 0=32-bit / 1=16-bit representation.
namespace oavk {

[[nodiscard]] inline oa::Result<oa::U32> resolveStorageDtypeSpecConstant(
	oa::ScalarType inStorageDtype)
{
	switch (inStorageDtype) {
		case oa::ScalarType::Float32:
			return 0U;
		case oa::ScalarType::BFloat16:
			return 1U;
		case oa::ScalarType::Float16:
			return oa::Status::error(oa::StatusCode::DtypeMismatch,
				"Float16 is not the BFloat16 dispatch ABI");
		case oa::ScalarType::Float64:
			return oa::Status::error(oa::StatusCode::Unimplemented,
				"Float64 requires an admitted FP64 kernel route");
		default:
			return oa::Status::error(
				oa::StatusCode::InvalidArgument,
				"generic dispatch storage dtype must be Float32 or BFloat16");
	}
}

} // namespace oavk
