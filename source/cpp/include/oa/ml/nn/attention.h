#pragma once

#include <oa/ml/type.h>

namespace oa {

// Execution policy for scaled dot-product attention. Auto selects only an
// internally proven route and otherwise preserves the compositional reference
// path; Flash explicitly requests the fused causal implementation.
enum class AttentionBackend : oa::U8 {
	Auto,
	Standard,
	Flash,
};

// Token-visibility contract for self-attention.
enum class AttentionMode : oa::U8 {
	Causal,
	Bidirectional,
};

} // namespace oa
