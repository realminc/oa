#pragma once

#include <Oa/Core/Types.h>

enum class OaActivation : OaU8 {
	None,
	Relu,
	Gelu,
};

enum class OaUpsampleMode : OaU8 {
	Nearest,
	Bilinear,
};

/// Execution policy for scaled dot-product attention. Auto selects the fused
/// causal implementation when its exact contract is supported and otherwise
/// preserves the compositional reference path.
enum class OaAttentionBackend : OaU8 {
	Auto,
	Standard,
	Flash,
};

/// Token-visibility contract for self-attention.
enum class OaAttentionMode : OaU8 {
	Causal,
	Bidirectional,
};
