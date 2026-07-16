#pragma once

#include "ContextTypes.h"

class OaContext;

// Per-domain default-context accessors. These are the internal hooks used by
// OaFnMatrix, OaFnLoss, etc. to reach the thread-local default context.
namespace OaFnMatrix { [[nodiscard]] OaContext& GetContext(); }
namespace OaFnLoss   { [[nodiscard]] OaContext& GetContext(); }
namespace OaFnAudio  { [[nodiscard]] OaContext& GetContext(); }
namespace OaFnUi     { [[nodiscard]] OaContext& GetContext(); }
namespace OaFnCrypto { [[nodiscard]] OaContext& GetContext(); }
