// Translation unit consumed by the developer-documentation generator.
//
// It deliberately includes every installed module umbrella so clang-doc sees
// the same C++ declarations as an external OA consumer. It is not linked into
// the library and must not acquire implementation-only include dependencies.

#include <oa/audio.h>
#include <oa/core.h>
#include <oa/crypto.h>
#include <oa/data.h>
#include <oa/mcp.h>
#include <oa/vision.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/network.h>
#include <oa/render.h>
#include <oa/runtime.h>
#include <oa/ui.h>
#include <oa/vision.h>
