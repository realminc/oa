#pragma once

#include <oa/core/fnMatrix.h>

namespace oa { class ExecutionSession; }

// Internal batching and autograd plumbing. These are lowering helpers used by
// the optimizer, gradient tape, and graph conformance tests; they are not part
// of the public stateless oa::FnMatrix operation vocabulary.
namespace oa {

namespace FnMatrix {

[[nodiscard]] oa::Status completeRecordedWork(oa::ExecutionSession& inContext);
void multiFill(oa::Span<oa::Matrix> inTensors, oa::F32 inValue);
void multiAdd(oa::Span<oa::Matrix> inDst, oa::Span<const oa::Matrix> inSrc);
void flushDeferredAccum();

} // namespace FnMatrix

} // namespace oa
