// OaTrainingProgram — fixed-shape training-step capture and replay.
//
// The program owns an independently compiled OaComputeGraph. Capture consumes
// the dispatches currently recorded in an OaContext only after compilation
// succeeds; subsequent replay bypasses forward/autograd/optimizer graph
// reconstruction entirely. Dynamic eager training remains the fallback.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Runtime/ComputeGraph.h>

class OaContext;
class OaComputeEngine;

class OaTrainingProgramOptions {
public:
	// Emit the final compute-to-host visibility edge. Keep enabled when loss or
	// metrics are read on the CPU after every replay; disable for GPU-only chunks.
	OaBool HostReadbackRequired = true;

	// Fail capture when a known host-mutated or nondeterministically frozen op is
	// present. This prevents a graph from silently reusing one dropout mask or a
	// push-constant optimizer step forever.
	OaBool ValidateReplaySafety = true;

	// Bracket the complete captured GPU program with timestamp queries. Training
	// waits after each step, satisfying the graph's single-flight timing contract.
	OaBool EnableGpuTiming = false;
};

class OaTrainingProgram {
public:
	OaTrainingProgram() = default;
	~OaTrainingProgram();

	OaTrainingProgram(const OaTrainingProgram&) = delete;
	OaTrainingProgram& operator=(const OaTrainingProgram&) = delete;
	OaTrainingProgram(OaTrainingProgram&&) = delete;
	OaTrainingProgram& operator=(OaTrainingProgram&&) = delete;

	// Compile the context's currently recorded fixed-shape step. On success the
	// source graph is cleared without execution and this program becomes its
	// sole executable owner. On failure the source graph is left intact so the
	// caller may execute it eagerly or diagnose the rejected node. The context's
	// compute runtime and Vulkan device must outlive this program.
	[[nodiscard]] OaStatus Capture(
		OaContext& InContext,
		const OaTrainingProgramOptions& InOptions = {});

	// Non-blocking submit. Same-queue replays are ordered by Vulkan; call Wait()
	// before mapped host reads or resource mutation from the CPU.
	[[nodiscard]] OaStatus Replay();
	[[nodiscard]] OaResult<OaCompletionToken> ReplayAsync();
	[[nodiscard]] OaStatus Wait();

	// Wait for pending work, release the compiled plan and all retained buffers.
	[[nodiscard]] OaStatus Reset();

	[[nodiscard]] OaBool IsCaptured() const noexcept { return Captured_; }
	[[nodiscard]] OaU32 NodeCount() const noexcept { return Graph_.NodeCount(); }
	[[nodiscard]] OaGraphStats Stats() const { return Graph_.GetStats(); }
	[[nodiscard]] OaF64 LastGpuMs() const noexcept { return Graph_.LastReplayGpuMs(); }
	[[nodiscard]] OaString DebugReportJson(OaStringView InName = "TrainingStep") const {
		return Graph_.DebugReportJson(InName);
	}

private:
	[[nodiscard]] static OaStatus Validate_(const OaComputeGraph& InGraph);
	[[nodiscard]] OaStatus PrepareReplayRng_(OaComputeEngine& InRuntime);

	OaComputeGraph Graph_;
	OaVec<OaMatrix> RngStates_;
	OaComputeEngine* Runtime_ = nullptr;
	OaBool Captured_ = false;
};
