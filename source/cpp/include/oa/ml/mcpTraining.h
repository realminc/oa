#pragma once

#include <oa/mcp.h>
#include <oa/ml/trainingSession.h>

namespace oa {

struct McpTrainingConfig {
  // Calling registerTools is the opt-in boundary. This switch permits safe-point
  // pause/resume/checkpoint/evaluate/parameter/recapture commands.
  oa::Bool enableCommands = true;
  // Stop is separately gated because it changes the training terminal path.
  oa::Bool enableStop = false;
};

/// Registers an explicitly included MCP view over an existing training session.
/// The server borrows `inSession` through its handlers and must be closed or
/// destroyed first. `oa/ml.h` deliberately does not opt applications into the
/// network control surface. Rebuild and preview remain absent until their
/// checkpoint and completion contracts are admitted.
class McpTraining {
public:
  [[nodiscard]] static oa::Status registerTools(
      McpServer &inServer,
      oa::TrainingSession &inSession,
      McpTrainingConfig inConfig = {});
};

} // namespace oa
