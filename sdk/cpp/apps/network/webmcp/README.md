# oa::Network - WebMCP for native Vulkan applications

**Status:** experimental SDK application; the bridge and host-side control
checkpoints and the complete installed training/checkpoint loop are proven on
Intel Iris Xe. Brave 150 experimental-WebMCP discovery and end-to-end
invocation are proven; ChatGPT built-in-browser and live-deployment
qualification remain open.

This application is the first native-to-page checkpoint for OA's WebMCP
challenge entry. It creates one real headless `oa::Engine` and serves a
TypeScript Training Lab through an application-owned loopback gateway. The
human interface and WebMCP use the same `oa::McpServer` handlers and the same
stable application controller.

Every run begins paused at step zero. `training_start`, pause, resume,
checkpoint, evaluation, and the typed `learning_rate` update use the bounded
`oa::TrainingSession` safe-point queue. `training_restart` accepts exactly six
bounded fields: total steps, context length, model width, hidden width, batch
size, and initial learning rate. It cooperatively stops an active run; only the
engine-owning thread destroys the old model and constructs the new one. Status
includes a monotonically increasing run ID, the effective configuration,
parameter count, and restart state so clients cannot confuse snapshots from
different models.

At natural completion, the engine-owning thread saves the model and optimizer
to the application-owned `OA_VAR_DIR/webmcp/transformer_byte.oam` slot (or the
resolved OA var directory), reloads them into fresh objects, checks exact
parameter and optimizer state, repeats evaluation and bounded greedy
generation, and publishes the result. The canonical 300-step 32/64-width run
retains its numerical quality gate. Custom runs report checkpoint equivalence,
accuracy, and language-quality independently; a short run is not mislabeled as
language-qualified. `training_generate` only reads the immutable result and
never performs GPU work on the gateway thread. Recapture, arbitrary
parameters, arbitrary model graphs, and arbitrary checkpoint paths are not
registered.

## Build and run

```sh
cmake --preset release
cmake --build build/release --target oa-webmcp TestWebMcp
ctest --test-dir build/release -R '^TestWebMcp$' --output-on-failure
./bin/release/sdk/apps/network/oa-webmcp
```

Open the exact URL printed by the application. It contains a per-launch
credential in the URL fragment. The page reads the credential, removes the
fragment from browser history, and sends it only in the `Authorization` header
to the same-origin `/api/mcp` route.

For an external HTTPS reverse proxy or tunnel, keep the application listener
on loopback and configure the one exact browser origin:

```sh
./bin/release/sdk/apps/network/oa-webmcp \
  --external-origin https://webmcp.example.com
```

The origin must be a canonical lowercase ASCII HTTPS origin without a path,
query, fragment, user information, trailing dot, or explicit default port.
The application prints the external credential-bearing launch URL and the
separate loopback upstream address. The proxy must preserve the external
`Host` header and terminate TLS; the application then requires that exact Host,
the exact HTTPS `Origin`, and the per-launch bearer credential. This option
does not bind a public interface, install a tunnel, supply proxy authentication
or rate limits, or by itself qualify an internet-facing deployment.

The ordinary page works in browsers without WebMCP. In a supported ChatGPT
built-in browser, the top-level script feature-detects
`document.modelContext.registerTool`, obtains descriptors from `tools/list`,
and registers only the fixed allowlist: `vulkan_status`, `training_status`,
`training_metrics`, `training_results`, `training_start`, `training_pause`,
`training_resume`, `training_checkpoint`, `training_evaluate`,
`training_set_parameter`, `training_restart`, and `training_generate`. It does not dynamically
publish arbitrary future MCP tools.

The checked-in browser-adapter integration test runs the generated JavaScript
against a deterministic page and MCP fixture. It proves credential removal,
native status rendering, the exact allowlist, projection of the current
WebMCP annotation dictionary, exact callback `AbortSignal` forwarding,
cancellation before gateway acceptance, typed invocation forwarding,
generation-result validation, shared agent-activity rendering, refresh, and
ordinary-browser fallback. It also covers the one-argument callback used by
Brave 150's older experimental implementation; that compatibility path has no
browser-provided cancellation signal to forward. Agent calls update the same
visible device, training, artifact, and activity surfaces used by the human
controls. This is an automated contract test, not evidence that a particular
ChatGPT browser build exposes WebMCP.

The adapter follows the
[2026-08-26 WebMCP draft](https://webmachinelearning.github.io/webmcp/): every
registered callback forwards the exact cancellation signal to `fetch` when the
browser supplies it. It also tolerates the older experimental one-argument
callback without inventing cancellation semantics. Cancellation before HTTP
acceptance prevents the native request. A mutating command that already
received a sequence remains audit-visible and follows the training session's
cooperative safe-point contract even if the browser no longer waits for its
response.

## TypeScript

The native binary embeds checked-in generated JavaScript, so running the
application requires no Node.js installation. When editing `web/webmcp.ts`,
regenerate and type-check the asset with:

```sh
cd sdk/cpp/apps/network/webmcp/web
npm install
npm run build
npm run check
npm test
```

`web/generated/webmcp.js` is generated from `web/webmcp.ts` and must be
committed with the source change. CMake embeds the HTML, CSS, and generated JS
into `oa-webmcp`; the installed application does not read from the source tree.

## Current real-device evidence

A fresh temporary-prefix install completed the frozen 300-step workload on an
Intel Iris Xe TGL GT2 using the Intel open-source Mesa 26.1.7 driver and Vulkan
1.4.354. The installed binary was byte-identical to the verified Release
binary. The reloaded checkpoint preserved optimizer step 300 and parameter
hash `7d1d1207b2b2c380`, reproduced `0.926758` evaluation accuracy, and passed
both checkpoint-round-trip and generation-quality gates. This is one named
device qualification, not a cross-vendor claim.

Brave Browser 150.1.92.134 was launched with its experimental WebMCP and
browser-agent testing features against a fresh temporary-prefix install. Its
top-level `document.modelContext` registered exactly the eleven allowlisted
tools. The browser-agent testing surface invoked `vulkan_status`,
`training_start`, and `training_generate`; the calls reached the native MCP
registry, completed all 300 Vulkan training steps, reloaded the checkpoint,
and visibly updated the shared page. The browser run reproduced parameter hash
`7d1d1207b2b2c380`, `0.926758` accuracy, and the qualified bounded generation.
The installed executable was byte-identical to the verified build-tree binary
with SHA-256 `0a6de3b9530f17a408a35f3438e8d10bb344ab59c8ea95de68c529086624e3df`.
This proves the experimental Chromium path on that named browser build. It is
not evidence for the current ChatGPT built-in browser, which remains a separate
manual gate.

## Current security boundary

- loopback-only bind, ephemeral port by default;
- optional canonical external HTTPS origin for exact reverse-proxy Host/Origin
  validation without changing the loopback bind;
- 256-bit operating-system random launch credential;
- exact Host, Origin, bearer, and JSON media-type checks;
- 16 KiB header and 64 KiB body bounds;
- no chunked transfer, duplicate Content-Length, keep-alive, path parameters,
  directory serving, upload, URL fetch, shell, or shader route;
- immutable embedded assets, `no-store`, `nosniff`, same-origin resource
  policy, no-referrer, and a restrictive content security policy;
- one synchronously handled request per connection;
- narrowly gated start/pause/resume/checkpoint/evaluate, typed learning-rate
  mutations, and bounded fresh-run configuration;
- steps `[1, 2000]`, context multiples of four in `[4, 64]`, model width
  multiples of eight in `[8, 256]`, hidden width multiples of eight in
  `[8, 1024]`, batch `[1, 256]`, learning rate `[0.000001, 1]`, and an
  additional `batch * context <= 8192` resource bound;
- no arbitrary model graph, parameter, path, URL, upload, shader, or process
  operation;
- one application-owned checkpoint slot; WebMCP cannot supply or inspect a
  filesystem path;
- a stable application controller that synchronizes gateway borrows while the
  engine-owning thread alone replaces TrainingSession/model/optimizer state;
- explicit Engine/TrainingSession/MCP/listener/thread shutdown ownership.

This is a local demonstration boundary, not a qualified internet-facing
server. Public challenge deployment still requires an authenticated HTTPS
front end, isolation, quotas, and separate deployment evidence.
