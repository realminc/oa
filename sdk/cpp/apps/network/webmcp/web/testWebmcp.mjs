import assert from "node:assert/strict";
import test from "node:test";

const token = "a".repeat(64);
const toolNames = [
  "vulkan_status",
  "training_status",
  "training_metrics",
  "training_results",
  "training_start",
  "training_pause",
  "training_resume",
  "training_checkpoint",
  "training_evaluate",
  "training_set_parameter",
  "training_restart",
  "training_generate",
];

class FakeElement {
  constructor() {
    this.className = "";
    this.disabled = false;
    this.listeners = new Map();
    this.style = {};
    this.textContent = "";
    this.value = "0.0100";
  }

  addEventListener(name, listener) {
    this.listeners.set(name, listener);
  }
}

function toolResult(name) {
  if (name === "vulkan_status") {
    return {
      content: [{type: "text", text: "ready"}],
      structuredContent: {
        deviceName: "Test Vulkan GPU",
        deviceVendor: "OA",
        deviceType: "integrated",
        driverName: "test-driver",
        driverVersion: "1.0",
        vulkanApiVersion: "1.4.0",
        hasCompute: true,
        hasGraphics: true,
        deviceVramBytes: 1024,
        memoryTotalBytes: 1024,
        memoryUsedBytes: 256,
        memoryFreeBytes: 768,
      },
    };
  }
  if (name === "training_status") {
    return {
      content: [{type: "text", text: "paused"}],
      structuredContent: {
        runId: 1,
        revision: 1,
        state: "paused",
        step: 0,
        epoch: 0,
        totalSteps: 300,
        contextLength: 16,
        modelWidth: 32,
        hiddenWidth: 64,
        batchSize: 64,
        parameterCount: 27648,
        restartPending: false,
        learningRate: 0.01,
        loss: null,
        gpuMs: null,
        wallMs: null,
      },
    };
  }
  if (name === "training_generate") {
    return {
      content: [{type: "text", text: "qualified"}],
      structuredContent: {
        checkpointRoundTrip: true,
        generationQualityPassed: true,
        accuracy: 0.95,
        optimizerStep: 300,
        parameterHash: "0123456789abcdef",
        prompt: "to be",
        generated: "to be or not to be",
      },
    };
  }
  if (name === "training_restart") {
    return {
      content: [{type: "text", text: "restart accepted"}],
      structuredContent: {requestId: 3},
    };
  }
  return {
    content: [{type: "text", text: "accepted"}],
    structuredContent: {sequence: 7},
  };
}

function installBrowser({withWebMcp}) {
  const elements = new Map();
  const requests = [];
  const registered = [];
  let replacedUrl = "";
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new FakeElement());
      return elements.get(id);
    },
  };
  if (withWebMcp) {
    document.modelContext = {
      async registerTool(tool) {
        registered.push(tool);
      },
    };
  }
  globalThis.document = document;
  globalThis.window = {
    location: {hash: `#token=${token}`, pathname: "/", search: ""},
    setInterval() { return 1; },
  };
  globalThis.history = {
    replaceState(_state, _unused, url) { replacedUrl = url; },
  };
  globalThis.fetch = async (url, options) => {
    if (options.signal?.aborted) {
      throw options.signal.reason;
    }
    const message = JSON.parse(options.body);
    requests.push({url, options, message});
    let result;
    if (message.method === "tools/list") {
      result = {
        tools: [
          ...toolNames.map((name) => ({
            name,
            title: name,
            description: `test ${name}`,
            inputSchema: {type: "object"},
            annotations: {readOnlyHint: name.endsWith("status")},
          })),
          {
            name: "not_allowlisted",
            inputSchema: {type: "object"},
          },
        ],
      };
    } else {
      result = toolResult(message.params.name);
    }
    return {
      ok: true,
      status: 200,
      async json() { return {jsonrpc: "2.0", id: message.id, result}; },
    };
  };
  return {
    elements,
    registered,
    requests,
    replacedUrl: () => replacedUrl,
  };
}

async function waitFor(predicate) {
  for (let attempt = 0; attempt < 100; ++attempt) {
    if (predicate()) return;
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.fail("browser adapter did not settle");
}

test("registers only the fixed WebMCP tools and invokes the native registry", async () => {
  const browser = installBrowser({withWebMcp: true});
  await import(`./generated/webmcp.js?case=webmcp-${Date.now()}`);
  await waitFor(() => browser.registered.length === toolNames.length);

  assert.equal(browser.replacedUrl(), "/");
  assert.deepEqual(browser.registered.map((tool) => tool.name), toolNames);
  assert.equal(
    browser.elements.get("webmcp-state").textContent,
    `webmcp · ${toolNames.length} tools registered`,
  );
  assert.equal(browser.elements.get("native-state").textContent, "native · connected");
  assert.equal(browser.elements.get("device-name").textContent, "Test Vulkan GPU · integrated");
  assert.deepEqual(
    browser.registered.find((tool) => tool.name === "training_status").annotations,
    {readOnlyHint: true, untrustedContentHint: false},
  );

  for (const request of browser.requests) {
    assert.equal(request.url, "/api/mcp");
    assert.equal(request.options.headers.Authorization, `Bearer ${token}`);
    assert.equal(
      request.message.params._meta["io.modelcontextprotocol/protocolVersion"],
      "2026-07-28",
    );
  }

  const pause = browser.registered.find((tool) => tool.name === "training_pause");
  assert.ok(pause);
  assert.deepEqual(pause.annotations, {
    readOnlyHint: false,
    untrustedContentHint: false,
  });
  assert.equal("destructiveHint" in pause.annotations, false);
  assert.equal("idempotentHint" in pause.annotations, false);
  assert.equal("openWorldHint" in pause.annotations, false);
  const controller = new AbortController();
  assert.deepEqual(
    await pause.execute({expectedRevision: 1}, {signal: controller.signal}),
    {sequence: 7},
  );
  const pauseRequest = browser.requests.at(-1);
  assert.equal(pauseRequest.options.signal, controller.signal);
  assert.equal(pauseRequest.message.method, "tools/call");
  assert.equal(pauseRequest.message.params.name, "training_pause");
  assert.deepEqual(pauseRequest.message.params.arguments, {expectedRevision: 1});
  assert.equal(browser.elements.get("activity-source").textContent, "agent · training_pause");
  assert.equal(browser.elements.get("activity-source").className, "badge ready");
  assert.deepEqual(JSON.parse(browser.elements.get("result").textContent), {
    source: "webmcp",
    tool: "training_pause",
    result: {sequence: 7},
  });

  assert.deepEqual(
    await pause.execute({expectedRevision: 1}),
    {sequence: 7},
  );
  const legacyPauseRequest = browser.requests.at(-1);
  assert.equal(legacyPauseRequest.options.signal, undefined);
  assert.equal(legacyPauseRequest.message.params.name, "training_pause");
  assert.deepEqual(
    legacyPauseRequest.message.params.arguments,
    {expectedRevision: 1},
  );

  const aborted = new AbortController();
  aborted.abort(new Error("test cancellation"));
  const requestCountBeforeAbort = browser.requests.length;
  await assert.rejects(
    pause.execute({expectedRevision: 1}, {signal: aborted.signal}),
    /test cancellation/,
  );
  assert.equal(browser.requests.length, requestCountBeforeAbort);
  assert.equal(
    browser.elements.get("activity-source").textContent,
    "agent · training_pause failed",
  );
  assert.equal(browser.elements.get("activity-source").className, "badge error");
  assert.equal(browser.elements.get("result").textContent, "test cancellation");

  const generation = browser.registered.find((tool) => tool.name === "training_generate");
  assert.ok(generation);
  assert.deepEqual(
    await generation.execute({}, {signal: new AbortController().signal}),
    toolResult("training_generate").structuredContent,
  );
  assert.equal(browser.elements.get("activity-source").textContent, "agent · training_generate");
  assert.equal(browser.elements.get("checkpoint-roundtrip").textContent, "verified");
  assert.equal(browser.elements.get("evaluation-accuracy").textContent, "95.00%");
  assert.equal(browser.elements.get("generation-output").textContent, "to be or not to be");

  const requestCount = browser.requests.length;
  browser.elements.get("refresh").listeners.get("click")();
  await waitFor(() => browser.requests.length >= requestCount + 2);
  assert.equal(browser.elements.get("native-state").textContent, "native · connected");

  browser.elements.get("config-total-steps").value = "12";
  browser.elements.get("config-context-length").value = "8";
  browser.elements.get("config-model-width").value = "16";
  browser.elements.get("config-hidden-width").value = "32";
  browser.elements.get("config-batch-size").value = "8";
  browser.elements.get("config-learning-rate").value = "0.02";
  const beforeRestart = browser.requests.length;
  browser.elements.get("restart-training").listeners.get("click")();
  await waitFor(() => browser.requests.length >= beforeRestart + 2);
  const restartRequest = browser.requests.findLast(
    (request) => request.message.params?.name === "training_restart",
  );
  assert.ok(restartRequest);
  assert.deepEqual(restartRequest.message.params.arguments, {
    totalSteps: 12,
    contextLength: 8,
    modelWidth: 16,
    hiddenWidth: 32,
    batchSize: 8,
    learningRate: 0.02,
  });
});

test("keeps the ordinary page usable when WebMCP is unavailable", async () => {
  const browser = installBrowser({withWebMcp: false});
  await import(`./generated/webmcp.js?case=fallback-${Date.now()}`);
  await waitFor(() => browser.elements.get("webmcp-state")?.textContent.includes("unavailable"));

  assert.equal(browser.registered.length, 0);
  assert.equal(browser.elements.get("native-state").textContent, "native · connected");
  assert.equal(browser.elements.get("webmcp-state").textContent, "webmcp · browser unavailable");
  assert.equal(browser.elements.get("training-state").textContent, "paused");
});
