function isObject(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
function isToolCallResult(value) {
    if (!isObject(value))
        return false;
    if (value.isError !== undefined && typeof value.isError !== "boolean")
        return false;
    if (value.structuredContent !== undefined && !isObject(value.structuredContent))
        return false;
    if (value.content !== undefined && (!Array.isArray(value.content) ||
        !value.content.every((item) => isObject(item) && typeof item.type === "string" &&
            (item.text === undefined || typeof item.text === "string"))))
        return false;
    return true;
}
function isVulkanStatus(value) {
    if (!isObject(value))
        return false;
    const strings = ["deviceName", "deviceVendor", "deviceType", "driverName", "driverVersion", "vulkanApiVersion"];
    const numbers = ["deviceVramBytes", "memoryTotalBytes", "memoryUsedBytes", "memoryFreeBytes"];
    return strings.every((name) => typeof value[name] === "string") &&
        numbers.every((name) => typeof value[name] === "number" && Number.isFinite(value[name]) && value[name] >= 0) &&
        typeof value.hasCompute === "boolean" && typeof value.hasGraphics === "boolean";
}
function isTrainingStatus(value) {
    if (!isObject(value))
        return false;
    const integers = [
        "runId", "revision", "step", "epoch", "totalSteps", "contextLength",
        "modelWidth", "hiddenWidth", "batchSize", "parameterCount",
    ];
    const nullableNumbers = ["learningRate", "loss", "gpuMs", "wallMs"];
    return integers.every((name) => typeof value[name] === "number" &&
        Number.isSafeInteger(value[name]) && value[name] >= 0) &&
        nullableNumbers.every((name) => value[name] === null ||
            (typeof value[name] === "number" && Number.isFinite(value[name]))) &&
        typeof value.state === "string" && typeof value.restartPending === "boolean" &&
        value.runId >= 1 && value.totalSteps >= 1;
}
function isTrainingGeneration(value) {
    return isObject(value) &&
        typeof value.checkpointRoundTrip === "boolean" &&
        typeof value.generationQualityPassed === "boolean" &&
        typeof value.accuracy === "number" && Number.isFinite(value.accuracy) &&
        value.accuracy >= 0 && value.accuracy <= 1 &&
        typeof value.optimizerStep === "number" && Number.isSafeInteger(value.optimizerStep) &&
        value.optimizerStep >= 0 &&
        typeof value.parameterHash === "string" && /^[0-9a-f]{16}$/.test(value.parameterHash) &&
        typeof value.prompt === "string" && typeof value.generated === "string";
}
function isMcpTool(value) {
    return isObject(value) && typeof value.name === "string" &&
        isObject(value.inputSchema) &&
        (value.title === undefined || typeof value.title === "string") &&
        (value.description === undefined || typeof value.description === "string") &&
        (value.annotations === undefined || isObject(value.annotations));
}
const protocolVersion = "2026-07-28";
const allowedTools = new Set([
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
]);
let requestId = 0;
let trainingStartAccepted = false;
let lastTrainingStatus = null;
let renderedRunId = 0;
let generationRendered = false;
function element(id) {
    const value = document.getElementById(id);
    if (!value)
        throw new Error(`missing UI element: ${id}`);
    return value;
}
function setBadge(id, text, state) {
    const badge = element(id);
    badge.textContent = text;
    badge.className = `badge ${state}`;
}
function takeCredential() {
    const fragment = new URLSearchParams(window.location.hash.slice(1));
    const token = fragment.get("token") ?? "";
    history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
    if (!/^[0-9a-f]{64}$/.test(token)) {
        throw new Error("missing or invalid launch credential; open the exact URL printed by oa-webmcp");
    }
    return token;
}
const credential = takeCredential();
async function mcpRequest(method, fields = {}, signal) {
    const response = await fetch("/api/mcp", {
        method: "POST",
        headers: {
            "Authorization": `Bearer ${credential}`,
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            jsonrpc: "2.0",
            id: ++requestId,
            method,
            params: {
                _meta: {
                    "io.modelcontextprotocol/protocolVersion": protocolVersion,
                    "io.modelcontextprotocol/clientCapabilities": {},
                },
                ...fields,
            },
        }),
        signal,
    });
    if (!response.ok)
        throw new Error(`native gateway returned HTTP ${response.status}`);
    const message = await response.json();
    if (message.error)
        throw new Error(`MCP ${message.error.code}: ${message.error.message}`);
    if (message.result === undefined)
        throw new Error("MCP response omitted result");
    return message.result;
}
async function callTool(name, argumentsValue = {}, signal) {
    if (!allowedTools.has(name))
        throw new Error(`tool is not allowlisted: ${name}`);
    const result = await mcpRequest("tools/call", {
        name,
        arguments: argumentsValue,
    }, signal);
    if (!isToolCallResult(result))
        throw new Error("native MCP tool returned an invalid result shape");
    return result;
}
async function callVulkanStatus(argumentsValue = {}) {
    return callTool("vulkan_status", argumentsValue);
}
function gibibytes(bytes) {
    return `${(bytes / (1024 ** 3)).toFixed(2)} GiB`;
}
function statusFromResult(result) {
    if (result.isError) {
        throw new Error(result.content?.[0]?.text ?? "native MCP tool failed");
    }
    const status = result.structuredContent;
    if (!isVulkanStatus(status))
        throw new Error("vulkan_status returned invalid structuredContent");
    return status;
}
function structuredFromResult(result) {
    if (result.isError) {
        throw new Error(result.content?.[0]?.text ?? "native MCP tool failed");
    }
    if (!isObject(result.structuredContent)) {
        throw new Error("native MCP tool omitted structuredContent");
    }
    return result.structuredContent;
}
function trainingStatusFromResult(result) {
    const status = structuredFromResult(result);
    if (!isTrainingStatus(status))
        throw new Error("training_status returned invalid structuredContent");
    return status;
}
function renderVulkan(result) {
    const status = statusFromResult(result);
    element("device-name").textContent = `${status.deviceName} · ${status.deviceType}`;
    element("device-vendor").textContent = status.deviceVendor;
    element("vulkan-version").textContent = status.vulkanApiVersion;
    element("driver").textContent = `${status.driverName} · ${status.driverVersion}`;
    element("compute").textContent = status.hasCompute ? "ready" : "unavailable";
    element("memory").textContent = `${gibibytes(status.memoryUsedBytes)} / ${gibibytes(status.memoryTotalBytes)}`;
    element("result").textContent = JSON.stringify(result, null, 2);
}
function metric(value, suffix = "") {
    return value === null ? "—" : `${value.toFixed(3)}${suffix}`;
}
function renderTraining(result) {
    const status = trainingStatusFromResult(result);
    if (renderedRunId !== status.runId) {
        renderedRunId = status.runId;
        trainingStartAccepted = false;
        generationRendered = false;
        element("config-total-steps").value = String(status.totalSteps);
        element("config-model-width").value = String(status.modelWidth);
        element("config-hidden-width").value = String(status.hiddenWidth);
        element("config-context-length").value = String(status.contextLength);
        element("config-batch-size").value = String(status.batchSize);
        if (status.learningRate !== null) {
            element("config-learning-rate").value = String(status.learningRate);
            element("learning-rate").value = String(status.learningRate);
        }
    }
    lastTrainingStatus = status;
    element("training-run").textContent = `#${status.runId}`;
    element("training-state").textContent = status.state;
    element("training-step").textContent = `${status.step} / ${status.totalSteps}`;
    element("training-loss").textContent = metric(status.loss);
    element("training-gpu").textContent = metric(status.gpuMs, " ms");
    element("training-wall").textContent = metric(status.wallMs, " ms");
    element("training-lr").textContent = status.learningRate === null
        ? "—"
        : status.learningRate.toExponential(3);
    element("training-parameters").textContent = status.parameterCount.toLocaleString("en-US");
    element("training-shape").textContent =
        `B${status.batchSize} × T${status.contextLength} × D${status.modelWidth}/${status.hiddenWidth}`;
    element("training-progress").style.width = `${Math.min(100, status.totalSteps === 0 ? 0 : (status.step / status.totalSteps) * 100)}%`;
    const start = element("start-training");
    start.disabled = trainingStartAccepted || status.state !== "paused" || status.step !== 0;
    start.textContent = trainingStartAccepted && status.state === "paused"
        ? "Start accepted"
        : status.state === "paused" && status.step === 0
            ? "Start training"
            : status.state === "completed" ? "Training complete" : "Training running";
    element("pause-training").disabled = status.state !== "running";
    element("resume-training").disabled =
        status.state !== "paused" || status.step === 0;
    const active = status.state === "running" || status.state === "paused";
    element("checkpoint-training").disabled = !active;
    element("evaluate-training").disabled = !active;
    element("apply-learning-rate").disabled =
        !active;
    element("read-generation").disabled = status.state !== "completed";
    const restart = element("restart-training");
    restart.disabled = status.restartPending;
    restart.textContent = status.restartPending ? "Restart accepted" : "Create new run";
    return status;
}
function renderGeneration(result) {
    const value = structuredFromResult(result);
    if (!isTrainingGeneration(value)) {
        throw new Error("training_generate returned invalid structuredContent");
    }
    element("checkpoint-roundtrip").textContent = value.checkpointRoundTrip ? "verified" : "failed";
    element("evaluation-accuracy").textContent = `${(value.accuracy * 100).toFixed(2)}%`;
    element("generation-quality").textContent = value.generationQualityPassed ? "passed" : "failed";
    element("generation-output").textContent = value.generated;
    generationRendered = true;
}
function renderAgentActivity(name, result) {
    setBadge("activity-source", `agent · ${name}`, "ready");
    element("result").textContent = JSON.stringify({
        source: "webmcp",
        tool: name,
        result,
    }, null, 2);
}
function renderAgentFailure(name, error) {
    setBadge("activity-source", `agent · ${name} failed`, "error");
    element("result").textContent = error instanceof Error ? error.message : String(error);
}
async function executeWebMcpTool(tool, argumentsValue, options) {
    try {
        const result = await callTool(tool.name, argumentsValue, options?.signal);
        const structured = structuredFromResult(result);
        if (tool.name === "vulkan_status") {
            renderVulkan(result);
        }
        else if (tool.name === "training_status") {
            renderTraining(result);
        }
        else if (tool.name === "training_generate") {
            renderGeneration(result);
        }
        renderAgentActivity(tool.name, structured);
        return structured;
    }
    catch (error) {
        renderAgentFailure(tool.name, error);
        throw error;
    }
}
async function refresh() {
    const button = element("refresh");
    button.disabled = true;
    try {
        const [vulkan, training] = await Promise.all([
            callVulkanStatus(),
            callTool("training_status"),
        ]);
        renderVulkan(vulkan);
        renderTraining(training);
        setBadge("native-state", "native · connected", "ready");
    }
    catch (error) {
        setBadge("native-state", "native · failed", "error");
        element("result").textContent = error instanceof Error ? error.message : String(error);
        throw error;
    }
    finally {
        button.disabled = false;
    }
}
async function refreshTraining() {
    try {
        const status = renderTraining(await callTool("training_status"));
        if (status.state === "completed" && !generationRendered) {
            renderGeneration(await callTool("training_generate"));
        }
    }
    catch (error) {
        setBadge("native-state", "native · failed", "error");
        element("result").textContent = error instanceof Error ? error.message : String(error);
    }
}
async function readGeneration() {
    try {
        const result = await callTool("training_generate");
        renderGeneration(result);
        element("result").textContent = JSON.stringify(result, null, 2);
    }
    catch (error) {
        element("result").textContent = error instanceof Error ? error.message : String(error);
    }
}
async function startTraining() {
    const button = element("start-training");
    button.disabled = true;
    try {
        const result = await callTool("training_start");
        trainingStartAccepted = true;
        element("result").textContent = JSON.stringify(result, null, 2);
        await refreshTraining();
    }
    catch (error) {
        element("result").textContent = error instanceof Error ? error.message : String(error);
        await refreshTraining();
    }
}
async function trainingCommand(name, argumentsValue) {
    try {
        const result = await callTool(name, argumentsValue);
        element("result").textContent = JSON.stringify(result, null, 2);
    }
    catch (error) {
        element("result").textContent = error instanceof Error ? error.message : String(error);
    }
    await refreshTraining();
}
function expectedRevision() {
    if (!lastTrainingStatus)
        throw new Error("training status is unavailable");
    return lastTrainingStatus.revision;
}
async function setLearningRate() {
    const value = Number(element("learning-rate").value);
    if (!Number.isFinite(value) || value <= 0 || value > 1) {
        element("result").textContent = "Learning rate must be finite and in (0, 1].";
        return;
    }
    await trainingCommand("training_set_parameter", {
        name: "learning_rate",
        value,
        expectedRevision: expectedRevision(),
    });
}
function integerInput(id) {
    const value = Number(element(id).value);
    if (!Number.isSafeInteger(value))
        throw new Error(`${id} must be an integer`);
    return value;
}
async function restartTraining() {
    const button = element("restart-training");
    button.disabled = true;
    try {
        const config = {
            totalSteps: integerInput("config-total-steps"),
            contextLength: integerInput("config-context-length"),
            modelWidth: integerInput("config-model-width"),
            hiddenWidth: integerInput("config-hidden-width"),
            batchSize: integerInput("config-batch-size"),
            learningRate: Number(element("config-learning-rate").value),
        };
        if (!Number.isFinite(config.learningRate)) {
            throw new Error("initial learning rate must be finite");
        }
        const result = await callTool("training_restart", config);
        element("result").textContent = JSON.stringify(result, null, 2);
        await refreshTraining();
    }
    catch (error) {
        element("result").textContent = error instanceof Error ? error.message : String(error);
        await refreshTraining();
    }
    finally {
        button.disabled = lastTrainingStatus?.restartPending ?? false;
    }
}
async function registerWebMcp() {
    if (typeof document.modelContext?.registerTool !== "function") {
        setBadge("webmcp-state", "webmcp · browser unavailable", "unavailable");
        return;
    }
    const listed = await mcpRequest("tools/list");
    if (!isObject(listed) || !Array.isArray(listed.tools) || !listed.tools.every(isMcpTool)) {
        throw new Error("native MCP registry returned an invalid tool list");
    }
    const tools = listed.tools.filter((candidate) => allowedTools.has(candidate.name));
    if (tools.length !== allowedTools.size) {
        throw new Error("native MCP registry omitted an allowlisted tool");
    }
    for (const tool of tools) {
        if (typeof tool.description !== "string" || tool.description.length === 0) {
            throw new Error(`native MCP tool omitted its WebMCP description: ${tool.name}`);
        }
        const annotations = {
            readOnlyHint: tool.annotations?.readOnlyHint ?? false,
            untrustedContentHint: false,
        };
        await document.modelContext.registerTool({
            name: tool.name,
            title: tool.title,
            description: tool.description,
            inputSchema: tool.inputSchema,
            annotations,
            execute: async (argumentsValue, options) => executeWebMcpTool(tool, argumentsValue, options),
        });
    }
    setBadge("webmcp-state", `webmcp · ${tools.length} tools registered`, "ready");
}
element("refresh").addEventListener("click", () => { void refresh(); });
element("start-training").addEventListener("click", () => { void startTraining(); });
element("pause-training").addEventListener("click", () => {
    void trainingCommand("training_pause", { expectedRevision: expectedRevision() });
});
element("resume-training").addEventListener("click", () => {
    void trainingCommand("training_resume", { expectedRevision: expectedRevision() });
});
element("checkpoint-training").addEventListener("click", () => {
    void trainingCommand("training_checkpoint", { expectedRevision: expectedRevision() });
});
element("evaluate-training").addEventListener("click", () => {
    void trainingCommand("training_evaluate", { expectedRevision: expectedRevision() });
});
element("apply-learning-rate").addEventListener("click", () => { void setLearningRate(); });
element("restart-training").addEventListener("click", () => { void restartTraining(); });
element("read-generation").addEventListener("click", () => { void readGeneration(); });
void (async () => {
    try {
        await refresh();
        await registerWebMcp();
        window.setInterval(() => { void refreshTraining(); }, 750);
    }
    catch (error) {
        setBadge("webmcp-state", "webmcp · registration failed", "error");
        element("result").textContent = error instanceof Error ? error.message : String(error);
    }
})();
export {};
