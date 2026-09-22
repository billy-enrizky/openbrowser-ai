// src/extension/laya-core.js
var QTYPES = Object.freeze({ choice: 0, score: 1, noul: 2 });
var MAX_SEARCH_WINDOW_BLOCKS = 128;
var DEFAULT_LAYA_BROWSER_CONFIG = Object.freeze({
  model: "mizchi/laya-multilingual-onnx",
  modelBaseUrl: "https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/d9d003d543e63d6d3375c21d44624136bd1e0bad/",
  max_len: 8192,
  head_max_len: 512,
  temperature: [1, 1, 1],
  temperature_by_options: {},
  max_batch_tokens: 8192,
  max_batch_sequences: 16,
  search_window_blocks: MAX_SEARCH_WINDOW_BLOCKS,
  threshold: 0.58,
  ambiguity_margin: 0.05
});

// src/extension/laya-assets.js
var MODEL_CACHE_NAME = "context-atlas-laya-model-d9d003d543e63d6d3375c21d44624136bd1e0bad";
var MODEL_FILES = Object.freeze({
  model: "model.onnx",
  tokenizer: "tokenizer/tokenizer.json",
  tokenizerConfig: "tokenizer/tokenizer_config.json",
  rlConfig: "rl_agent_config.json"
});
function withTrailingSlash(value) {
  return String(value).endsWith("/") ? String(value) : `${value}/`;
}
function browserModelUrls(config = DEFAULT_LAYA_BROWSER_CONFIG) {
  const baseUrl = withTrailingSlash(config.modelBaseUrl);
  return Object.fromEntries(Object.entries(MODEL_FILES).map(([key, file]) => [key, new URL(file, baseUrl).toString()]));
}
async function fetchResponse(url, fetchImpl, cacheStorage) {
  if (cacheStorage && typeof cacheStorage.open === "function") {
    try {
      const cache = await cacheStorage.open(MODEL_CACHE_NAME);
      const cached = await cache.match(url);
      if (cached) return cached;
      const response2 = await fetchResponse(url, fetchImpl, null);
      try {
        await cache.put(url, response2.clone());
      } catch (_error) {
      }
      return response2;
    } catch (_error) {
    }
  }
  const response = await fetchImpl(url, { credentials: "omit" });
  if (!response?.ok) throw new Error(`Could not download the local Laya asset (${response?.status || "network error"}).`);
  return response;
}
function resolveCacheStorage(globalLike = globalThis) {
  try {
    const cacheStorage = globalLike?.caches;
    return cacheStorage && typeof cacheStorage.open === "function" ? cacheStorage : null;
  } catch (_error) {
    return null;
  }
}
async function loadLayaBrowserAssets({ config = DEFAULT_LAYA_BROWSER_CONFIG, fetchImpl = globalThis.fetch, cacheStorage } = {}) {
  if (typeof fetchImpl !== "function") throw new Error("The browser fetch API is required for local Laya assets.");
  const availableCacheStorage = cacheStorage === void 0 ? resolveCacheStorage() : cacheStorage;
  const urls = browserModelUrls(config);
  const [modelResponse, tokenizerResponse, tokenizerConfigResponse, rlConfigResponse] = await Promise.all([
    fetchResponse(urls.model, fetchImpl, availableCacheStorage),
    fetchResponse(urls.tokenizer, fetchImpl, availableCacheStorage),
    fetchResponse(urls.tokenizerConfig, fetchImpl, availableCacheStorage),
    fetchResponse(urls.rlConfig, fetchImpl, availableCacheStorage)
  ]);
  return {
    modelBytes: new Uint8Array(await modelResponse.arrayBuffer()),
    tokenizerJson: await tokenizerResponse.json(),
    tokenizerConfig: await tokenizerConfigResponse.json(),
    rlConfig: await rlConfigResponse.json()
  };
}

// src/extension/offscreen.js
var CHANNEL = "context-atlas-laya-sandbox-v1";
var SANDBOX_READY_TIMEOUT_MS = 3e4;
var runtimePromise;
var sandboxFrame;
var sandboxReadyPromise;
var requestSequence = 0;
var pendingRequests = /* @__PURE__ */ new Map();
function errorMessage(error) {
  if (error instanceof Error && error.message) return error.message;
  return "The local Laya runtime could not start.";
}
function rejectPending(error) {
  for (const { reject } of pendingRequests.values()) reject(error);
  pendingRequests.clear();
}
function acceptSandboxMessage(event) {
  if (!sandboxFrame || event.source !== sandboxFrame.contentWindow) return;
  const message = event.data;
  if (!message || message.channel !== CHANNEL) return;
  if (message.type === "ready") {
    if (sandboxReadyPromise?.resolve) {
      clearTimeout(sandboxReadyPromise.timeout);
      sandboxReadyPromise.resolve(sandboxFrame);
    }
    return;
  }
  const pending = pendingRequests.get(message.requestId);
  if (!pending) return;
  if (message.ok) pending.resolve(message.payload || {});
  else pending.reject(new Error(message.error || "The local Laya sandbox rejected the request."));
  pendingRequests.delete(message.requestId);
}
window.addEventListener("message", acceptSandboxMessage);
function ensureSandbox() {
  if (sandboxReadyPromise) return sandboxReadyPromise.promise;
  let resolveReady;
  let rejectReady;
  const promise = new Promise((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
  });
  const timeout = setTimeout(() => {
    const error = new Error("The local Laya sandbox did not become ready.");
    sandboxReadyPromise?.reject(error);
    rejectPending(error);
    sandboxReadyPromise = null;
    sandboxFrame?.remove();
    sandboxFrame = null;
  }, SANDBOX_READY_TIMEOUT_MS);
  sandboxReadyPromise = { promise, resolve: resolveReady, reject: rejectReady, timeout };
  sandboxFrame = document.createElement("iframe");
  sandboxFrame.hidden = true;
  sandboxFrame.title = "Context Atlas local model sandbox";
  sandboxFrame.src = chrome.runtime.getURL("sandbox.html");
  sandboxFrame.addEventListener("error", () => {
    const error = new Error("The local Laya sandbox could not be loaded.");
    clearTimeout(sandboxReadyPromise?.timeout);
    sandboxReadyPromise?.reject(error);
    rejectPending(error);
    sandboxReadyPromise = null;
    sandboxFrame?.remove();
    sandboxFrame = null;
  });
  document.body.append(sandboxFrame);
  return promise;
}
async function requestSandbox(type, payload = {}, transfer = []) {
  const frame = await ensureSandbox();
  const requestId = `r${++requestSequence}`;
  return new Promise((resolve, reject) => {
    pendingRequests.set(requestId, { resolve, reject });
    frame.contentWindow.postMessage({ channel: CHANNEL, requestId, type, ...payload }, "*", transfer);
  });
}
async function initializeRuntime() {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      const assets = await loadLayaBrowserAssets();
      const modelBuffer = assets.modelBytes.buffer;
      return requestSandbox("initialize", {
        assets: { modelBytes: modelBuffer, tokenizerJson: assets.tokenizerJson, tokenizerConfig: assets.tokenizerConfig, rlConfig: assets.rlConfig }
      }, [modelBuffer]);
    })().catch((error) => {
      runtimePromise = null;
      throw error;
    });
  }
  return runtimePromise;
}
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || !["context_atlas.offscreen.initialize", "context_atlas.offscreen.runtime_status", "context_atlas.offscreen.search"].includes(message.type)) return false;
  (async () => {
    try {
      const runtime = await initializeRuntime();
      if (message.type === "context_atlas.offscreen.initialize" || message.type === "context_atlas.offscreen.runtime_status") {
        sendResponse({ ok: true, payload: runtime });
        return;
      }
      const payload = await requestSandbox("search", { query: message.query, blocks: message.blocks });
      sendResponse({ ok: true, payload });
    } catch (error) {
      sendResponse({ ok: false, error: errorMessage(error) });
    }
  })();
  return true;
});
