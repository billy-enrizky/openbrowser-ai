import { createLayaBrowserRuntime } from "./laya-browser.js";

const CHANNEL = "context-atlas-laya-sandbox-v1";
let runtimePromise;
let parentOrigin = "*";

function sendResponse(requestId, payload) {
  window.parent.postMessage({ channel: CHANNEL, requestId, ok: true, payload }, parentOrigin);
}

function runtimeInfo(runtime) {
  return { model: runtime.model, modelId: runtime.modelId, provider: runtime.provider, max_len: runtime.max_len, head_max_len: runtime.head_max_len };
}

function errorMessage(error) {
  if (error instanceof Error && error.message) return error.message;
  return "The local Laya runtime could not start.";
}

window.addEventListener("message", (event) => {
  if (event.source !== window.parent || event.data?.channel !== CHANNEL) return;
  if (parentOrigin !== "*" && event.origin !== parentOrigin) return;
  if (parentOrigin === "*") parentOrigin = event.origin;
  const { requestId, type } = event.data;
  (async () => {
    if (type === "initialize") {
      const sourceAssets = event.data.assets || {};
      runtimePromise = createLayaBrowserRuntime({ assets: { ...sourceAssets, modelBytes: new Uint8Array(sourceAssets.modelBytes) }, wasmPath: "./ort/" });
      sendResponse(requestId, runtimeInfo(await runtimePromise));
      return;
    }
    if (type === "search") {
      if (!runtimePromise) throw new Error("The local Laya runtime has not been initialized.");
      const runtime = await runtimePromise;
      const payload = await runtime.search.search(event.data.query, event.data.blocks);
      sendResponse(requestId, { ...payload, provider: runtime.provider });
    }
  })().catch((error) => {
    window.parent.postMessage({ channel: CHANNEL, requestId, ok: false, error: errorMessage(error) }, parentOrigin);
  });
});

window.parent.postMessage({ channel: CHANNEL, type: "ready" }, "*");
