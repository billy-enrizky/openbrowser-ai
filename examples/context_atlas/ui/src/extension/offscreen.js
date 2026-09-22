import { loadLayaBrowserAssets } from "./laya-assets.js";

const CHANNEL = "context-atlas-laya-sandbox-v1";
const SANDBOX_READY_TIMEOUT_MS = 30000;
let runtimePromise;
let sandboxFrame;
let sandboxReadyPromise;
let requestSequence = 0;
const pendingRequests = new Map();

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
        assets: { modelBytes: modelBuffer, tokenizerJson: assets.tokenizerJson, tokenizerConfig: assets.tokenizerConfig, rlConfig: assets.rlConfig },
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
