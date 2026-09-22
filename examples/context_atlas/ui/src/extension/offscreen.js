import { loadLayaBrowserAssets } from "./laya-assets.js";

const CHANNEL = "context-atlas-laya-sandbox-v1";
const SANDBOX_READY_TIMEOUT_MS = 30000;
const SANDBOX_REQUEST_TIMEOUT_MS = 120000;
const INITIALIZATION_TIMEOUT_MS = 120000;
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
  for (const { reject, timeout } of pendingRequests.values()) {
    clearTimeout(timeout);
    reject(error);
  }
  pendingRequests.clear();
}

function withTimeout(promise, timeoutMs, message, onTimeout) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      const error = new Error(message);
      try {
        onTimeout?.(error);
      } catch (_error) {
      }
      reject(error);
    }, timeoutMs);

    promise.then(
      (value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        resolve(value);
      },
      (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        reject(error);
      },
    );
  });
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
  clearTimeout(pending.timeout);
  pendingRequests.delete(message.requestId);
  if (message.ok) pending.resolve(message.payload || {});
  else pending.reject(new Error(message.error || "The local Laya sandbox rejected the request."));
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
    const pending = { resolve, reject, timeout: null };
    pendingRequests.set(requestId, pending);
    pending.timeout = setTimeout(() => {
      const pending = pendingRequests.get(requestId);
      if (!pending) return;
      pendingRequests.delete(requestId);
      pending.reject(new Error("The local Laya sandbox request timed out."));
    }, SANDBOX_REQUEST_TIMEOUT_MS);
    try {
      frame.contentWindow.postMessage({ channel: CHANNEL, requestId, type, ...payload }, "*", transfer);
    } catch (error) {
      clearTimeout(pending.timeout);
      pendingRequests.delete(requestId);
      reject(error);
    }
  });
}

async function initializeRuntime() {
  if (!runtimePromise) {
    const controller = typeof AbortController === "function" ? new AbortController() : null;
    const initialization = (async () => {
      const assets = await loadLayaBrowserAssets(controller ? { signal: controller.signal } : {});
      const modelBuffer = assets.modelBytes.buffer;
      return requestSandbox("initialize", {
        assets: { modelBytes: modelBuffer, tokenizerJson: assets.tokenizerJson, tokenizerConfig: assets.tokenizerConfig, rlConfig: assets.rlConfig },
      }, [modelBuffer]);
    })();
    runtimePromise = withTimeout(
      initialization,
      INITIALIZATION_TIMEOUT_MS,
      "The local Laya runtime initialization timed out.",
      (error) => {
        controller?.abort();
        rejectPending(error);
      },
    ).catch((error) => {
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
