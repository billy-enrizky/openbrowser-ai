importScripts("jev-client.js", "source-snapshot.js");

const SUPPORTED_PROVIDERS = new Set(["jev", "laya"]);
const SUPPORTED_TYPES = new Set([
  "context_atlas.provider_status",
  "context_atlas.request_provider_access",
  "context_atlas.set_provider",
  "context_atlas.key_status",
  "context_atlas.save_key",
  "context_atlas.clear_key",
  "context_atlas.runtime_status",
  "context_atlas.search",
  "context_atlas.publish_source",
  "context_atlas.get_source",
]);
const PROVIDER_KEY = "context_atlas.provider";
const JEV_STORAGE_KEY = "context_atlas.jev_api_key";
const SOURCE_STORAGE_KEY = "context_atlas.current_source";
const OFFSCREEN_TARGET = "context_atlas.offscreen";
const OFFSCREEN_URL = "offscreen.html";
const PROVIDER_ORIGINS = Object.freeze({
  jev: ["https://api.typesafe.ai/*"],
  laya: [
    "https://huggingface.co/*",
    "https://*.cdn.hf.co/*",
  ],
});
let requestSequence = 0;

chrome.action.onClicked.addListener(async (tab) => {
  if (!tab?.id) return;
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["context-atlas-extension.js"],
    });
  } catch (_error) {
    // The content script reports unsupported pages in its own panel when possible.
  }
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || !SUPPORTED_TYPES.has(message.type)) return false;
  const requestId = ++requestSequence;

  // Keep this call in the synchronous user-gesture turn. Awaiting anything
  // before chrome.permissions.request() would make Chrome reject the request.
  if (message.type === "context_atlas.request_provider_access") {
    return requestProviderAccess(message, sendResponse, requestId);
  }

  (async () => {
    try {
      const payload = await handleMessage(message);
      sendResponse({ ok: true, requestId, payload });
    } catch (error) {
      sendResponse({ ok: false, requestId, error: safeError(error), code: error?.code || "request_failed" });
    }
  })();
  return true;
});

function requestProviderAccess(message, sendResponse, requestId) {
  let permissionPromise;
  try {
    permissionPromise = chrome.permissions.request({ origins: providerOrigins(message.provider) });
  } catch (error) {
    sendResponse({ ok: false, requestId, error: safeError(error), code: error?.code || "request_failed" });
    return true;
  }

  (async () => {
    try {
      const granted = await permissionPromise;
      if (!granted) {
        const error = new Error(permissionDeniedMessage(message.provider));
        error.code = "permission_denied";
        throw error;
      }
      sendResponse({ ok: true, requestId, payload: { granted: true, provider: message.provider } });
    } catch (error) {
      sendResponse({ ok: false, requestId, error: safeError(error), code: error?.code || "request_failed" });
    }
  })();
  return true;
}

function providerOrigins(provider) {
  if (!SUPPORTED_PROVIDERS.has(provider)) throw new Error("Unsupported Context Atlas provider.");
  return PROVIDER_ORIGINS[provider];
}

function permissionDeniedMessage(provider) {
  return provider === "jev"
    ? "Cloud access is needed for Cloud search. Allow access and try again."
    : "Local model access is needed for Local search. Allow access and try again.";
}

async function handleMessage(message) {
  switch (message.type) {
    case "context_atlas.provider_status":
      return providerStatus();
    case "context_atlas.set_provider":
      return setProvider(message.provider);
    case "context_atlas.key_status":
      return keyStatus();
    case "context_atlas.save_key":
      return saveKey(message.api_key);
    case "context_atlas.clear_key":
      return clearKey();
    case "context_atlas.runtime_status":
      return runtimeStatus();
    case "context_atlas.search":
      return search(message.query, message.blocks);
    case "context_atlas.publish_source":
      return publishSource(message.source);
    case "context_atlas.get_source":
      return getSource();
    default:
      throw new Error("Unsupported Context Atlas request");
  }
}

async function publishSource(source) {
  const snapshot = ContextAtlasSourceSnapshot.serializeSourceSnapshot(source);
  await sourceStorage().set({ [SOURCE_STORAGE_KEY]: snapshot });
  return { saved: true, revision: snapshot.revision || null };
}

async function getSource() {
  const stored = await sourceStorage().get({ [SOURCE_STORAGE_KEY]: null });
  if (!stored[SOURCE_STORAGE_KEY]) {
    const error = new Error("The current page source is not ready. Open Context Atlas on a webpage first.");
    error.code = "not_ready";
    error.status = 404;
    throw error;
  }
  return ContextAtlasSourceSnapshot.validateSourceSnapshot(stored[SOURCE_STORAGE_KEY]);
}

function sourceStorage() {
  return chrome.storage.session || chrome.storage.local;
}

async function providerStatus() {
  const settings = await chrome.storage.local.get({ [PROVIDER_KEY]: "laya" });
  return { provider: normalizeProvider(settings[PROVIDER_KEY]) };
}

async function setProvider(provider) {
  if (!SUPPORTED_PROVIDERS.has(provider)) throw new Error("Unsupported Context Atlas provider.");
  await chrome.storage.local.set({ [PROVIDER_KEY]: provider });
  return { provider };
}

async function keyStatus() {
  const settings = await chrome.storage.local.get({ [JEV_STORAGE_KEY]: "" });
  return { configured: typeof settings[JEV_STORAGE_KEY] === "string" && Boolean(settings[JEV_STORAGE_KEY].trim()) };
}

async function saveKey(apiKey) {
  if (typeof apiKey !== "string" || !apiKey.trim()) throw new Error("Enter a Jev key before saving.");
  await chrome.storage.local.set({ [JEV_STORAGE_KEY]: apiKey.trim() });
  return { configured: true };
}

async function clearKey() {
  await chrome.storage.local.remove(JEV_STORAGE_KEY);
  return { configured: false };
}

async function runtimeStatus() {
  const { provider } = await providerStatus();
  if (provider === "jev") return { provider, ...(await keyStatus()) };
  return sendOffscreenMessage({ type: "context_atlas.offscreen.runtime_status" });
}

async function search(query, blocks) {
  const { provider } = await providerStatus();
  if (provider === "jev") {
    const settings = await chrome.storage.local.get({ [JEV_STORAGE_KEY]: "" });
    return ContextAtlasJev.search({ apiKey: settings[JEV_STORAGE_KEY], query, passages: blocks });
  }
  return sendOffscreenMessage({
    type: "context_atlas.offscreen.search",
    query,
    blocks,
  });
}

async function sendOffscreenMessage(message) {
  await ensureOffscreenDocument();
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({ target: OFFSCREEN_TARGET, ...message }, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error("The local Laya runtime is unavailable."));
        return;
      }
      if (!response?.ok) {
        reject(new Error(response?.error || "The local Laya runtime failed."));
        return;
      }
      resolve(response.payload || {});
    });
  });
}

async function ensureOffscreenDocument() {
  if (await hasOffscreenDocument()) return;
  try {
    await chrome.offscreen.createDocument({
      url: OFFSCREEN_URL,
      reasons: ["WORKERS"],
      justification: "Run the bundled ONNX Runtime Web model for local Context Atlas search.",
    });
  } catch (error) {
    if (!(await hasOffscreenDocument())) throw error;
  }
}

async function hasOffscreenDocument() {
  if (typeof chrome.runtime.getContexts === "function") {
    const contexts = await chrome.runtime.getContexts({
      contextTypes: ["OFFSCREEN_DOCUMENT"],
      documentUrls: [chrome.runtime.getURL(OFFSCREEN_URL)],
    });
    return contexts.length > 0;
  }
  if (typeof chrome.offscreen?.hasDocument === "function") return chrome.offscreen.hasDocument();
  return false;
}

function normalizeProvider(value) {
  return SUPPORTED_PROVIDERS.has(value) ? value : "laya";
}

function safeError(error) {
  if (error instanceof Error && error.message === "Enter a Jev key before saving.") return "Enter a Cloud access key before saving.";
  if (error?.code === "authentication" || error?.status === 401 || error?.status === 403) {
    return "The saved Cloud access key was rejected. Check it and try again.";
  }
  if (error?.code === "rate_limit" || error?.status === 429) return "Cloud search is busy. Try again in a moment.";
  if (error?.code === "provider_limit" || error?.status === 413) return "Cloud search rejected the source size. Retry with fewer visible passages.";
  if (error?.code === "not_configured" || error?.status === 503 || error?.message === "Jev is not configured. Save a key first.") return "Cloud is not configured. Save an access key first.";
  if (error?.code === "invalid_input" || error?.status === 400) return "The page source or question is invalid. Check it and try again.";
  if (error?.code === "invalid_source") return "The current page source could not be prepared. Refresh the page and try again.";
  if (error?.code === "not_ready" || error?.status === 404) return "The current page is not ready. Open Context Atlas on a webpage first.";
  if (error?.code === "network") return "Cloud search is unavailable. Retry the request.";
  if (error?.code === "permission_denied") return error.message;
  if (error?.message === "Unsupported Context Atlas provider.") return error.message;
  if (error?.message === "The local Laya runtime is unavailable.") return error.message;
  if (error?.message === "The local Laya runtime failed.") return error.message;
  return "Context Atlas could not complete the request. Retry the search.";
}
