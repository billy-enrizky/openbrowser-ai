const CHANNEL = "context_atlas";
const ALLOWED_ORIGINS = new Set([
  "http://127.0.0.1:8765",
  "http://localhost:8765",
]);

export function createWindowBridgeAdapter({ windowObject = globalThis.window, timeoutMs = 15000 } = {}) {
  if (!windowObject || typeof windowObject.addEventListener !== "function" || typeof windowObject.postMessage !== "function") {
    throw new Error("A browser window is required for the Context Atlas bridge.");
  }
  const origin = windowObject.location?.origin;
  const unsupportedOrigin = !ALLOWED_ORIGINS.has(origin);
  let sequence = 0;

  function request(type, fields = {}) {
    if (unsupportedOrigin) return Promise.reject(new Error("Context Atlas must run on the supported localhost address."));
    const requestId = `context-atlas-${Date.now().toString(36)}-${(++sequence).toString(36)}`;
    return new Promise((resolve, reject) => {
      let settled = false;
      const finishOnTimeout = () => finish(new Error("The Context Atlas extension did not respond."));
      const timer = typeof windowObject.setTimeout === "function"
        ? windowObject.setTimeout(finishOnTimeout, timeoutMs)
        : setTimeout(finishOnTimeout, timeoutMs);
      const onMessage = (event) => {
        const response = event?.data;
        if (event?.source !== windowObject || event.origin !== origin || response?.channel !== CHANNEL || response?.requestId !== requestId) return;
        if (response.type && response.type !== type) return;
        if (response.ok) finish(null, response.payload || {});
        else {
          const error = new Error(response.error || "The Context Atlas request failed.");
          error.code = response.code;
          finish(error);
        }
      };
      function finish(error, payload) {
        if (settled) return;
        settled = true;
        if (typeof windowObject.clearTimeout === "function") windowObject.clearTimeout(timer);
        else clearTimeout(timer);
        windowObject.removeEventListener("message", onMessage);
        if (error) reject(error);
        else resolve(payload);
      }
      windowObject.addEventListener("message", onMessage);
      windowObject.postMessage({ channel: CHANNEL, type, requestId, ...fields }, origin);
    });
  }

  return {
    ensureProviderAccess: (provider) => request("context_atlas.request_provider_access", { provider }),
    getProvider: async () => (await request("context_atlas.provider_status")).provider,
    setProvider: async (provider) => {
      const payload = await request("context_atlas.set_provider", { provider });
      return payload.provider;
    },
    getKeyStatus: () => request("context_atlas.key_status"),
    saveKey: (apiKey) => request("context_atlas.save_key", { api_key: apiKey }),
    clearKey: () => request("context_atlas.clear_key"),
    getRuntimeStatus: () => request("context_atlas.runtime_status"),
    search: (query, blocks) => request("context_atlas.search", { query, blocks }),
    publishSource: (source) => request("context_atlas.publish_source", { source }),
    getSource: () => request("context_atlas.get_source"),
    refreshSource: () => request("context_atlas.get_source"),
  };
}
