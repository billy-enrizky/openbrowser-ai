import { isProvider } from "./state.js";

export function createFetchAdapter({ fetchImpl = globalThis.fetch, baseUrl = "" } = {}) {
  if (typeof fetchImpl !== "function") throw new Error("A fetch implementation is required.");
  const request = async (path, options = {}) => {
    let response;
    try {
      response = await fetchImpl(`${baseUrl}${path}`, options);
    } catch (_error) {
      const error = new Error("The local Context Atlas server is unavailable.");
      error.status = 0;
      throw error;
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(payload?.error || "The local Context Atlas server rejected the request.");
      error.status = response.status;
      throw error;
    }
    return payload;
  };
  return {
    ensureProviderAccess: async () => ({ granted: true }),
    getKeyStatus: () => request("/api/key/status"),
    saveKey: (apiKey) => request("/api/key", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: apiKey }),
    }),
    clearKey: () => request("/api/key", { method: "DELETE" }),
    search: (query, blocks) => request("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, blocks }),
    }),
  };
}

export function createRuntimeAdapter({ sendMessage }) {
  if (typeof sendMessage !== "function") throw new Error("A runtime message function is required.");
  const request = (message) => new Promise((resolve, reject) => {
    sendMessage(message, (response) => {
      if (globalThis.chrome?.runtime?.lastError) {
        reject(new Error(globalThis.chrome.runtime.lastError.message));
        return;
      }
      if (!response?.ok) {
        const error = new Error(response?.error || "The extension request failed.");
        error.status = response?.status;
        error.code = response?.code;
        reject(error);
        return;
      }
      resolve(response.payload || {});
    });
  });
  return {
    ensureProviderAccess: (provider) => request({ type: "context_atlas.request_provider_access", provider }),
    getProvider: async () => (await request({ type: "context_atlas.provider_status" })).provider,
    setProvider: async (provider) => {
      if (!isProvider(provider)) throw new Error("Unsupported Context Atlas provider.");
      return (await request({ type: "context_atlas.set_provider", provider })).provider;
    },
    getKeyStatus: () => request({ type: "context_atlas.key_status" }),
    saveKey: (apiKey) => request({ type: "context_atlas.save_key", api_key: apiKey }),
    clearKey: () => request({ type: "context_atlas.clear_key" }),
    getRuntimeStatus: () => request({ type: "context_atlas.runtime_status" }),
    search: (query, blocks) => request({ type: "context_atlas.search", query, blocks }),
    publishSource: (source) => request({ type: "context_atlas.publish_source", source }),
    getSource: () => request({ type: "context_atlas.get_source" }),
    refreshSource: () => request({ type: "context_atlas.get_source" }),
  };
}
