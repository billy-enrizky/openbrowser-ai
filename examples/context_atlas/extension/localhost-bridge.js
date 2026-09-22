(function () {
  const CHANNEL = "context_atlas";
  const ALLOWED_ORIGINS = new Set([
    "http://127.0.0.1:8765",
  ]);
  const ALLOWED_TYPES = new Set([
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

  if (!ALLOWED_ORIGINS.has(location.origin)) return;

  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== location.origin) return;
    const request = event.data;
    if (!request || request.channel !== CHANNEL || !ALLOWED_TYPES.has(request.type)) return;
    if (typeof request.requestId !== "string" || request.requestId.length < 1 || request.requestId.length > 120) return;
    if (typeof request.ok === "boolean") return;
    const message = copyAllowedFields(request);
    chrome.runtime.sendMessage(message, (response) => {
      const runtimeError = chrome.runtime.lastError;
      const payload = {
        channel: CHANNEL,
        type: request.type,
        requestId: request.requestId,
        ok: !runtimeError && response?.ok === true,
      };
      if (runtimeError) {
        payload.error = "The Context Atlas extension is unavailable.";
      } else if (response?.ok) {
        payload.payload = response.payload || {};
      } else {
        payload.error = response?.error || "The Context Atlas request failed.";
        if (response?.code) payload.code = response.code;
      }
      window.postMessage(payload, location.origin);
    });
  });

  function copyAllowedFields(request) {
    const message = { type: request.type };
    if (request.type === "context_atlas.request_provider_access" && typeof request.provider === "string") message.provider = request.provider;
    if (request.type === "context_atlas.set_provider" && typeof request.provider === "string") message.provider = request.provider;
    if (request.type === "context_atlas.save_key" && typeof request.api_key === "string") message.api_key = request.api_key;
    if (request.type === "context_atlas.search") {
      if (typeof request.query === "string") message.query = request.query;
      if (Array.isArray(request.blocks)) message.blocks = request.blocks;
    }
    if (request.type === "context_atlas.publish_source" && request.source && typeof request.source === "object") message.source = request.source;
    return message;
  }
})();
