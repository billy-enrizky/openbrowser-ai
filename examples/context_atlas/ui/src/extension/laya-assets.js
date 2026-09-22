import { DEFAULT_LAYA_BROWSER_CONFIG } from "./laya-core.js";

const MODEL_CACHE_NAME = "context-atlas-laya-model-d9d003d543e63d6d3375c21d44624136bd1e0bad";
export const MODEL_FILES = Object.freeze({
  model: "model.onnx",
  tokenizer: "tokenizer/tokenizer.json",
  tokenizerConfig: "tokenizer/tokenizer_config.json",
  rlConfig: "rl_agent_config.json",
});

function withTrailingSlash(value) {
  return String(value).endsWith("/") ? String(value) : `${value}/`;
}

export function browserModelUrls(config = DEFAULT_LAYA_BROWSER_CONFIG) {
  const baseUrl = withTrailingSlash(config.modelBaseUrl);
  return Object.fromEntries(Object.entries(MODEL_FILES).map(([key, file]) => [key, new URL(file, baseUrl).toString()]));
}

export async function fetchResponse(url, fetchImpl, cacheStorage) {
  if (cacheStorage && typeof cacheStorage.open === "function") {
    try {
      const cache = await cacheStorage.open(MODEL_CACHE_NAME);
      const cached = await cache.match(url);
      if (cached) return cached;
      const response = await fetchResponse(url, fetchImpl, null);
      try { await cache.put(url, response.clone()); } catch (_error) { /* Cache is an optimization. */ }
      return response;
    } catch (_error) {
      // Cache Storage failures fall back to an immutable direct fetch.
    }
  }
  const response = await fetchImpl(url, { credentials: "omit" });
  if (!response?.ok) throw new Error(`Could not download the local Laya asset (${response?.status || "network error"}).`);
  return response;
}

export function resolveCacheStorage(globalLike = globalThis) {
  try {
    const cacheStorage = globalLike?.caches;
    return cacheStorage && typeof cacheStorage.open === "function" ? cacheStorage : null;
  } catch (_error) {
    return null;
  }
}

export async function loadLayaBrowserAssets({ config = DEFAULT_LAYA_BROWSER_CONFIG, fetchImpl = globalThis.fetch, cacheStorage } = {}) {
  if (typeof fetchImpl !== "function") throw new Error("The browser fetch API is required for local Laya assets.");
  const availableCacheStorage = cacheStorage === undefined ? resolveCacheStorage() : cacheStorage;
  const urls = browserModelUrls(config);
  const [modelResponse, tokenizerResponse, tokenizerConfigResponse, rlConfigResponse] = await Promise.all([
    fetchResponse(urls.model, fetchImpl, availableCacheStorage),
    fetchResponse(urls.tokenizer, fetchImpl, availableCacheStorage),
    fetchResponse(urls.tokenizerConfig, fetchImpl, availableCacheStorage),
    fetchResponse(urls.rlConfig, fetchImpl, availableCacheStorage),
  ]);
  return {
    modelBytes: new Uint8Array(await modelResponse.arrayBuffer()),
    tokenizerJson: await tokenizerResponse.json(),
    tokenizerConfig: await tokenizerConfigResponse.json(),
    rlConfig: await rlConfigResponse.json(),
  };
}
