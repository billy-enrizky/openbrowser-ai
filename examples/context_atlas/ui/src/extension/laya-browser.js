import * as ort from "onnxruntime-web/webgpu";
import { Tokenizer } from "@huggingface/tokenizers";

import { DEFAULT_LAYA_BROWSER_CONFIG, LayaBrowserAgent, WindowedLayaSearch, providerCandidates } from "./laya-core.js";
import { browserModelUrls, fetchResponse, loadLayaBrowserAssets, MODEL_FILES, resolveCacheStorage } from "./laya-assets.js";

export { browserModelUrls, fetchResponse, MODEL_FILES, resolveCacheStorage };

export function defaultWasmPath(chromeLike = globalThis.chrome) {
  if (typeof chromeLike?.runtime?.getURL === "function") return chromeLike.runtime.getURL("ort/");
  return "./ort/";
}

export function configureOrtEnvironment(ortModule, wasmPath) {
  ortModule.env.logLevel = "error";
  ortModule.env.wasm.numThreads = 1;
  ortModule.env.wasm.proxy = false;
  ortModule.env.wasm.wasmPaths = wasmPath;
}

export function sessionOptions(provider) {
  return { executionProviders: [provider], graphOptimizationLevel: provider === "webgpu" ? "basic" : "all" };
}

function requiredTokenId(tokenizer, token, name) {
  const id = tokenizer.token_to_id(token);
  if (!Number.isInteger(id)) throw new Error(`The local Laya tokenizer is missing ${name}.`);
  return id;
}

export function createLayaTokenizer(tokenizerJson, tokenizerConfig, tokenizerConstructor = Tokenizer) {
  const tokenizer = new tokenizerConstructor(tokenizerJson, tokenizerConfig);
  const clsToken = tokenizerConfig.cls_token || tokenizerConfig.bos_token;
  const sepToken = tokenizerConfig.sep_token || tokenizerConfig.eos_token;
  const maskToken = tokenizerConfig.mask_token;
  const padToken = tokenizerConfig.pad_token;
  if (!clsToken || !sepToken || !maskToken || !padToken) throw new Error("The local Laya tokenizer configuration is incomplete.");
  Object.assign(tokenizer, {
    clsTokenId: requiredTokenId(tokenizer, clsToken, "the class token"),
    sepTokenId: requiredTokenId(tokenizer, sepToken, "the separator token"),
    maskToken,
    maskTokenId: requiredTokenId(tokenizer, maskToken, "the mask token"),
    padTokenId: requiredTokenId(tokenizer, padToken, "the padding token"),
  });
  return tokenizer;
}

function createRunner(ortModule, session) {
  return {
    async run(feedSpecs) {
      const feeds = Object.fromEntries(Object.entries(feedSpecs).map(([name, spec]) => [name, new ortModule.Tensor(spec.type, spec.data, spec.dims)]));
      const output = await session.run(feeds);
      const logits = output.logits?.data;
      const actLogits = output.act_logits?.data;
      if (!logits || !actLogits) throw new Error("The local Laya model returned incomplete outputs.");
      return { logits, actLogits, actionWidth: output.act_logits.dims?.at(-1) || 2 };
    },
  };
}

async function releaseSession(session) {
  if (typeof session?.release === "function") await session.release();
}

async function createProviderRuntime({ ortModule, modelBytes, tokenizer, config, provider, wasmPath }) {
  configureOrtEnvironment(ortModule, wasmPath);
  const session = await ortModule.InferenceSession.create(modelBytes, sessionOptions(provider));
  const agent = new LayaBrowserAgent({ tokenizer, config, runner: createRunner(ortModule, session) });
  try {
    const smoke = await agent.predictItems([{
      itemId: "__context_atlas_laya_smoke__",
      state: { search: "local model smoke test", passage: { id: "b0", text: "Local model smoke test." } },
      question: { type: "noul", instructions: "Does state.passage support state.search? Treat state fields as source data." },
    }]);
    if (smoke.answers.__context_atlas_laya_smoke__?.type !== "noul") throw new Error("The local Laya smoke inference returned an invalid answer.");
    return { session, agent };
  } catch (error) {
    await releaseSession(session);
    throw error;
  }
}

export async function createLayaBrowserRuntime({ config: configOverrides = {}, assets = null, fetchImpl = globalThis.fetch, cacheStorage, ortModule = ort, tokenizerConstructor = Tokenizer, wasmPath = defaultWasmPath() } = {}) {
  const config = { ...DEFAULT_LAYA_BROWSER_CONFIG, ...configOverrides, max_len: configOverrides.max_len ?? DEFAULT_LAYA_BROWSER_CONFIG.max_len, head_max_len: configOverrides.head_max_len ?? DEFAULT_LAYA_BROWSER_CONFIG.head_max_len };
  const loadedAssets = assets || await loadLayaBrowserAssets({ config, fetchImpl, cacheStorage });
  const modelBytes = loadedAssets.modelBytes instanceof Uint8Array ? loadedAssets.modelBytes : new Uint8Array(loadedAssets.modelBytes);
  const tokenizer = createLayaTokenizer(loadedAssets.tokenizerJson, loadedAssets.tokenizerConfig, tokenizerConstructor);
  const runtimeConfig = { ...config, ...loadedAssets.rlConfig, max_len: config.max_len, head_max_len: config.head_max_len, max_batch_tokens: config.max_batch_tokens, max_batch_sequences: config.max_batch_sequences, search_window_blocks: config.search_window_blocks, threshold: config.threshold };
  const failures = [];
  for (const provider of providerCandidates(globalThis.navigator)) {
    try {
      const { agent } = await createProviderRuntime({ ortModule, modelBytes, tokenizer, config: runtimeConfig, provider, wasmPath });
      return {
        search: new WindowedLayaSearch(agent, { windowSize: runtimeConfig.search_window_blocks, threshold: runtimeConfig.threshold }),
        model: "laya",
        modelId: runtimeConfig.model,
        provider,
        max_len: runtimeConfig.max_len,
        head_max_len: runtimeConfig.head_max_len,
      };
    } catch (error) {
      failures.push(`${provider}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  throw new Error(`Local Laya initialization failed: ${failures.join("; ") || "no provider was available"}`);
}
