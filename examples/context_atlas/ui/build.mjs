import { build } from "esbuild";
import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const root = resolve(import.meta.dirname);

const common = {
  bundle: true,
  minify: false,
  sourcemap: false,
  format: "iife",
  platform: "browser",
  target: ["es2022", "chrome110"],
  jsx: "automatic",
  loader: { ".css": "text" },
  define: { "process.env.NODE_ENV": '"production"' },
  legalComments: "none",
};

const builds = [
  {
    entryPoints: [resolve(root, "src/standalone/main.jsx")],
    outfile: resolve(root, "../web/assets/context-atlas-standalone.js"),
  },
  {
    entryPoints: [resolve(root, "src/extension/main.jsx")],
    outfile: resolve(root, "../extension/context-atlas-extension.js"),
  },
  {
    entryPoints: [resolve(root, "src/extension/offscreen.js")],
    outfile: resolve(root, "../extension/context-atlas-offscreen.js"),
    format: "esm",
  },
  {
    entryPoints: [resolve(root, "src/extension/sandbox.js")],
    outfile: resolve(root, "../extension/context-atlas-sandbox.js"),
    format: "esm",
  },
];

await Promise.all(builds.map(async (options) => {
  await mkdir(dirname(options.outfile), { recursive: true });
  await build({ ...common, ...options });
}));

const wasmDirectory = resolve(root, "../extension/ort");
await mkdir(wasmDirectory, { recursive: true });
await copyFile(
  resolve(root, "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm"),
  resolve(wasmDirectory, "ort-wasm-simd-threaded.asyncify.wasm"),
);
await copyFile(
  resolve(root, "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs"),
  resolve(wasmDirectory, "ort-wasm-simd-threaded.asyncify.mjs"),
);

console.log("Built Context Atlas standalone, extension, and local Laya runtime bundles.");
