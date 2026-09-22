// manifest.json is the canonical origin list. Classify Jev by its API hostname
// and keep every other manifest origin for the Local model.
const manifest = globalThis.chrome?.runtime?.getManifest?.();
const optionalHostPermissions = Array.isArray(manifest?.optional_host_permissions)
  ? manifest.optional_host_permissions.filter((origin) => typeof origin === "string")
  : [];
const jevOrigins = optionalHostPermissions.filter((origin) => {
  try {
    return new URL(origin.replace(/\/\*$/, "")).hostname === "api.typesafe.ai";
  } catch (_error) {
    return false;
  }
});
const layaOrigins = optionalHostPermissions.filter((origin) => !jevOrigins.includes(origin));

globalThis.ContextAtlasProviderOrigins = Object.freeze({
  jev: Object.freeze(jevOrigins),
  laya: Object.freeze(layaOrigins),
});
