// manifest.json is the canonical origin list. Keep Jev's single entry first,
// followed by all Local model entries.
const manifest = globalThis.chrome?.runtime?.getManifest?.();
const optionalHostPermissions = Array.isArray(manifest?.optional_host_permissions)
  ? manifest.optional_host_permissions.filter((origin) => typeof origin === "string")
  : [];
const [jevOrigin, ...layaOrigins] = optionalHostPermissions;

globalThis.ContextAtlasProviderOrigins = Object.freeze({
  jev: Object.freeze(jevOrigin ? [jevOrigin] : []),
  laya: Object.freeze(layaOrigins),
});
