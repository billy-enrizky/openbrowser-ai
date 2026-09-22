export function normalizeSourceText(value) {
  return String(value || "").normalize("NFKC").replace(/\s+/g, " ").trim().toLocaleLowerCase();
}

export function stableBlockId({ path = "", text = "", occurrence = 0 } = {}) {
  const key = `${path || `occurrence:${occurrence}`}\u0000${normalizeSourceText(text)}`;
  return `b${hashString(key).toString(36)}`;
}

export function sourceRevision(blocks) {
  const serialized = JSON.stringify((Array.isArray(blocks) ? blocks : []).map((block) => ({
    id: block?.id,
    section: block?.section,
    text: block?.text,
    sentences: Array.isArray(block?.sentences) ? block.sentences.map((sentence) => ({ id: sentence?.id, index: sentence?.index, text: sentence?.text })) : [],
  })));
  return `r${hashString(serialized).toString(36)}`;
}

function hashString(value) { let hash = 2166136261; for (let index = 0; index < value.length; index += 1) { hash ^= value.charCodeAt(index); hash = Math.imul(hash, 16777619); } return hash >>> 0; }
