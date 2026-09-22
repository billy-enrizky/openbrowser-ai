export const MAX_QUERY_CANDIDATE_BLOCKS = 1024;

const TOKEN_PATTERN = /[\p{L}\p{N}]+/gu;

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

export function selectQueryCandidates(query, blocks, { limit = MAX_QUERY_CANDIDATE_BLOCKS, neighborRadius = 1 } = {}) {
  if (!Array.isArray(blocks)) throw new Error("blocks must be an array");
  if (!Number.isInteger(limit) || limit < 1) throw new Error("limit must be a positive integer");
  if (!Number.isInteger(neighborRadius) || neighborRadius < 0) throw new Error("neighborRadius must be a nonnegative integer");
  if (blocks.length <= limit) return { blocks: [...blocks], selectedIndices: blocks.map((_block, index) => index), omittedIndices: [], omittedRanges: [] };
  const queryTerms = scalarTerms(query);
  const scores = blocks.map((block, index) => ({ index, score: queryOverlap(queryTerms, `${block?.title || ""} ${block?.section || ""} ${block?.text || ""}`) }));
  const selected = new Set();
  const add = (index) => { if (index < 0 || index >= blocks.length || selected.size >= limit) return false; selected.add(index); return true; };
  add(0);
  add(blocks.length - 1);
  const ranked = [...scores].filter((item) => item.score > 0).sort((left, right) => right.score - left.score || left.index - right.index);
  for (const item of ranked) add(item.index);
  for (const item of ranked) {
    for (let distance = 1; distance <= neighborRadius && selected.size < limit; distance += 1) {
      add(item.index - distance);
      add(item.index + distance);
    }
    if (selected.size >= limit) break;
  }
  const remaining = limit - selected.size;
  for (let slot = 1; slot <= remaining; slot += 1) add(Math.round((slot * (blocks.length - 1)) / (remaining + 1)));
  for (let index = 0; selected.size < limit && index < blocks.length; index += 1) add(index);
  const selectedIndices = [...selected].sort((left, right) => left - right);
  const omittedIndices = blocks.map((_block, index) => index).filter((index) => !selected.has(index));
  return { blocks: selectedIndices.map((index) => blocks[index]), selectedIndices, omittedIndices, omittedRanges: contiguousRanges(omittedIndices) };
}

function scalarTerms(value) { return new Set(normalizeSourceText(value).match(TOKEN_PATTERN) || []); }
function queryOverlap(queryTerms, text) { let score = 0; const textTerms = scalarTerms(text); for (const term of queryTerms) if (textTerms.has(term)) score += 1; return score; }
function contiguousRanges(indices) { const ranges = []; for (const index of indices) { const previous = ranges[ranges.length - 1]; if (previous && index === previous[1] + 1) previous[1] = index; else ranges.push([index, index]); } return ranges; }
function hashString(value) { let hash = 2166136261; for (let index = 0; index < value.length; index += 1) { hash ^= value.charCodeAt(index); hash = Math.imul(hash, 16777619); } return hash >>> 0; }
