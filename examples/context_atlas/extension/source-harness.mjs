import jevClient from "./jev-client.js";

const { meaningfulTerms, scorePassage } = jevClient;

export const MAX_SOURCE_SCAN_PASSAGES = 1024;
export const MAX_PLANNED_PASSAGES = 160;
export const MAX_PLANNED_TEXT_LENGTH = 60000;
export const SOURCE_BOUNDARY_PASSAGES = 8;

export function planSourcePassages(query, passages) {
  if (!Array.isArray(passages) || !passages.length) return [];
  const terms = meaningfulTerms(query);
  const ranked = passages.map((passage, index) => ({ passage, index, score: scorePassage(passage, terms) }));
  const selected = new Set();
  let selectedLength = 0;

  const add = (index) => {
    if (selected.has(index) || selected.size >= MAX_PLANNED_PASSAGES || index < 0 || index >= passages.length) return false;
    const length = String(passages[index]?.text || "").length;
    if (!length || selectedLength + length > MAX_PLANNED_TEXT_LENGTH) return false;
    selected.add(index);
    selectedLength += length;
    return true;
  };

  for (let offset = 0; offset < SOURCE_BOUNDARY_PASSAGES; offset += 1) {
    add(offset);
    add(passages.length - 1 - offset);
  }
  const relevant = ranked.filter((item) => item.score > 0).sort((left, right) => right.score - left.score || left.index - right.index);
  relevant.forEach(({ index }) => { add(index); add(index - 1); add(index + 1); });
  ranked.slice().sort((left, right) => right.score - left.score || left.index - right.index).forEach(({ index }) => add(index));
  return [...selected].sort((left, right) => left - right).map((index) => passages[index]);
}
