export function findTextNodeSegments(textNodes, start, end) {
  const rangeStart = Number.isFinite(start) ? Math.max(0, start) : 0;
  const rangeEnd = Number.isFinite(end) ? Math.max(rangeStart, end) : rangeStart;
  if (rangeEnd <= rangeStart) return [];
  const segments = [];
  let cursor = 0;
  for (const node of Array.isArray(textNodes) ? textNodes : []) {
    const length = String(node?.nodeValue || "").length;
    const segmentStart = Math.max(0, rangeStart - cursor);
    const segmentEnd = Math.min(length, rangeEnd - cursor);
    if (segmentStart < segmentEnd) segments.push({ node, start: segmentStart, end: segmentEnd });
    cursor += length;
    if (cursor >= rangeEnd) break;
  }
  return segments;
}
