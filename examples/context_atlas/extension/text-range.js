(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
	root.ContextAtlasTextRange = api;
})(typeof globalThis === "object" ? globalThis : this, function () {
  function segmentText(source) {
    if (typeof source !== "string" || !source.trim()) return [];
    if (typeof Intl !== "undefined" && Intl.Segmenter) {
      const segmenter = new Intl.Segmenter(undefined, { granularity: "sentence" });
      const segments = [];
      for (const item of segmenter.segment(source)) {
        const raw = item.segment;
        const leftTrim = raw.length - raw.trimStart().length;
        const rightTrim = raw.trimEnd().length;
        const text = raw.trim();
        if (text) {
          segments.push({
            index: segments.length,
            text,
            start: item.index + leftTrim,
            end: item.index + rightTrim,
          });
        }
      }
      return segments.map(({ index, text }) => ({ index, text }));
    }

    return scanSentences(source).map(({ index, text }) => ({ index, text }));
  }

  function findSentenceOffset(source, sentenceIndex, expectedText) {
    if (typeof source !== "string" || !Number.isInteger(sentenceIndex)) return null;
    const segments = getSegments(source);
    const selected = segments.find((item) => item.index === sentenceIndex);
    if (!selected || (typeof expectedText === "string" && selected.text !== expectedText)) {
      return null;
    }
    return { start: selected.start, end: selected.end, text: selected.text };
  }

  function isCurrentRequest(currentRequest, responseRequest) {
    return Number.isInteger(currentRequest) && currentRequest === responseRequest;
  }

  function getSegments(source) {
    if (typeof Intl !== "undefined" && Intl.Segmenter) {
      const segmenter = new Intl.Segmenter(undefined, { granularity: "sentence" });
      const segments = [];
      for (const item of segmenter.segment(source)) {
        const raw = item.segment;
        const text = raw.trim();
        if (!text) continue;
        const leftTrim = raw.length - raw.trimStart().length;
        const rightTrim = raw.trimEnd().length;
        segments.push({
          index: segments.length,
          text,
          start: item.index + leftTrim,
          end: item.index + rightTrim,
        });
      }
      return segments;
    }
    return scanSentences(source);
  }

  function scanSentences(source) {
    const sentences = [];
    let start = 0;
    for (let position = 0; position < source.length; position += 1) {
      const character = source[position];
      if (!".!?。！？｡．".includes(character) || !endsSentence(source, position)) continue;
      const end = sentenceBoundaryEnd(source, position);
      const raw = source.slice(start, end);
      const text = raw.trim();
      if (text) {
        const leftTrim = raw.length - raw.trimStart().length;
        sentences.push({
          index: sentences.length,
          text,
          start: start + leftTrim,
          end: start + raw.trimEnd().length,
        });
      }
      start = end;
      position = end - 1;
    }
    const raw = source.slice(start);
    const text = raw.trim();
    if (text) {
      const leftTrim = raw.length - raw.trimStart().length;
      sentences.push({
        index: sentences.length,
        text,
        start: start + leftTrim,
        end: start + raw.trimEnd().length,
      });
    }
    return sentences;
  }

  function endsSentence(source, position) {
    if ("。！？｡．".includes(source[position])) return true;
    const next = sentenceBoundaryEnd(source, position);
    return next === source.length || /\s/.test(source[next]);
  }

  function sentenceBoundaryEnd(source, position) {
    const closing = "\"'”’)]}»〉》」』】〕］）｝";
    let next = position + 1;
    while (next < source.length && closing.includes(source[next])) next += 1;
    return next;
  }

  return { findSentenceOffset, isCurrentRequest, segmentText };
});
