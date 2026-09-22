export const MAX_QUERY_LENGTH = 400;
export const MAX_BLOCKS = 160;
export const MAX_PASSAGE_LENGTH = 2200;
export const MAX_TOTAL_TEXT_LENGTH = 60000;

export function segmentText(source) {
  return getSentenceSegments(source).map(({ index, text }) => ({ index, text }));
}
export function findSentenceOffset(source, sentenceIndex, sentenceText) {
  const sentences = getSentenceSegments(source);
  const sentence = sentences.find((item) => item.index === sentenceIndex && item.text === sentenceText);
  return sentence ? { start: sentence.start, end: sentence.end, text: sentence.text } : null;
}

function getSentenceSegments(source) {
  if (typeof source !== "string" || !source.trim()) return [];
  if (typeof Intl !== "undefined" && typeof Intl.Segmenter === "function") {
    const segmenter = new Intl.Segmenter(undefined, { granularity: "sentence" });
    const sentences = [];
    for (const item of segmenter.segment(source)) {
      const raw = item.segment;
      const text = raw.trim();
      if (!text) continue;
      const leftTrim = raw.length - raw.trimStart().length;
      sentences.push({
        index: sentences.length,
        start: item.index + leftTrim,
        end: item.index + raw.trimEnd().length,
        text,
      });
    }
    return sentences;
  }

  return scanSentences(source);
}

function scanSentences(source) {
  const sentences = [];
  let start = 0;
  for (let index = 0; index < source.length; index += 1) {
    if (!isSentenceTerminator(source[index]) || !endsSentence(source, index)) continue;
    const end = sentenceBoundaryEnd(source, index);
    const raw = source.slice(start, end);
    const text = raw.trim();
    if (text) {
      const offset = raw.indexOf(text);
      sentences.push({ index: sentences.length, start: start + offset, end: start + offset + text.length, text });
    }
    start = end;
    index = end - 1;
  }
  const rawTail = source.slice(start);
  const textTail = rawTail.trim();
  if (textTail) {
    const offset = rawTail.indexOf(textTail);
    sentences.push({ index: sentences.length, start: start + offset, end: start + offset + textTail.length, text: textTail });
  }
  return sentences;
}

function isSentenceTerminator(character) {
  return ".!?。！？｡．".includes(character);
}

function endsSentence(source, index) {
  if ("。！？｡．".includes(source[index])) return true;
  const next = sentenceBoundaryEnd(source, index);
  return next === source.length || /\s/.test(source[next]);
}

function sentenceBoundaryEnd(source, index) {
  const closing = "\"'”’)]}»〉》」』】〕］）｝";
  let next = index + 1;
  while (next < source.length && closing.includes(source[next])) next += 1;
  return next;
}
