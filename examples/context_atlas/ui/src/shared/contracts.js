export const MAX_QUERY_LENGTH = 400;
export const MAX_BLOCKS = 160;
export const MAX_PASSAGE_LENGTH = 2200;
export const MAX_TOTAL_TEXT_LENGTH = 60000;

export function segmentText(source) {
  if (typeof source !== "string" || !source.trim()) return [];
  const sentences = [];
  let start = 0;
  for (let index = 0; index < source.length; index += 1) {
    if (!".!?".includes(source[index]) || !endsSentence(source, index)) continue;
    const text = source.slice(start, index + 1).trim();
    if (text) sentences.push({ index: sentences.length, text });
    start = index + 1;
  }
  const tail = source.slice(start).trim();
  if (tail) sentences.push({ index: sentences.length, text: tail });
  return sentences;
}
export function findSentenceOffset(source, sentenceIndex, sentenceText) {
  const sentences = segmentTextWithOffsets(source);
  const sentence = sentences.find((item) => item.index === sentenceIndex && item.text === sentenceText);
  return sentence ? { start: sentence.start, end: sentence.end, text: sentence.text } : null;
}

function segmentTextWithOffsets(source) {
  if (typeof source !== "string" || !source.trim()) return [];
  const sentences = [];
  let start = 0;
  for (let index = 0; index < source.length; index += 1) {
    if (!".!?".includes(source[index]) || !endsSentence(source, index)) continue;
    const raw = source.slice(start, index + 1);
    const text = raw.trim();
    if (text) {
      const offset = raw.indexOf(text);
      sentences.push({ index: sentences.length, start: start + offset, end: start + offset + text.length, text });
    }
    start = index + 1;
  }
  const rawTail = source.slice(start);
  const textTail = rawTail.trim();
  if (textTail) {
    const offset = rawTail.indexOf(textTail);
    sentences.push({ index: sentences.length, start: start + offset, end: start + offset + textTail.length, text: textTail });
  }
  return sentences;
}

function endsSentence(source, index) {
  const closing = "\"'”’)]}";
  let next = index + 1;
  while (next < source.length && closing.includes(source[next])) next += 1;
  return next === source.length || /\s/.test(source[next]);
}
