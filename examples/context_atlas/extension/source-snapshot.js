(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root && typeof root === "object") root.ContextAtlasSourceSnapshot = api;
})(typeof globalThis === "object" ? globalThis : this, function () {
  const MAX_TITLE_LENGTH = 240;
  const MAX_URL_LENGTH = 2048;
  const MAX_REVISION_LENGTH = 512;
  const MAX_PASSAGES = 160;
  const MAX_PASSAGE_ID_LENGTH = 160;
  const MAX_SECTION_LENGTH = 240;
  const MAX_PASSAGE_TITLE_LENGTH = 240;
  const MAX_PASSAGE_LENGTH = 2200;
  const MAX_TOTAL_TEXT_LENGTH = 60000;
  const MAX_SENTENCES_PER_PASSAGE = 128;
  const MAX_SENTENCE_ID_LENGTH = 160;
  const MAX_SENTENCE_LENGTH = 2200;
  const MAX_SOURCE_SNAPSHOT_BYTES = 1000000;

  function serializeSourceSnapshot(source) {
    if (!source || typeof source !== "object" || Array.isArray(source)) {
      throw invalid("Source snapshot is invalid.");
    }
    const snapshot = {};
    if (Object.hasOwn(source, "title")) snapshot.title = cleanText(source.title, MAX_TITLE_LENGTH, "Source title");
    if (Object.hasOwn(source, "url")) snapshot.url = cleanUrl(source.url);
    if (Object.hasOwn(source, "revision")) snapshot.revision = cleanText(source.revision, MAX_REVISION_LENGTH, "Source revision");
    snapshot.passages = normalizePassages(source.passages);
    return validateSourceSnapshot(snapshot);
  }

  function validateSourceSnapshot(source) {
    if (!source || typeof source !== "object" || Array.isArray(source)) {
      throw invalid("Source snapshot is invalid.");
    }
    const snapshot = {};
    if (Object.hasOwn(source, "title")) snapshot.title = cleanText(source.title, MAX_TITLE_LENGTH, "Source title");
    if (Object.hasOwn(source, "url")) snapshot.url = cleanUrl(source.url);
    if (Object.hasOwn(source, "revision")) snapshot.revision = cleanText(source.revision, MAX_REVISION_LENGTH, "Source revision");
    snapshot.passages = normalizePassages(source.passages);
    if (utf8ByteLength(JSON.stringify(snapshot)) > MAX_SOURCE_SNAPSHOT_BYTES) throw invalid("Source snapshot is too large.");
    return snapshot;
  }

  function normalizePassages(passages) {
    if (!Array.isArray(passages) || passages.length > MAX_PASSAGES) {
      throw invalid("Source passages are invalid.");
    }
    const seen = new Set();
    let totalLength = 0;
    let totalSentenceLength = 0;
    return passages.map((passage) => {
      if (!passage || typeof passage !== "object" || Array.isArray(passage)) throw invalid("Source passage is invalid.");
      const id = cleanText(passage.id, MAX_PASSAGE_ID_LENGTH, "Source passage ID");
      if (!id || seen.has(id)) throw invalid("Source passage IDs must be unique.");
      seen.add(id);
      const text = cleanText(passage.text, MAX_PASSAGE_LENGTH, "Source passage text");
      if (!text) throw invalid("Source passage text is invalid.");
      totalLength += text.length;
      if (totalLength > MAX_TOTAL_TEXT_LENGTH) throw invalid("Source snapshot is too large.");
      const normalized = { id, text };
      if (Object.hasOwn(passage, "title")) {
        const title = cleanText(passage.title, MAX_PASSAGE_TITLE_LENGTH, "Source passage title");
        if (title) normalized.title = title;
      }
      if (Object.hasOwn(passage, "section")) {
        const section = cleanText(passage.section, MAX_SECTION_LENGTH, "Source section");
        if (section) normalized.section = section;
      }
      normalized.sentences = normalizeSentences(passage.sentences, text);
      totalSentenceLength += normalized.sentences.reduce((sum, sentence) => sum + sentence.text.length, 0);
      if (totalSentenceLength > MAX_TOTAL_TEXT_LENGTH) throw invalid("Source snapshot is too large.");
      return normalized;
    });
  }

  function normalizeSentences(sentences, passageText) {
    if (!Array.isArray(sentences) || !sentences.length || sentences.length > MAX_SENTENCES_PER_PASSAGE) {
      throw invalid("Source sentence data is invalid.");
    }
    return sentences.map((sentence, index) => {
      if (!sentence || typeof sentence !== "object" || sentence.index !== index) {
        throw invalid("Source sentence data is invalid.");
      }
      const text = cleanText(sentence.text, MAX_SENTENCE_LENGTH, "Source sentence text");
      if (!text || !passageText.includes(text)) throw invalid("Source sentence data is invalid.");
      const normalized = { index, text };
      if (Object.hasOwn(sentence, "id")) {
        const id = cleanText(sentence.id, MAX_SENTENCE_ID_LENGTH, "Source sentence ID");
        if (id) normalized.id = id;
      }
      return normalized;
    });
  }

  function cleanText(value, maxLength, label) {
    if (typeof value !== "string") throw invalid(`${label} is invalid.`);
    const text = value.trim();
    if (text.length > maxLength) throw invalid(`${label} is too long.`);
    return text;
  }

  function cleanUrl(value) {
    const url = cleanText(value, MAX_URL_LENGTH, "Source URL");
    if (url && !/^https?:\/\//i.test(url)) throw invalid("Source URL is invalid.");
    return url;
  }

  function utf8ByteLength(value) {
    if (typeof TextEncoder === "function") return new TextEncoder().encode(value).length;
    let length = 0;
    for (const character of String(value)) {
      const codePoint = character.codePointAt(0);
      length += codePoint <= 0x7f ? 1 : codePoint <= 0x7ff ? 2 : codePoint <= 0xffff ? 3 : 4;
    }
    return length;
  }

  function invalid(message) {
    const error = new Error(message);
    error.code = "invalid_source";
    error.status = 400;
    return error;
  }

  return {
    MAX_SOURCE_SNAPSHOT_BYTES,
    MAX_PASSAGES,
    MAX_PASSAGE_LENGTH,
    MAX_TOTAL_TEXT_LENGTH,
    serializeSourceSnapshot,
    validateSourceSnapshot,
  };
});
