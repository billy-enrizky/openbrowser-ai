export const SUPPORTED_PROVIDERS = Object.freeze(["jev", "laya"]);

export function isProvider(value) {
  return SUPPORTED_PROVIDERS.includes(value);
}

export function normalizeProvider(value) {
  return isProvider(value) ? value : "laya";
}

export function invalidateProviderRequest(requestRef, clearResult = () => {}) {
  if (requestRef && typeof requestRef === "object" && Object.hasOwn(requestRef, "requestId")) {
    return {
      ...requestRef,
      requestId: requestRef.requestId + 1,
      result: null,
      matches: [],
      current: 0,
      traceOpen: false,
    };
  }
  requestRef.current += 1;
  clearResult();
  return requestRef.current;
}

export function isCurrentRequest(requestId, currentRequestId) {
  return requestId === currentRequestId;
}
export function sourceFingerprint(passages, sourceUrl = "") {
  return JSON.stringify(
    {
      url: String(sourceUrl || ""),
      passages: (Array.isArray(passages) ? passages : []).map((passage) => ({
        id: passage?.id,
        text: passage?.text,
        sentences: Array.isArray(passage?.sentences)
          ? passage.sentences.map((sentence) => ({ index: sentence?.index, text: sentence?.text }))
          : [],
      })),
    },
  );
}

export function findSourceMatch(result, passages, resultIndex = 0) {
  const candidates = Array.isArray(result?.matches) && result.matches.length
    ? result.matches
    : Array.isArray(result?.scores) ? result.scores : [];
  const match = candidates[resultIndex];
  if (!match || !Array.isArray(passages)) return null;
  const passage = passages.find((item) => item?.id === match.passage_id);
  const sentence = passage?.sentences?.find(
    (item) => item?.index === match.sentence_index && item?.text === match.sentence_text,
  );
  return passage && sentence ? { match, passage, sentence } : null;
}

export function validSourceMatches(result, passages) {
  const candidates = Array.isArray(result?.matches) && result.matches.length
    ? result.matches
    : Array.isArray(result?.scores) ? result.scores : [];
  return candidates
    .map((_, index) => findSourceMatch({ matches: candidates }, passages, index))
    .filter(Boolean);
}

export function buildProvenanceThread(result, passages, query) {
  const selected = findSourceMatch(result, passages);
  if (!selected) return [];
  const { match, passage, sentence } = selected;
  return [
    { kind: "query", label: "Query", text: String(query || ""), passageId: null, sentenceIndex: null, probability: null },
    { kind: "passage", label: "Matched passage", text: passage.text, passageId: passage.id, sentenceIndex: null, probability: match.probability },
    { kind: "sentence", label: "Selected sentence", text: sentence.text, passageId: passage.id, sentenceIndex: match.sentence_index, probability: match.probability },
    { kind: "source", label: "Source", text: passage.title || passage.id, passageId: passage.id, sentenceIndex: match.sentence_index, probability: match.probability },
  ];
}

export function normalizeSearchError(error) {
  const status = Number(error?.status);
  const rawMessage = error?.message || "The request could not be completed.";
  const codeMessages = {
    authentication: "The saved Cloud access key was rejected. Check it and try again.",
    not_configured: "Cloud is not configured. Save an access key first.",
    invalid_input: "The page source or question is invalid. Check it and try again.",
    invalid_source: "The current page source could not be prepared. Refresh the page and try again.",
    not_ready: "The current page is not ready. Open Context Atlas on a webpage first.",
    permission_required: "Allow access in the new Context Atlas window, then choose this provider again.",
  };
  if (Object.hasOwn(codeMessages, error?.code)) {
    return { message: codeMessages[error.code], retryable: false };
  }
  if (error?.code === "permission_denied") {
    return { message: String(rawMessage), retryable: false };
  }
  const message = String(rawMessage).toLowerCase();
  if (/local laya|laya model|local model/.test(message)) {
    return { message: "Local search is unavailable. Retry the local search.", retryable: true };
  }
  if (/jev rate limit|rate limit/.test(message)) {
    return { message: "Cloud search is busy. Try again in a moment.", retryable: true };
  }
  if (/saved jev key was rejected|jev key was rejected/.test(message)) {
    return { message: "The saved Cloud access key was rejected. Check it and try again.", retryable: false };
  }
  if (/jev rejected the source size/.test(message)) {
    return { message: "Cloud search rejected the source size. Retry with fewer visible passages.", retryable: true };
  }
  if (/jev is not configured/.test(message)) {
    return { message: "Cloud is not configured. Save an access key first.", retryable: false };
  }
  if (/max[_ ]tokens|context length|provider input|provider limit|input too large/.test(message)) {
    return {
      message: "Cloud search rejected the source size. Retry with fewer visible passages.",
      retryable: true,
    };
  }
  if (status === 429 || message.includes("rate limit")) {
    return { message: "Cloud search is busy. Try again in a moment.", retryable: true };
  }
  if (status === 503 || message.includes("not configured")) {
    return { message: "Cloud is not configured. Save an access key first.", retryable: true };
  }
  if (status === 400) {
    return { message: "The local server rejected this source. Check the page text and try again.", retryable: false };
  }
  return { message: String(rawMessage), retryable: true };
}
