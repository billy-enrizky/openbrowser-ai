(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  root.ContextAtlasViewModel = api;
})(typeof globalThis === "object" ? globalThis : this, function () {
  function selectFocusedMatch(result) {
    if (!result) return null;
    if (Array.isArray(result.matches) && result.matches[0]) return result.matches[0];
    if (Array.isArray(result.scores) && result.scores[0]) return result.scores[0];
    return null;
  }

  function buildProvenanceThread(result, blocks, query) {
    const match = selectFocusedMatch(result);
    if (!match || !Array.isArray(blocks)) return [];
    const passage = blocks.find((block) => block && block.id === match.passage_id);
    if (!passage) return [];
    const sentences = Array.isArray(passage.sentences) ? passage.sentences : [];
    const sentence = sentences.find(
      (item) => item && item.index === match.sentence_index && item.text === match.sentence_text,
    );
    if (typeof passage.text !== "string" || !sentence) return [];
    return [
      {
        kind: "query",
        label: "Query",
        text: String(query || ""),
        passageId: null,
        sentenceIndex: null,
        probability: null,
      },
      {
        kind: "passage",
        label: "Matched passage",
        text: passage.text,
        passageId: passage.id,
        sentenceIndex: null,
        probability: match.probability,
      },
      {
        kind: "sentence",
        label: "Selected sentence",
        text: sentence.text,
        passageId: passage.id,
        sentenceIndex: match.sentence_index,
        probability: match.probability,
      },
      {
        kind: "source",
        label: "Source",
        text: passage.title || passage.id,
        passageId: passage.id,
        sentenceIndex: match.sentence_index,
        probability: match.probability,
      },
    ];
  }

  return { buildProvenanceThread, selectFocusedMatch };
});
