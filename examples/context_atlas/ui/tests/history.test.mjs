import { webcrypto } from "node:crypto";
import test from "node:test";
import assert from "node:assert/strict";

import {
  SEARCH_HISTORY_PREFIX,
  createChromeHistoryStorage,
  createHistorySubjectHash,
  createLocalHistoryStorage,
  createSearchHistory,
  normalizeHistoryQuery,
  restoreCachedSearch,
} from "../src/shared/history.js";

const jevSubject = {
  surface: "extension",
  locator: "https://example.com/long-page#details",
  revision: "visible source revision",
  provider: "jev",
};

const layaSubject = { ...jevSubject, provider: "laya" };

const passages = [
  {
    id: "b1",
    text: "First answer. Second answer.",
    sentences: [
      { id: "s-first", index: 0, text: "First answer." },
      { id: "s-second", index: 1, text: "Second answer." },
    ],
  },
];

const ambiguousResult = {
  ambiguous: true,
  ambiguity_gap: 0.01,
  matches: [
    { passage_id: "b1", probability: 0.91, sentence_index: 0, sentence_text: "First answer.", ignored: "do not persist" },
    { passage_id: "b1", probability: 0.9, sentence_index: 1, sentence_text: "Second answer." },
  ],
};

test("normalizes queries and hashes page, revision, and provider without exposing them", async () => {
  assert.equal(normalizeHistoryQuery("  Find   THIS\nanswer  "), "find this answer");

  const first = await createHistorySubjectHash(jevSubject, webcrypto);
  const same = await createHistorySubjectHash({ ...jevSubject }, webcrypto);
  const changedUrl = await createHistorySubjectHash({ ...jevSubject, locator: "https://example.com/other" }, webcrypto);
  const changedRevision = await createHistorySubjectHash({ ...jevSubject, revision: "changed source" }, webcrypto);
  const changedProvider = await createHistorySubjectHash(layaSubject, webcrypto);

  assert.equal(first, same);
  assert.match(first, /^[a-f0-9]{64}$/);
  assert.notEqual(first, changedUrl);
  assert.notEqual(first, changedRevision);
  assert.notEqual(first, changedProvider);
  assert.doesNotMatch(first, /example|visible|jev/);
});

test("saves, finds, selects, and restores source-owned references without raw text", async () => {
  const storage = createMemoryHistoryStorage();
  const history = createSearchHistory(storage, { cryptoLike: webcrypto, now: () => 1234 });

  const saved = await history.save({ subject: jevSubject, query: "Which answer matters?", result: ambiguousResult, passages });
  assert.equal(saved.selected, null);
  assert.equal(saved.query, undefined);
  assert.equal(saved.normalizedQuery, "which answer matters?");
  assert.deepEqual(saved.result.matches[0], {
    passage_id: "b1",
    probability: 0.91,
    sentence_id: "s-first",
  });

  const storedText = JSON.stringify(await storage.entries());
  assert.doesNotMatch(storedText, /https:\/\/example\.com/);
  assert.doesNotMatch(storedText, /visible source revision/);
  assert.doesNotMatch(storedText, /First answer/);
  assert.doesNotMatch(storedText, /sentence_index/);
  assert.doesNotMatch(storedText, /Which answer matters\?/);
  assert.match(storedText, /which answer matters\?/);
  assert.doesNotMatch(storedText, /do not persist/);
  assert.ok((await storage.entries()).every(([key]) => key.startsWith(SEARCH_HISTORY_PREFIX)));

  assert.equal((await history.find(jevSubject, "  WHICH answer matters? ")).normalizedQuery, "which answer matters?");
  assert.equal((await history.findLatest(jevSubject)).normalizedQuery, "which answer matters?");
  assert.equal(await history.find(layaSubject, "Which answer matters?"), null);

  await history.select(jevSubject, "which answer matters?", { passage_id: "b1", sentence_id: "s-second" }, passages);
  const selected = await history.find(jevSubject, "Which answer matters?");
  const restored = restoreCachedSearch(selected, passages);

  assert.equal(restored.current, 1);
  assert.equal(restored.ambiguousSelection, 1);
  assert.equal(restored.query, "which answer matters?");
  assert.equal(restored.matches[1].sentence_text, "Second answer.");
  assert.equal(restored.selected.sentence.text, "Second answer.");
  assert.equal(selected.selected.sentence_id, "s-second");

  const reorderedPassages = [{
    ...passages[0],
    sentences: [
      { id: "s-second", index: 0, text: "Second answer." },
      { id: "s-first", index: 1, text: "First answer." },
    ],
  }];
  const reordered = restoreCachedSearch(selected, reorderedPassages);
  assert.equal(reordered.selected.sentence.text, "Second answer.");
  assert.equal(reordered.selected.match.sentence_index, 0);
});

test("sanitizes out-of-range probabilities before writing history", async () => {
  const storage = createMemoryHistoryStorage();
  const history = createSearchHistory(storage, { cryptoLike: webcrypto });
  const saved = await history.save({
    subject: jevSubject,
    query: "score",
    result: { ambiguous: false, matches: [{ passage_id: "b1", probability: 2, sentence_index: 0, sentence_text: "First answer." }] },
    passages,
  });

  assert.equal(saved, null);
  assert.equal(await history.find(jevSubject, "score"), null);
});

test("rejects stale and malformed records", async () => {
  const storage = createMemoryHistoryStorage();
  const history = createSearchHistory(storage, { cryptoLike: webcrypto });
  await history.save({ subject: jevSubject, query: "find", result: ambiguousResult, passages });
  const [[key, record]] = await storage.entries();

  assert.equal(restoreCachedSearch(record, [{ ...passages[0], id: "changed" }]), null);
  await storage.set(key, { ...record, normalizedQuery: "different query" });
  assert.equal(await history.find(jevSubject, "find"), null);
  await storage.set(key, { ...record, result: { ...record.result, matches: [{ ...record.result.matches[0], probability: "bad" }] } });
  assert.equal(await history.find(jevSubject, "find"), null);
});

test("finds the newest source-valid record when a newer record is stale", async () => {
  let timestamp = 100;
  const storage = createMemoryHistoryStorage();
  const history = createSearchHistory(storage, { cryptoLike: webcrypto, now: () => timestamp++ });
  await history.save({ subject: jevSubject, query: "first", result: ambiguousResult, passages });
  await history.save({
    subject: jevSubject,
    query: "stale",
    result: { ambiguous: false, ambiguity_gap: null, matches: [{ passage_id: "missing", probability: 0.99, sentence_index: 0, sentence_text: "Stale answer." }] },
    passages: [{ id: "missing", text: "Stale answer.", sentences: [{ id: "stale-sentence", index: 0, text: "Stale answer." }] }],
  });

  assert.equal((await history.findLatest(jevSubject)).normalizedQuery, "stale");
  assert.equal((await history.findLatest(jevSubject, passages)).normalizedQuery, "first");
});

test("resolves query-planned passages before validating the latest record", async () => {
  const storage = createMemoryHistoryStorage();
  const history = createSearchHistory(storage, { cryptoLike: webcrypto, now: () => 100 });
  await history.save({ subject: jevSubject, query: "which answer", result: ambiguousResult, passages });

  const initialWindow = [{ id: "boundary", text: "Boundary only.", sentences: [{ id: "boundary-sentence", index: 0, text: "Boundary only." }] }];
  const latest = await history.findLatest(jevSubject, initialWindow, async (query) => query === "which answer" ? passages : []);

  assert.equal(latest.normalizedQuery, "which answer");
});

test("orders same-millisecond writes by their serialized update time", async () => {
  const storage = createMemoryHistoryStorage();
  const history = createSearchHistory(storage, { cryptoLike: webcrypto, now: () => 100 });
  await history.save({ subject: jevSubject, query: "first", result: ambiguousResult, passages });
  await history.save({ subject: jevSubject, query: "second", result: ambiguousResult, passages });

  assert.equal((await history.findLatest(jevSubject)).normalizedQuery, "second");
});

test("derives a stable sentence ID when a source does not provide one", async () => {
  const sourceWithoutIds = [{
    id: "b-no-id",
    text: "Generated identity.",
    sentences: [{ index: 0, text: "Generated identity." }],
  }];
  const storage = createMemoryHistoryStorage();
  const history = createSearchHistory(storage, { cryptoLike: webcrypto });
  await history.save({
    subject: jevSubject,
    query: "generated",
    result: { ambiguous: false, matches: [{ passage_id: "b-no-id", probability: 0.9, sentence_index: 0, sentence_text: "Generated identity." }] },
    passages: sourceWithoutIds,
  });

  const record = await history.find(jevSubject, "generated");
  assert.match(record.result.matches[0].sentence_id, /^s[a-z0-9]+$/u);
  assert.equal(restoreCachedSearch(record, sourceWithoutIds).matches[0].sentence_text, "Generated identity.");
});

test("ignores an empty ambiguous record instead of restoring it", async () => {
  const storage = createMemoryHistoryStorage();
  const history = createSearchHistory(storage, { cryptoLike: webcrypto });
  await history.save({ subject: jevSubject, query: "empty", result: ambiguousResult, passages });
  const [[key, record]] = await storage.entries();
  await storage.set(key, { ...record, result: { ...record.result, matches: [], ambiguous: true } });

  assert.equal(await history.find(jevSubject, "empty"), null);
});

test("evicts oldest records and fails open when storage is unavailable", async () => {
  let timestamp = 100;
  const storage = createMemoryHistoryStorage();
  const history = createSearchHistory(storage, { cryptoLike: webcrypto, maxEntries: 2, now: () => timestamp++ });
  await history.save({ subject: jevSubject, query: "first", result: ambiguousResult, passages });
  await history.save({ subject: jevSubject, query: "second", result: ambiguousResult, passages });
  await history.save({ subject: jevSubject, query: "third", result: ambiguousResult, passages });

  assert.equal((await storage.entries()).length, 2);
  assert.equal(await history.find(jevSubject, "first"), null);
  assert.equal((await history.findLatest(jevSubject)).normalizedQuery, "third");

  const failure = async () => { throw new Error("storage unavailable"); };
  const unavailable = createSearchHistory({ entries: failure, get: failure, set: failure, remove: failure }, { cryptoLike: webcrypto });
  assert.equal(await unavailable.find(jevSubject, "find"), null);
  assert.equal(await unavailable.findLatest(jevSubject), null);
  assert.equal(await unavailable.save({ subject: jevSubject, query: "find", result: ambiguousResult, passages }), null);
  assert.equal(await unavailable.select(jevSubject, "find", { passage_id: "b1", sentence_index: 0, sentence_text: "First answer." }, passages), null);
});

test("adapts Chrome storage local operations", async () => {
  const values = {};
  const storage = createChromeHistoryStorage({
    async get(key) {
      if (key === null) return { ...values };
      return Object.hasOwn(values, key) ? { [key]: values[key] } : {};
    },
    async set(next) { Object.assign(values, next); },
    async remove(keys) { for (const key of [].concat(keys)) delete values[key]; },
  });

  await storage.set("record", { value: 1 });
  assert.deepEqual(await storage.get("record"), { value: 1 });
  assert.deepEqual(await storage.entries(), [["record", { value: 1 }]]);
  await storage.remove(["record"]);
  assert.equal(await storage.get("record"), null);
});

test("adapts localStorage without exposing its raw keys", async () => {
  const values = new Map();
  const storage = createLocalHistoryStorage({
    get length() { return values.size; },
    key(index) { return [...values.keys()][index] ?? null; },
    getItem(key) { return values.has(key) ? values.get(key) : null; },
    setItem(key, value) { values.set(key, String(value)); },
    removeItem(key) { values.delete(key); },
  });

  await storage.set(`${SEARCH_HISTORY_PREFIX}hash.query`, { value: 1 });
  assert.deepEqual(await storage.get(`${SEARCH_HISTORY_PREFIX}hash.query`), { value: 1 });
  assert.deepEqual(await storage.entries(), [[`${SEARCH_HISTORY_PREFIX}hash.query`, { value: 1 }]]);
  await storage.remove([`${SEARCH_HISTORY_PREFIX}hash.query`]);
  assert.equal(await storage.get(`${SEARCH_HISTORY_PREFIX}hash.query`), null);
});

function createMemoryHistoryStorage() {
  const values = new Map();
  return {
    async entries() { return [...values.entries()].map(([key, value]) => [key, structuredClone(value)]); },
    async get(key) { return values.has(key) ? structuredClone(values.get(key)) : null; },
    async set(key, value) { values.set(key, structuredClone(value)); },
    async remove(keys) { for (const key of keys) values.delete(key); },
  };
}
