import { findSourceMatch } from "./state.js";

export const SEARCH_HISTORY_PREFIX = "context-atlas.history.v1.";
export const MAX_SEARCH_HISTORY_ENTRIES = 100;

const HISTORY_VERSION = 1;
const PROVIDERS = new Set(["jev", "laya"]);
const SURFACES = new Set(["extension", "standalone"]);

export function normalizeHistoryQuery(value) {
	return String(value ?? "").normalize("NFKC").replace(/\s+/gu, " ").trim().toLowerCase();
}

export async function createHistorySubjectHash(subject, cryptoLike = globalThis.crypto) {
	const canonical = JSON.stringify([
		HISTORY_VERSION,
		String(subject?.surface ?? ""),
		String(subject?.locator ?? ""),
		String(subject?.revision ?? ""),
		String(subject?.provider ?? ""),
	]);
	return sha256Hex(canonical, cryptoLike);
}

export function createChromeHistoryStorage(storageArea) {
	return {
		async entries() {
			const values = await storageArea.get(null);
			return Object.entries(values || {});
		},
		async get(key) {
			const values = await storageArea.get(key);
			return Object.hasOwn(values || {}, key) ? values[key] : null;
		},
		async set(key, value) {
			await storageArea.set({ [key]: value });
		},
		async remove(keys) {
			if (keys.length) await storageArea.remove(keys);
		},
	};
}

export function createLocalHistoryStorage(storageArea = null) {
	function resolveStorage() {
		if (storageArea) return storageArea;
		try {
			return globalThis.localStorage;
		} catch (_error) {
			return null;
		}
	}

	return {
		async entries() {
			const target = resolveStorage();
			if (!target) throw new Error("localStorage is unavailable");
			const entries = [];
			for (let index = 0; index < target.length; index += 1) {
				const key = target.key(index);
				if (!key) continue;
				const raw = target.getItem(key);
				if (raw === null) continue;
				try {
					entries.push([key, JSON.parse(raw)]);
				} catch (_error) {
					entries.push([key, null]);
				}
			}
			return entries;
		},
		async get(key) {
			const target = resolveStorage();
			if (!target) throw new Error("localStorage is unavailable");
			const raw = target.getItem(key);
			if (raw === null) return null;
			try {
				return JSON.parse(raw);
			} catch (_error) {
				return null;
			}
		},
		async set(key, value) {
			const target = resolveStorage();
			if (!target) throw new Error("localStorage is unavailable");
			target.setItem(key, JSON.stringify(value));
		},
		async remove(keys) {
			const target = resolveStorage();
			if (!target) throw new Error("localStorage is unavailable");
			for (const key of keys) target.removeItem(key);
		},
	};
}

export function createSearchHistory(storage, {
	maxEntries = MAX_SEARCH_HISTORY_ENTRIES,
	now = Date.now,
	cryptoLike = globalThis.crypto,
} = {}) {
	const retention = Number.isInteger(maxEntries) && maxEntries > 0
		? maxEntries
		: MAX_SEARCH_HISTORY_ENTRIES;
	let writeQueue = Promise.resolve();

	async function recordIdentity(subject, query) {
		const surface = String(subject?.surface ?? "");
		const provider = String(subject?.provider ?? "");
		const normalizedQuery = normalizeHistoryQuery(query);
		if (!SURFACES.has(surface) || !PROVIDERS.has(provider) || !normalizedQuery) return null;
		const subjectHash = await createHistorySubjectHash(subject, cryptoLike);
		const queryHash = await sha256Hex(normalizedQuery, cryptoLike);
		return {
			subjectHash,
			surface,
			provider,
			normalizedQuery,
			key: `${SEARCH_HISTORY_PREFIX}${subjectHash}.${queryHash}`,
		};
	}

	async function find(subject, query) {
		try {
			const identity = await recordIdentity(subject, query);
			if (!identity) return null;
			const record = normalizeRecord(await storage.get(identity.key));
			if (!matchesIdentity(record, identity)) return null;
			return record;
		} catch (_error) {
			return null;
		}
	}

	async function findLatest(subject, sourcePassages = null, resolvePassages = null) {
		try {
			const surface = String(subject?.surface ?? "");
			const provider = String(subject?.provider ?? "");
			if (!SURFACES.has(surface) || !PROVIDERS.has(provider)) return null;
			const subjectHash = await createHistorySubjectHash(subject, cryptoLike);
			const records = [];
			for (const [key, value] of await storage.entries()) {
				if (!key.startsWith(SEARCH_HISTORY_PREFIX)) continue;
				const record = normalizeRecord(value);
				if (!record || record.subjectHash !== subjectHash || record.surface !== surface || record.provider !== provider) continue;
				const queryHash = await sha256Hex(record.normalizedQuery, cryptoLike);
				if (key !== `${SEARCH_HISTORY_PREFIX}${record.subjectHash}.${queryHash}`) continue;
				records.push({ key, record });
			}
			records.sort((left, right) => right.record.updatedAt - left.record.updatedAt || right.key.localeCompare(left.key));
			for (const { record } of records) {
				if (!Array.isArray(sourcePassages) || restoreCachedSearch(record, sourcePassages)) return record;
				if (typeof resolvePassages === "function") {
					try {
						const resolvedPassages = await resolvePassages(record.normalizedQuery);
						if (restoreCachedSearch(record, resolvedPassages)) return record;
					} catch (_error) {
						// A source resolver is advisory. Try the next record.
					}
				}
			}
			return null;
		} catch (_error) {
			return null;
		}
	}

	function enqueue(operation) {
		const pending = writeQueue.catch(() => null).then(operation).catch(() => null);
		writeQueue = pending;
		return pending;
	}

	function save({ subject, query, result, selected = null, passages = null }) {
		return enqueue(async () => {
			const identity = await recordIdentity(subject, query);
			if (!identity) return null;
			const sanitizedResult = sanitizeResult(result, passages);
			if (!sanitizedResult) return null;
			const existing = normalizeRecord(await storage.get(identity.key));
			const timestamp = await nextUpdatedAt(storage, now);
			const record = {
				version: HISTORY_VERSION,
				surface: identity.surface,
				provider: identity.provider,
				subjectHash: identity.subjectHash,
				normalizedQuery: identity.normalizedQuery,
				result: sanitizedResult,
				selected: sanitizeSelection(selected, passages),
				createdAt: existing?.createdAt ?? timestamp,
				updatedAt: timestamp,
			};
			await storage.set(identity.key, record);
			await pruneHistory(storage, retention);
			return record;
		});
	}

	function select(subject, query, selected, passages = null) {
		return enqueue(async () => {
			const identity = await recordIdentity(subject, query);
			if (!identity) return null;
			const existing = normalizeRecord(await storage.get(identity.key));
			if (!existing || !matchesIdentity(existing, identity)) return null;
			const sanitizedSelection = sanitizeSelection(selected, passages);
			if (!sanitizedSelection) return null;
			const record = {
				...existing,
				selected: sanitizedSelection,
				updatedAt: await nextUpdatedAt(storage, now),
			};
			await storage.set(identity.key, record);
			await pruneHistory(storage, retention);
			return record;
		});
	}

	return { find, findLatest, save, select };
}

export function restoreCachedSearch(record, passages) {
	const normalized = normalizeRecord(record);
	if (!normalized || !Array.isArray(passages)) return null;
	const matches = normalized.result.matches.map((match) => restoreSourceMatch(match, passages));
	if (matches.some((match) => match === null)) return null;
	const result = { ...normalized.result, matches };
	const selectedIndex = normalized.selected
		? matches.findIndex((match) => (
			match.passage_id === normalized.selected.passage_id
			&& match.sentence_id === normalized.selected.sentence_id
		))
		: -1;
	const current = selectedIndex >= 0 ? selectedIndex : 0;
	return {
		query: normalized.normalizedQuery,
		result,
		matches,
		current,
		ambiguousSelection: result.ambiguous && selectedIndex < 0 ? null : current,
		selected: selectedIndex >= 0 ? findSourceMatch({ matches }, passages, selectedIndex) : null,
	};
}

function restoreSourceMatch(match, passages) {
	const passage = passages.find((item) => item?.id === match.passage_id);
	const sentence = passage?.sentences?.map((item, index) => ({ item, index }))
		.find(({ item, index }) => sentenceIdentity(passage, item, index) === match.sentence_id)?.item;
	if (!passage || !sentence || typeof sentence.text !== "string" || !sentence.text) return null;
	return { ...match, sentence_index: Number.isInteger(sentence.index) ? sentence.index : passage.sentences.indexOf(sentence), sentence_text: sentence.text };
}

function matchesIdentity(record, identity) {
	return Boolean(
		record
		&& record.subjectHash === identity.subjectHash
		&& record.surface === identity.surface
		&& record.provider === identity.provider
		&& record.normalizedQuery === identity.normalizedQuery,
	);
}

async function sha256Hex(value, cryptoLike) {
	if (!cryptoLike?.subtle?.digest) throw new Error("Web Crypto is unavailable");
	const digest = await cryptoLike.subtle.digest("SHA-256", new TextEncoder().encode(String(value)));
	return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function sanitizeResult(result, passages) {
	if (!Array.isArray(passages) || !passages.length) return null;
	const matches = (Array.isArray(result?.matches) ? result.matches : [])
		.map((match) => {
			const source = findSourceSentence(match, passages);
			const probability = Number(match?.probability);
			if (!source || !Number.isFinite(probability) || probability < 0 || probability > 1) return null;
			return {
				passage_id: source.passage.id,
				sentence_id: sentenceIdentity(source.passage, source.sentence, source.index),
				probability,
			};
		})
		.filter(Boolean);
	if (!matches.length) return null;
	const ambiguityGap = Number(result?.ambiguity_gap);
	const ambiguous = result?.ambiguous === true;
	if (ambiguous && matches.length < 2) return null;
	return {
		matches,
		ambiguous,
		ambiguity_gap: Number.isFinite(ambiguityGap) ? ambiguityGap : null,
	};
}

function sanitizeSelection(selected, passages) {
	const source = findSourceSentence(selected, passages);
	if (!source) return null;
	return {
		passage_id: source.passage.id,
		sentence_id: sentenceIdentity(source.passage, source.sentence, source.index),
	};
}

function normalizeRecord(value) {
	if (!value || value.version !== HISTORY_VERSION) return null;
	if (!/^[a-f0-9]{64}$/u.test(value.subjectHash || "")) return null;
	if (!SURFACES.has(value.surface) || !PROVIDERS.has(value.provider)) return null;
	if (Object.hasOwn(value, "query")) return null;
	if (typeof value.normalizedQuery !== "string" || !value.normalizedQuery) return null;
	if (normalizeHistoryQuery(value.normalizedQuery) !== value.normalizedQuery) return null;
	const createdAt = Number(value.createdAt);
	const updatedAt = Number(value.updatedAt);
	if (!Number.isFinite(createdAt) || !Number.isFinite(updatedAt)) return null;
	const result = normalizeStoredResult(value.result);
	if (!result) return null;
	const selected = value.selected === null ? null : normalizeStoredSelection(value.selected);
	if (value.selected !== null && !selected) return null;
	return {
		version: HISTORY_VERSION,
		surface: value.surface,
		provider: value.provider,
		subjectHash: value.subjectHash,
		normalizedQuery: value.normalizedQuery,
		result,
		selected,
		createdAt,
		updatedAt,
	};
}

function normalizeStoredResult(value) {
	if (!value || typeof value !== "object" || !Array.isArray(value.matches) || typeof value.ambiguous !== "boolean") return null;
	if (!value.matches.length || (value.ambiguous && value.matches.length < 2)) return null;
	const ambiguityGap = value.ambiguity_gap === null ? null : Number(value.ambiguity_gap);
	if (value.ambiguity_gap !== null && !Number.isFinite(ambiguityGap)) return null;
	const matches = value.matches.map((match) => {
		if (
			typeof match?.passage_id !== "string"
			|| !match.passage_id
			|| typeof match?.sentence_id !== "string"
			|| !match.sentence_id
			|| Object.hasOwn(match, "sentence_index")
			|| !Number.isFinite(match?.probability)
			|| match.probability < 0
			|| match.probability > 1
		) return null;
		return {
			passage_id: match.passage_id,
			sentence_id: match.sentence_id,
			probability: match.probability,
		};
	});
	if (matches.some((match) => match === null)) return null;
	return { matches, ambiguous: value.ambiguous, ambiguity_gap: ambiguityGap };
}

function normalizeStoredSelection(value) {
	if (
		!value
		|| typeof value.passage_id !== "string"
		|| !value.passage_id
		|| typeof value.sentence_id !== "string"
		|| !value.sentence_id
		|| Object.hasOwn(value, "sentence_index")
	) return null;
	return { passage_id: value.passage_id, sentence_id: value.sentence_id };
}

async function pruneHistory(storage, maxEntries) {
	const records = (await storage.entries())
		.filter(([key]) => key.startsWith(SEARCH_HISTORY_PREFIX))
		.map(([key, value]) => ({ key, record: normalizeRecord(value) }))
		.filter((item) => item.record)
		.sort((left, right) => right.record.updatedAt - left.record.updatedAt || right.key.localeCompare(left.key));
	const staleKeys = records.slice(maxEntries).map((item) => item.key);
	if (staleKeys.length) await storage.remove(staleKeys);
}

async function nextUpdatedAt(storage, now) {
	const requested = Number(now());
	const base = Number.isFinite(requested) ? requested : Date.now();
	let latest = null;
	for (const [key, value] of await storage.entries()) {
		if (!key.startsWith(SEARCH_HISTORY_PREFIX)) continue;
		const record = normalizeRecord(value);
		if (record && (latest === null || record.updatedAt > latest)) latest = record.updatedAt;
	}
	return Math.max(base, latest === null ? base : latest + 1);
}

function findSourceSentence(match, passages) {
	if (!Array.isArray(passages) || typeof match?.passage_id !== "string" || !match.passage_id) return null;
	const passage = passages.find((item) => item?.id === match.passage_id);
	if (!passage || !Array.isArray(passage.sentences)) return null;
	const sentenceId = typeof match.sentence_id === "string" && match.sentence_id ? match.sentence_id : null;
	const sentenceIndex = Number(match.sentence_index);
	const sentenceText = typeof match.sentence_text === "string" ? match.sentence_text : null;
	const source = passage.sentences.map((sentence, index) => ({ sentence, index })).find(({ sentence, index }) => (
		(sentenceId
			? sentenceIdentity(passage, sentence, index) === sentenceId
			: Number(sentence?.index ?? index) === sentenceIndex
				&& (!sentenceText || sentence?.text === sentenceText))
	));
	return source ? { passage, sentence: source.sentence, index: source.index } : null;
}

function sentenceIdentity(passage, sentence, index) {
	if (typeof sentence?.id === "string" && sentence.id) return sentence.id;
	const normalized = normalizeHistoryQuery(sentence?.text);
	let occurrence = 0;
	for (let prior = 0; prior < index; prior += 1) {
		if (normalizeHistoryQuery(passage?.sentences?.[prior]?.text) === normalized) occurrence += 1;
	}
	return `s${hashString(`${passage?.id || ""}\u0000${normalized}\u0000${occurrence}`).toString(36)}`;
}

function hashString(value) {
	let hash = 2166136261;
	for (let index = 0; index < value.length; index += 1) {
		hash ^= value.charCodeAt(index);
		hash = Math.imul(hash, 16777619);
	}
	return hash >>> 0;
}
