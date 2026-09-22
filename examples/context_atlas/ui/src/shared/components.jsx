import React, { useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronLeft, ChevronRight, KeyRound, RefreshCw, Route, X } from "lucide-react";

import { findSentenceOffset } from "./contracts.js";
import {
  buildProvenanceThread,
  findSourceMatch,
  invalidateProviderRequest,
  isCurrentRequest,
  normalizeProvider,
  normalizeSearchError,
  sourceFingerprint,
  validSourceMatches,
} from "./state.js";

function BusyProgress({ label = "Loading", value = 50, showValue = false }) {
  const progress = Math.min(100, Math.max(0, Number.isFinite(Number(value)) ? Number(value) : 50));
  return <>
    <span className="context-atlas-loading-progress" role="progressbar" aria-label={label} aria-valuemin="0" aria-valuemax="100" aria-valuenow={progress} style={{ "--context-atlas-progress": progress / 100 }} />
    {showValue ? <span className="context-atlas-progress-value">{progress}%</span> : null}
  </>;
}

export function ContextAtlasSurface({
  adapter,
  source,
  surface = "standalone",
  sourceEditor = null,
  sourceStatus = null,
  onRefreshSource = null,
  prepareSearch = null,
  onClose = null,
}) {
  const sourcePassages = Array.isArray(source?.passages) ? source.passages : [];
  const [activePassages, setActivePassages] = useState(sourcePassages);
  const passages = activePassages;
  const isExtension = surface === "extension";
  const providerEnabled = typeof adapter?.getProvider === "function" && typeof adapter?.setProvider === "function";
  const [provider, setProvider] = useState("laya");
  const [providerBusy, setProviderBusy] = useState(false);
  const providerSelectionVersionRef = useRef(0);
  const fingerprint = sourceFingerprint(sourcePassages, source?.url);
  const requestRef = useRef(0);
  const previousFingerprint = useRef(fingerprint);
  const lastSearchRef = useRef(null);
  const [apiKey, setApiKey] = useState("");
  const [keyConfigured, setKeyConfigured] = useState(null);
  const [keyBusy, setKeyBusy] = useState(false);
  const [keyAction, setKeyAction] = useState(null);
  const keyMutationVersionRef = useRef(0);
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [matches, setMatches] = useState([]);
  const [current, setCurrent] = useState(0);
  const [ambiguousSelection, setAmbiguousSelection] = useState(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const [searchProgress, setSearchProgress] = useState(0);
  const [status, setStatus] = useState({ kind: "idle", message: "Ready when you are.", retryable: false });

  useEffect(() => {
    if (!providerEnabled) return undefined;
    let active = true;
    const selectionVersion = providerSelectionVersionRef.current;
    (async () => {
      try {
        const savedProvider = normalizeProvider(await adapter.getProvider());
        if (active && selectionVersion === providerSelectionVersionRef.current) setProvider(savedProvider);
      } catch (_error) {
        if (active && selectionVersion === providerSelectionVersionRef.current) setProvider("laya");
      }
    })();
    return () => { active = false; };
  }, [adapter, providerEnabled]);

  useEffect(() => {
    if ((providerEnabled && provider !== "jev") || typeof adapter?.getKeyStatus !== "function") return undefined;
    let active = true;
    const mutationVersion = keyMutationVersionRef.current;
    (async () => {
      try {
        const payload = await adapter.getKeyStatus();
        if (active && mutationVersion === keyMutationVersionRef.current) {
          const configured = payload?.configured === true;
          setKeyConfigured(configured);
          setStatus({
            kind: "idle",
            message: configured ? "Ready to search with Cloud." : "Get your Cloud key at console.typesafe.ai.",
            retryable: false,
          });
        }
      } catch (error) {
        if (active && mutationVersion === keyMutationVersionRef.current) {
          setKeyConfigured(false);
          setStatus({ kind: "error", ...normalizeSearchError(error) });
        }
      }
    })();
    return () => { active = false; };
  }, [adapter, providerEnabled, provider]);

  useEffect(() => {
    if (previousFingerprint.current === fingerprint) return;
    previousFingerprint.current = fingerprint;
    invalidateProviderRequest(requestRef);
    lastSearchRef.current = null;
    setActivePassages(sourcePassages);
    setResult(null);
    setMatches([]);
    setCurrent(0);
    setAmbiguousSelection(null);
    setTraceOpen(false);
    setSearchProgress(0);
    setStatus({ kind: "idle", message: "The page changed. Search again.", retryable: false });
  }, [fingerprint, sourcePassages]);

  async function chooseProvider(nextProvider) {
    const next = normalizeProvider(nextProvider);
    if (!providerEnabled || next === provider || providerBusy) return;
    const selectionVersion = providerSelectionVersionRef.current + 1;
    providerSelectionVersionRef.current = selectionVersion;
    let accessPromise = Promise.resolve({ granted: true });
    if (next === "jev" && typeof adapter?.ensureProviderAccess === "function") {
      try {
        accessPromise = adapter.ensureProviderAccess(next);
      } catch (error) {
        setStatus({ kind: "error", ...normalizeSearchError(error) });
        return;
      }
    }
    setProviderBusy(true);
    setSearchProgress(0);
    setStatus({ kind: "loading", message: "Switching search…", retryable: false });
    try {
      await accessPromise;
      await adapter.setProvider(next);
      if (selectionVersion !== providerSelectionVersionRef.current) return;
      invalidateProviderRequest(requestRef, () => {
        setResult(null);
        setMatches([]);
        setCurrent(0);
        setAmbiguousSelection(null);
        setTraceOpen(false);
      });
      setProvider(next);
      setStatus({
        kind: "idle",
        message: next === "laya"
          ? "Local search stays in this browser. The model prepares on first use."
          : "Cloud search is ready when you save an access key.",
        retryable: false,
      });
    } catch (error) {
      if (selectionVersion === providerSelectionVersionRef.current) {
        setStatus({ kind: "error", ...normalizeSearchError(error) });
      }
    } finally {
      if (selectionVersion === providerSelectionVersionRef.current) setProviderBusy(false);
    }
  }

  const activeProvider = providerEnabled ? provider : "jev";
  const focusedIndex = result?.ambiguous && ambiguousSelection === null ? -1 : current;
  const focused = focusedIndex >= 0 ? matches[focusedIndex] || null : null;
  const focusedSource = useMemo(
    () => findSourceMatch({ matches: focused ? [focused] : [] }, passages),
    [focused, passages],
  );

  async function saveKey(event) {
    event?.preventDefault();
    const cleanKey = apiKey.trim();
    if (!cleanKey) {
      setStatus({ kind: "error", message: "Enter a Cloud access key before saving.", retryable: false });
      return;
    }
    const mutationVersion = keyMutationVersionRef.current + 1;
    keyMutationVersionRef.current = mutationVersion;
    setKeyBusy(true);
    setKeyAction("save");
    setSearchProgress(0);
    setStatus({ kind: "loading", message: "Saving your access key…", retryable: false });
    try {
      const payload = await adapter.saveKey(cleanKey);
      if (mutationVersion !== keyMutationVersionRef.current) return;
      setApiKey("");
      setKeyConfigured(payload?.configured === true);
      setStatus({ kind: "success", message: "Access key saved. Cloud search is ready.", retryable: false });
    } catch (error) {
      if (mutationVersion === keyMutationVersionRef.current) {
        setStatus({ kind: "error", ...normalizeSearchError(error) });
      }
    } finally {
      if (mutationVersion === keyMutationVersionRef.current) {
        setKeyBusy(false);
        setKeyAction(null);
      }
    }
  }

  async function clearKey() {
    const mutationVersion = keyMutationVersionRef.current + 1;
    keyMutationVersionRef.current = mutationVersion;
    setKeyBusy(true);
    setKeyAction("clear");
    setSearchProgress(0);
    setStatus({ kind: "loading", message: "Removing your saved key…", retryable: false });
    try {
      const payload = await adapter.clearKey();
      if (mutationVersion !== keyMutationVersionRef.current) return;
      setKeyConfigured(payload?.configured === true);
      setStatus({ kind: "success", message: "Saved access key cleared. Cloud search is paused.", retryable: false });
    } catch (error) {
      if (mutationVersion === keyMutationVersionRef.current) {
        setStatus({ kind: "error", ...normalizeSearchError(error) });
      }
    } finally {
      if (mutationVersion === keyMutationVersionRef.current) {
        setKeyBusy(false);
        setKeyAction(null);
      }
    }
  }

  async function searchSource(event, queryOverride = null) {
    event?.preventDefault();
    if (activeProvider === "jev" && keyConfigured !== true) {
      setStatus({ kind: "error", message: "Save your Cloud access key before searching.", retryable: false });
      return;
    }
    const cleanQuery = String(queryOverride ?? query).trim();
    if (!cleanQuery) {
      setStatus({ kind: "error", message: "Enter a question first.", retryable: false });
      return;
    }
    let accessPromise = Promise.resolve({ granted: true });
    if (activeProvider === "laya" && typeof adapter?.ensureProviderAccess === "function") {
      try {
        accessPromise = adapter.ensureProviderAccess(activeProvider);
      } catch (error) {
        setStatus({ kind: "error", ...normalizeSearchError(error) });
        return;
      }
    }
    setSearchProgress(12);
    setStatus({ kind: "loading", message: "Finding results…", retryable: false });
    let searchPassages = sourcePassages;
    try {
      await accessPromise;
      if (typeof prepareSearch === "function") {
        const prepared = await prepareSearch(cleanQuery);
        if (prepared && Array.isArray(prepared.passages)) {
          searchPassages = prepared.passages;
          setActivePassages(searchPassages);
          setSearchProgress(32);
        }
      }
    } catch (error) {
      setSearchProgress(0);
      setStatus({ kind: "error", ...normalizeSearchError(error) });
      return;
    }
    if (!searchPassages.length) {
      setSearchProgress(0);
      setStatus({ kind: "error", message: "No page text is ready yet. Refresh from the current page.", retryable: true });
      return;
    }
    const requestId = requestRef.current + 1;
    requestRef.current = requestId;
    const sourceAtRequest = sourceFingerprint(searchPassages);
    lastSearchRef.current = { query: cleanQuery, sourceAtRequest };
    setResult(null);
    setMatches([]);
    setCurrent(0);
    setAmbiguousSelection(null);
    setTraceOpen(false);
    setSearchProgress(55);
    try {
      const payload = await adapter.search(cleanQuery, searchPassages);
      if (!isCurrentRequest(requestId, requestRef.current) || sourceFingerprint(searchPassages) !== sourceAtRequest) return;
      const validMatches = validSourceMatches(payload, searchPassages);
      const returnedMatches = Array.isArray(payload?.matches) ? payload.matches : [];
      if (returnedMatches.length && !validMatches.length) {
        setSearchProgress(0);
        setStatus({ kind: "error", message: "The page changed while searching. Search again.", retryable: true });
        return;
      }
      const normalizedMatches = validMatches.map((item) => item.match);
      const ambiguous = isAmbiguous(normalizedMatches, payload?.threshold);
      const normalizedResult = { ...payload, ambiguous, matches: normalizedMatches };
      setResult(normalizedResult);
      setMatches(normalizedResult.matches);
      if (!ambiguous) focusResult(normalizedMatches, searchPassages, 0);
      setSearchProgress(100);
      setStatus({
        kind: "success",
        message: ambiguous
          ? "We found a few close matches. Choose one."
          : `${normalizedResult.matches.length} source-backed result${normalizedResult.matches.length === 1 ? "" : "s"}.`,
        retryable: false,
      });
    } catch (error) {
      if (!isCurrentRequest(requestId, requestRef.current)) return;
      setSearchProgress(0);
      setStatus({ kind: "error", ...normalizeSearchError(error) });
    }
  }

  function focusResult(candidateMatches, candidatePassages, index) {
    const selected = findSourceMatch({ matches: candidateMatches }, candidatePassages, index);
    if (selected) adapter.focusMatch?.(selected);
  }

  function selectResult(index) {
    const nextIndex = Math.max(0, Math.min(index, matches.length - 1));
    setCurrent(nextIndex);
    setAmbiguousSelection(nextIndex);
    setTraceOpen(false);
    focusResult(matches, passages, nextIndex);
  }

  function retry() {
    const previous = lastSearchRef.current?.query ?? query;
    setQuery(previous);
    void searchSource(null, previous);
  }

  const provenance = buildProvenanceThread(
    { matches: focused ? [focused] : [] },
    passages,
    query,
  );
  const showSearchProgress = searchProgress > 0 && (status.kind === "loading" || status.kind === "success");

  return (
    <div className={`context-atlas context-atlas-${surface}`} data-testid="context-atlas-surface">
      <header className="context-atlas-header">
        <div>
          {surface !== "extension" ? <p className="context-atlas-eyebrow">CONTEXT ATLAS</p> : null}
          <h1>{surface === "extension" ? "Find what matters" : "Find what matters in your source."}</h1>
          <p className="context-atlas-lede">Ask in plain language. Find the passage that answers it.</p>
        </div>
        {onClose ? <button className="context-atlas-icon-button" type="button" onClick={onClose} aria-label="Close"><X size={18} /></button> : null}
      </header>

      {providerEnabled ? <section className="context-atlas-card context-atlas-provider-card" aria-label="Search provider">
        <div className="context-atlas-provider-tabs" role="tablist" aria-label="Search provider">
          <button id="context-atlas-provider-tab-laya" className="context-atlas-provider-tab" type="button" role="tab" aria-selected={provider === "laya"} aria-controls="context-atlas-provider-panel-laya" tabIndex={provider === "laya" ? 0 : -1} onClick={() => void chooseProvider("laya")} disabled={providerBusy}>Local</button>
          <button id="context-atlas-provider-tab-jev" className="context-atlas-provider-tab" type="button" role="tab" aria-selected={provider === "jev"} aria-controls="context-atlas-provider-panel-jev" tabIndex={provider === "jev" ? 0 : -1} onClick={() => void chooseProvider("jev")} disabled={providerBusy}>Cloud</button>
        </div>
        <div id="context-atlas-provider-panel-laya" className="context-atlas-provider-status" role="tabpanel" aria-labelledby="context-atlas-provider-tab-laya" hidden={provider !== "laya"}>
          <p className="context-atlas-eyebrow">LOCAL</p>
          <p className="context-atlas-provider-status-title">Private browser search</p>
          <p className="context-atlas-card-copy">Runs in your browser.</p>
        </div>
      </section> : null}

      {activeProvider === "jev" && typeof adapter?.getKeyStatus === "function" ? <section className="context-atlas-card context-atlas-key-card" role={providerEnabled ? "tabpanel" : undefined} id={providerEnabled ? "context-atlas-provider-panel-jev" : undefined} aria-labelledby={providerEnabled ? "context-atlas-provider-tab-jev" : "context-atlas-key-heading"} hidden={providerEnabled && provider !== "jev"}>
        <div className="context-atlas-section-heading">
          <div><p className="context-atlas-eyebrow">CLOUD ACCESS</p><h2 id="context-atlas-key-heading">Cloud access key</h2></div>
          <span className={`context-atlas-badge ${keyConfigured ? "is-success" : keyConfigured === false ? "is-muted" : "is-warn"}`}>{keyConfigured ? "Saved locally" : keyConfigured === false ? "Not set up" : "Checking…"}</span>
        </div>
        <p className="context-atlas-card-copy" id="context-atlas-key-help">Your key stays local.</p>
        <form className="context-atlas-key-form" onSubmit={saveKey}>
          <label htmlFor="context-atlas-api-key">Cloud access key</label>
          <div className="context-atlas-input-row">
            <div className="context-atlas-input-with-icon"><KeyRound size={16} aria-hidden="true" /><input id="context-atlas-api-key" type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} autoComplete="off" spellCheck="false" placeholder="Paste your access key" aria-describedby="context-atlas-key-help" /></div>
            <button className="context-atlas-primary-button" type="submit" disabled={keyBusy}><span className="context-atlas-button-content">{keyAction === "save" ? <BusyProgress label="Saving access key" /> : <Check size={16} aria-hidden="true" />}Save key</span></button>
          </div>
        </form>
        <div className="context-atlas-key-footer"><p className="context-atlas-field-note" role="status" aria-live="polite">{keyConfigured ? "Ready to search with Cloud." : <>Get your Cloud key at <a className="context-atlas-inline-link" href="https://console.typesafe.ai/login" target="_blank" rel="noopener noreferrer">console.typesafe.ai</a>.</>}</p><button className="context-atlas-quiet-button" type="button" onClick={clearKey} disabled={keyBusy || !keyConfigured}><span className="context-atlas-button-content">{keyAction === "clear" ? <BusyProgress label="Removing saved key" /> : null}Clear saved key</span></button></div>
      </section> : null}

      <section className="context-atlas-card context-atlas-search-card" aria-labelledby="context-atlas-search-heading">
        <div className="context-atlas-section-heading"><div><p className="context-atlas-eyebrow">SEARCH</p><h2 id="context-atlas-search-heading">Ask a question</h2></div><span className="context-atlas-source-count">{passages.length} {passages.length === 1 ? "section" : "sections"}</span></div>
        {sourceEditor}
        {sourceStatus ? <div className={`context-atlas-source-status is-${sourceStatus.kind || "idle"}`} role="status" aria-live="polite" aria-busy={sourceStatus.kind === "loading"}><span className="context-atlas-status-message">{sourceStatus.kind === "loading" ? <BusyProgress label={sourceStatus.message} /> : null}<span>{sourceStatus.message}</span></span>{onRefreshSource ? <button className="context-atlas-quiet-button" type="button" onClick={() => void onRefreshSource()} disabled={sourceStatus.kind === "loading"}><RefreshCw size={14} aria-hidden="true" />Refresh from current page</button> : null}</div> : null}
        <form className="context-atlas-query-form" onSubmit={searchSource}>
          <label htmlFor="context-atlas-query">What do you want to find?</label>
          <div className="context-atlas-input-row"><input id="context-atlas-query" value={query} maxLength={400} onChange={(event) => setQuery(event.target.value)} placeholder="Ask about this page" autoComplete="off" /><button className="context-atlas-primary-button" type="submit" disabled={status.kind === "loading"}><span className="context-atlas-button-content"><Route size={16} aria-hidden="true" />Find it</span></button></div>
          <div className="context-atlas-query-footer"><span className="context-atlas-field-note">{query.length} / 400</span></div>
        </form>
        <div className={`context-atlas-status is-${status.kind}`} role="status" aria-live="polite" aria-busy={status.kind === "loading"}><span className="context-atlas-status-message">{showSearchProgress ? <span className="context-atlas-status-progress"><BusyProgress label={status.message} value={status.kind === "success" ? 100 : searchProgress} showValue /></span> : null}<span className="context-atlas-status-copy">{status.kind === "loading" ? <progress className="context-atlas-status-spinner" aria-hidden="true" /> : null}<span>{status.message}</span></span></span>{status.retryable ? <button className="context-atlas-retry-button" type="button" onClick={retry}><RefreshCw size={14} aria-hidden="true" />Retry</button> : null}</div>
      </section>

      <div className="context-atlas-results-layout">
        <article className="context-atlas-card context-atlas-focused-card" aria-live="polite">
          <div className="context-atlas-section-heading"><div><p className="context-atlas-eyebrow">RESULT</p><h2>{focused ? "Best match" : result?.ambiguous ? "Choose a match" : "Your result will appear here"}</h2></div><span className="context-atlas-field-note">{focused ? `${confidence(focused.probability)}% relevant` : ""}</span></div>
          {result?.ambiguous && ambiguousSelection === null ? <div className="context-atlas-ambiguity" role="group" aria-label="Close source matches"><p className="context-atlas-empty">We found a few close matches. Choose one to see it on the page.</p><ol className="context-atlas-ambiguity-list">{matches.slice(0, 5).map((match, index) => { const candidate = findSourceMatch({ matches: [match] }, passages); if (!candidate) return null; return <li key={`${match.passage_id}:${match.sentence_index}`}><button className="context-atlas-ambiguity-button" type="button" onClick={() => selectResult(index)}><span>{candidate.sentence.text}</span><span className="context-atlas-field-note">{confidence(match.probability)}%</span></button></li>; })}</ol></div> : focusedSource ? <blockquote className="context-atlas-quote">{focusedSource.sentence.text}</blockquote> : <p className="context-atlas-empty">Ask a question to get started.</p>}
          {focusedSource ? <p className="context-atlas-source-byline">From {focusedSource.passage.title || source?.title || "this source"}</p> : null}
          <div className="context-atlas-result-actions">{focused ? <button className="context-atlas-trace-button" type="button" aria-expanded={traceOpen} onClick={() => setTraceOpen((value) => !value)}><Route size={15} aria-hidden="true" />{traceOpen ? "Hide details" : "Show why"}</button> : <span />}<span className="context-atlas-field-note">{matches.length ? `${current + 1} of ${matches.length}` : ""}</span></div>
          {traceOpen && provenance.length ? <ol className="context-atlas-provenance" aria-label="How this result was found">{provenance.map((node) => <li key={node.kind}><span className="context-atlas-provenance-label">{node.label}</span><p>{node.text}</p>{typeof node.probability === "number" ? <span className="context-atlas-provenance-score">{confidence(node.probability)}%</span> : null}</li>)}</ol> : null}
          <div className="context-atlas-navigation"><button className="context-atlas-quiet-button" type="button" onClick={() => selectResult(current - 1)} disabled={!matches.length || current === 0}><ChevronLeft size={15} aria-hidden="true" />Previous</button><button className="context-atlas-quiet-button" type="button" onClick={() => selectResult(current + 1)} disabled={!matches.length || current === matches.length - 1}>Next<ChevronRight size={15} aria-hidden="true" /></button></div>
        </article>

        {surface !== "extension" ? <article className="context-atlas-card context-atlas-source-card" aria-labelledby="context-atlas-source-heading"><div className="context-atlas-section-heading"><div><h2 id="context-atlas-source-heading">{source?.title || "Current page"}</h2></div><span className="context-atlas-source-rule" /></div><div className="context-atlas-source-text">{passages.length ? passages.map((passage) => <p key={passage.id}>{renderPassage(passage, focused)}</p>) : <p className="context-atlas-empty">Open the extension on a webpage, then refresh this page.</p>}</div></article> : null}
      </div>
      {surface !== "extension" ? <footer className="context-atlas-footer">Every result remains tied to the exact words in the current page.</footer> : null}
    </div>
  );
}

function isAmbiguous(matches, threshold = 0.58) {
  const parsedThreshold = Number(threshold);
  const relevanceThreshold = Number.isFinite(parsedThreshold) && parsedThreshold >= 0 && parsedThreshold <= 1
    ? parsedThreshold
    : 0.58;
  const eligible = (Array.isArray(matches) ? matches : [])
    .filter((match) => {
      const probability = Number(match?.probability);
      return Number.isFinite(probability) && probability >= relevanceThreshold;
    })
    .sort((left, right) => Number(right.probability) - Number(left.probability));
  if (eligible.length < 2) return false;
  return Number(eligible[0].probability) - Number(eligible[1].probability) < 0.05;
}

function confidence(value) {
  const probability = Number(value);
  return Number.isFinite(probability) ? Math.round(probability * 100) : 0;
}

function renderPassage(passage, focused) {
  const offset = focused && focused.passage_id === passage.id ? findSentenceOffset(passage.text, focused.sentence_index, focused.sentence_text) : null;
  if (!offset) return passage.text;
  return <>{passage.text.slice(0, offset.start)}<mark className="context-atlas-highlight">{passage.text.slice(offset.start, offset.end)}</mark>{passage.text.slice(offset.end)}</>;
}
