import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

import { createRuntimeAdapter } from "../shared/adapters.js";
import { segmentText, findSentenceOffset, MAX_BLOCKS, MAX_PASSAGE_LENGTH } from "../shared/contracts.js";
import { ContextAtlasSurface } from "../shared/components.jsx";
import { findTextNodeSegments } from "./range-utils.js";
import { stableBlockId, sourceRevision } from "./search-context.js";
import { MAX_SOURCE_SCAN_PASSAGES, planSourcePassages } from "../../../extension/source-harness.mjs";
import styles from "../shared/styles.css";

const PANEL_ID = "context-atlas-extension-panel";
const PAGE_HIGHLIGHT_NAME = "context-atlas-page-highlight";
const PAGE_HIGHLIGHT_STYLE_ID = "context-atlas-page-highlight-style";
const SELECTOR = "p,li,pre,blockquote,figcaption";
const HEADING_SELECTOR = "h1,h2,h3";
const NOISE_ANCESTOR_SELECTOR = "nav,header,footer,aside,form,figure,table,[role='navigation'],[role='complementary'],[hidden],[aria-hidden='true'],[contenteditable='true'],.thumb,.infobox,.navbox,.metadata,.mw-editsection";
const SOURCE_REFRESH_DEBOUNCE_MS = 120;
const SOURCE_URL_POLL_MS = 250;
const NAVIGATION_EVENT = "context-atlas:navigation";
let internalMutationPending = false;

function collectPageSource(query = "") {
  const rawPassages = [];
  const rawElements = new Map();
  let section = "";
  document.querySelectorAll(`${HEADING_SELECTOR},${SELECTOR}`).forEach((element) => {
    if (element.matches(HEADING_SELECTOR)) {
      if (isUsable(element)) section = (element.textContent || "").trim();
      return;
    }
    if (rawPassages.length >= MAX_SOURCE_SCAN_PASSAGES || !isUsable(element) || hasPassageAncestor(element)) return;
    const text = (element.textContent || "").trim();
    if (text.length < 12 || text.length > MAX_PASSAGE_LENGTH) return;
    const sentences = segmentText(text);
    if (!sentences.length) return;
    const id = stableBlockId({ path: elementPath(element), text, occurrence: rawPassages.length });
    const passage = { id, title: document.title || location.hostname, section, text, sentences };
    rawPassages.push(passage);
    rawElements.set(id, element);
  });
  const passages = planSourcePassages(query, rawPassages).slice(0, MAX_BLOCKS);
  const elements = new Map(passages.map((passage) => [passage.id, rawElements.get(passage.id)]));
  return {
    title: document.title || location.hostname,
    url: location.href,
    passages,
    elements,
    revision: sourceRevision(passages),
  };
}

function serializableSource(source) {
  return {
    title: source.title,
    url: source.url,
    revision: source.revision,
    passages: source.passages.map(({ id, title, section, text, sentences }) => ({ id, title, section, text, sentences })),
  };
}

function publishPageSource(adapter, source) {
  if (typeof adapter.publishSource !== "function") return;
  void adapter.publishSource(serializableSource(source)).catch(() => {});
}

function isUsable(element) {
  if (element.localName !== "figcaption" && element.closest(NOISE_ANCESTOR_SELECTOR)) return false;
  const computed = getComputedStyle(element);
  return computed.display !== "none" && computed.visibility !== "hidden" && element.getClientRects().length > 0;
}

function hasPassageAncestor(element) {
  let parent = element.parentElement;
  while (parent) {
    if (parent.matches(SELECTOR)) return true;
    parent = parent.parentElement;
  }
  return false;
}

function elementPath(element) {
  const parts = [];
  let current = element;
  while (current && current !== document.body) {
    let position = 1;
    let sibling = current.previousElementSibling;
    while (sibling) {
      position += 1;
      sibling = sibling.previousElementSibling;
    }
    parts.unshift(`${current.localName}[${position}]`);
    current = current.parentElement;
  }
  return `body/${parts.join("/")}`;
}

function ExtensionApp({ adapter, initialSource, onSourceChange }) {
  const [source, setSource] = useState(initialSource);
  const sourceRef = useRef(initialSource);
  sourceRef.current = source;
  useEffect(() => onSourceChange((nextSource) => {
    sourceRef.current = nextSource;
    setSource(nextSource);
  }), [onSourceChange]);
  const prepareSearch = useCallback((query) => {
    const nextSource = collectPageSource(query);
    sourceRef.current = nextSource;
    return nextSource;
  }, []);
  const surfaceAdapter = useMemo(() => ({
    ...adapter,
    focusMatch: ({ passage, sentence }) => highlightMatch(sourceRef.current.elements, passage, sentence),
  }), [adapter]);
  return <ContextAtlasSurface adapter={surfaceAdapter} source={source} surface="extension" prepareSearch={prepareSearch} onClose={() => window.__contextAtlasClose?.()} />;
}

function highlightMatch(elements, passage, sentence) {
  clearPageHighlights();
  const element = elements.get(passage.id);
  if (!element) return;
  const offset = findSentenceOffset(passage.text, sentence.index, sentence.text);
  if (!offset) return;
  const rawText = element.textContent || "";
  const passageOffset = rawText.indexOf(passage.text);
  if (passageOffset < 0) return;
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);
  const segments = findTextNodeSegments(textNodes, passageOffset + offset.start, passageOffset + offset.end);
  if (!segments.length) return;
  const ranges = segments.map(({ node, start, end }) => {
    const range = document.createRange();
    range.setStart(node, start);
    range.setEnd(node, end);
    return range;
  });
  if (supportsCustomHighlights()) {
    ensurePageHighlightStyle();
    const highlight = new Highlight(...ranges);
    highlight.priority = 1;
    CSS.highlights.set(PAGE_HIGHLIGHT_NAME, highlight);
  } else {
    try {
      internalMutationPending = true;
      for (const range of [...ranges].reverse()) {
        const mark = document.createElement("mark");
        mark.className = PAGE_HIGHLIGHT_NAME;
        mark.style.background = "#ffd4c5";
        mark.style.color = "#9c2f24";
        mark.style.borderRadius = "3px";
        range.surroundContents(mark);
      }
    } finally {
      window.setTimeout(() => { internalMutationPending = false; }, 0);
    }
  }
  element.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
}

function supportsCustomHighlights() {
  return typeof CSS !== "undefined" && CSS.highlights && typeof Highlight === "function";
}

function ensurePageHighlightStyle() {
  if (document.getElementById(PAGE_HIGHLIGHT_STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = PAGE_HIGHLIGHT_STYLE_ID;
  style.textContent = `::highlight(${PAGE_HIGHLIGHT_NAME}) { background: #ffd4c5; color: #9c2f24; }`;
  (document.head || document.documentElement).appendChild(style);
}

function clearPageHighlights() {
  internalMutationPending = true;
  if (typeof CSS !== "undefined" && CSS.highlights) CSS.highlights.delete(PAGE_HIGHLIGHT_NAME);
  document.querySelectorAll("mark.context-atlas-page-highlight").forEach((mark) => mark.replaceWith(document.createTextNode(mark.textContent || "")));
  window.setTimeout(() => { internalMutationPending = false; }, 0);
}

function mutationTouchesPanel(record, host) {
  if (host.contains(record.target)) return true;
  return [...record.addedNodes, ...record.removedNodes].some((node) => node === host || host.contains(node));
}

function mountExtension() {
  if (!/^https?:$/.test(location.protocol) || document.getElementById(PANEL_ID)) return;
  window.__contextAtlasClose?.();
  const initialSource = collectPageSource();
  const host = document.createElement("aside");
  host.id = PANEL_ID;
  host.setAttribute("aria-label", "Context Atlas");
  const shadow = host.attachShadow({ mode: "open" });
  const style = document.createElement("style");
  style.textContent = styles;
  const mount = document.createElement("div");
  shadow.append(style, mount);
  document.documentElement.appendChild(host);
  const adapter = createRuntimeAdapter({ sendMessage: (message, callback) => chrome.runtime.sendMessage(message, callback) });
  publishPageSource(adapter, initialSource);
  let stopSourceObserver = () => {};
  let refreshTimer = null;
  const root = createRoot(mount);
  root.render(<ExtensionApp adapter={adapter} initialSource={initialSource} onSourceChange={(replaceSource) => {
    let observedUrl = location.href;
    const scheduleSourceRefresh = () => {
      if (refreshTimer !== null) window.clearTimeout(refreshTimer);
      refreshTimer = window.setTimeout(() => {
        refreshTimer = null;
        clearPageHighlights();
        const nextSource = collectPageSource();
        observedUrl = nextSource.url;
        publishPageSource(adapter, nextSource);
        replaceSource(nextSource);
      }, SOURCE_REFRESH_DEBOUNCE_MS);
    };
    const checkUrl = () => {
      const currentUrl = location.href;
      if (currentUrl === observedUrl) return;
      observedUrl = currentUrl;
      scheduleSourceRefresh();
    };
    const onNavigation = (event = null) => {
      const nextUrl = event?.destination?.url || location.href;
      if (nextUrl === observedUrl && location.href === observedUrl) return;
      observedUrl = nextUrl;
      scheduleSourceRefresh();
    };
    const observer = new MutationObserver((records) => {
      if (internalMutationPending || records.some((record) => mutationTouchesPanel(record, host))) return;
      scheduleSourceRefresh();
    });
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
    window.addEventListener("popstate", onNavigation);
    window.addEventListener("hashchange", onNavigation);
    window.addEventListener(NAVIGATION_EVENT, onNavigation);
    const navigationObject = window.navigation;
    if (navigationObject && typeof navigationObject.addEventListener === "function") {
      navigationObject.addEventListener("navigate", onNavigation);
    }
    const historyObject = window.history;
    const originalPushState = historyObject?.pushState;
    const originalReplaceState = historyObject?.replaceState;
    const wrappedPushState = typeof originalPushState === "function"
      ? function (...args) {
        const result = originalPushState.apply(this, args);
        window.dispatchEvent(new Event(NAVIGATION_EVENT));
        return result;
      }
      : null;
    const wrappedReplaceState = typeof originalReplaceState === "function"
      ? function (...args) {
        const result = originalReplaceState.apply(this, args);
        window.dispatchEvent(new Event(NAVIGATION_EVENT));
        return result;
      }
      : null;
    if (wrappedPushState) historyObject.pushState = wrappedPushState;
    if (wrappedReplaceState) historyObject.replaceState = wrappedReplaceState;
    const urlPollTimer = window.setInterval(checkUrl, SOURCE_URL_POLL_MS);
    stopSourceObserver = () => {
      observer.disconnect();
      window.removeEventListener("popstate", onNavigation);
      window.removeEventListener("hashchange", onNavigation);
      window.removeEventListener(NAVIGATION_EVENT, onNavigation);
      if (navigationObject && typeof navigationObject.removeEventListener === "function") {
        navigationObject.removeEventListener("navigate", onNavigation);
      }
      if (refreshTimer !== null) {
        window.clearTimeout(refreshTimer);
        refreshTimer = null;
      }
      window.clearInterval(urlPollTimer);
      if (wrappedPushState && historyObject.pushState === wrappedPushState) historyObject.pushState = originalPushState;
      if (wrappedReplaceState && historyObject.replaceState === wrappedReplaceState) historyObject.replaceState = originalReplaceState;
    };
    return stopSourceObserver;
  }} />);
  window.__contextAtlasClose = () => {
    stopSourceObserver();
    clearPageHighlights();
    root.unmount();
    host.remove();
    delete window.__contextAtlasClose;
  };
}

mountExtension();
