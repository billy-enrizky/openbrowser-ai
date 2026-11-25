(() => {
  const ACTIVE_CLASSES = ["text-primary", "dark:text-primary-light"];
  const TOC_ID = "table-of-contents-content";

  const previous = window.__openbrowserTocScrollspy;
  if (previous && typeof previous.cleanup === "function") previous.cleanup();

  const state = {
    toc: null,
    targets: [],
    activeSlug: "",
    frame: 0,
    observer: null,
    listeners: [],
    cleanup() {
      if (this.frame) cancelAnimationFrame(this.frame);
      this.frame = 0;
      if (this.interval) clearInterval(this.interval);
      this.interval = 0;
      if (this.observer) this.observer.disconnect();
      for (const [target, type, handler, options] of this.listeners) {
        target.removeEventListener(type, handler, options);
      }
      this.listeners = [];
    },
  };
  window.__openbrowserTocScrollspy = state;

  const decode = (value) => {
    try { return decodeURIComponent(value); } catch { return value; }
  };

  const currentOffset = () => {
    const raw = getComputedStyle(document.documentElement).getPropertyValue("--scroll-mt");
    const parsed = parseFloat(raw);
    return Number.isFinite(parsed) ? parsed + 12 : 128;
  };

  const clearToc = (toc) => {
    for (const node of toc.querySelectorAll("[data-active], [data-active-deepest]")) {
      node.removeAttribute("data-active");
      node.removeAttribute("data-active-deepest");
    }
    for (const link of toc.querySelectorAll("a")) {
      link.removeAttribute("aria-current");
      link.classList.remove(...ACTIVE_CLASSES);
    }
  };

  const activate = (entry) => {
    if (!entry) return;
    clearToc(state.toc);
    entry.link.setAttribute("aria-current", "location");
    entry.li.setAttribute("data-active-deepest", "true");
    let node = entry.li;
    while (node && state.toc.contains(node)) {
      node.setAttribute("data-active", "true");
      const direct = node.querySelector(":scope > a");
      if (direct) {
        direct.setAttribute("data-active", "true");
        direct.classList.add(...ACTIVE_CLASSES);
      }
      node = node.parentElement ? node.parentElement.closest("li") : null;
    }
    state.activeSlug = entry.slug;
  };

  const resolve = () => {
    const toc = document.getElementById(TOC_ID);
    if (!toc) return false;
    const links = Array.from(toc.querySelectorAll('a[data-layout-item="toc-item-link"], a[href^="#"]'));
    const seen = new Set();
    const targets = [];
    for (const link of links) {
      const href = link.getAttribute("href") || "";
      if (!href.startsWith("#")) continue;
      const slug = decode(href.slice(1));
      if (!slug || seen.has(slug)) continue;
      const element = document.getElementById(slug);
      if (!element) continue;
      seen.add(slug);
      targets.push({ slug, element, link, li: link.closest("li") });
    }
    if (!targets.length) return false;
    state.toc = toc;
    state.targets = targets;
    state.activeSlug = "";
    return true;
  };

  const update = () => {
    state.frame = 0;
    if (!state.toc || !state.targets.length) {
      if (!resolve()) return;
    }
    const offset = currentOffset();
    const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
    let current = state.targets[0];
    if (atBottom) {
      current = state.targets[state.targets.length - 1];
    } else {
      for (const candidate of state.targets) {
        const top = candidate.element.getBoundingClientRect().top;
        if (top <= offset + 1) current = candidate;
        else break;
      }
    }
    const applied = current.link.getAttribute("aria-current") === "location" &&
      current.link.hasAttribute("data-active") &&
      current.li.hasAttribute("data-active-deepest");
    if (current.slug !== state.activeSlug || !applied) activate(current);
  };

  const schedule = () => {
    if (state.frame) return;
    state.frame = requestAnimationFrame(update);
  };

  const refresh = () => {
    schedule();
    setTimeout(schedule, 50);
    setTimeout(schedule, 200);
    setTimeout(schedule, 600);
  };

  const listen = (target, type, handler, options) => {
    target.addEventListener(type, handler, options);
    state.listeners.push([target, type, handler, options]);
  };

  const init = () => {
    if (!resolve()) return;
    if (!state.toc.dataset.obScrollspy) state.toc.dataset.obScrollspy = "true";
    listen(window, "scroll", schedule, { passive: true });
    listen(window, "resize", schedule, { passive: true });
    listen(window, "hashchange", refresh);
    listen(window, "popstate", refresh);
    listen(document, "click", (event) => {
      const link = event.target && event.target.closest ? event.target.closest('#table-of-contents-content a[href^="#"]') : null;
      if (link) refresh();
    }, true);
    if (state.observer) state.observer.disconnect();
    state.observer = new MutationObserver(() => {
      const toc = document.getElementById(TOC_ID);
      if (!toc) return;
      if (toc !== state.toc || toc.querySelectorAll("a").length !== state.targets.length) {
        state.cleanup();
        init();
        schedule();
      }
    });
    state.observer.observe(document.body, { childList: true, subtree: true });
    if (state.interval) clearInterval(state.interval);
    state.interval = setInterval(schedule, 500);
    refresh();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
  window.addEventListener("load", () => setTimeout(schedule, 50), { once: true });
})();
