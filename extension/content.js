/**
 * OpenBrowser Content Script
 *
 * Runs on OpenBrowser frontend pages. Discovers the backend WebSocket URL
 * from a <meta> tag or derives it from the current page URL, then relays
 * it to the background service worker.
 */

(function () {
  "use strict";

  let lastSentUrl = null;

  /**
   * Read the WebSocket URL from the page meta tag.
   *
   * Returns null if the meta tag is not found -- this means the page is
   * not an OpenBrowser frontend and we should NOT send a URL to the
   * background worker (avoids overriding a correct URL with a wrong guess).
   */
  function resolveBackendUrl() {
    const meta = document.querySelector('meta[name="openbrowser-ws-url"]');
    if (meta && meta.content) {
      return meta.content.trim();
    }
    return null;
  }

  /**
   * Send the backend URL to the background service worker if it has
   * changed since the last time we sent it.
   */
  function sendBackendUrl() {
    const url = resolveBackendUrl();
    if (!url || url === lastSentUrl) {
      return;
    }
    lastSentUrl = url;

    chrome.runtime.sendMessage(
      { type: "set_backend_url", url: url },
      function (_response) {
        // Suppress errors when the background script is not ready
        if (chrome.runtime.lastError) {
          // Will retry on the next interval tick
          lastSentUrl = null;
        }
      }
    );
  }

  // Send on initial load
  sendBackendUrl();

  // Poll every 5 seconds to detect SPA navigation or meta tag changes
  setInterval(sendBackendUrl, 5000);
})();
