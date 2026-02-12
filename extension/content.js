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
   * Derive the default WebSocket backend URL from the current page origin.
   * e.g. http://localhost:3000 -> ws://localhost:8000/ws/extension
   *      https://app.openbrowser.me -> wss://api.openbrowser.me/ws/extension
   */
  function deriveBackendUrl() {
    const loc = window.location;
    const isSecure = loc.protocol === "https:";
    const wsProtocol = isSecure ? "wss" : "ws";

    if (loc.hostname === "localhost" || loc.hostname === "127.0.0.1") {
      return `${wsProtocol}://${loc.hostname}:8000/ws/extension`;
    }

    // For production: replace app/www subdomain with api subdomain
    const apiHost = loc.hostname.replace(/^(app|www)\./, "api.");
    return `${wsProtocol}://${apiHost}/ws/extension`;
  }

  /**
   * Read the WebSocket URL from the page meta tag, falling back to
   * a derived URL based on the current page origin.
   */
  function resolveBackendUrl() {
    const meta = document.querySelector('meta[name="openbrowser-ws-url"]');
    if (meta && meta.content) {
      return meta.content.trim();
    }
    return deriveBackendUrl();
  }

  /**
   * Send the backend URL to the background service worker if it has
   * changed since the last time we sent it.
   */
  function sendBackendUrl() {
    const url = resolveBackendUrl();
    if (url === lastSentUrl) {
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
