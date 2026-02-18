/**
 * OpenBrowser Popup Script
 *
 * Reads connection state from the background service worker and
 * chrome.storage.local, updates the popup UI, and provides manual
 * connect/disconnect controls.
 */

(function () {
  "use strict";

  // -----------------------------------------------------------------------
  // DOM references
  // -----------------------------------------------------------------------

  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const connectedInfo = document.getElementById("connectedInfo");
  const backendUrlDisplay = document.getElementById("backendUrlDisplay");
  const tabIdDisplay = document.getElementById("tabIdDisplay");

  const overrideToggle = document.getElementById("overrideToggle");
  const overrideArrow = document.getElementById("overrideArrow");
  const overrideBody = document.getElementById("overrideBody");

  const manualUrlInput = document.getElementById("manualUrl");
  const connectBtn = document.getElementById("connectBtn");
  const disconnectBtn = document.getElementById("disconnectBtn");

  // -----------------------------------------------------------------------
  // UI update
  // -----------------------------------------------------------------------

  function updateUI(state) {
    const connected = state.connected === true;

    if (connected) {
      statusDot.className = "status-dot connected";
      statusText.textContent = "Connected";
      connectedInfo.style.display = "block";
      backendUrlDisplay.textContent = state.backendUrl || "--";
      tabIdDisplay.textContent =
        state.activeTabId != null ? "Tab " + state.activeTabId : "None";
    } else {
      statusDot.className = "status-dot disconnected";
      statusText.textContent = "Disconnected";
      connectedInfo.style.display = "none";
      backendUrlDisplay.textContent = "--";
      tabIdDisplay.textContent = "--";
    }

    // Pre-fill manual URL input if we have a stored backend URL
    if (state.backendUrl && !manualUrlInput.value) {
      manualUrlInput.value = state.backendUrl;
    }
  }

  // -----------------------------------------------------------------------
  // Load initial state
  // -----------------------------------------------------------------------

  function loadState() {
    // Ask the background worker directly for live status
    chrome.runtime.sendMessage({ type: "get_status" }, function (response) {
      if (chrome.runtime.lastError) {
        // Background worker may not be active yet; fall back to storage
        chrome.storage.local.get(
          ["connected", "backendUrl", "activeTabId", "debuggerAttached"],
          function (data) {
            updateUI(data);
          }
        );
        return;
      }
      if (response) {
        updateUI(response);
      }
    });
  }

  loadState();

  // -----------------------------------------------------------------------
  // Listen for storage changes (real-time status updates)
  // -----------------------------------------------------------------------

  chrome.storage.onChanged.addListener(function (changes, area) {
    if (area !== "local") return;

    // Re-read full state whenever anything relevant changes
    if (
      changes.connected ||
      changes.backendUrl ||
      changes.activeTabId ||
      changes.debuggerAttached
    ) {
      loadState();
    }
  });

  // -----------------------------------------------------------------------
  // Collapsible manual override section
  // -----------------------------------------------------------------------

  overrideToggle.addEventListener("click", function () {
    const isOpen = overrideBody.classList.toggle("open");
    if (isOpen) {
      overrideArrow.classList.add("open");
    } else {
      overrideArrow.classList.remove("open");
    }
  });

  // -----------------------------------------------------------------------
  // Manual connect / disconnect
  // -----------------------------------------------------------------------

  connectBtn.addEventListener("click", function () {
    const url = manualUrlInput.value.trim();
    if (!url) {
      return;
    }

    chrome.runtime.sendMessage(
      { type: "manual_connect", url: url },
      function (_response) {
        if (chrome.runtime.lastError) {
          console.error("[OpenBrowser Popup] Connect error:", chrome.runtime.lastError.message);
        }
        // State will update via storage listener
      }
    );
  });

  disconnectBtn.addEventListener("click", function () {
    chrome.runtime.sendMessage(
      { type: "manual_disconnect" },
      function (_response) {
        if (chrome.runtime.lastError) {
          console.error("[OpenBrowser Popup] Disconnect error:", chrome.runtime.lastError.message);
        }
        // State will update via storage listener
      }
    );
  });
})();
