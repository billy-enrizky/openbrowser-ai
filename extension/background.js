/**
 * OpenBrowser Background Service Worker
 *
 * Maintains a WebSocket connection to the OpenBrowser backend and relays
 * Chrome DevTools Protocol (CDP) commands between the backend and the
 * browser via the chrome.debugger API.
 *
 * Key responsibilities:
 *   - WebSocket lifecycle (connect, ping keepalive, reconnect)
 *   - CDP command relay (backend -> chrome.debugger -> backend)
 *   - CDP event forwarding (chrome.debugger events -> backend)
 *   - Session/target ID mapping for multi-target debugging
 *   - Tab attach/detach management
 *
 * NOTE: Uses async/await (Promise-based) chrome.debugger API for MV3
 * compatibility. The callback-based API can silently drop results in
 * service workers.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/** @type {WebSocket|null} */
let ws = null;

/** @type {string|null} */
let backendUrl = null;

/** @type {number|null} Current tab being debugged */
let activeTabId = null;

/** @type {boolean} Whether the debugger is attached to the active tab */
let debuggerAttached = false;

/** CDP sessionId -> targetId */
const sessionToTarget = new Map();

/** targetId -> CDP sessionId */
const targetToSession = new Map();

/** Ping interval handle */
let pingInterval = null;

/** Reconnect timer handle */
let reconnectTimer = null;

/** Reconnect delay in ms -- grows exponentially, capped at 30s */
let reconnectDelay = 1000;

const MAX_RECONNECT_DELAY = 30000;
const PING_INTERVAL_MS = 25000;
const CDP_PROTOCOL_VERSION = "1.3";

// ---------------------------------------------------------------------------
// Persistence helpers
// ---------------------------------------------------------------------------

function persistState() {
  chrome.storage.local.set({
    backendUrl: backendUrl,
    connected: ws !== null && ws.readyState === WebSocket.OPEN,
    activeTabId: activeTabId,
    debuggerAttached: debuggerAttached,
  });
}

// ---------------------------------------------------------------------------
// WebSocket management
// ---------------------------------------------------------------------------

function connect(url) {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    if (url === backendUrl) {
      return; // already connected / connecting to same URL
    }
    ws.close();
  }

  backendUrl = url;
  persistState();

  try {
    ws = new WebSocket(url);
  } catch (err) {
    console.error("[OpenBrowser] Failed to create WebSocket:", err);
    scheduleReconnect();
    return;
  }

  ws.onopen = function () {
    console.log("[OpenBrowser] WebSocket connected to", url);
    reconnectDelay = 1000; // reset backoff
    persistState();
    startPing();
    sendBrowserInfo();
  };

  ws.onmessage = function (event) {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (err) {
      console.error("[OpenBrowser] Invalid JSON from backend:", err);
      return;
    }
    handleBackendMessage(msg);
  };

  ws.onclose = function (event) {
    console.log("[OpenBrowser] WebSocket closed. Code:", event.code, "Reason:", event.reason);
    cleanup();
    scheduleReconnect();
  };

  ws.onerror = function (err) {
    console.error("[OpenBrowser] WebSocket error:", err);
    // onclose will fire after onerror, which triggers reconnect
  };
}

function disconnect() {
  clearReconnect();
  if (ws) {
    ws.close();
  }
  cleanup();
}

function cleanup() {
  stopPing();
  ws = null;
  persistState();
}

function startPing() {
  stopPing();
  pingInterval = setInterval(function () {
    if (ws && ws.readyState === WebSocket.OPEN) {
      wsSend({ type: "ping" });
    }
  }, PING_INTERVAL_MS);
}

function stopPing() {
  if (pingInterval !== null) {
    clearInterval(pingInterval);
    pingInterval = null;
  }
}

function scheduleReconnect() {
  clearReconnect();
  if (!backendUrl) return;

  console.log("[OpenBrowser] Reconnecting in", reconnectDelay, "ms");
  reconnectTimer = setTimeout(function () {
    connect(backendUrl);
  }, reconnectDelay);

  reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
}

function clearReconnect() {
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

function wsSend(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// ---------------------------------------------------------------------------
// Send browser info on connect
// ---------------------------------------------------------------------------

async function sendBrowserInfo() {
  try {
    const tabs = await chrome.tabs.query({});
    wsSend({
      type: "extension_connected",
      info: {
        userAgent: navigator.userAgent,
        tabCount: tabs ? tabs.length : 0,
      },
    });
  } catch (err) {
    console.error("[OpenBrowser] sendBrowserInfo error:", err);
  }
}

// ---------------------------------------------------------------------------
// Backend message handler
// ---------------------------------------------------------------------------

function handleBackendMessage(msg) {
  switch (msg.type) {
    case "cdp_command":
      handleCdpCommand(msg);
      break;
    case "attach_tab":
      handleAttachTab(msg);
      break;
    case "detach_tab":
    case "detach_debugger":
      handleDetachTab(msg);
      break;
    case "get_browser_info":
      handleGetBrowserInfo(msg);
      break;
    case "pong":
      // keepalive ack, nothing to do
      break;
    default:
      console.warn("[OpenBrowser] Unknown message type from backend:", msg.type);
  }
}

// ---------------------------------------------------------------------------
// CDP command relay (async/await -- MV3 Promise-based API)
// ---------------------------------------------------------------------------

async function handleCdpCommand(msg) {
  const { id, method, params, sessionId } = msg;

  let target;
  if (sessionId) {
    const targetId = sessionToTarget.get(sessionId);
    if (targetId) {
      target = { targetId: targetId };
    } else {
      // Fallback: send on main tab
      if (!activeTabId) {
        wsSend({ type: "cdp_response", id: id, error: "No active tab and unknown sessionId" });
        return;
      }
      target = { tabId: activeTabId };
    }
  } else {
    if (!activeTabId) {
      wsSend({ type: "cdp_response", id: id, error: "No active tab attached" });
      return;
    }
    target = { tabId: activeTabId };
  }

  try {
    // Use Promise-based API (MV3) instead of callback-based.
    // The callback-based API can silently return undefined for results.
    const result = await chrome.debugger.sendCommand(target, method, params || {});

    // Track session <-> target mapping from Target.attachToTarget response
    if (method === "Target.attachToTarget" && result && result.sessionId) {
      const newTargetId = params && params.targetId ? params.targetId : null;
      if (newTargetId) {
        sessionToTarget.set(result.sessionId, newTargetId);
        targetToSession.set(newTargetId, result.sessionId);

        // Also attach chrome.debugger to the new target so we receive its events
        try {
          await chrome.debugger.attach({ targetId: newTargetId }, CDP_PROTOCOL_VERSION);
        } catch (attachErr) {
          // Target may already be attached -- not fatal
          console.warn(
            "[OpenBrowser] Could not attach debugger to target",
            newTargetId,
            attachErr.message
          );
        }
      }
    }

    // Send response back. Ensure result is never undefined in JSON
    // (JSON.stringify omits undefined values).
    wsSend({
      type: "cdp_response",
      id: id,
      result: result !== undefined && result !== null ? result : {},
    });
  } catch (err) {
    console.error("[OpenBrowser] CDP command error:", method, err.message);
    wsSend({
      type: "cdp_response",
      id: id,
      error: err.message || String(err),
    });
  }
}

// ---------------------------------------------------------------------------
// Tab attach / detach (async/await)
// ---------------------------------------------------------------------------

async function handleAttachTab(msg) {
  const requestedTabId = msg.tabId || null;

  async function attachToTab(tabId) {
    try {
      await chrome.debugger.attach({ tabId: tabId }, CDP_PROTOCOL_VERSION);
      activeTabId = tabId;
      debuggerAttached = true;
      persistState();

      // Enable Runtime and Page domains so we receive events
      try {
        await chrome.debugger.sendCommand({ tabId: tabId }, "Runtime.enable", {});
      } catch (e) {
        console.warn("[OpenBrowser] Runtime.enable failed:", e.message);
      }
      try {
        await chrome.debugger.sendCommand({ tabId: tabId }, "Page.enable", {});
      } catch (e) {
        console.warn("[OpenBrowser] Page.enable failed:", e.message);
      }

      wsSend({
        type: "tab_attached",
        tabId: tabId,
        success: true,
      });
    } catch (err) {
      wsSend({
        type: "tab_attached",
        tabId: tabId,
        success: false,
        error: err.message || String(err),
      });
    }
  }

  if (requestedTabId) {
    await attachToTab(requestedTabId);
  } else {
    // Create a new tab with about:blank (avoids chrome://newtab which
    // can cause debugger detachment) and attach
    try {
      const tab = await chrome.tabs.create({ active: true, url: "about:blank" });
      if (!tab) {
        wsSend({
          type: "tab_attached",
          tabId: null,
          success: false,
          error: "Failed to create tab",
        });
        return;
      }
      await attachToTab(tab.id);
    } catch (err) {
      wsSend({
        type: "tab_attached",
        tabId: null,
        success: false,
        error: err.message || String(err),
      });
    }
  }
}

async function handleDetachTab(_msg) {
  if (!activeTabId) {
    return;
  }

  try {
    await chrome.debugger.detach({ tabId: activeTabId });
  } catch (err) {
    console.warn("[OpenBrowser] Detach error:", err.message);
  }
  activeTabId = null;
  debuggerAttached = false;
  sessionToTarget.clear();
  targetToSession.clear();
  persistState();
}

// ---------------------------------------------------------------------------
// Browser info
// ---------------------------------------------------------------------------

async function handleGetBrowserInfo(msg) {
  try {
    const tabs = await chrome.tabs.query({});
    wsSend({
      type: "browser_info",
      id: msg.id || null,
      info: {
        userAgent: navigator.userAgent,
        tabCount: tabs ? tabs.length : 0,
        activeTabId: activeTabId,
        debuggerAttached: debuggerAttached,
      },
    });
  } catch (err) {
    console.error("[OpenBrowser] handleGetBrowserInfo error:", err);
  }
}

// ---------------------------------------------------------------------------
// chrome.debugger event forwarding
// ---------------------------------------------------------------------------

chrome.debugger.onEvent.addListener(function (source, method, params) {
  // Track Target.attachedToTarget events for session mapping
  if (method === "Target.attachedToTarget" && params) {
    const sessionId = params.sessionId;
    const targetInfo = params.targetInfo;
    if (sessionId && targetInfo && targetInfo.targetId) {
      sessionToTarget.set(sessionId, targetInfo.targetId);
      targetToSession.set(targetInfo.targetId, sessionId);
    }
  }

  // Track Target.detachedFromTarget events to clean up mappings
  if (method === "Target.detachedFromTarget" && params) {
    const sessionId = params.sessionId;
    if (sessionId) {
      const targetId = sessionToTarget.get(sessionId);
      if (targetId) {
        targetToSession.delete(targetId);
      }
      sessionToTarget.delete(sessionId);
    }
  }

  // Determine sessionId from source
  let sessionId = null;
  if (source.targetId) {
    sessionId = targetToSession.get(source.targetId) || null;
  }

  wsSend({
    type: "cdp_event",
    method: method,
    params: params || {},
    sessionId: sessionId,
  });
});

// ---------------------------------------------------------------------------
// chrome.debugger detach notification
// ---------------------------------------------------------------------------

chrome.debugger.onDetach.addListener(function (source, reason) {
  console.log("[OpenBrowser] Debugger detached. Source:", source, "Reason:", reason);

  // If the detached target is our active tab, reset state
  if (source.tabId && source.tabId === activeTabId) {
    activeTabId = null;
    debuggerAttached = false;
    sessionToTarget.clear();
    targetToSession.clear();
    persistState();
  }

  // If the detached target is a child target, clean up its mapping
  if (source.targetId) {
    const sessionId = targetToSession.get(source.targetId);
    if (sessionId) {
      sessionToTarget.delete(sessionId);
      targetToSession.delete(source.targetId);
    }
  }

  wsSend({
    type: "debugger_detached",
    reason: reason,
    tabId: source.tabId || null,
    targetId: source.targetId || null,
  });
});

// ---------------------------------------------------------------------------
// Message listener (from content script and popup)
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener(function (message, _sender, sendResponse) {
  if (!message || !message.type) {
    return;
  }

  switch (message.type) {
    case "set_backend_url":
      if (message.url) {
        connect(message.url);
      }
      sendResponse({ ok: true });
      break;

    case "get_status":
      sendResponse({
        connected: ws !== null && ws.readyState === WebSocket.OPEN,
        backendUrl: backendUrl,
        activeTabId: activeTabId,
        debuggerAttached: debuggerAttached,
      });
      break;

    case "manual_connect":
      if (message.url) {
        connect(message.url);
      }
      sendResponse({ ok: true });
      break;

    case "manual_disconnect":
      disconnect();
      sendResponse({ ok: true });
      break;

    default:
      break;
  }

  // Return true to indicate async sendResponse is possible
  return true;
});

// ---------------------------------------------------------------------------
// Startup: do NOT auto-connect from cached URL.
//
// The content script will detect an OpenBrowser frontend page and send the
// correct backend URL via "set_backend_url" message.  Auto-connecting from
// cache caused stale-URL issues (e.g. connecting to port 8000 when the
// backend moved to 8001).
// ---------------------------------------------------------------------------
