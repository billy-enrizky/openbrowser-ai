import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";

import { ContextAtlasSurface } from "../shared/components.jsx";
import { createWindowBridgeAdapter } from "./window-bridge.js";
import styles from "../shared/styles.css";

function StandaloneApp() {
  const adapter = useMemo(() => createWindowBridgeAdapter(), []);
  const [source, setSource] = useState(null);
  const [sourceStatus, setSourceStatus] = useState({ kind: "loading", message: "Getting the current page…" });

  const refreshSource = useCallback(async () => {
    setSourceStatus({ kind: "loading", message: "Getting the current page…" });
    try {
      const nextSource = await adapter.refreshSource();
      setSource(nextSource);
      setSourceStatus({ kind: "success", message: "Using the current page from your browser." });
    } catch (error) {
      setSource(null);
      setSourceStatus({ kind: "error", message: error?.code === "not_ready" ? "Open Context Atlas on a webpage, then refresh this page." : (error?.message || "The current page is not available yet.") });
    }
  }, [adapter]);

  useEffect(() => {
    void refreshSource();
  }, [refreshSource]);

  return <ContextAtlasSurface adapter={adapter} source={source || { title: "Current page", passages: [] }} sourceStatus={sourceStatus} onRefreshSource={refreshSource} surface="standalone" />;
}

function mountStyles() {
  if (document.getElementById("context-atlas-standalone-styles")) return;
  const style = document.createElement("style");
  style.id = "context-atlas-standalone-styles";
  style.textContent = styles;
  document.head.appendChild(style);
}

const mount = document.getElementById("context-atlas-root");
if (mount) {
  mountStyles();
  createRoot(mount).render(<StandaloneApp />);
}
