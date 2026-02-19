"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Monitor,
  Maximize2,
  Minimize2,
  X,
  ExternalLink,
  RefreshCw,
  Loader2,
  GripVertical,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store";
import { useAuth } from "@/components/auth/AuthProvider";
import { Button } from "@/components/ui";
import { API_BASE_URL, BROWSER_VIEWER_DEFAULT_WIDTH, BROWSER_VIEWER_MIN_WIDTH } from "@/lib/config";
import dynamic from "next/dynamic";

// Dynamically import VncScreen to avoid SSR issues
const VncScreen = dynamic(
  () => import("react-vnc").then((mod) => mod.VncScreen),
  { ssr: false }
);

interface BrowserViewerProps {
  className?: string;
}

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

export function BrowserViewer({ className }: BrowserViewerProps) {
  const {
    vncInfo,
    browserViewerOpen,
    setBrowserViewerOpen,
    browserViewerMode,
    setBrowserViewerMode,
    latestScreenshot,
  } = useAppStore();

  const { authEnabled, getValidIdToken } = useAuth();

  const containerRef = useRef<HTMLDivElement>(null);
  const vncContainerRef = useRef<HTMLDivElement>(null);
  const isMountedRef = useRef(true);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("disconnected");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [key, setKey] = useState(0); // Used to force reconnection
  const [panelWidth, setPanelWidth] = useState(BROWSER_VIEWER_DEFAULT_WIDTH);
  const [isResizing, setIsResizing] = useState(false);
  const [isInteractive, setIsInteractive] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [resolvedVncUrl, setResolvedVncUrl] = useState<string>("");

  // Build full WebSocket URL from the relative proxy path returned by the backend.
  // Re-runs on vncInfo change (new session) and key change (reconnect) to get a fresh token.
  useEffect(() => {
    let cancelled = false;
    async function buildUrl() {
      if (!vncInfo) {
        setResolvedVncUrl("");
        return;
      }
      const rawUrl = vncInfo.vnc_url;

      // Already a full ws:// URL (e.g. local dev connecting directly to websockify)
      if (rawUrl.startsWith("ws://") || rawUrl.startsWith("wss://")) {
        setResolvedVncUrl(rawUrl);
        return;
      }

      // Relative path -- construct full WebSocket URL via the API proxy
      const wsBase = API_BASE_URL.replace(/^https/, "wss").replace(/^http/, "ws");
      let token = "";
      if (authEnabled) {
        token = (await getValidIdToken()) || "";
      }
      if (cancelled) return;
      const separator = rawUrl.includes("?") ? "&" : "?";
      const tokenParam = token ? `${separator}token=${token}` : "";
      setResolvedVncUrl(`${wsBase}${rawUrl}${tokenParam}`);
    }
    buildUrl();
    return () => { cancelled = true; };
  }, [vncInfo, key, authEnabled, getValidIdToken]);

  // Track mounted state
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // Handle connection events - only update if mounted
  const handleConnect = useCallback(() => {
    if (isMountedRef.current) {
      setConnectionStatus("connected");
      setErrorMessage(null);
      setIsReconnecting(false);
    }
  }, []);

  const handleDisconnect = useCallback(() => {
    if (isMountedRef.current && !isReconnecting) {
      setConnectionStatus("disconnected");
    }
  }, [isReconnecting]);

  const handleSecurityFailure = useCallback((e: CustomEvent<{ status: number; reason?: string }>) => {
    if (isMountedRef.current) {
      setConnectionStatus("error");
      setErrorMessage(`Security error: ${e.detail.reason || "Unknown"}`);
      setIsReconnecting(false);
    }
  }, []);

  // Force reconnection
  const reconnect = useCallback(() => {
    setConnectionStatus("connecting");
    setErrorMessage(null);
    setKey((prev) => prev + 1);
  }, []);

  // Handle panel resize
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
    
    const startX = e.clientX;
    const startWidth = panelWidth;
    
    const handleMouseMove = (e: MouseEvent) => {
      const delta = startX - e.clientX;
      const newWidth = Math.max(BROWSER_VIEWER_MIN_WIDTH, Math.min(startWidth + delta, window.innerWidth * 0.8));
      setPanelWidth(newWidth);
    };
    
    const handleMouseUp = () => {
      setIsResizing(false);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
    
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  }, [panelWidth]);

  // Handle opening in popup window
  const openInPopup = useCallback(() => {
    if (!vncInfo || !resolvedVncUrl) return;

    const width = vncInfo.width || 1280;
    const height = vncInfo.height || 1024;
    const left = (window.screen.width - width) / 2;
    const top = (window.screen.height - height) / 2;

    const popup = window.open(
      "",
      "BrowserViewer",
      `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=no`
    );

    if (popup) {
      popup.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
          <title>OpenBrowser - Live View</title>
          <style>
            body { margin: 0; padding: 0; background: #000; overflow: hidden; }
            #vnc-container { width: 100vw; height: 100vh; }
          </style>
        </head>
        <body>
          <div id="vnc-container"></div>
          <script type="module">
            import RFB from 'https://cdn.jsdelivr.net/npm/@novnc/novnc@1.4.0/core/rfb.js';

            const container = document.getElementById('vnc-container');
            const rfb = new RFB(container, '${resolvedVncUrl}', {
              credentials: { password: '${vncInfo.password}' },
              shared: true,
            });
            rfb.scaleViewport = true;
            rfb.resizeSession = false;
          </script>
        </body>
        </html>
      `);
      popup.document.close();

      // Close embedded viewer
      setBrowserViewerMode("popup");
      setBrowserViewerOpen(false);
    }
  }, [vncInfo, resolvedVncUrl, setBrowserViewerMode, setBrowserViewerOpen]);

  // Toggle fullscreen
  const toggleFullscreen = useCallback(() => {
    setIsFullscreen((prev) => !prev);
  }, []);

  // Toggle interactive mode - need to reconnect for viewOnly change to take effect
  const toggleInteractive = useCallback(() => {
    setIsReconnecting(true);
    setConnectionStatus("connecting");
    setIsInteractive((prev) => !prev);
    // Delay key change to allow proper cleanup
    setTimeout(() => {
      if (isMountedRef.current) {
        setKey((k) => k + 1);
      }
    }, 100);
  }, []);

  // Set connecting status when VNC info changes
  useEffect(() => {
    if (vncInfo && browserViewerOpen) {
      setConnectionStatus("connecting");
    }
  }, [vncInfo, browserViewerOpen]);

  // Reset state when viewer closes
  useEffect(() => {
    if (!browserViewerOpen) {
      setConnectionStatus("disconnected");
      setErrorMessage(null);
      setIsInteractive(false);
    }
  }, [browserViewerOpen]);

  if (!browserViewerOpen) {
    return null;
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 20 }}
        className={cn(
          "flex flex-col bg-zinc-900 border-l border-zinc-800/50",
          isFullscreen ? "fixed inset-0 z-50" : "relative",
          isResizing && "select-none",
          className
        )}
        style={!isFullscreen ? { width: panelWidth, minWidth: BROWSER_VIEWER_MIN_WIDTH } : undefined}
      >
        {/* Resize handle */}
        {!isFullscreen && (
          <div
            className="absolute left-0 top-0 bottom-0 w-2 cursor-ew-resize hover:bg-cyan-500/20 transition-colors flex items-center justify-center group z-10"
            onMouseDown={handleMouseDown}
          >
            <GripVertical className="w-3 h-3 text-zinc-600 group-hover:text-cyan-400 transition-colors" />
          </div>
        )}

        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800/50 bg-zinc-900/95">
          <div className="flex items-center gap-2">
            <Monitor className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-medium text-zinc-200">Live Browser</span>
            {connectionStatus === "connected" && (
              <span className="flex items-center gap-1 px-2 py-0.5 text-xs bg-green-500/20 text-green-400 rounded-full">
                <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
                Connected
              </span>
            )}
            {connectionStatus === "connecting" && (
              <span className="flex items-center gap-1 px-2 py-0.5 text-xs bg-yellow-500/20 text-yellow-400 rounded-full">
                <Loader2 className="w-3 h-3 animate-spin" />
                Connecting
              </span>
            )}
            {connectionStatus === "error" && (
              <span className="px-2 py-0.5 text-xs bg-red-500/20 text-red-400 rounded-full">
                Error
              </span>
            )}
          </div>

          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              onClick={reconnect}
              disabled={connectionStatus === "connecting"}
              title="Reconnect"
              className="w-7 h-7"
            >
              <RefreshCw className={cn("w-4 h-4", connectionStatus === "connecting" && "animate-spin")} />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={openInPopup}
              title="Open in new window"
              className="w-7 h-7"
            >
              <ExternalLink className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleFullscreen}
              title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
              className="w-7 h-7"
            >
              {isFullscreen ? (
                <Minimize2 className="w-4 h-4" />
              ) : (
                <Maximize2 className="w-4 h-4" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setBrowserViewerOpen(false)}
              title="Close"
              className="w-7 h-7"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Browser View Container - screenshot fallback when VNC unavailable */}
        <div ref={containerRef} className="flex-1 relative bg-black overflow-hidden">
          {latestScreenshot ? (
            /* Show latest screenshot from task execution */
            <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
              <img
                src={`data:image/png;base64,${latestScreenshot}`}
                alt="Browser view"
                className="max-w-full max-h-full object-contain"
              />
              <div className="absolute top-2 right-2 px-2 py-1 bg-zinc-900/80 rounded text-xs text-zinc-400 backdrop-blur-sm">
                Live screenshot
              </div>
            </div>
          ) : vncInfo && resolvedVncUrl && connectionStatus !== "error" ? (
            /* Try VNC connection when available */
            <div ref={vncContainerRef} className="absolute inset-0 flex items-center justify-center overflow-hidden">
              <div
                className="relative"
                style={{
                  width: "100%",
                  height: "100%",
                  maxWidth: "100%",
                  maxHeight: "100%",
                }}
              >
                <VncScreen
                  key={key}
                  url={resolvedVncUrl}
                  rfbOptions={{
                    credentials: {
                      password: vncInfo.password,
                      username: "",
                      target: "",
                    },
                    shared: true,
                  }}
                  scaleViewport={true}
                  clipViewport={true}
                  resizeSession={false}
                  viewOnly={!isInteractive}
                  background="#000000"
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                  }}
                  onConnect={handleConnect}
                  onDisconnect={handleDisconnect}
                  onSecurityFailure={handleSecurityFailure}
                />
              </div>

              {/* Take control button overlay */}
              {connectionStatus === "connected" && !isInteractive && (
                <div className="absolute bottom-4 right-4 z-20">
                  <Button
                    onClick={toggleInteractive}
                    className="bg-cyan-500 hover:bg-cyan-600 text-white shadow-lg"
                  >
                    <Sparkles className="w-4 h-4 mr-2" />
                    Take control
                  </Button>
                </div>
              )}
            </div>
          ) : (
            /* No browser session active */
            <div className="absolute inset-0 flex flex-col items-center justify-center text-zinc-500">
              <Monitor className="w-12 h-12 mb-4 opacity-50" />
              <p className="text-sm">No browser session active</p>
              <p className="text-xs mt-1 text-zinc-600">Start a task to view the browser</p>
            </div>
          )}
        </div>

        {/* Footer with info */}
        {vncInfo && connectionStatus === "connected" && (
          <div className="px-4 py-1.5 border-t border-zinc-800/50 bg-zinc-900/95 flex items-center justify-between">
            <p className="text-xs text-zinc-500">
              {vncInfo.width}x{vncInfo.height} - {isInteractive ? "Click and type to interact" : "View only mode"}
            </p>
            {isInteractive && (
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleInteractive}
                className="text-xs h-6 px-2 text-zinc-400 hover:text-zinc-200"
              >
                Release control
              </Button>
            )}
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
