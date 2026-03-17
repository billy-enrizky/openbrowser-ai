"use client";

import React from "react";
import { motion } from "framer-motion";
import { Monitor, LogOut } from "lucide-react";
import { Button } from "@/components/ui";
import { ModelSelector } from "./ModelSelector";
import { useAppStore } from "@/store";
import { useAuth } from "@/components/auth";
import { cn } from "@/lib/utils";
import { appConfig } from "@/lib/ui-config";

export function Header() {
  const { vncInfo, browserViewerOpen, toggleBrowserViewer } = useAppStore();
  const { authEnabled, logout } = useAuth();
  const hasVncSession = !!vncInfo;

  return (
    <header className="h-16 border-b border-zinc-800/50 bg-zinc-900/50 backdrop-blur-xl relative z-[9999]">
      <div className="h-full flex items-center justify-between px-6">
        {/* Left: Version selector + Model selector + View Browser button */}
        <div className="flex items-center gap-3">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-800/50 text-zinc-300 hover:bg-zinc-700/50 transition-colors"
          >
            <span className="text-sm font-medium">{appConfig.appName} {appConfig.version}</span>
            <svg
              className="w-4 h-4 text-zinc-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </motion.button>

          {/* Model Selector */}
          <ModelSelector />

          {/* View Browser Button - Always clickable, shows empty state if no VNC session */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={toggleBrowserViewer}
            className={cn(
              "flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all",
              browserViewerOpen
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                : hasVncSession
                  ? "bg-zinc-800/50 text-zinc-300 hover:bg-cyan-500/10 hover:text-cyan-300 hover:border-cyan-500/20 border border-transparent"
                  : "bg-zinc-800/50 text-zinc-400 hover:bg-zinc-700/50 border border-transparent"
            )}
          >
            <Monitor className={cn("w-4 h-4", hasVncSession && "text-cyan-400")} />
            <span className="text-sm font-medium">
              {browserViewerOpen ? "Hide Browser" : "View Browser"}
            </span>
            {hasVncSession && (
              <span className={cn(
                "w-2 h-2 rounded-full",
                browserViewerOpen ? "bg-cyan-400" : "bg-green-400 animate-pulse"
              )} />
            )}
          </motion.button>
        </div>

        {/* Right: Sign Out */}
        <div className="flex items-center gap-3">
          {authEnabled && (
            <Button
              variant="ghost"
              size="sm"
              onClick={logout}
              className="text-zinc-300 hover:text-zinc-100"
              title="Sign out"
            >
              <LogOut className="w-4 h-4" />
              <span className="ml-2">Sign out</span>
            </Button>
          )}
        </div>
      </div>
    </header>
  );
}
