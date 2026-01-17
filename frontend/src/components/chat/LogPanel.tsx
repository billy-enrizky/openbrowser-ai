"use client";

import React, { useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Terminal, X, ChevronDown, ChevronUp, AlertCircle, Info, AlertTriangle, Bug } from "lucide-react";
import { cn } from "@/lib/utils";
import type { LogEntry } from "@/types";

interface LogPanelProps {
  logs: LogEntry[];
  isOpen: boolean;
  onToggle: () => void;
  onClear: () => void;
}

function getLogIcon(level: string) {
  switch (level) {
    case "error":
      return <AlertCircle className="w-3 h-3 text-red-400" />;
    case "warning":
      return <AlertTriangle className="w-3 h-3 text-yellow-400" />;
    case "debug":
      return <Bug className="w-3 h-3 text-purple-400" />;
    default:
      return <Info className="w-3 h-3 text-cyan-400" />;
  }
}

function getLogColor(level: string) {
  switch (level) {
    case "error":
      return "text-red-400";
    case "warning":
      return "text-yellow-400";
    case "debug":
      return "text-purple-400";
    default:
      return "text-zinc-300";
  }
}

export function LogPanel({ logs, isOpen, onToggle, onClear }: LogPanelProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (containerRef.current && isOpen) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs, isOpen]);

  return (
    <div className="border-t border-zinc-800/50">
      {/* Header */}
      <div
        role="button"
        tabIndex={0}
        onClick={onToggle}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onToggle();
          }
        }}
        className="w-full flex items-center justify-between px-4 py-2 bg-zinc-900/80 hover:bg-zinc-800/80 transition-colors cursor-pointer select-none"
      >
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-cyan-400" />
          <span className="text-sm font-medium text-zinc-300">Backend Logs</span>
          {logs.length > 0 && (
            <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded-full">
              {logs.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isOpen && logs.length > 0 && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onClear();
              }}
              className="text-xs text-zinc-500 hover:text-zinc-300 px-2 py-1 rounded hover:bg-zinc-700/50 transition-colors"
            >
              Clear
            </button>
          )}
          {isOpen ? (
            <ChevronDown className="w-4 h-4 text-zinc-500" />
          ) : (
            <ChevronUp className="w-4 h-4 text-zinc-500" />
          )}
        </div>
      </div>

      {/* Log Content */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div
              ref={containerRef}
              className="h-48 overflow-y-auto bg-zinc-950 font-mono text-xs scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent"
            >
              {logs.length === 0 ? (
                <div className="flex items-center justify-center h-full text-zinc-600">
                  No logs yet. Start a task to see backend activity.
                </div>
              ) : (
                <div className="p-2 space-y-0.5">
                  {logs.map((log) => (
                    <div
                      key={log.id}
                      className={cn(
                        "flex items-start gap-2 py-0.5 px-2 rounded hover:bg-zinc-900/50",
                        getLogColor(log.level)
                      )}
                    >
                      <span className="shrink-0 mt-0.5">{getLogIcon(log.level)}</span>
                      <span className="text-zinc-600 shrink-0">
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </span>
                      {log.stepNumber !== undefined && log.stepNumber !== null && (
                        <span className="text-cyan-600 shrink-0">[Step {log.stepNumber}]</span>
                      )}
                      {log.source && (
                        <span className="text-zinc-600 shrink-0 truncate max-w-[150px]" title={log.source}>
                          [{log.source.split(".").pop()}]
                        </span>
                      )}
                      <span className="break-all">{log.message}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

