"use client";

import React, { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Sidebar, Header } from "@/components/layout";
import { ChatInput, ChatMessages, getFileTypeFromName, LogPanel } from "@/components/chat";
import { useAppStore } from "@/store";
import { useWebSocket } from "@/hooks/useWebSocket";
import type { Message, WSMessage, FileAttachment, LogEntry } from "@/types";
import { cn } from "@/lib/utils";

// Helper function to get file type - use backend-provided type or derive from filename
function getFileType(backendType?: string, filename?: string): FileAttachment["type"] {
  // If backend provides a valid type, use it
  if (backendType) {
    const validTypes = ["csv", "json", "text", "image", "pdf", "code", "unknown"];
    if (validTypes.includes(backendType)) {
      return backendType as FileAttachment["type"];
    }
  }
  // Otherwise derive from filename
  if (filename) {
    return getFileTypeFromName(filename);
  }
  return "unknown";
}

// Helper function to parse file attachments from backend data
function parseAttachments(data: Record<string, unknown>): FileAttachment[] {
  const attachments: FileAttachment[] = [];
  
  // Handle different attachment formats from backend
  if (data.attachments) {
    const rawAttachments = data.attachments as Array<{
      name?: string;
      filename?: string;
      content?: string;
      data?: string;
      url?: string;
      type?: string;
      mime_type?: string;
      size?: number;
    } | string>;
    
    for (const attachment of rawAttachments) {
      if (typeof attachment === "string") {
        // Legacy format: just a filename or path
        const filename = attachment.split("/").pop() || attachment;
        attachments.push({
          id: crypto.randomUUID(),
          name: filename,
          type: getFileTypeFromName(filename),
          content: undefined,
          url: attachment,
        });
      } else if (typeof attachment === "object") {
        // New format: object with metadata
        const filename = attachment.name || attachment.filename || "file";
        // Use backend-provided type if available, otherwise derive from filename
        const fileType = getFileType(attachment.type, filename);
        attachments.push({
          id: crypto.randomUUID(),
          name: filename,
          type: fileType,
          content: attachment.content || attachment.data,
          // Don't use file:// URLs as they don't work in browser
          url: attachment.url?.startsWith("file://") ? undefined : attachment.url,
          mimeType: attachment.mime_type,
          size: attachment.size,
        });
      }
    }
  }
  
  // Handle files array (alternative format)
  if (data.files) {
    const rawFiles = data.files as Array<{
      name?: string;
      filename?: string;
      content?: string;
      data?: string;
      url?: string;
      type?: string;
      mime_type?: string;
      size?: number;
    }>;
    
    for (const file of rawFiles) {
      const filename = file.name || file.filename || "file";
      // Use backend-provided type if available, otherwise derive from filename
      const fileType = getFileType(file.type, filename);
      attachments.push({
        id: crypto.randomUUID(),
        name: filename,
        type: fileType,
        content: file.content || file.data,
        // Don't use file:// URLs as they don't work in browser
        url: file.url?.startsWith("file://") ? undefined : file.url,
        mimeType: file.mime_type,
        size: file.size,
      });
    }
  }
  
  return attachments;
}

export default function Home() {
  const { sidebarOpen, messages, addMessage, agentType, maxSteps, useVision, logs, addLog, clearLogs, showLogs, setShowLogs } = useAppStore();
  const [isLoading, setIsLoading] = useState(false);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);

  // Handle incoming WebSocket messages
  const handleWSMessage = useCallback((wsMessage: WSMessage) => {
    const { type, task_id, data } = wsMessage;

    switch (type) {
      case "task_started":
        setCurrentTaskId(task_id || null);
        // Clear logs when a new task starts
        clearLogs();
        break;

      case "log": {
        // Handle backend log messages
        const logEntry: LogEntry = {
          id: crypto.randomUUID(),
          level: (data.level as "info" | "warning" | "error" | "debug") || "info",
          message: data.message as string,
          source: data.source as string | undefined,
          stepNumber: data.step_number as number | undefined,
          timestamp: new Date(),
        };
        addLog(logEntry);
        break;
      }

      case "step_update":
        // Update with step info
        if (data.thinking) {
          addMessage({
            id: crypto.randomUUID(),
            role: "assistant",
            content: data.thinking as string,
            timestamp: new Date(),
            taskId: task_id,
            metadata: {
              stepNumber: data.step_number as number,
              isThinking: true,
            },
          });
        }
        break;

      case "output":
        // Skip final outputs - they will be included in task_completed with attachments
        if (data.is_final) {
          break;
        }
        addMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.content as string,
          timestamp: new Date(),
          taskId: task_id,
          metadata: {
            stepNumber: data.step_number as number,
          },
        });
        break;

      case "screenshot":
        // Add screenshot to the last message or create new one
        addMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          content: "Browser screenshot captured",
          timestamp: new Date(),
          taskId: task_id,
          metadata: {
            stepNumber: data.step_number as number,
            screenshot: data.base64 as string,
          },
        });
        break;

      case "task_completed": {
        setIsLoading(false);
        const attachments = parseAttachments(data);
        addMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.result as string || "Task completed successfully.",
          timestamp: new Date(),
          taskId: task_id,
          metadata: {
            attachments: attachments.length > 0 ? attachments : undefined,
          },
        });
        break;
      }

      case "task_failed":
      case "error":
        setIsLoading(false);
        addMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          content: `Error: ${data.error}`,
          timestamp: new Date(),
          taskId: task_id,
          metadata: {
            isError: true,
          },
        });
        break;

      case "task_cancelled":
        setIsLoading(false);
        addMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          content: "Task was cancelled",
          timestamp: new Date(),
          taskId: task_id,
        });
        break;
    }
  }, [addMessage, addLog, clearLogs]);

  const { isConnected, sendMessage } = useWebSocket({
    onMessage: handleWSMessage,
    autoConnect: true,
  });

  const handleSendMessage = useCallback((content: string) => {
    // Add user message
    addMessage({
      id: crypto.randomUUID(),
      role: "user",
      content,
      timestamp: new Date(),
    });

    // Send to backend
    setIsLoading(true);
    const sent = sendMessage("start_task", undefined, {
      task: content,
      agent_type: agentType,
      max_steps: maxSteps,
      use_vision: useVision,
    });

    if (!sent) {
      setIsLoading(false);
      addMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        content: "Failed to connect to the server. Please try again.",
        timestamp: new Date(),
        metadata: { isError: true },
      });
    }
  }, [addMessage, sendMessage, agentType, maxSteps, useVision]);

  const hasMessages = messages.length > 0;

  return (
    <div className="flex h-screen overflow-hidden bg-zinc-950">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <main
        className={cn(
          "flex-1 flex flex-col transition-all duration-200",
          sidebarOpen ? "ml-[280px]" : "ml-16"
        )}
      >
        {/* Header */}
        <Header />

        {/* Chat Area */}
        <div className="flex-1 flex flex-col min-h-0">
          {hasMessages ? (
            <>
              {/* Messages - scrollable area */}
              <div className="flex-1 min-h-0 overflow-hidden">
                <ChatMessages messages={messages} />
              </div>

              {/* Log Panel - fixed height when open */}
              <LogPanel
                logs={logs}
                isOpen={showLogs}
                onToggle={() => setShowLogs(!showLogs)}
                onClear={clearLogs}
              />

              {/* Input at bottom - fixed */}
              <div className="shrink-0 border-t border-zinc-800/50 bg-zinc-900/50 backdrop-blur-xl p-4">
                <ChatInput
                  onSend={handleSendMessage}
                  isLoading={isLoading}
                  placeholder="Continue the conversation..."
                />
              </div>
            </>
          ) : (
            /* Welcome Screen */
            <div className="flex-1 flex flex-col items-center justify-center p-8">
              {/* Background effects */}
              <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl" />
                <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl" />
              </div>

              {/* Content */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="relative z-10 w-full max-w-3xl"
              >
                {/* Greeting */}
                <motion.h1
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                  className="text-4xl md:text-5xl font-bold text-center mb-12 tracking-tight"
                >
                  <span className="bg-gradient-to-r from-zinc-100 via-zinc-300 to-zinc-100 bg-clip-text text-transparent">
                    What can I do for you?
                  </span>
                </motion.h1>

                {/* Input */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  <ChatInput
                    onSend={handleSendMessage}
                    isLoading={isLoading}
                  />
                </motion.div>

                {/* Connection status */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5 }}
                  className="mt-8 flex items-center justify-center gap-2"
                >
                  <div
                    className={cn(
                      "w-2 h-2 rounded-full",
                      isConnected ? "bg-green-500" : "bg-red-500"
                    )}
                  />
                  <span className="text-sm text-zinc-500">
                    {isConnected ? "Connected to server" : "Connecting..."}
                  </span>
                </motion.div>

                {/* Framework Documentation Card */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                  className="mt-16"
                >
                  <a
                    href="https://docs.openbrowser.me/introduction"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group block"
                  >
                    <div className="relative overflow-hidden rounded-2xl border border-cyan-500/20 bg-gradient-to-br from-zinc-900/80 via-zinc-900/60 to-cyan-950/30 p-8 transition-all duration-500 hover:border-cyan-400/40 hover:shadow-[0_0_60px_-12px_rgba(6,182,212,0.3)] hover:scale-[1.02]">
                      {/* Animated background grid */}
                      <div className="absolute inset-0 opacity-20 group-hover:opacity-30 transition-opacity duration-500">
                        <div className="absolute inset-0" style={{
                          backgroundImage: `linear-gradient(rgba(6,182,212,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(6,182,212,0.1) 1px, transparent 1px)`,
                          backgroundSize: '32px 32px'
                        }} />
                      </div>
                      
                      {/* Glow orbs */}
                      <div className="absolute -top-20 -right-20 w-40 h-40 bg-cyan-500/20 rounded-full blur-3xl group-hover:bg-cyan-400/30 transition-all duration-700" />
                      <div className="absolute -bottom-20 -left-20 w-40 h-40 bg-blue-500/10 rounded-full blur-3xl group-hover:bg-blue-400/20 transition-all duration-700" />
                      
                      <div className="relative z-10 flex items-center justify-between gap-6">
                        <div className="flex-1">
                          {/* Badge */}
                          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-4">
                            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                            <span className="text-xs font-medium text-cyan-400 uppercase tracking-wider">Documentation</span>
                          </div>
                          
                          <h3 className="text-2xl font-bold text-zinc-100 mb-2 group-hover:text-white transition-colors">
                            Explore the OpenBrowser Framework
                          </h3>
                          <p className="text-zinc-400 group-hover:text-zinc-300 transition-colors leading-relaxed">
                            Automate browser tasks with AI. Dive into our comprehensive docs to build powerful AI-native web automation.
                          </p>
                          
                          {/* Link indicator */}
                          <div className="mt-4 inline-flex items-center gap-2 text-cyan-400 font-medium group-hover:gap-3 transition-all duration-300">
                            <span>Read the docs</span>
                            <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform duration-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                            </svg>
                          </div>
                        </div>
                        
                        {/* Icon/Logo area */}
                        <div className="shrink-0 w-28 h-28 rounded-2xl bg-gradient-to-br from-cyan-500/20 via-cyan-600/10 to-blue-600/20 border border-cyan-500/20 flex items-center justify-center group-hover:scale-110 group-hover:rotate-3 transition-all duration-500 shadow-lg shadow-cyan-500/10">
                          <div className="relative">
                            {/* Browser icon */}
                            <svg className="w-12 h-12 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                            </svg>
                            {/* AI sparkle */}
                            <div className="absolute -top-1 -right-1 w-4 h-4">
                              <svg className="w-full h-full text-cyan-300 animate-pulse" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M12 2L13.09 8.26L18 6L14.74 10.91L21 12L14.74 13.09L18 18L13.09 15.74L12 22L10.91 15.74L6 18L9.26 13.09L3 12L9.26 10.91L6 6L10.91 8.26L12 2Z" />
                              </svg>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Bottom gradient line */}
                      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                    </div>
                  </a>
                </motion.div>
              </motion.div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
