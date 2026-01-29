"use client";

import React from "react";
import { motion } from "framer-motion";
import { Bot, User, AlertCircle, Loader2, Image as ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Message } from "@/types";
import { FileAttachment } from "./FileAttachment";

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";
  const isError = message.metadata?.isError;
  const isThinking = message.metadata?.isThinking;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "flex gap-4 px-4 py-6",
        isUser ? "bg-transparent" : "bg-zinc-800/20"
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "w-8 h-8 rounded-lg flex items-center justify-center shrink-0",
          isUser
            ? "bg-gradient-to-br from-cyan-500 to-blue-600"
            : isError
            ? "bg-red-500/20 text-red-400"
            : "bg-zinc-700/50 text-zinc-400"
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : isError ? (
          <AlertCircle className="w-4 h-4" />
        ) : isThinking ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <Bot className="w-4 h-4" />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        {/* Header */}
        <div className="flex items-center gap-2 mb-2">
          <span className={cn(
            "text-sm font-medium",
            isUser ? "text-cyan-400" : "text-zinc-400"
          )}>
            {isUser ? "You" : "OpenBrowser"}
          </span>
          {message.metadata?.stepNumber && (
            <span className="text-xs text-zinc-600 px-2 py-0.5 bg-zinc-800 rounded-full">
              Step {message.metadata.stepNumber}
            </span>
          )}
          <span className="text-xs text-zinc-600">
            {new Date(message.timestamp).toLocaleTimeString()}
          </span>
        </div>

        {/* Message Content */}
        <div
          className={cn(
            "prose prose-invert prose-sm max-w-none",
            "text-zinc-300 leading-relaxed",
            isError && "text-red-400"
          )}
        >
          {isThinking ? (
            <div className="flex items-center gap-2 text-zinc-500">
              <span>Thinking</span>
              <span className="flex gap-1">
                <span className="w-1.5 h-1.5 bg-zinc-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="w-1.5 h-1.5 bg-zinc-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="w-1.5 h-1.5 bg-zinc-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              </span>
            </div>
          ) : (
            <MessageContent content={message.content} />
          )}
        </div>

        {/* Screenshot */}
        {message.metadata?.screenshot && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mt-4"
          >
            <div className="relative rounded-xl overflow-hidden border border-zinc-700/50 bg-zinc-800/50">
              <div className="absolute top-2 left-2 flex items-center gap-1.5 px-2 py-1 bg-zinc-900/80 rounded-lg backdrop-blur-sm">
                <ImageIcon className="w-3 h-3 text-zinc-500" />
                <span className="text-xs text-zinc-500">Screenshot</span>
              </div>
              <img
                src={`data:image/png;base64,${message.metadata.screenshot}`}
                alt="Browser screenshot"
                className="w-full max-w-2xl"
              />
            </div>
          </motion.div>
        )}

        {/* File Attachments */}
        {message.metadata?.attachments && message.metadata.attachments.length > 0 && (
          <div className="mt-4 space-y-2">
            {message.metadata.attachments.map((attachment) => (
              <FileAttachment key={attachment.id} file={attachment} />
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

function MessageContent({ content }: { content: string }) {
  // Simple markdown-like rendering
  const parts = content.split(/(```[\s\S]*?```)/g);

  return (
    <>
      {parts.map((part, index) => {
        if (part.startsWith("```") && part.endsWith("```")) {
          const code = part.slice(3, -3);
          const [lang, ...lines] = code.split("\n");
          const codeContent = lines.join("\n");

          return (
            <div key={index} className="my-4">
              <div className="flex items-center justify-between px-4 py-2 bg-zinc-900 rounded-t-lg border border-b-0 border-zinc-700/50">
                <span className="text-xs text-zinc-500">{lang || "code"}</span>
                <CopyButton text={codeContent || lang} />
              </div>
              <pre className="p-4 bg-zinc-900/50 rounded-b-lg border border-zinc-700/50 overflow-x-auto">
                <code className="text-sm text-cyan-300 font-mono">
                  {codeContent || lang}
                </code>
              </pre>
            </div>
          );
        }

        // Check if this part contains a markdown table (lines starting with |)
        const lines = part.split('\n');
        const hasTable = lines.some(line => line.trim().startsWith('|') && line.trim().endsWith('|'));
        
        if (hasTable) {
          // Render with horizontal scroll for tables
          return (
            <div 
              key={index} 
              className="my-2 rounded-lg border border-zinc-700/50 bg-zinc-900/30"
            >
              <div 
                className="overflow-x-auto"
                style={{ 
                  maxWidth: '100%',
                  overflowX: 'auto',
                }}
              >
                <div className="inline-block min-w-full">
                  {lines.map((line, lineIndex) => {
                    // Handle inline code within lines
                    const inlineCodeParts = line.split(/(`[^`]+`)/g);
                    return (
                      <p key={lineIndex} className="whitespace-nowrap font-mono text-sm px-3 py-1">
                        {inlineCodeParts.map((segment, segIndex) => {
                          if (segment.startsWith("`") && segment.endsWith("`")) {
                            return (
                              <code
                                key={segIndex}
                                className="px-1.5 py-0.5 bg-zinc-800 rounded text-cyan-300 text-sm font-mono"
                              >
                                {segment.slice(1, -1)}
                              </code>
                            );
                          }
                          return segment;
                        })}
                      </p>
                    );
                  })}
                </div>
              </div>
            </div>
          );
        }

        // Handle inline code for non-table content
        const inlineCodeParts = part.split(/(`[^`]+`)/g);
        
        return (
          <p key={index} className="whitespace-pre-wrap">
            {inlineCodeParts.map((segment, segIndex) => {
              if (segment.startsWith("`") && segment.endsWith("`")) {
                return (
                  <code
                    key={segIndex}
                    className="px-1.5 py-0.5 bg-zinc-800 rounded text-cyan-300 text-sm font-mono"
                  >
                    {segment.slice(1, -1)}
                  </code>
                );
              }
              return segment;
            })}
          </p>
        );
      })}
    </>
  );
}

// Copy Button Component
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}
