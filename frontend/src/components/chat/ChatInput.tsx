"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Plus,
  Link2,
  Mic,
  Presentation,
  Globe,
  Smartphone,
  Palette,
  MoreHorizontal,
  X,
} from "lucide-react";
import { Button } from "@/components/ui";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store";

const quickActions = [
  { icon: Presentation, label: "Create slides", color: "from-purple-500/20 to-violet-500/20" },
  { icon: Globe, label: "Build website", color: "from-cyan-500/20 to-blue-500/20" },
  { icon: Smartphone, label: "Develop apps", color: "from-green-500/20 to-emerald-500/20" },
  { icon: Palette, label: "Design", color: "from-pink-500/20 to-rose-500/20" },
  { icon: MoreHorizontal, label: "More", color: "from-zinc-500/20 to-zinc-600/20" },
];

const integrations = [
  { name: "Chrome", icon: "/icons/chrome.svg" },
  { name: "Gmail", icon: "/icons/gmail.svg" },
  { name: "Calendar", icon: "/icons/calendar.svg" },
  { name: "Drive", icon: "/icons/drive.svg" },
  { name: "Slack", icon: "/icons/slack.svg" },
  { name: "GitHub", icon: "/icons/github.svg" },
  { name: "Notion", icon: "/icons/notion.svg" },
];

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading?: boolean;
  placeholder?: string;
}

export function ChatInput({ onSend, isLoading = false, placeholder }: ChatInputProps) {
  const [message, setMessage] = useState("");
  const [showIntegrations, setShowIntegrations] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { agentType, setAgentType } = useAppStore();

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [message]);

  const handleSubmit = () => {
    if (message.trim() && !isLoading) {
      onSend(message.trim());
      setMessage("");
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Main Input Container */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={cn(
          "relative bg-zinc-800/30 border border-zinc-700/50 rounded-2xl",
          "backdrop-blur-xl shadow-2xl shadow-black/20",
          "focus-within:border-cyan-500/50 focus-within:ring-2 focus-within:ring-cyan-500/20",
          "transition-all duration-300"
        )}
      >
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || "Assign a task or ask anything"}
          disabled={isLoading}
          rows={1}
          className={cn(
            "w-full bg-transparent text-zinc-100 placeholder:text-zinc-500",
            "px-5 pt-4 pb-14 resize-none",
            "focus:outline-none",
            "text-base leading-relaxed",
            "min-h-[60px] max-h-[200px]"
          )}
        />

        {/* Bottom Actions Bar */}
        <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between">
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="w-8 h-8 text-zinc-500 hover:text-zinc-300"
              onClick={() => {}}
            >
              <Plus className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="w-8 h-8 text-zinc-500 hover:text-zinc-300"
              onClick={() => setShowIntegrations(!showIntegrations)}
            >
              <Link2 className="w-4 h-4" />
            </Button>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="w-8 h-8 text-zinc-500 hover:text-zinc-300"
            >
              <Mic className="w-4 h-4" />
            </Button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleSubmit}
              disabled={!message.trim() || isLoading}
              className={cn(
                "w-9 h-9 rounded-xl flex items-center justify-center",
                "transition-all duration-200",
                message.trim() && !isLoading
                  ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg shadow-cyan-500/25"
                  : "bg-zinc-700/50 text-zinc-500"
              )}
            >
              {isLoading ? (
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </motion.button>
          </div>
        </div>
      </motion.div>

      {/* Integrations Bar */}
      <AnimatePresence>
        {showIntegrations && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-3 flex items-center justify-between px-2"
          >
            <div className="flex items-center gap-2">
              <Link2 className="w-4 h-4 text-zinc-500" />
              <span className="text-sm text-zinc-500">Connect your tools to OpenBrowser</span>
            </div>
            <div className="flex items-center gap-2">
              {integrations.map((integration) => (
                <motion.button
                  key={integration.name}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className="w-7 h-7 rounded-lg bg-zinc-800/50 flex items-center justify-center hover:bg-zinc-700/50 transition-colors"
                  title={integration.name}
                >
                  <span className="text-xs">{integration.name[0]}</span>
                </motion.button>
              ))}
              <Button
                variant="ghost"
                size="icon"
                className="w-7 h-7"
                onClick={() => setShowIntegrations(false)}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quick Actions */}
      <div className="mt-6 flex items-center justify-center gap-3 flex-wrap">
        {quickActions.map((action) => (
          <motion.button
            key={action.label}
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            className={cn(
              "flex items-center gap-2 px-4 py-2.5 rounded-xl",
              "bg-gradient-to-r border border-zinc-700/50",
              action.color,
              "text-zinc-300 hover:text-zinc-100",
              "transition-all duration-200",
              "shadow-lg shadow-black/10"
            )}
          >
            <action.icon className="w-4 h-4" />
            <span className="text-sm font-medium">{action.label}</span>
          </motion.button>
        ))}
      </div>
    </div>
  );
}
