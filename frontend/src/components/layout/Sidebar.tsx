"use client";

import React from "react";
import Link from "next/link";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";
import {
  PenSquare,
  Search,
  Library,
  FolderPlus,
  MessageSquare,
  ChevronLeft,
  ChevronRight,
  Gift,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store";
import { Button } from "@/components/ui";
import type { ChatConversation } from "@/types";

const navItems = [
  { icon: Search, label: "Search", href: "/search" },
  { icon: Library, label: "Library", href: "/library" },
];

interface SidebarProps {
  onNewChat?: () => void;
  onSelectConversation?: (conversationId: string) => void;
  chatsLoading?: boolean;
}

export function Sidebar({ onNewChat, onSelectConversation, chatsLoading = false }: SidebarProps) {
  const { sidebarOpen, toggleSidebar, conversations, activeConversationId } = useAppStore();

  return (
    <motion.aside
      initial={false}
      animate={{ width: sidebarOpen ? 280 : 64 }}
      transition={{ duration: 0.2, ease: "easeInOut" }}
      className={cn(
        "h-screen bg-zinc-900/95 border-r border-zinc-800/50",
        "flex flex-col backdrop-blur-xl",
        "fixed left-0 top-0 z-40"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-zinc-800/50">
        {/* Logo - always visible */}
        <div className={cn(
          "w-8 h-8 rounded-lg overflow-hidden shrink-0",
          !sidebarOpen && "mx-auto"
        )}>
          <Image
            src="/favicon.svg"
            alt="OpenBrowser"
            width={32}
            height={32}
            className="w-full h-full"
          />
        </div>
        <AnimatePresence mode="wait">
          {sidebarOpen && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="font-semibold text-zinc-100 tracking-tight flex-1 ml-2"
            >
              OpenBrowser
            </motion.span>
          )}
        </AnimatePresence>
        {sidebarOpen && (
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleSidebar}
            className="shrink-0"
          >
            <ChevronLeft className="w-4 h-4" />
          </Button>
        )}
      </div>

      {/* Collapsed toggle button */}
      {!sidebarOpen && (
        <div className="p-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleSidebar}
            className="w-full"
          >
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>
      )}

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-2">
        <div className="space-y-1">
          <motion.button
            type="button"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onNewChat}
            className={cn(
              "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl",
              "text-cyan-400 hover:text-cyan-300 hover:bg-zinc-800/50",
              "transition-colors cursor-pointer bg-gradient-to-r from-cyan-500/10 to-blue-600/10"
            )}
          >
            <PenSquare className="w-5 h-5 shrink-0" />
            <AnimatePresence mode="wait">
              {sidebarOpen && (
                <motion.span
                  initial={{ opacity: 0, width: 0 }}
                  animate={{ opacity: 1, width: "auto" }}
                  exit={{ opacity: 0, width: 0 }}
                  className="text-sm font-medium whitespace-nowrap overflow-hidden"
                >
                  New chat
                </motion.span>
              )}
            </AnimatePresence>
          </motion.button>

          {navItems.map((item) => (
            <Link key={item.label} href={item.href}>
              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-xl",
                  "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/50",
                  "transition-colors cursor-pointer",
                )}
              >
                <item.icon className="w-5 h-5 shrink-0" />
                <AnimatePresence mode="wait">
                  {sidebarOpen && (
                    <motion.span
                      initial={{ opacity: 0, width: 0 }}
                      animate={{ opacity: 1, width: "auto" }}
                      exit={{ opacity: 0, width: 0 }}
                      className="text-sm font-medium whitespace-nowrap overflow-hidden"
                    >
                      {item.label}
                    </motion.span>
                  )}
                </AnimatePresence>
              </motion.div>
            </Link>
          ))}
        </div>

        {/* Projects Section */}
        <AnimatePresence mode="wait">
          {sidebarOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-6"
            >
              <div className="flex items-center justify-between px-3 mb-2">
                <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                  Projects
                </span>
                <Button variant="ghost" size="icon" className="w-6 h-6">
                  <FolderPlus className="w-3.5 h-3.5" />
                </Button>
              </div>

              <div className="space-y-1">
                <Link href="/projects/new">
                  <div className="flex items-center gap-3 px-3 py-2 rounded-xl text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/50 transition-colors cursor-pointer">
                    <FolderPlus className="w-4 h-4" />
                    <span className="text-sm">New project</span>
                  </div>
                </Link>
              </div>

              {/* Conversations */}
              <div className="mt-4">
                <span className="px-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                  Chats
                </span>
                <div className="mt-2 space-y-1">
                  {chatsLoading && (
                    <div className="flex items-center gap-2 px-3 py-2 text-xs text-zinc-500">
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      Loading chats...
                    </div>
                  )}
                  {!chatsLoading && conversations.length === 0 && (
                    <div className="px-3 py-2 text-xs text-zinc-500">No saved chats yet</div>
                  )}
                  {conversations.map((conversation: ChatConversation) => (
                    <button
                      type="button"
                      key={conversation.id}
                      onClick={() => onSelectConversation?.(conversation.id)}
                      className={cn(
                        "w-full flex items-center gap-3 px-3 py-2 rounded-xl transition-colors text-left",
                        conversation.id === activeConversationId
                          ? "bg-cyan-500/10 text-cyan-300"
                          : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/50"
                      )}
                    >
                      <MessageSquare className="w-4 h-4 shrink-0" />
                      <span className="text-sm truncate">{conversation.title}</span>
                    </button>
                  ))}
                </div>
              </div>

            </motion.div>
          )}
        </AnimatePresence>
      </nav>

      {/* Footer */}
      <AnimatePresence mode="wait">
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="p-4 border-t border-zinc-800/50"
          >
            <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-gradient-to-r from-amber-500/10 to-orange-500/10 text-amber-400 cursor-pointer hover:from-amber-500/20 hover:to-orange-500/20 transition-colors">
              <Gift className="w-5 h-5" />
              <div className="flex-1">
                <div className="text-sm font-medium">Share OpenBrowser</div>
                <div className="text-xs text-amber-500/70">Get free credits</div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.aside>
  );
}
