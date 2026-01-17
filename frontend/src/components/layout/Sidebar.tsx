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
  ListFilter,
  Settings,
  ChevronLeft,
  ChevronRight,
  Gift,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store";
import { Button } from "@/components/ui";

const navItems = [
  { icon: PenSquare, label: "New task", href: "/" },
  { icon: Search, label: "Search", href: "/search" },
  { icon: Library, label: "Library", href: "/library" },
];

export function Sidebar() {
  const { sidebarOpen, toggleSidebar, tasks } = useAppStore();

  // Get recent tasks (last 5)
  const recentTasks = tasks.slice(0, 5);

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
          {navItems.map((item) => (
            <Link key={item.label} href={item.href}>
              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-xl",
                  "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/50",
                  "transition-colors cursor-pointer",
                  item.label === "New task" && "bg-gradient-to-r from-cyan-500/10 to-blue-600/10 text-cyan-400 hover:text-cyan-300"
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
                <Link href="/tasks">
                  <div className="flex items-center gap-3 px-3 py-2 rounded-xl text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/50 transition-colors cursor-pointer">
                    <ListFilter className="w-4 h-4" />
                    <span className="text-sm">All tasks</span>
                  </div>
                </Link>
              </div>

              {/* Recent Tasks */}
              {recentTasks.length > 0 && (
                <div className="mt-4">
                  <span className="px-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                    Recent
                  </span>
                  <div className="mt-2 space-y-1">
                    {recentTasks.map((task) => (
                      <Link key={task.id} href={`/task/${task.id}`}>
                        <div className="flex items-center gap-3 px-3 py-2 rounded-xl text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/50 transition-colors cursor-pointer">
                          <Settings className="w-4 h-4 shrink-0" />
                          <span className="text-sm truncate">
                            {task.task.slice(0, 30)}...
                          </span>
                        </div>
                      </Link>
                    ))}
                  </div>
                </div>
              )}
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
