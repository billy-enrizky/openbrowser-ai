"use client";

import React from "react";
import { motion } from "framer-motion";
import { Bell, Sparkles, User } from "lucide-react";
import { Button } from "@/components/ui";

export function Header() {
  return (
    <header className="h-16 border-b border-zinc-800/50 bg-zinc-900/50 backdrop-blur-xl">
      <div className="h-full flex items-center justify-between px-6">
        {/* Left: Version selector */}
        <div className="flex items-center gap-2">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-800/50 text-zinc-300 hover:bg-zinc-700/50 transition-colors"
          >
            <span className="text-sm font-medium">OpenBrowser 1.0</span>
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
        </div>

        {/* Center: Plan status */}
        <div className="flex items-center gap-3">
          <span className="text-sm text-zinc-500">Free plan</span>
          <span className="text-zinc-700">|</span>
          <Button variant="ghost" size="sm" className="text-cyan-400 hover:text-cyan-300">
            Start free trial
          </Button>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" className="relative">
            <Bell className="w-5 h-5" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-cyan-500 rounded-full" />
          </Button>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gradient-to-r from-cyan-500/10 to-blue-600/10 border border-cyan-500/20">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-medium text-cyan-300">2,500</span>
          </div>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="w-9 h-9 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center"
          >
            <User className="w-5 h-5 text-white" />
          </motion.button>
        </div>
      </div>
    </header>
  );
}
