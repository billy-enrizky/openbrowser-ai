"use client";

import React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ */
/* SVG Icons                                                          */
/* ------------------------------------------------------------------ */

function TerminalIcon({ className }: { className?: string }) {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <polyline points="4 17 10 11 4 5" />
      <line x1="12" y1="19" x2="20" y2="19" />
    </svg>
  );
}

function BrainIcon({ className }: { className?: string }) {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M12 2a5 5 0 0 1 5 5c0 1.5-.7 2.8-1.7 3.7L12 14l-3.3-3.3A5 5 0 0 1 12 2z" />
      <path d="M12 14v8" />
      <path d="M8 18h8" />
      <circle cx="12" cy="7" r="1" />
    </svg>
  );
}

function WrenchIcon({ className }: { className?: string }) {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
    </svg>
  );
}

function MonitorIcon({ className }: { className?: string }) {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
      <line x1="8" y1="21" x2="16" y2="21" />
      <line x1="12" y1="17" x2="12" y2="21" />
    </svg>
  );
}

function ServerIcon({ className }: { className?: string }) {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
      <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
      <line x1="6" y1="6" x2="6.01" y2="6" />
      <line x1="6" y1="18" x2="6.01" y2="18" />
    </svg>
  );
}

function CodeIcon({ className }: { className?: string }) {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* Token comparison bars (Card 1 visual)                              */
/* ------------------------------------------------------------------ */

function TokenBars() {
  return (
    <div className="mt-6 space-y-3">
      {/* Chrome DevTools MCP */}
      <div>
        <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
          <span>Chrome DevTools MCP</span>
          <span>310,856 tokens</span>
        </div>
        <div className="h-8 rounded-lg bg-red-500/80 w-full flex items-center px-3">
          <span className="text-xs font-medium text-white/90">310,856</span>
        </div>
      </div>
      {/* Playwright MCP */}
      <div>
        <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
          <span>Playwright MCP</span>
          <span>150,248 tokens</span>
        </div>
        <div className="h-8 rounded-lg bg-yellow-500/80 w-[48%] flex items-center px-3">
          <span className="text-xs font-medium text-white/90">150,248</span>
        </div>
      </div>
      {/* OpenBrowser MCP */}
      <div>
        <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
          <span>OpenBrowser MCP</span>
          <span>49,423 tokens</span>
        </div>
        <div className="h-8 rounded-lg bg-cyan-500 w-[16%] min-w-[5rem] flex items-center px-2">
          <span className="text-xs font-medium text-white/90">49,423</span>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Provider pills (Card 2 visual)                                     */
/* ------------------------------------------------------------------ */

function ProviderPills() {
  const providers = ["Gemini", "GPT-4", "Claude", "LiteLLM"];
  return (
    <div className="flex flex-wrap gap-2 mt-4">
      {providers.map((p) => (
        <span
          key={p}
          className="rounded-full bg-white/5 border border-white/10 px-3 py-1 text-xs text-slate-300"
        >
          {p}
        </span>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Code snippet (Card 3 visual)                                       */
/* ------------------------------------------------------------------ */

function CodeSnippet() {
  return (
    <div className="mt-4 bg-zinc-950 rounded-lg p-3 font-mono text-sm overflow-x-auto">
      <span className="text-cyan-400">await</span>
      <span className="text-slate-300"> navigate</span>
      <span className="text-slate-400">(</span>
      <span className="text-green-400">&quot;https://news.ycombinator.com&quot;</span>
      <span className="text-slate-400">)</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Browser mockup (Card 4 visual)                                     */
/* ------------------------------------------------------------------ */

function BrowserMockup() {
  return (
    <div className="mt-6 rounded-lg border border-white/10 overflow-hidden">
      {/* Title bar */}
      <div className="flex items-center gap-1.5 px-3 py-2 bg-zinc-800/50 border-b border-white/5">
        <span className="w-2.5 h-2.5 rounded-full bg-red-400/70" />
        <span className="w-2.5 h-2.5 rounded-full bg-yellow-400/70" />
        <span className="w-2.5 h-2.5 rounded-full bg-green-400/70" />
        <span className="ml-3 text-xs text-slate-500">openbrowser.me</span>
      </div>
      {/* Content area */}
      <div className="h-28 bg-gradient-to-br from-cyan-950/30 via-zinc-900 to-violet-950/30 flex items-center justify-center">
        <div className="flex items-center gap-2 text-slate-500 text-sm">
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          >
            <circle cx="8" cy="8" r="6" />
            <path d="M8 5v3l2 2" />
          </svg>
          Live VNC stream
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Feature card wrapper                                               */
/* ------------------------------------------------------------------ */

interface FeatureCardProps {
  children: React.ReactNode;
  className?: string;
  index: number;
}

function FeatureCard({ children, className, index }: FeatureCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      className={cn(
        "relative group rounded-xl border border-white/[0.08] bg-zinc-900/60 p-6 md:p-8 hover:border-white/15 transition-colors overflow-hidden",
        className
      )}
    >
      {/* Hover glow */}
      <div className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 bg-gradient-to-br from-cyan-500/5 via-transparent to-violet-500/5" />
      <div className="relative z-10">{children}</div>
    </motion.div>
  );
}

/* ------------------------------------------------------------------ */
/* Features section                                                   */
/* ------------------------------------------------------------------ */

export function Features() {
  return (
    <section id="features" className="relative py-24 px-6 overflow-hidden">
      {/* Subtle radial glow */}
      <div className="pointer-events-none absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-gradient-to-b from-cyan-500/[0.04] to-transparent rounded-full blur-3xl" />
      <div className="max-w-7xl mx-auto">
        {/* Section heading */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-bold text-white">
            Why OpenBrowser
          </h2>
          <p className="text-lg text-slate-400 mt-4">
            Built different from the ground up.
          </p>
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Card 1: Text-First Architecture (wide) */}
          <FeatureCard index={0} className="lg:col-span-2">
            <TerminalIcon className="text-cyan-400 mb-4" />
            <h3 className="text-xl font-semibold text-white">
              Text-First Architecture
            </h3>
            <p className="text-slate-400 mt-2">
              3x fewer tokens than Playwright MCP. 6.3x fewer than Chrome
              DevTools MCP. Your AI agent processes text, not screenshots.
            </p>
            <TokenBars />
          </FeatureCard>

          {/* Card 2: Any LLM Provider */}
          <FeatureCard index={1}>
            <BrainIcon className="text-violet-400 mb-4" />
            <h3 className="text-xl font-semibold text-white">
              Any LLM Provider
            </h3>
            <p className="text-slate-400 mt-2">
              Works with Google Gemini, OpenAI GPT, Anthropic Claude, and any
              OpenAI-compatible endpoint.
            </p>
            <ProviderPills />
          </FeatureCard>

          {/* Card 3: One Tool, Full Control */}
          <FeatureCard index={2}>
            <WrenchIcon className="text-cyan-400 mb-4" />
            <h3 className="text-xl font-semibold text-white">
              One Tool, Full Control
            </h3>
            <p className="text-slate-400 mt-2">
              One <code className="text-cyan-400/80 text-sm">execute_code</code> tool, persistent
              Python namespace. Navigate, click, type, extract.
            </p>
            <CodeSnippet />
          </FeatureCard>

          {/* Card 4: Live Browser View (wide) */}
          <FeatureCard index={3} className="lg:col-span-2">
            <MonitorIcon className="text-violet-400 mb-4" />
            <h3 className="text-xl font-semibold text-white">
              Live Browser View
            </h3>
            <p className="text-slate-400 mt-2">
              Watch the agent browse in real-time via VNC streaming. See every
              click, scroll, and navigation as it happens.
            </p>
            <BrowserMockup />
          </FeatureCard>

          {/* Card 5: Production Ready */}
          <FeatureCard index={4}>
            <ServerIcon className="text-cyan-400 mb-4" />
            <h3 className="text-xl font-semibold text-white">
              Production Ready
            </h3>
            <p className="text-slate-400 mt-2">
              Docker, Kubernetes, cloud deployment. Battle-tested infrastructure
              for any scale.
            </p>
          </FeatureCard>

          {/* Card 6: Open Source */}
          <FeatureCard index={5} className="lg:col-span-2">
            <CodeIcon className="text-violet-400 mb-4" />
            <h3 className="text-xl font-semibold text-white">Open Source</h3>
            <p className="text-slate-400 mt-2">
              MIT licensed. Community-driven. Fully extensible. Build on top of
              OpenBrowser.
            </p>
            <a
              href="https://github.com/billy-enrizky/openbrowser-ai"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 mt-4 rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300 hover:bg-white/10 transition"
            >
              Star on GitHub
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M2.5 9.5L9.5 2.5M9.5 2.5H4M9.5 2.5V8"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </a>
          </FeatureCard>
        </div>
      </div>
    </section>
  );
}
