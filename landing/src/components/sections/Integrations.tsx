"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { Marquee } from "@/components/ui/marquee";

/* ------------------------------------------------------------------ */
/* Integration card                                                   */
/* ------------------------------------------------------------------ */

interface IntegrationCardProps {
  name: string;
  abbr: string;
  color: string;
  icon?: React.ReactNode;
}

function IntegrationCard({ name, abbr, color, icon }: IntegrationCardProps) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-white/[0.08] bg-zinc-900/60 px-6 py-4 mx-2 hover:border-white/15 transition-colors">
      <div
        className={cn(
          "flex h-10 w-10 items-center justify-center rounded-lg text-white text-xs font-bold",
          color
        )}
      >
        {icon ?? abbr}
      </div>
      <span className="text-sm font-medium text-slate-300">{name}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* SVG Icons                                                          */
/* ------------------------------------------------------------------ */

function TerminalSvg() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="4 17 10 11 4 5" />
      <line x1="12" y1="19" x2="20" y2="19" />
    </svg>
  );
}

function CursorSvg() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M4 4l7.07 17 2.51-7.39L21 11.07z" />
    </svg>
  );
}

function WaveSvg() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M2 12c2-3 4-6 6-3s4 3 6 0 4-3 6 0" />
    </svg>
  );
}

function CodeBracketsSvg() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  );
}

function WorkflowSvg() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="3" y="3" width="6" height="6" rx="1" />
      <rect x="15" y="15" width="6" height="6" rx="1" />
      <path d="M6 9v3a3 3 0 0 0 3 3h3" />
      <path d="M18 9v3a3 3 0 0 1-3 3h-3" />
    </svg>
  );
}

function PlugSvg() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 22v-5" />
      <path d="M9 8V2" />
      <path d="M15 8V2" />
      <path d="M18 8v5a6 6 0 0 1-12 0V8z" />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* Integrations data                                                  */
/* ------------------------------------------------------------------ */

const integrations: IntegrationCardProps[] = [
  { name: "Claude Code", abbr: "C", color: "bg-orange-600/80", icon: <TerminalSvg /> },
  { name: "Cursor", abbr: "Cu", color: "bg-blue-600/80", icon: <CursorSvg /> },
  { name: "Windsurf", abbr: "W", color: "bg-teal-600/80", icon: <WaveSvg /> },
  { name: "VS Code", abbr: "VS", color: "bg-sky-600/80", icon: <CodeBracketsSvg /> },
  { name: "n8n", abbr: "n8n", color: "bg-rose-600/80", icon: <WorkflowSvg /> },
  { name: "Any MCP Client", abbr: "MCP", color: "bg-violet-600/80", icon: <PlugSvg /> },
  { name: "Cline", abbr: "Cl", color: "bg-emerald-600/80", icon: <TerminalSvg /> },
  { name: "Roo Code", abbr: "R", color: "bg-amber-600/80", icon: <TerminalSvg /> },
];

/* Split into two rows */
const row1 = integrations;
const row2 = [...integrations].reverse();

/* ------------------------------------------------------------------ */
/* Integrations section                                               */
/* ------------------------------------------------------------------ */

export function Integrations() {
  return (
    <section id="integrations" className="relative py-24 px-6 overflow-hidden">
      <div className="pointer-events-none absolute bottom-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-gradient-to-t from-cyan-500/[0.03] to-transparent rounded-full blur-3xl" />
      <div className="max-w-7xl mx-auto text-center">
        {/* Section heading */}
        <div className="mb-12">
          <h2 className="text-3xl md:text-5xl font-bold text-white">
            Works With Your Favorite Tools
          </h2>
          <p className="text-lg text-slate-400 mt-4">
            OpenBrowser integrates with any MCP-compatible client.
          </p>
        </div>

        {/* Marquee rows */}
        <div className="space-y-4">
          <Marquee pauseOnHover className="[--duration:30s]">
            {row1.map((item) => (
              <IntegrationCard key={item.name} {...item} />
            ))}
          </Marquee>

          <Marquee pauseOnHover reverse className="[--duration:30s]">
            {row2.map((item) => (
              <IntegrationCard key={item.name} {...item} />
            ))}
          </Marquee>
        </div>
      </div>
    </section>
  );
}
