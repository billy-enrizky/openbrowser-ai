"use client";

import React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { AuroraBackground } from "@/components/ui/aurora-background";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";

export function Hero() {
  const handleWaitlistClick = () => {
    document.getElementById("waitlist")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <AuroraBackground className="min-h-screen bg-zinc-950 dark:bg-zinc-950">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.2 }}
        className="relative z-10 flex flex-col items-center text-center gap-6 px-6"
      >
        {/* Open Source badge */}
        <a
          href="https://github.com/billy-enrizky/openbrowser-ai"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-sm text-slate-300 hover:bg-white/10 transition"
        >
          Open Source
          <svg
            width="12"
            height="12"
            viewBox="0 0 12 12"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className="ml-1"
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

        {/* Headline */}
        <div className="max-w-4xl">
          <TextGenerateEffect
            words="The General-Purpose Agentic Browser"
            className={cn(
              "text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight",
              "[&_div]:!text-5xl [&_div]:md:!text-6xl [&_div]:lg:!text-7xl",
              "[&_div]:!leading-tight [&_div]:!tracking-tight"
            )}
          />
        </div>

        {/* Subheadline */}
        <p className="text-lg md:text-xl text-slate-400 max-w-2xl">
          Control any browser with natural language. 877x more token-efficient
          than alternatives.
        </p>

        {/* CTA buttons */}
        <div className="flex flex-col sm:flex-row gap-4 mt-4">
          <button
            onClick={handleWaitlistClick}
            className="bg-gradient-to-r from-cyan-500 to-violet-500 text-white font-semibold rounded-full px-8 py-3 text-lg hover:opacity-90 transition shadow-lg shadow-cyan-500/25"
          >
            Join the Waitlist
          </button>
          <a
            href="https://docs.openbrowser.me"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-white/10 px-8 py-3 text-lg text-slate-300 hover:bg-white/5 transition text-center"
          >
            Read the Docs
          </a>
        </div>
      </motion.div>
    </AuroraBackground>
  );
}
