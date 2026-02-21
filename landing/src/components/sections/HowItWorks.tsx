"use client";

import React from "react";
import { motion } from "framer-motion";
import { TracingBeam } from "@/components/ui/tracing-beam";

/* ------------------------------------------------------------------ */
/* Code blocks                                                        */
/* ------------------------------------------------------------------ */

function InstallBlock() {
  return (
    <div className="mt-3 rounded-lg border border-white/5 bg-zinc-900 p-4 font-mono text-sm text-slate-300">
      pip install openbrowser-ai
    </div>
  );
}

function TaskCodeBlock() {
  return (
    <div className="mt-3 rounded-lg border border-white/5 bg-zinc-900 p-4 font-mono text-sm leading-relaxed overflow-x-auto">
      <div>
        <span className="text-cyan-400">from</span>
        <span className="text-slate-300"> openbrowser </span>
        <span className="text-cyan-400">import</span>
        <span className="text-slate-300"> Agent, ChatBrowserUse</span>
      </div>
      <div>
        <span className="text-cyan-400">from</span>
        <span className="text-slate-300"> dotenv </span>
        <span className="text-cyan-400">import</span>
        <span className="text-slate-300"> load_dotenv</span>
      </div>
      <div>
        <span className="text-cyan-400">import</span>
        <span className="text-slate-300"> asyncio</span>
      </div>
      <div className="mt-2">
        <span className="text-violet-400">load_dotenv</span>
        <span className="text-slate-400">()</span>
      </div>
      <div className="mt-2">
        <span className="text-cyan-400">async def</span>
        <span className="text-violet-400"> main</span>
        <span className="text-slate-400">():</span>
      </div>
      <div>
        <span className="text-slate-300">    llm = </span>
        <span className="text-violet-400">ChatBrowserUse</span>
        <span className="text-slate-400">()</span>
      </div>
      <div>
        <span className="text-slate-300">    agent = </span>
        <span className="text-violet-400">Agent</span>
        <span className="text-slate-400">(</span>
      </div>
      <div>
        <span className="text-slate-300">        task=</span>
        <span className="text-green-400">&quot;Find the #1 post on Show HN&quot;</span>
        <span className="text-slate-400">,</span>
      </div>
      <div>
        <span className="text-slate-300">        llm=llm</span>
        <span className="text-slate-400">,</span>
      </div>
      <div>
        <span className="text-slate-300">    </span>
        <span className="text-slate-400">)</span>
      </div>
      <div>
        <span className="text-slate-300">    </span>
        <span className="text-cyan-400">await</span>
        <span className="text-slate-300"> agent.</span>
        <span className="text-violet-400">run</span>
        <span className="text-slate-400">()</span>
      </div>
      <div className="mt-2">
        <span className="text-slate-300">asyncio.</span>
        <span className="text-violet-400">run</span>
        <span className="text-slate-400">(</span>
        <span className="text-violet-400">main</span>
        <span className="text-slate-400">())</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Step component                                                     */
/* ------------------------------------------------------------------ */

interface StepProps {
  number: number;
  title: string;
  children: React.ReactNode;
  isLast?: boolean;
}

function Step({ number, title, children, isLast = false }: StepProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: number * 0.15 }}
      className={isLast ? "" : "mb-16"}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-r from-cyan-500 to-violet-500 text-white font-bold text-lg">
        {number}
      </div>
      <h3 className="mt-4 text-xl font-semibold text-white">{title}</h3>
      {children}
    </motion.div>
  );
}

/* ------------------------------------------------------------------ */
/* HowItWorks section                                                 */
/* ------------------------------------------------------------------ */

export function HowItWorks() {
  return (
    <section id="how-it-works" className="py-24 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Section heading */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-bold text-white">
            Get Started in 3 Steps
          </h2>
          <p className="text-lg text-slate-400 mt-4">
            From zero to autonomous browsing in minutes.
          </p>
        </div>

        {/* Timeline with tracing beam */}
        <div className="max-w-3xl mx-auto">
          <TracingBeam>
            <div className="pl-8 md:pl-12">
              <Step number={1} title="Install">
                <InstallBlock />
              </Step>

              <Step number={2} title="Write a Task">
                <TaskCodeBlock />
              </Step>

              <Step number={3} title="Watch It Browse" isLast>
                <p className="text-slate-400 mt-3">
                  The agent navigates, clicks, types, and extracts data
                  autonomously. Watch it work in real-time through the live
                  browser view.
                </p>
              </Step>
            </div>
          </TracingBeam>
        </div>
      </div>
    </section>
  );
}
