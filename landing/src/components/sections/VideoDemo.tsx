"use client";

import React from "react";
import { motion } from "framer-motion";

/* ------------------------------------------------------------------ */
/* VideoDemo section                                                  */
/* ------------------------------------------------------------------ */

export function VideoDemo() {
  return (
    <section id="demo" className="relative py-24 px-6 overflow-hidden">
      <div className="pointer-events-none absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[500px] bg-gradient-to-br from-cyan-500/[0.05] via-transparent to-violet-500/[0.05] rounded-full blur-3xl" />
      <div className="max-w-7xl mx-auto text-center">
        {/* Section heading */}
        <div className="mb-12">
          <h2 className="text-3xl md:text-5xl font-bold text-white">
            See OpenBrowser in Action
          </h2>
          <p className="text-lg text-slate-400 mt-4">
            Watch a real agent browse the web autonomously.
          </p>
        </div>

        {/* Video container */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="max-w-4xl mx-auto"
        >
          <div
            className="rounded-2xl overflow-hidden border border-white/[0.08]"
            style={{
              boxShadow:
                "0 0 60px rgba(14, 165, 233, 0.12), 0 0 120px rgba(139, 92, 246, 0.08)",
            }}
          >
            <iframe
              className="aspect-video w-full"
              src="https://www.youtube.com/embed/4_I53tt1S4Q"
              title="OpenBrowser Demo"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
        </motion.div>
      </div>
    </section>
  );
}
