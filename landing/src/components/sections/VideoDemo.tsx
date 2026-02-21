"use client";

import React from "react";
import { motion } from "framer-motion";

/* ------------------------------------------------------------------ */
/* VideoDemo section                                                  */
/* ------------------------------------------------------------------ */

export function VideoDemo() {
  return (
    <section id="demo" className="py-24 px-6">
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
            className="rounded-2xl overflow-hidden border border-white/10"
            style={{
              boxShadow:
                "0 0 80px rgba(14, 165, 233, 0.15), 0 0 160px rgba(139, 92, 246, 0.1)",
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
