"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";

export function Waitlist() {
  const [form, setForm] = useState({ fullName: "", email: "", useCase: "" });
  const [submitState, setSubmitState] = useState<
    "idle" | "submitting" | "success" | "error"
  >("idle");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitState("submitting");
    try {
      const apiUrl = process.env.NEXT_PUBLIC_WAITLIST_API_URL;
      if (!apiUrl) {
        // If no API URL configured, simulate success for dev
        await new Promise((r) => setTimeout(r, 1000));
        setSubmitState("success");
        return;
      }
      const res = await fetch(apiUrl + "/waitlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fullName: form.fullName,
          email: form.email,
          useCase: form.useCase,
        }),
      });
      if (!res.ok) throw new Error("Failed");
      setSubmitState("success");
    } catch {
      setSubmitState("error");
    }
  };

  return (
    <section id="waitlist" className="py-24 px-6">
      <div className="max-w-2xl mx-auto text-center">
        {/* Heading */}
        <div className="mb-12">
          <h2 className="text-3xl md:text-5xl font-bold text-white">
            Get Early Access
          </h2>
          <p className="text-lg text-slate-400 mt-4">
            Be the first to try the hosted version of OpenBrowser.
          </p>
        </div>

        {/* Success state */}
        {submitState === "success" ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="w-16 h-16 rounded-full bg-green-500/10 mx-auto flex items-center justify-center">
              <svg
                width="32"
                height="32"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-green-400"
              >
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </div>
            <h3 className="text-2xl font-bold text-white mt-6">
              You&apos;re on the list!
            </h3>
            <p className="text-slate-400 mt-2">
              We&apos;ll be in touch soon with early access details.
            </p>
          </motion.div>
        ) : (
          /* Form */
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <form
              onSubmit={handleSubmit}
              className="text-left space-y-4"
            >
              {/* Full Name */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1.5">
                  Full Name
                </label>
                <input
                  type="text"
                  required
                  value={form.fullName}
                  onChange={(e) =>
                    setForm({ ...form, fullName: e.target.value })
                  }
                  placeholder="Jane Doe"
                  className="w-full bg-zinc-900 border border-white/10 rounded-lg px-4 py-3 text-white placeholder:text-slate-500 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 focus:outline-none transition"
                />
              </div>

              {/* Email */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1.5">
                  Email
                </label>
                <input
                  type="email"
                  required
                  value={form.email}
                  onChange={(e) =>
                    setForm({ ...form, email: e.target.value })
                  }
                  placeholder="jane@example.com"
                  className="w-full bg-zinc-900 border border-white/10 rounded-lg px-4 py-3 text-white placeholder:text-slate-500 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 focus:outline-none transition"
                />
              </div>

              {/* Use Case */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1.5">
                  Use Case
                </label>
                <textarea
                  rows={3}
                  value={form.useCase}
                  onChange={(e) =>
                    setForm({ ...form, useCase: e.target.value })
                  }
                  placeholder="What would you use OpenBrowser for?"
                  className="w-full bg-zinc-900 border border-white/10 rounded-lg px-4 py-3 text-white placeholder:text-slate-500 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 focus:outline-none transition resize-none"
                />
              </div>

              {/* Submit button */}
              <button
                type="submit"
                disabled={submitState === "submitting"}
                className="w-full mt-2 bg-gradient-to-r from-cyan-500 to-violet-500 text-white font-semibold rounded-lg py-3 hover:opacity-90 transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {submitState === "submitting"
                  ? "Submitting..."
                  : "Join the Waitlist"}
              </button>

              {/* Error state */}
              {submitState === "error" && (
                <p
                  className="text-sm text-red-400 mt-2 text-center cursor-pointer"
                  onClick={() => setSubmitState("idle")}
                >
                  Something went wrong. Please try again.
                </p>
              )}
            </form>
          </motion.div>
        )}
      </div>
    </section>
  );
}
