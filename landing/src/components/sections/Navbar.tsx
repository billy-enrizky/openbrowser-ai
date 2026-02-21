"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import Link from "next/link";
import { cn } from "@/lib/utils";

const navLinks = [
  {
    name: "Docs",
    href: "https://docs.openbrowser.me",
    external: true,
  },
  {
    name: "GitHub",
    href: "https://github.com/billy-enrizky/openbrowser-ai",
    external: true,
  },
  {
    name: "Discord",
    href: "https://discord.gg/YRXzbJjq9K",
    external: true,
  },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleWaitlistClick = () => {
    setMobileOpen(false);
    document.getElementById("waitlist")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <>
      <nav
        className={cn(
          "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
          scrolled
            ? "backdrop-blur-lg bg-zinc-950/80 border-b border-white/5"
            : "bg-transparent"
        )}
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between px-6 h-16">
          {/* Left: Logo + Brand */}
          <Link href="/" className="flex items-center gap-2">
            <Image
              src="/logo.svg"
              alt="OpenBrowser"
              width={28}
              height={28}
            />
            <span className="font-bold text-lg text-white">OpenBrowser</span>
          </Link>

          {/* Center: Desktop links */}
          <div className="hidden md:flex items-center gap-6">
            {navLinks.map((link) => (
              <a
                key={link.name}
                href={link.href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-slate-300 hover:text-white transition font-medium"
              >
                {link.name}
              </a>
            ))}
          </div>

          {/* Right: CTA + Mobile hamburger */}
          <div className="flex items-center gap-4">
            <button
              onClick={handleWaitlistClick}
              className="hidden md:block text-sm font-medium bg-gradient-to-r from-cyan-500 to-violet-500 text-white rounded-full px-4 py-2 hover:opacity-90 transition"
            >
              Join Waitlist
            </button>

            {/* Mobile hamburger */}
            <button
              className="md:hidden flex flex-col justify-center items-center gap-1.5 w-8 h-8"
              onClick={() => setMobileOpen(!mobileOpen)}
              aria-label="Toggle menu"
            >
              <span
                className={cn(
                  "block w-5 h-0.5 bg-white transition-all duration-300",
                  mobileOpen && "rotate-45 translate-y-2"
                )}
              />
              <span
                className={cn(
                  "block w-5 h-0.5 bg-white transition-all duration-300",
                  mobileOpen && "opacity-0"
                )}
              />
              <span
                className={cn(
                  "block w-5 h-0.5 bg-white transition-all duration-300",
                  mobileOpen && "-rotate-45 -translate-y-2"
                )}
              />
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile menu overlay */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-40 bg-zinc-950/95 backdrop-blur-lg pt-20 px-6 md:hidden"
          >
            <div className="flex flex-col items-center gap-8">
              {navLinks.map((link) => (
                <a
                  key={link.name}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-lg text-slate-300 hover:text-white transition"
                  onClick={() => setMobileOpen(false)}
                >
                  {link.name}
                </a>
              ))}
              <button
                onClick={handleWaitlistClick}
                className="text-sm font-medium bg-gradient-to-r from-cyan-500 to-violet-500 text-white rounded-full px-6 py-3 hover:opacity-90 transition mt-4"
              >
                Join Waitlist
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
