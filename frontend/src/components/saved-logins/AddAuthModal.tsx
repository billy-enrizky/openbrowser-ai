"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Loader2 } from "lucide-react";
import { startAuthSession, saveAuthProfile } from "@/lib/auth-profiles-api";
import { useAppStore } from "@/store";

interface AddAuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  token: string | null;
}

export function AddAuthModal({ isOpen, onClose, token }: AddAuthModalProps) {
  const [domain, setDomain] = useState("");
  const [label, setLabel] = useState("");
  const [taskId, setTaskId] = useState<string | null>(null);
  const [vncUrl, setVncUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const addAuthProfile = useAppStore((s) => s.addAuthProfile);

  const handleStartSession = async () => {
    if (!domain.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const result = await startAuthSession(token, domain.trim(), label.trim() || domain.trim());
      setTaskId(result.task_id);
      setVncUrl(result.vnc_url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start session");
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!taskId) return;
    setLoading(true);
    setError(null);
    try {
      const profile = await saveAuthProfile(token, taskId, domain.trim(), label.trim() || domain.trim());
      addAuthProfile(profile);
      onClose();
      resetState();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save profile");
    } finally {
      setLoading(false);
    }
  };

  const resetState = () => {
    setDomain("");
    setLabel("");
    setTaskId(null);
    setVncUrl(null);
    setError(null);
  };

  const handleClose = () => {
    onClose();
    resetState();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={handleClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 10 }}
            transition={{ duration: 0.2 }}
            className="relative w-full max-w-lg rounded-xl border border-zinc-700/50 bg-zinc-900 p-6 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              onClick={handleClose}
              className="absolute right-4 top-4 rounded p-1 text-zinc-500 hover:text-zinc-300 transition-colors"
              aria-label="Close"
            >
              <X size={18} />
            </button>

            <h2 className="mb-4 text-lg font-semibold text-zinc-100">Save Login</h2>

            {!taskId ? (
              <div className="flex flex-col gap-4">
                <div>
                  <label htmlFor="auth-domain" className="mb-1.5 block text-sm font-medium text-zinc-400">
                    Domain
                  </label>
                  <input
                    id="auth-domain"
                    type="text"
                    value={domain}
                    onChange={(e) => setDomain(e.target.value)}
                    placeholder="example.com"
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-500 transition-colors"
                  />
                </div>
                <div>
                  <label htmlFor="auth-label" className="mb-1.5 block text-sm font-medium text-zinc-400">
                    Label (optional)
                  </label>
                  <input
                    id="auth-label"
                    type="text"
                    value={label}
                    onChange={(e) => setLabel(e.target.value)}
                    placeholder="My account"
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-500 transition-colors"
                  />
                </div>
                {error && <p className="text-sm text-red-400">{error}</p>}
                <button
                  type="button"
                  onClick={handleStartSession}
                  disabled={loading || !domain.trim()}
                  className="flex items-center justify-center gap-2 rounded-lg bg-zinc-100 px-4 py-2.5 text-sm font-medium text-zinc-900 transition-colors hover:bg-white disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading && <Loader2 size={16} className="animate-spin" />}
                  Start Login Session
                </button>
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                <p className="text-sm text-zinc-400">
                  Log in to <span className="font-medium text-zinc-200">{domain}</span> in the browser below, then click Save.
                </p>
                {vncUrl && (
                  <div className="aspect-video w-full overflow-hidden rounded-lg border border-zinc-700 bg-black">
                    <iframe
                      src={vncUrl}
                      title="Browser session"
                      className="h-full w-full"
                      sandbox="allow-scripts allow-same-origin"
                    />
                  </div>
                )}
                {error && <p className="text-sm text-red-400">{error}</p>}
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={handleClose}
                    className="flex-1 rounded-lg border border-zinc-700 px-4 py-2.5 text-sm font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={handleSave}
                    disabled={loading}
                    className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-zinc-100 px-4 py-2.5 text-sm font-medium text-zinc-900 transition-colors hover:bg-white disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading && <Loader2 size={16} className="animate-spin" />}
                    Save
                  </button>
                </div>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
