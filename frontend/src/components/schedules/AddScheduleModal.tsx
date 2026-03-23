"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Loader2 } from "lucide-react";
import { createScheduledJob } from "@/lib/schedules-api";
import { useAppStore } from "@/store";
import { SchedulePicker } from "./SchedulePicker";

interface AddScheduleModalProps {
  isOpen: boolean;
  onClose: () => void;
  token: string | null;
}

export function AddScheduleModal({ isOpen, onClose, token }: AddScheduleModalProps) {
  const [title, setTitle] = useState("");
  const [taskDescription, setTaskDescription] = useState("");
  const [expression, setExpression] = useState("0 9 * * ? *");
  const [timezone, setTimezone] = useState("UTC");
  const [authProfileId, setAuthProfileId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const authProfiles = useAppStore((s) => s.authProfiles);
  const addScheduledJob = useAppStore((s) => s.addScheduledJob);

  const handleSubmit = async () => {
    if (!title.trim() || !taskDescription.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const job = await createScheduledJob(token, {
        title: title.trim(),
        task_description: taskDescription.trim(),
        schedule_expression: expression,
        schedule_timezone: timezone,
        auth_profile_id: authProfileId || undefined,
      });
      addScheduledJob(job);
      onClose();
      resetState();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create schedule");
    } finally {
      setLoading(false);
    }
  };

  const resetState = () => {
    setTitle("");
    setTaskDescription("");
    setExpression("0 9 * * ? *");
    setTimezone("UTC");
    setAuthProfileId(null);
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
            className="relative w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-xl border border-zinc-700/50 bg-zinc-900 p-6 shadow-2xl"
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

            <h2 className="mb-4 text-lg font-semibold text-zinc-100">New Schedule</h2>

            <div className="flex flex-col gap-4">
              <div>
                <label htmlFor="schedule-title" className="mb-1.5 block text-sm font-medium text-zinc-400">
                  Title
                </label>
                <input
                  id="schedule-title"
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Daily price check"
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-500 transition-colors"
                />
              </div>

              <div>
                <label htmlFor="schedule-task" className="mb-1.5 block text-sm font-medium text-zinc-400">
                  Task description
                </label>
                <textarea
                  id="schedule-task"
                  value={taskDescription}
                  onChange={(e) => setTaskDescription(e.target.value)}
                  placeholder="Go to example.com and check the price of..."
                  rows={4}
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-500 transition-colors resize-none"
                />
              </div>

              <div>
                <label className="mb-1.5 block text-sm font-medium text-zinc-400">Schedule</label>
                <SchedulePicker
                  expression={expression}
                  timezone={timezone}
                  onExpressionChange={setExpression}
                  onTimezoneChange={setTimezone}
                />
              </div>

              {authProfiles.length > 0 && (
                <div>
                  <label htmlFor="schedule-auth" className="mb-1.5 block text-sm font-medium text-zinc-400">
                    Saved login (optional)
                  </label>
                  <select
                    id="schedule-auth"
                    value={authProfileId ?? ""}
                    onChange={(e) => setAuthProfileId(e.target.value || null)}
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 focus:border-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-500 transition-colors"
                  >
                    <option value="">None</option>
                    {authProfiles.map((profile) => (
                      <option key={profile.id} value={profile.id}>
                        {profile.label} ({profile.domain})
                      </option>
                    ))}
                  </select>
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
                  onClick={handleSubmit}
                  disabled={loading || !title.trim() || !taskDescription.trim()}
                  className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-zinc-100 px-4 py-2.5 text-sm font-medium text-zinc-900 transition-colors hover:bg-white disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading && <Loader2 size={16} className="animate-spin" />}
                  Test & Schedule
                </button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
