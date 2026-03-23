"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Pause, Play, Trash2, Loader2 } from "lucide-react";
import { fetchExecutions } from "@/lib/schedules-api";
import type { JobExecution, ScheduledJob } from "@/types";

const statusColors: Record<string, string> = {
  running: "bg-blue-400",
  success: "bg-emerald-400",
  failed: "bg-red-400",
  auth_expired: "bg-amber-400",
};

const statusLabels: Record<string, string> = {
  running: "Running",
  success: "Success",
  failed: "Failed",
  auth_expired: "Auth Expired",
};

interface ExecutionHistoryProps {
  job: ScheduledJob | null;
  isOpen: boolean;
  onClose: () => void;
  onTogglePause: () => void;
  onDelete: () => void;
  token: string | null;
}

export function ExecutionHistory({ job, isOpen, onClose, onTogglePause, onDelete, token }: ExecutionHistoryProps) {
  const [executions, setExecutions] = useState<JobExecution[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!job || !isOpen) return;
    setLoading(true);
    fetchExecutions(token, job.id)
      .then(setExecutions)
      .catch(() => setExecutions([]))
      .finally(() => setLoading(false));
  }, [job?.id, isOpen, token]);

  if (!job) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={onClose}
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
              onClick={onClose}
              className="absolute right-4 top-4 rounded p-1 text-zinc-500 hover:text-zinc-300 transition-colors"
              aria-label="Close"
            >
              <X size={18} />
            </button>

            <h2 className="mb-1 text-lg font-semibold text-zinc-100">{job.title}</h2>
            <p className="mb-4 text-sm text-zinc-500">{job.scheduleExpression} ({job.scheduleTimezone})</p>

            <div className="mb-4 flex items-center gap-2">
              <span className={`h-2.5 w-2.5 rounded-full ${statusColors[job.status] ?? "bg-zinc-500"}`} />
              <span className="text-sm font-medium text-zinc-300 capitalize">{job.status}</span>
              <div className="ml-auto flex gap-2">
                {(job.status === "active" || job.status === "paused") && (
                  <button
                    type="button"
                    onClick={onTogglePause}
                    className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
                  >
                    {job.status === "active" ? <Pause size={14} /> : <Play size={14} />}
                    {job.status === "active" ? "Pause" : "Resume"}
                  </button>
                )}
                <button
                  type="button"
                  onClick={onDelete}
                  className="flex items-center gap-1.5 rounded-lg border border-red-800/50 px-3 py-1.5 text-xs font-medium text-red-400 transition-colors hover:bg-red-900/20"
                >
                  <Trash2 size={14} />
                  Delete
                </button>
              </div>
            </div>

            <div className="mb-3 rounded-lg border border-zinc-800 bg-zinc-800/40 p-3">
              <p className="text-xs font-medium text-zinc-500 mb-1">Task</p>
              <p className="text-sm text-zinc-300 whitespace-pre-wrap">{job.taskDescription}</p>
            </div>

            <h3 className="mb-2 text-sm font-semibold text-zinc-400">Recent runs</h3>
            {loading ? (
              <div className="flex items-center justify-center py-6">
                <Loader2 size={20} className="animate-spin text-zinc-500" />
              </div>
            ) : executions.length === 0 ? (
              <p className="py-4 text-center text-xs text-zinc-500">No executions yet</p>
            ) : (
              <div className="flex flex-col gap-1.5">
                {executions.map((ex) => (
                  <div
                    key={ex.id}
                    className="flex items-center gap-2.5 rounded-lg bg-zinc-800/40 px-3 py-2 text-sm"
                  >
                    <span className={`h-2 w-2 shrink-0 rounded-full ${statusColors[ex.status] ?? "bg-zinc-500"}`} />
                    <span className="text-zinc-300">
                      {statusLabels[ex.status] ?? ex.status}
                    </span>
                    <span className="ml-auto text-xs text-zinc-500">
                      {new Date(ex.startedAt).toLocaleString()}
                    </span>
                    {ex.completedAt && ex.startedAt && (
                      <span className="text-xs text-zinc-600">
                        {Math.round((new Date(ex.completedAt).getTime() - new Date(ex.startedAt).getTime()) / 1000)}s
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
