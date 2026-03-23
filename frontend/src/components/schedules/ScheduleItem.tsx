"use client";

import { Pause, Play, Trash2 } from "lucide-react";
import type { ScheduledJob } from "@/types";

const statusColors: Record<string, string> = {
  active: "bg-emerald-400",
  paused: "bg-amber-400",
  testing: "bg-blue-400",
  failed: "bg-red-400",
};

interface ScheduleItemProps {
  job: ScheduledJob;
  onSelect: () => void;
  onTogglePause: () => void;
  onDelete: () => void;
}

export function ScheduleItem({ job, onSelect, onTogglePause, onDelete }: ScheduleItemProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className="group flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm text-zinc-400 transition-colors duration-200 hover:bg-zinc-800/60 hover:text-zinc-200"
    >
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium">{job.title}</div>
        <div className="truncate text-xs text-zinc-500">{job.scheduleExpression}</div>
      </div>
      <span className={`h-2 w-2 shrink-0 rounded-full ${statusColors[job.status] ?? "bg-zinc-500"}`} />
      <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
        {(job.status === "active" || job.status === "paused") && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onTogglePause();
            }}
            className="rounded p-0.5 text-zinc-500 hover:text-zinc-300"
            aria-label={job.status === "active" ? "Pause" : "Resume"}
          >
            {job.status === "active" ? <Pause size={14} /> : <Play size={14} />}
          </button>
        )}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className="rounded p-0.5 text-zinc-500 hover:text-red-400"
          aria-label="Delete schedule"
        >
          <Trash2 size={14} />
        </button>
      </div>
    </button>
  );
}
