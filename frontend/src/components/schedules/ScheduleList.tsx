"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, Plus } from "lucide-react";
import { useAppStore } from "@/store";
import { ScheduleItem } from "./ScheduleItem";

interface ScheduleListProps {
  onAdd: () => void;
  onSelect: (jobId: string) => void;
  onTogglePause: (jobId: string, currentStatus: string) => void;
  onDelete: (jobId: string) => void;
}

export function ScheduleList({ onAdd, onSelect, onTogglePause, onDelete }: ScheduleListProps) {
  const [isOpen, setIsOpen] = useState(true);
  const scheduledJobs = useAppStore((s) => s.scheduledJobs);

  return (
    <div className="flex flex-col">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between px-3 py-2 text-xs font-semibold uppercase tracking-wider text-zinc-500 hover:text-zinc-300 transition-colors duration-200"
      >
        <span className="flex items-center gap-1.5">
          {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          Schedules
        </span>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onAdd();
          }}
          className="rounded p-0.5 text-zinc-500 hover:bg-zinc-700/50 hover:text-zinc-300 transition-colors duration-200"
          aria-label="Add schedule"
        >
          <Plus size={14} />
        </button>
      </button>
      {isOpen && (
        <div className="max-h-48 overflow-y-auto px-1.5 pb-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-700">
          {scheduledJobs.length === 0 ? (
            <p className="px-3 py-3 text-xs text-zinc-500">No scheduled jobs yet</p>
          ) : (
            <div className="flex flex-col gap-0.5">
              {scheduledJobs.map((job) => (
                <ScheduleItem
                  key={job.id}
                  job={job}
                  onSelect={() => onSelect(job.id)}
                  onTogglePause={() => onTogglePause(job.id, job.status)}
                  onDelete={() => onDelete(job.id)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
