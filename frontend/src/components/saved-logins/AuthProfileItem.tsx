"use client";

import { Trash2 } from "lucide-react";
import type { AuthProfile } from "@/types";

const statusColors: Record<string, string> = {
  active: "bg-emerald-400",
  expired: "bg-amber-400",
  revoked: "bg-red-400",
};

interface AuthProfileItemProps {
  profile: AuthProfile;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
}

export function AuthProfileItem({ profile, isSelected, onSelect, onDelete }: AuthProfileItemProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`group flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm transition-colors duration-200 ${
        isSelected
          ? "bg-zinc-700/60 text-zinc-100"
          : "text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200"
      }`}
    >
      <img
        src={`https://www.google.com/s2/favicons?domain=${profile.domain}&sz=16`}
        alt=""
        width={16}
        height={16}
        className="shrink-0 rounded-sm"
      />
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium">{profile.label}</div>
        <div className="truncate text-xs text-zinc-500">{profile.domain}</div>
      </div>
      <span className={`h-2 w-2 shrink-0 rounded-full ${statusColors[profile.status] ?? "bg-zinc-500"}`} />
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        className="shrink-0 rounded p-0.5 text-zinc-500 opacity-0 transition-opacity duration-200 hover:text-red-400 group-hover:opacity-100"
        aria-label="Delete profile"
      >
        <Trash2 size={14} />
      </button>
    </button>
  );
}
