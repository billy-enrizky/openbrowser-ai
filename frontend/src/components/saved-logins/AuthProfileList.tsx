"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, Plus } from "lucide-react";
import { useAppStore } from "@/store";
import { AuthProfileItem } from "./AuthProfileItem";

interface AuthProfileListProps {
  onAdd: () => void;
  onDelete: (id: string) => void;
}

export function AuthProfileList({ onAdd, onDelete }: AuthProfileListProps) {
  const [isOpen, setIsOpen] = useState(true);
  const authProfiles = useAppStore((s) => s.authProfiles);
  const selectedAuthProfileId = useAppStore((s) => s.selectedAuthProfileId);
  const setSelectedAuthProfileId = useAppStore((s) => s.setSelectedAuthProfileId);

  return (
    <div className="flex flex-col">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between px-3 py-2 text-xs font-semibold uppercase tracking-wider text-zinc-500 hover:text-zinc-300 transition-colors duration-200"
      >
        <span className="flex items-center gap-1.5">
          {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          Saved Logins
        </span>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onAdd();
          }}
          className="rounded p-0.5 text-zinc-500 hover:bg-zinc-700/50 hover:text-zinc-300 transition-colors duration-200"
          aria-label="Add auth profile"
        >
          <Plus size={14} />
        </button>
      </button>
      {isOpen && (
        <div className="max-h-48 overflow-y-auto px-1.5 pb-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-700">
          {authProfiles.length === 0 ? (
            <p className="px-3 py-3 text-xs text-zinc-500">No saved logins yet</p>
          ) : (
            <div className="flex flex-col gap-0.5">
              {authProfiles.map((profile) => (
                <AuthProfileItem
                  key={profile.id}
                  profile={profile}
                  isSelected={selectedAuthProfileId === profile.id}
                  onSelect={() =>
                    setSelectedAuthProfileId(
                      selectedAuthProfileId === profile.id ? null : profile.id
                    )
                  }
                  onDelete={() => onDelete(profile.id)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
