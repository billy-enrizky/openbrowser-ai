"use client";

import { useState } from "react";
import { KeyRound } from "lucide-react";
import { AddAuthModal } from "./AddAuthModal";

interface SaveAuthButtonProps {
  isVisible: boolean;
  token: string | null;
}

export function SaveAuthButton({ isVisible, token }: SaveAuthButtonProps) {
  const [showModal, setShowModal] = useState(false);

  if (!isVisible) return null;

  return (
    <>
      <button
        type="button"
        onClick={() => setShowModal(true)}
        className="flex items-center gap-1.5 rounded-lg border border-zinc-700/50 px-3 py-1.5 text-xs font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
      >
        <KeyRound size={14} />
        Save Login
      </button>
      <AddAuthModal isOpen={showModal} onClose={() => setShowModal(false)} token={token} />
    </>
  );
}
