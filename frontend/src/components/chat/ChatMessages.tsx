"use client";

import React, { useRef, useEffect } from "react";
import { AnimatePresence } from "framer-motion";
import { ChatMessage } from "./ChatMessage";
import type { Message } from "@/types";

interface ChatMessagesProps {
  messages: Message[];
}

export function ChatMessages({ messages }: ChatMessagesProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages]);

  if (messages.length === 0) {
    return null;
  }

  return (
    <div
      ref={containerRef}
      className="h-full overflow-y-auto scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent"
    >
      <AnimatePresence mode="popLayout">
        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} />
        ))}
      </AnimatePresence>
    </div>
  );
}
