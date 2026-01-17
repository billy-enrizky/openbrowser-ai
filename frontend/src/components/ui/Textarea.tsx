"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  autoResize?: boolean;
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, autoResize = false, onChange, ...props }, ref) => {
    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      if (autoResize) {
        e.target.style.height = "auto";
        e.target.style.height = `${e.target.scrollHeight}px`;
      }
      onChange?.(e);
    };

    return (
      <textarea
        ref={ref}
        onChange={handleChange}
        className={cn(
          "w-full bg-zinc-800/50 border border-zinc-700/50 rounded-xl",
          "text-zinc-100 placeholder:text-zinc-500",
          "focus:outline-none focus:ring-2 focus:ring-cyan-500/30 focus:border-cyan-500/50",
          "transition-all duration-200 resize-none",
          "px-4 py-3",
          autoResize && "overflow-hidden",
          className
        )}
        {...props}
      />
    );
  }
);

Textarea.displayName = "Textarea";
