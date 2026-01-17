"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "primary" | "secondary" | "ghost" | "outline";
  size?: "sm" | "md" | "lg" | "icon";
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "md", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center font-medium transition-all duration-200",
          "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500/50",
          "disabled:opacity-50 disabled:cursor-not-allowed",
          // Variants
          variant === "default" && "bg-zinc-800 text-zinc-100 hover:bg-zinc-700",
          variant === "primary" && "bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:from-cyan-400 hover:to-blue-500 shadow-lg shadow-cyan-500/25",
          variant === "secondary" && "bg-zinc-700/50 text-zinc-200 hover:bg-zinc-600/50",
          variant === "ghost" && "bg-transparent text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50",
          variant === "outline" && "border border-zinc-700 bg-transparent text-zinc-300 hover:bg-zinc-800/50 hover:border-zinc-600",
          // Sizes
          size === "sm" && "h-8 px-3 text-sm rounded-lg",
          size === "md" && "h-10 px-4 text-sm rounded-xl",
          size === "lg" && "h-12 px-6 text-base rounded-xl",
          size === "icon" && "h-10 w-10 rounded-xl",
          className
        )}
        {...props}
      />
    );
  }
);

Button.displayName = "Button";
