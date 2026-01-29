"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Cpu, Check, AlertCircle } from "lucide-react";
import { useAppStore } from "@/store";
import { cn } from "@/lib/utils";
import type { LLMModel } from "@/types";

// Provider icons/colors
const providerConfig: Record<string, { color: string; bgColor: string; label: string }> = {
  google: { color: "text-blue-400", bgColor: "bg-blue-500/10", label: "Google" },
  openai: { color: "text-green-400", bgColor: "bg-green-500/10", label: "OpenAI" },
  anthropic: { color: "text-orange-400", bgColor: "bg-orange-500/10", label: "Anthropic" },
};

export function ModelSelector() {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  
  const {
    availableModels,
    availableProviders,
    selectedModel,
    setSelectedModel,
    modelsLoading,
    modelsError,
  } = useAppStore();

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Get current model info
  const currentModel = availableModels.find((m) => m.id === selectedModel);
  const currentProvider = currentModel ? providerConfig[currentModel.provider] : null;

  // Group models by provider
  const modelsByProvider = availableModels.reduce((acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = [];
    }
    acc[model.provider].push(model);
    return acc;
  }, {} as Record<string, LLMModel[]>);

  // Handle model selection
  const handleSelectModel = (modelId: string) => {
    setSelectedModel(modelId);
    setIsOpen(false);
  };

  // Show loading state
  if (modelsLoading) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-800/50 text-zinc-400">
        <div className="w-4 h-4 border-2 border-zinc-600 border-t-zinc-400 rounded-full animate-spin" />
        <span className="text-sm">Loading models...</span>
      </div>
    );
  }

  // Show error state
  if (modelsError) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400">
        <AlertCircle className="w-4 h-4" />
        <span className="text-sm">No API keys configured</span>
      </div>
    );
  }

  // Show empty state if no models available
  if (availableModels.length === 0) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-800/50 text-zinc-500">
        <Cpu className="w-4 h-4" />
        <span className="text-sm">No models available</span>
      </div>
    );
  }

  return (
    <div className="relative z-[100]" ref={dropdownRef}>
      {/* Trigger Button */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all",
          "bg-zinc-800/50 hover:bg-zinc-700/50 border border-transparent",
          isOpen && "border-cyan-500/30 bg-zinc-700/50"
        )}
      >
        {currentProvider && (
          <div className={cn("w-2 h-2 rounded-full", currentProvider.color.replace("text-", "bg-"))} />
        )}
        <Cpu className="w-4 h-4 text-zinc-400" />
        <span className="text-sm font-medium text-zinc-300 max-w-[150px] truncate">
          {currentModel?.name || "Select Model"}
        </span>
        <ChevronDown className={cn(
          "w-4 h-4 text-zinc-500 transition-transform",
          isOpen && "rotate-180"
        )} />
      </motion.button>

      {/* Dropdown Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className={cn(
              "absolute top-full left-0 mt-2 z-[9999]",
              "w-72 max-h-[400px] overflow-y-auto",
              "bg-zinc-900 border border-zinc-700/50 rounded-xl",
              "shadow-2xl shadow-black/50"
            )}
          >
            {/* Header */}
            <div className="px-4 py-3 border-b border-zinc-800">
              <h3 className="text-sm font-medium text-zinc-300">Select Model</h3>
              <p className="text-xs text-zinc-500 mt-0.5">
                {availableProviders.length} provider{availableProviders.length !== 1 ? "s" : ""} available
              </p>
            </div>

            {/* Models grouped by provider */}
            <div className="py-2">
              {Object.entries(modelsByProvider).map(([provider, models]) => {
                const config = providerConfig[provider] || { color: "text-zinc-400", bgColor: "bg-zinc-500/10", label: provider };
                return (
                  <div key={provider} className="mb-2 last:mb-0">
                    {/* Provider Header */}
                    <div className="px-4 py-1.5 flex items-center gap-2">
                      <div className={cn("w-2 h-2 rounded-full", config.color.replace("text-", "bg-"))} />
                      <span className={cn("text-xs font-medium uppercase tracking-wider", config.color)}>
                        {config.label}
                      </span>
                    </div>

                    {/* Models */}
                    {models.map((model) => {
                      const isSelected = model.id === selectedModel;
                      return (
                        <button
                          key={model.id}
                          onClick={() => handleSelectModel(model.id)}
                          className={cn(
                            "w-full px-4 py-2 flex items-center justify-between",
                            "hover:bg-zinc-800/50 transition-colors",
                            isSelected && "bg-cyan-500/10"
                          )}
                        >
                          <div className="flex items-center gap-3">
                            <div className={cn(
                              "w-8 h-8 rounded-lg flex items-center justify-center",
                              config.bgColor
                            )}>
                              <Cpu className={cn("w-4 h-4", config.color)} />
                            </div>
                            <div className="text-left">
                              <div className={cn(
                                "text-sm font-medium",
                                isSelected ? "text-cyan-300" : "text-zinc-300"
                              )}>
                                {model.name}
                              </div>
                              <div className="text-xs text-zinc-500 font-mono">
                                {model.id}
                              </div>
                            </div>
                          </div>
                          {isSelected && (
                            <Check className="w-4 h-4 text-cyan-400" />
                          )}
                        </button>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
