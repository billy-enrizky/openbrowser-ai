"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Task, Project, Message, AgentType, LogEntry, VncInfo, BrowserViewerMode, LLMModel, ChatConversation } from "@/types";

interface AppState {
  // Sidebar
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;

  // Current task
  currentTaskId: string | null;
  setCurrentTaskId: (id: string | null) => void;

  // Tasks
  tasks: Task[];
  addTask: (task: Task) => void;
  updateTask: (id: string, updates: Partial<Task>) => void;
  removeTask: (id: string) => void;

  // Projects
  projects: Project[];
  currentProjectId: string | null;
  setCurrentProjectId: (id: string | null) => void;
  addProject: (project: Project) => void;
  updateProject: (id: string, updates: Partial<Project>) => void;
  removeProject: (id: string) => void;

  // Messages (conversation history)
  messages: Message[];
  addMessage: (message: Message) => void;
  setMessages: (messages: Message[]) => void;
  clearMessages: () => void;
  conversations: ChatConversation[];
  setConversations: (conversations: ChatConversation[]) => void;
  upsertConversation: (conversation: ChatConversation) => void;
  removeConversation: (id: string) => void;
  activeConversationId: string | null;
  setActiveConversationId: (id: string | null) => void;

  // Backend logs
  logs: LogEntry[];
  addLog: (log: LogEntry) => void;
  clearLogs: () => void;
  showLogs: boolean;
  setShowLogs: (show: boolean) => void;

  // VNC / Browser Viewer
  vncInfo: VncInfo | null;
  setVncInfo: (info: VncInfo | null) => void;
  browserViewerOpen: boolean;
  setBrowserViewerOpen: (open: boolean) => void;
  toggleBrowserViewer: () => void;
  browserViewerMode: BrowserViewerMode;
  setBrowserViewerMode: (mode: BrowserViewerMode) => void;
  latestScreenshot: string | null;
  setLatestScreenshot: (screenshot: string | null) => void;

  // Settings
  agentType: AgentType;
  setAgentType: (type: AgentType) => void;
  maxSteps: number;
  setMaxSteps: (steps: number) => void;
  useVision: boolean;
  setUseVision: (use: boolean) => void;

  // LLM Model Selection
  availableModels: LLMModel[];
  setAvailableModels: (models: LLMModel[]) => void;
  availableProviders: string[];
  setAvailableProviders: (providers: string[]) => void;
  selectedModel: string | null;
  setSelectedModel: (model: string | null) => void;
  modelsLoading: boolean;
  setModelsLoading: (loading: boolean) => void;
  modelsError: string | null;
  setModelsError: (error: string | null) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Sidebar
      sidebarOpen: true,
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

      // Current task
      currentTaskId: null,
      setCurrentTaskId: (id) => set({ currentTaskId: id }),

      // Tasks
      tasks: [],
      addTask: (task) =>
        set((state) => ({ tasks: [task, ...state.tasks] })),
      updateTask: (id, updates) =>
        set((state) => ({
          tasks: state.tasks.map((t) =>
            t.id === id ? { ...t, ...updates } : t
          ),
        })),
      removeTask: (id) =>
        set((state) => ({
          tasks: state.tasks.filter((t) => t.id !== id),
        })),

      // Projects
      projects: [],
      currentProjectId: null,
      setCurrentProjectId: (id) => set({ currentProjectId: id }),
      addProject: (project) =>
        set((state) => ({ projects: [project, ...state.projects] })),
      updateProject: (id, updates) =>
        set((state) => ({
          projects: state.projects.map((p) =>
            p.id === id ? { ...p, ...updates } : p
          ),
        })),
      removeProject: (id) =>
        set((state) => ({
          projects: state.projects.filter((p) => p.id !== id),
        })),

      // Messages
      messages: [],
      addMessage: (message) =>
        set((state) => ({ messages: [...state.messages, message] })),
      setMessages: (messages) => set({ messages }),
      clearMessages: () => set({ messages: [] }),
      conversations: [],
      setConversations: (conversations) => set({ conversations }),
      upsertConversation: (conversation) =>
        set((state) => {
          const existingIndex = state.conversations.findIndex((c) => c.id === conversation.id);
          if (existingIndex === -1) {
            return { conversations: [conversation, ...state.conversations] };
          }
          const next = [...state.conversations];
          next[existingIndex] = conversation;
          next.sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());
          return { conversations: next };
        }),
      removeConversation: (id) =>
        set((state) => ({
          conversations: state.conversations.filter((c) => c.id !== id),
        })),
      activeConversationId: null,
      setActiveConversationId: (id) => set({ activeConversationId: id }),

      // Backend logs
      logs: [],
      addLog: (log) =>
        set((state) => ({ logs: [...state.logs.slice(-200), log] })), // Keep last 200 logs
      clearLogs: () => set({ logs: [] }),
      showLogs: false,
      setShowLogs: (show) => set({ showLogs: show }),

      // VNC / Browser Viewer
      vncInfo: null,
      setVncInfo: (info) => set({ vncInfo: info }),
      browserViewerOpen: false,
      setBrowserViewerOpen: (open) => set({ browserViewerOpen: open }),
      toggleBrowserViewer: () => set((state) => ({ browserViewerOpen: !state.browserViewerOpen })),
      browserViewerMode: "embedded",
      setBrowserViewerMode: (mode) => set({ browserViewerMode: mode }),
      latestScreenshot: null,
      setLatestScreenshot: (screenshot) => set({ latestScreenshot: screenshot }),

      // Settings
      agentType: "code",
      setAgentType: (type) => set({ agentType: type }),
      maxSteps: 50,
      setMaxSteps: (steps) => set({ maxSteps: steps }),
      useVision: true,
      setUseVision: (use) => set({ useVision: use }),

      // LLM Model Selection
      availableModels: [],
      setAvailableModels: (models) => set({ availableModels: models }),
      availableProviders: [],
      setAvailableProviders: (providers) => set({ availableProviders: providers }),
      selectedModel: null,
      setSelectedModel: (model) => set({ selectedModel: model }),
      modelsLoading: false,
      setModelsLoading: (loading) => set({ modelsLoading: loading }),
      modelsError: null,
      setModelsError: (error) => set({ modelsError: error }),
    }),
    {
      name: "openbrowser-storage",
      version: 2,
      migrate: (persistedState: unknown) => {
        const state = persistedState as Record<string, unknown>;
        return {
          ...state,
          messages: [],
          conversations: [],
          activeConversationId: null,
        } as unknown as AppState;
      },
      partialize: (state) => ({
        tasks: state.tasks.slice(0, 50), // Keep last 50 tasks
        projects: state.projects,
        agentType: state.agentType,
        maxSteps: state.maxSteps,
        useVision: state.useVision,
        sidebarOpen: state.sidebarOpen,
        showLogs: state.showLogs,
        browserViewerMode: state.browserViewerMode,
        selectedModel: state.selectedModel, // Persist selected model
      }),
    }
  )
);
