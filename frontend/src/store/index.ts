"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Task, Project, Message, AgentType, LogEntry, VncInfo, BrowserViewerMode } from "@/types";

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
  clearMessages: () => void;

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

  // Settings
  agentType: AgentType;
  setAgentType: (type: AgentType) => void;
  maxSteps: number;
  setMaxSteps: (steps: number) => void;
  useVision: boolean;
  setUseVision: (use: boolean) => void;
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
      clearMessages: () => set({ messages: [] }),

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

      // Settings
      agentType: "code",
      setAgentType: (type) => set({ agentType: type }),
      maxSteps: 50,
      setMaxSteps: (steps) => set({ maxSteps: steps }),
      useVision: true,
      setUseVision: (use) => set({ useVision: use }),
    }),
    {
      name: "openbrowser-storage",
      partialize: (state) => ({
        tasks: state.tasks.slice(0, 50), // Keep last 50 tasks
        projects: state.projects,
        messages: state.messages.slice(-100), // Keep last 100 messages
        agentType: state.agentType,
        maxSteps: state.maxSteps,
        useVision: state.useVision,
        sidebarOpen: state.sidebarOpen,
        showLogs: state.showLogs,
        browserViewerMode: state.browserViewerMode,
      }),
    }
  )
);
