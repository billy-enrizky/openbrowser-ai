// API Configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
export const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";

// Agent Configuration
export const DEFAULT_MAX_STEPS = 50;
export const DEFAULT_AGENT_TYPE = "code" as const;

// UI Configuration
export const SIDEBAR_WIDTH = 280;
export const SIDEBAR_COLLAPSED_WIDTH = 64;

