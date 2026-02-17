// API Configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
export const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";

// Authentication Configuration
export const AUTH_ENABLED = process.env.NEXT_PUBLIC_AUTH_ENABLED === "true";
export const COGNITO_DOMAIN = (process.env.NEXT_PUBLIC_COGNITO_DOMAIN || "").replace(/\/$/, "");
export const COGNITO_CLIENT_ID = process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID || "";
export const COGNITO_REDIRECT_URI =
  process.env.NEXT_PUBLIC_COGNITO_REDIRECT_URI || "http://localhost:3000/auth/callback";
export const COGNITO_LOGOUT_URI =
  process.env.NEXT_PUBLIC_COGNITO_LOGOUT_URI || "http://localhost:3000/login";
export const COGNITO_SCOPES = process.env.NEXT_PUBLIC_COGNITO_SCOPES || "openid email profile";

// VNC Configuration
export const VNC_ENABLED = process.env.NEXT_PUBLIC_VNC_ENABLED !== "false"; // Enabled by default

// Agent Configuration
export const DEFAULT_MAX_STEPS = 50;
export const DEFAULT_AGENT_TYPE = "code" as const;

// UI Configuration
export const SIDEBAR_WIDTH = 280;
export const SIDEBAR_COLLAPSED_WIDTH = 64;
export const BROWSER_VIEWER_MIN_WIDTH = 400;
export const BROWSER_VIEWER_DEFAULT_WIDTH = 800;
