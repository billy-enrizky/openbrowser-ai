import type { AppConfig, QuickAction, Integration } from "@/types";

// Parse JSON from environment variable safely
function parseJsonEnv<T>(value: string | undefined, fallback: T): T {
  if (!value) return fallback;
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

// Default quick actions (can be overridden by env var)
const DEFAULT_QUICK_ACTIONS: QuickAction[] = [];

// Default integrations (can be overridden by env var)
const DEFAULT_INTEGRATIONS: Integration[] = [];

// App configuration from environment variables
export const appConfig: AppConfig = {
  appName: process.env.NEXT_PUBLIC_APP_NAME || "OpenBrowser",
  version: process.env.NEXT_PUBLIC_APP_VERSION || "1.0",
  quickActions: parseJsonEnv<QuickAction[]>(process.env.NEXT_PUBLIC_QUICK_ACTIONS, DEFAULT_QUICK_ACTIONS),
  integrations: parseJsonEnv<Integration[]>(process.env.NEXT_PUBLIC_INTEGRATIONS, DEFAULT_INTEGRATIONS),
};

// Default user info (will be overridden by API when available)
export const defaultUserInfo = {
  planName: process.env.NEXT_PUBLIC_DEFAULT_PLAN || "Free plan",
  planType: "free" as const,
  credits: 0,
  creditsUsed: 0,
  notifications: 0,
};

// Integration connection text
export const INTEGRATION_TEXT = process.env.NEXT_PUBLIC_INTEGRATION_TEXT || "Connect your tools to OpenBrowser";
