"use client";

import {
  COGNITO_CLIENT_ID,
  COGNITO_DOMAIN,
  COGNITO_LOGOUT_URI,
  COGNITO_REDIRECT_URI,
  COGNITO_SCOPES,
} from "@/lib/config";

const TOKEN_STORAGE_KEY = "openbrowser.auth.tokens";
const PKCE_STATE_KEY = "openbrowser.auth.pkce_state";
const PKCE_VERIFIER_KEY = "openbrowser.auth.pkce_verifier";

export interface CognitoTokenSet {
  idToken: string;
  accessToken: string;
  refreshToken?: string;
  expiresAt: number;
}

export interface AuthUser {
  sub: string;
  email?: string;
  name?: string;
  username?: string;
}

interface TokenResponse {
  access_token: string;
  id_token: string;
  refresh_token?: string;
  expires_in: number;
  token_type: string;
}

function base64UrlEncode(bytes: Uint8Array): string {
  const binary = Array.from(bytes)
    .map((b) => String.fromCharCode(b))
    .join("");

  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function randomString(byteLength = 32): string {
  const bytes = new Uint8Array(byteLength);
  crypto.getRandomValues(bytes);
  return base64UrlEncode(bytes);
}

function parseJwtPayload(token: string): Record<string, unknown> {
  const parts = token.split(".");
  if (parts.length !== 3) {
    throw new Error("Invalid JWT");
  }

  const payload = parts[1].replace(/-/g, "+").replace(/_/g, "/");
  const padded = payload + "=".repeat((4 - (payload.length % 4)) % 4);
  return JSON.parse(atob(padded));
}

export function getUserFromIdToken(idToken: string): AuthUser {
  const payload = parseJwtPayload(idToken);
  const sub = String(payload.sub || "");
  if (!sub) {
    throw new Error("Missing subject in id_token");
  }

  return {
    sub,
    email: payload.email ? String(payload.email) : undefined,
    name: payload.name ? String(payload.name) : undefined,
    username: payload["cognito:username"] ? String(payload["cognito:username"]) : undefined,
  };
}

function validateCognitoClientConfig(): void {
  if (!COGNITO_DOMAIN || !COGNITO_CLIENT_ID) {
    throw new Error(
      "Missing Cognito frontend config. Set NEXT_PUBLIC_COGNITO_DOMAIN and NEXT_PUBLIC_COGNITO_CLIENT_ID."
    );
  }
}

async function createCodeChallenge(verifier: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return base64UrlEncode(new Uint8Array(digest));
}

export async function buildLoginUrl(): Promise<string> {
  validateCognitoClientConfig();

  const state = randomString(24);
  const codeVerifier = randomString(64);
  const codeChallenge = await createCodeChallenge(codeVerifier);

  sessionStorage.setItem(PKCE_STATE_KEY, state);
  sessionStorage.setItem(PKCE_VERIFIER_KEY, codeVerifier);

  const params = new URLSearchParams({
    response_type: "code",
    client_id: COGNITO_CLIENT_ID,
    redirect_uri: COGNITO_REDIRECT_URI,
    scope: COGNITO_SCOPES,
    state,
    code_challenge_method: "S256",
    code_challenge: codeChallenge,
  });

  return `${COGNITO_DOMAIN}/oauth2/authorize?${params.toString()}`;
}

export async function exchangeCodeForTokens(code: string, state: string): Promise<CognitoTokenSet> {
  validateCognitoClientConfig();

  const expectedState = sessionStorage.getItem(PKCE_STATE_KEY);
  const codeVerifier = sessionStorage.getItem(PKCE_VERIFIER_KEY);

  sessionStorage.removeItem(PKCE_STATE_KEY);
  sessionStorage.removeItem(PKCE_VERIFIER_KEY);

  if (!expectedState || state !== expectedState) {
    throw new Error("Invalid OAuth state");
  }
  if (!codeVerifier) {
    throw new Error("Missing PKCE code_verifier");
  }

  const body = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: COGNITO_CLIENT_ID,
    code,
    redirect_uri: COGNITO_REDIRECT_URI,
    code_verifier: codeVerifier,
  });

  const response = await fetch(`${COGNITO_DOMAIN}/oauth2/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Token exchange failed: ${text}`);
  }

  const tokenResponse = (await response.json()) as TokenResponse;
  const expiresAt = Date.now() + tokenResponse.expires_in * 1000;

  return {
    idToken: tokenResponse.id_token,
    accessToken: tokenResponse.access_token,
    refreshToken: tokenResponse.refresh_token,
    expiresAt,
  };
}

export async function refreshTokens(refreshToken: string): Promise<CognitoTokenSet> {
  validateCognitoClientConfig();

  const body = new URLSearchParams({
    grant_type: "refresh_token",
    client_id: COGNITO_CLIENT_ID,
    refresh_token: refreshToken,
  });

  const response = await fetch(`${COGNITO_DOMAIN}/oauth2/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Token refresh failed: ${text}`);
  }

  const tokenResponse = (await response.json()) as TokenResponse;
  const expiresAt = Date.now() + tokenResponse.expires_in * 1000;

  return {
    idToken: tokenResponse.id_token,
    accessToken: tokenResponse.access_token,
    refreshToken,
    expiresAt,
  };
}

export function saveTokens(tokens: CognitoTokenSet): void {
  localStorage.setItem(TOKEN_STORAGE_KEY, JSON.stringify(tokens));
}

export function loadTokens(): CognitoTokenSet | null {
  const raw = localStorage.getItem(TOKEN_STORAGE_KEY);
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as CognitoTokenSet;
    if (!parsed.idToken || !parsed.accessToken || !parsed.expiresAt) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function clearTokens(): void {
  localStorage.removeItem(TOKEN_STORAGE_KEY);
  sessionStorage.removeItem(PKCE_STATE_KEY);
  sessionStorage.removeItem(PKCE_VERIFIER_KEY);
}

export function isTokenExpired(tokens: CognitoTokenSet, skewMs = 60_000): boolean {
  return Date.now() >= tokens.expiresAt - skewMs;
}

export function buildLogoutUrl(): string {
  if (!COGNITO_DOMAIN || !COGNITO_CLIENT_ID) {
    return COGNITO_LOGOUT_URI;
  }

  const params = new URLSearchParams({
    client_id: COGNITO_CLIENT_ID,
    logout_uri: COGNITO_LOGOUT_URI,
  });
  return `${COGNITO_DOMAIN}/logout?${params.toString()}`;
}

