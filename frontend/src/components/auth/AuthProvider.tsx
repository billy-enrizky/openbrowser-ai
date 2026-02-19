"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { AUTH_ENABLED } from "@/lib/config";
import {
  AuthUser,
  CognitoTokenSet,
  buildLoginUrl,
  buildLogoutUrl,
  clearTokens,
  exchangeCodeForTokens,
  getUserFromIdToken,
  isTokenExpired,
  loadTokens,
  refreshTokens,
  saveTokens,
} from "@/lib/auth/cognito";

interface AuthContextValue {
  isLoading: boolean;
  isAuthenticated: boolean;
  user: AuthUser | null;
  idToken: string | null;
  accessToken: string | null;
  authEnabled: boolean;
  login: () => Promise<void>;
  logout: () => void;
  completeSignIn: (code: string, state: string) => Promise<void>;
  getValidIdToken: () => Promise<string | null>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function toAuthState(tokens: CognitoTokenSet | null) {
  if (!tokens) {
    return { user: null, idToken: null, accessToken: null, isAuthenticated: false };
  }

  const user = getUserFromIdToken(tokens.idToken);
  return {
    user,
    idToken: tokens.idToken,
    accessToken: tokens.accessToken,
    isAuthenticated: true,
  };
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isLoading, setIsLoading] = useState(true);
  const [tokens, setTokens] = useState<CognitoTokenSet | null>(null);
  const [user, setUser] = useState<AuthUser | null>(null);

  const applyTokens = useCallback((nextTokens: CognitoTokenSet | null, clearAuthStorage = true) => {
    if (!nextTokens) {
      setTokens(null);
      setUser(null);
      if (clearAuthStorage) {
        clearTokens();
      }
      return;
    }

    saveTokens(nextTokens);
    setTokens(nextTokens);
    setUser(getUserFromIdToken(nextTokens.idToken));
  }, []);

  const hydrateAuth = useCallback(async () => {
    if (!AUTH_ENABLED) {
      setIsLoading(false);
      return;
    }

    try {
      const existingTokens = loadTokens();
      if (!existingTokens) {
        // Preserve PKCE session values for OAuth callback flow.
        applyTokens(null, false);
        return;
      }

      if (isTokenExpired(existingTokens) && existingTokens.refreshToken) {
        const refreshed = await refreshTokens(existingTokens.refreshToken);
        applyTokens(refreshed);
      } else if (isTokenExpired(existingTokens)) {
        applyTokens(null);
      } else {
        applyTokens(existingTokens);
      }
    } catch {
      applyTokens(null);
    } finally {
      setIsLoading(false);
    }
  }, [applyTokens]);

  useEffect(() => {
    void hydrateAuth();
  }, [hydrateAuth]);

  const getValidIdToken = useCallback(async (): Promise<string | null> => {
    if (!AUTH_ENABLED) {
      return null;
    }

    const currentTokens = tokens ?? loadTokens();
    if (!currentTokens) {
      return null;
    }

    if (!isTokenExpired(currentTokens)) {
      return currentTokens.idToken;
    }

    if (!currentTokens.refreshToken) {
      applyTokens(null);
      return null;
    }

    try {
      const refreshed = await refreshTokens(currentTokens.refreshToken);
      applyTokens(refreshed);
      return refreshed.idToken;
    } catch {
      applyTokens(null);
      return null;
    }
  }, [applyTokens, tokens]);

  const login = useCallback(async () => {
    if (!AUTH_ENABLED) {
      return;
    }

    const loginUrl = await buildLoginUrl();
    window.location.assign(loginUrl);
  }, []);

  const logout = useCallback(() => {
    applyTokens(null);
    if (!AUTH_ENABLED) {
      return;
    }

    window.location.assign(buildLogoutUrl());
  }, [applyTokens]);

  const completeSignIn = useCallback(
    async (code: string, state: string) => {
      const newTokens = await exchangeCodeForTokens(code, state);
      applyTokens(newTokens);
    },
    [applyTokens]
  );

  const contextValue = useMemo<AuthContextValue>(() => {
    const derived = AUTH_ENABLED ? toAuthState(tokens) : { user: null, idToken: null, accessToken: null, isAuthenticated: true };
    return {
      isLoading,
      isAuthenticated: derived.isAuthenticated,
      user: AUTH_ENABLED ? user : null,
      idToken: derived.idToken,
      accessToken: derived.accessToken,
      authEnabled: AUTH_ENABLED,
      login,
      logout,
      completeSignIn,
      getValidIdToken,
    };
  }, [completeSignIn, getValidIdToken, isLoading, login, logout, tokens, user]);

  return <AuthContext.Provider value={contextValue}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
