"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { useAuth } from "@/components/auth";

export default function LoginPage() {
  const router = useRouter();
  const { authEnabled, isLoading, isAuthenticated, login } = useAuth();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isLoading) {
      return;
    }

    if (!authEnabled || isAuthenticated) {
      router.replace("/");
    }
  }, [authEnabled, isAuthenticated, isLoading, router]);

  if (!authEnabled || isLoading) {
    return null;
  }

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 flex items-center justify-center px-6">
      <div className="w-full max-w-md rounded-2xl border border-zinc-800 bg-zinc-900/60 p-8 backdrop-blur">
        <h1 className="text-2xl font-semibold tracking-tight">Sign in to OpenBrowser</h1>
        <p className="mt-3 text-sm text-zinc-400">
          Authenticate with AWS Cognito to start browser sessions.
        </p>

        {error && (
          <p className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
            {error}
          </p>
        )}

        <button
          type="button"
          onClick={async () => {
            try {
              setError(null);
              await login();
            } catch (e) {
              setError(e instanceof Error ? e.message : "Failed to start sign-in.");
            }
          }}
          className="mt-6 w-full rounded-lg bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-zinc-950 transition hover:bg-cyan-400"
        >
          Continue with Cognito
        </button>

        <p className="mt-5 text-xs text-zinc-500">
          By continuing you agree to your organization&apos;s authentication policy.
        </p>

        <div className="mt-6 border-t border-zinc-800 pt-4">
          <Link href="/" className="text-xs text-zinc-400 hover:text-zinc-200 transition">
            Return to app
          </Link>
        </div>
      </div>
    </main>
  );
}

