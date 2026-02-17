"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";

import { useAuth } from "@/components/auth";

function AuthCallbackContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { authEnabled, completeSignIn } = useAuth();
  const [asyncError, setAsyncError] = useState<string | null>(null);

  const code = searchParams.get("code");
  const state = searchParams.get("state");
  const oauthError = searchParams.get("error");
  const oauthErrorDescription = searchParams.get("error_description");
  const staticError = oauthError
    ? oauthErrorDescription || oauthError
    : !code || !state
      ? "Missing authorization code or state."
      : null;
  const error = staticError || asyncError;

  useEffect(() => {
    if (!authEnabled) {
      router.replace("/");
      return;
    }

    if (staticError || !code || !state) {
      return;
    }

    (async () => {
      try {
        await completeSignIn(code, state);
        router.replace("/");
      } catch (e) {
        setAsyncError(e instanceof Error ? e.message : "Authentication failed.");
      }
    })();
  }, [authEnabled, code, completeSignIn, router, state, staticError]);

  if (!error) {
    return (
      <main className="min-h-screen bg-zinc-950 text-zinc-100 flex items-center justify-center px-6">
        <div className="w-full max-w-md rounded-2xl border border-zinc-800 bg-zinc-900/60 p-8 backdrop-blur text-center">
          <h1 className="text-lg font-semibold">Signing you in...</h1>
          <p className="mt-2 text-sm text-zinc-400">Finishing Cognito authentication.</p>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 flex items-center justify-center px-6">
      <div className="w-full max-w-md rounded-2xl border border-zinc-800 bg-zinc-900/60 p-8 backdrop-blur">
        <h1 className="text-xl font-semibold">Sign-in failed</h1>
        <p className="mt-3 text-sm text-red-300">{error}</p>

        <Link
          href="/login"
          className="mt-6 inline-flex rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-zinc-950 transition hover:bg-cyan-400"
        >
          Back to login
        </Link>
      </div>
    </main>
  );
}

function LoadingState() {
  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 flex items-center justify-center px-6">
      <div className="w-full max-w-md rounded-2xl border border-zinc-800 bg-zinc-900/60 p-8 backdrop-blur text-center">
        <h1 className="text-lg font-semibold">Signing you in...</h1>
        <p className="mt-2 text-sm text-zinc-400">Finishing Cognito authentication.</p>
      </div>
    </main>
  );
}

export default function AuthCallbackPage() {
  return (
    <Suspense fallback={<LoadingState />}>
      <AuthCallbackContent />
    </Suspense>
  );
}
