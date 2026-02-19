"use client";

/**
 * Polling-based task streaming hook.
 *
 * Uses:
 *   POST /api/v1/tasks/start              -> start a task (returns task_id)
 *   GET  /api/v1/tasks/{task_id}/events    -> poll for new events
 *   POST /api/v1/tasks/{task_id}/cancel    -> cancel a running task
 *
 * Previously this used SSE (Server-Sent Events) but API Gateway HTTP API
 * has a hard 30s integration timeout that kills long-lived connections.
 * Polling every ~1.5s avoids this limit.
 */

import { useCallback, useRef, useState } from "react";
import { API_BASE_URL } from "@/lib/config";
import type { WSMessage } from "@/types";

/** Interval between polls in milliseconds. */
const POLL_INTERVAL_MS = 1500;

interface UseTaskStreamOptions {
  /** Called for every event (same shape as the old WS messages). */
  onMessage?: (message: WSMessage) => void;
  /** Called when the polling loop starts. */
  onConnect?: () => void;
  /** Called when the polling loop ends. */
  onDisconnect?: () => void;
  /** Auth token getter -- called fresh for every request. */
  getToken?: () => Promise<string | null>;
}

export function useTaskStream(options: UseTaskStreamOptions = {}) {
  const { onMessage, onConnect, onDisconnect, getToken } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const activeTaskRef = useRef<string | null>(null);

  // ------------------------------------------------------------------
  // Start a task and begin polling for events.
  // Returns the task_id on success, null on failure.
  // ------------------------------------------------------------------
  const startTask = useCallback(
    async (params: {
      task: string;
      agent_type: string;
      max_steps: number;
      use_vision: boolean;
      llm_model?: string | null;
      use_current_browser?: boolean;
      conversation_id?: string | null;
    }): Promise<string | null> => {
      setIsStarting(true);

      try {
        const token = getToken ? await getToken() : null;

        // 1. POST to start the task
        const headers: HeadersInit = { "Content-Type": "application/json" };
        if (token) headers["Authorization"] = `Bearer ${token}`;

        const startResp = await fetch(`${API_BASE_URL}/api/v1/tasks/start`, {
          method: "POST",
          headers,
          body: JSON.stringify(params),
        });

        if (!startResp.ok) {
          const err = await startResp.text();
          throw new Error(`Failed to start task: ${startResp.status} ${err}`);
        }

        const { task_id } = (await startResp.json()) as { task_id: string };
        activeTaskRef.current = task_id;

        // 2. Start polling loop
        const abort = new AbortController();
        abortRef.current = abort;

        _pollEvents(task_id, abort.signal, token);
        return task_id;
      } catch (err) {
        console.error("startTask error:", err);
        return null;
      } finally {
        setIsStarting(false);
      }
    },
    [getToken, onMessage, onConnect, onDisconnect],
  );

  // ------------------------------------------------------------------
  // Cancel the active task.
  // ------------------------------------------------------------------
  const cancelTask = useCallback(
    async (taskId?: string) => {
      const id = taskId || activeTaskRef.current;
      if (!id) return;

      try {
        const token = getToken ? await getToken() : null;
        const headers: HeadersInit = { "Content-Type": "application/json" };
        if (token) headers["Authorization"] = `Bearer ${token}`;

        await fetch(`${API_BASE_URL}/api/v1/tasks/${id}/cancel`, {
          method: "POST",
          headers,
        });
      } catch (err) {
        console.error("cancelTask error:", err);
      }
    },
    [getToken],
  );

  // ------------------------------------------------------------------
  // Disconnect / abort the current polling loop.
  // ------------------------------------------------------------------
  const disconnect = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    activeTaskRef.current = null;
    setIsConnected(false);
  }, []);

  // ------------------------------------------------------------------
  // Internal: poll GET /tasks/{id}/events?since=N every POLL_INTERVAL_MS.
  // ------------------------------------------------------------------
  async function _pollEvents(
    taskId: string,
    signal: AbortSignal,
    token: string | null,
  ) {
    let lastEventId = 0;
    let consecutiveErrors = 0;
    const maxErrors = 10;

    setIsConnected(true);
    onConnect?.();

    try {
      while (!signal.aborted && consecutiveErrors < maxErrors) {
        try {
          const headers: HeadersInit = {};
          if (token) headers["Authorization"] = `Bearer ${token}`;

          const url = `${API_BASE_URL}/api/v1/tasks/${taskId}/events?since=${lastEventId}`;
          const resp = await fetch(url, { headers, signal });

          if (!resp.ok) {
            // 404 means task was cleaned up -- stop polling
            if (resp.status === 404) break;
            throw new Error(`Poll failed: ${resp.status}`);
          }

          consecutiveErrors = 0;

          const body = (await resp.json()) as {
            events: Array<{ id: number; type: string; data: Record<string, unknown> }>;
            complete: boolean;
          };

          // Dispatch events to onMessage
          for (const evt of body.events) {
            if (evt.id > lastEventId) {
              lastEventId = evt.id;
            }

            const message: WSMessage = {
              type: evt.type as WSMessage["type"],
              task_id: taskId,
              data: evt.data,
              timestamp: new Date().toISOString(),
            };
            onMessage?.(message);
          }

          // Task is done
          if (body.complete) break;

          // Wait before next poll
          await new Promise<void>((resolve, reject) => {
            const timer = setTimeout(resolve, POLL_INTERVAL_MS);
            signal.addEventListener("abort", () => {
              clearTimeout(timer);
              reject(new DOMException("Aborted", "AbortError"));
            }, { once: true });
          });
        } catch (err: unknown) {
          if (signal.aborted) break;
          if (err instanceof DOMException && err.name === "AbortError") break;

          consecutiveErrors++;
          if (consecutiveErrors >= maxErrors) {
            console.error("Polling: max consecutive errors reached", err);
            break;
          }

          // Backoff on errors
          const delay = Math.min(1000 * 2 ** (consecutiveErrors - 1), 10000);
          console.warn(`Poll error (${consecutiveErrors}/${maxErrors}), retrying in ${delay}ms...`, err);
          await new Promise((r) => setTimeout(r, delay));
        }
      }
    } finally {
      setIsConnected(false);
      onDisconnect?.();
      activeTaskRef.current = null;
    }
  }

  return {
    /** Whether the polling loop is active. */
    isConnected,
    /** Whether a start-task request is in flight. */
    isStarting,
    /** Currently active task ID (if any). */
    activeTaskId: activeTaskRef.current,
    /** Start a new task and begin polling. */
    startTask,
    /** Cancel the active task. */
    cancelTask,
    /** Abort the polling loop. */
    disconnect,
  };
}
