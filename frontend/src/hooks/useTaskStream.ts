"use client";

/**
 * SSE-based task streaming hook.
 *
 * Replaces useWebSocket for production where API Gateway HTTP API
 * does not support WebSocket protocol upgrade.
 *
 * Uses:
 *   POST /api/v1/tasks/start  -> start a task (returns task_id)
 *   GET  /api/v1/tasks/{task_id}/stream?token=...  -> SSE event stream
 *   POST /api/v1/tasks/{task_id}/cancel  -> cancel a running task
 */

import { useCallback, useRef, useState } from "react";
import { API_BASE_URL } from "@/lib/config";
import type { WSMessage } from "@/types";

interface UseTaskStreamOptions {
  /** Called for every SSE event (same shape as the old WS messages). */
  onMessage?: (message: WSMessage) => void;
  /** Called when the SSE connection opens. */
  onConnect?: () => void;
  /** Called when the SSE connection closes. */
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
  // Start a task and open an SSE stream for its events.
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

        // 2. Open SSE stream
        const abort = new AbortController();
        abortRef.current = abort;

        const streamUrl = new URL(`${API_BASE_URL}/api/v1/tasks/${task_id}/stream`);
        if (token) streamUrl.searchParams.set("token", token);

        _consumeSSE(streamUrl.toString(), abort.signal, task_id, token);
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
  // Disconnect / abort the current SSE stream.
  // ------------------------------------------------------------------
  const disconnect = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    activeTaskRef.current = null;
    setIsConnected(false);
  }, []);

  // ------------------------------------------------------------------
  // Internal: consume the SSE stream using fetch + ReadableStream.
  //
  // We use fetch instead of EventSource so we can pass auth headers
  // and have full control over reconnection.
  // ------------------------------------------------------------------
  async function _consumeSSE(
    url: string,
    signal: AbortSignal,
    taskId: string,
    token: string | null,
  ) {
    let lastEventId = 0;

    const attemptStream = async () => {
      // Build URL with lastEventId for reconnection
      const streamUrl = new URL(url);
      if (lastEventId > 0) {
        streamUrl.searchParams.set("lastEventId", String(lastEventId));
      }

      const resp = await fetch(streamUrl.toString(), { signal });
      if (!resp.ok || !resp.body) {
        throw new Error(`SSE stream failed: ${resp.status}`);
      }

      setIsConnected(true);
      onConnect?.();

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events from the buffer
        const parts = buffer.split("\n\n");
        // Last part is incomplete -- keep it in the buffer
        buffer = parts.pop() || "";

        for (const rawEvent of parts) {
          if (!rawEvent.trim()) continue;

          let eventId: string | null = null;
          let eventType: string | null = null;
          let eventData = "";

          for (const line of rawEvent.split("\n")) {
            if (line.startsWith("id: ")) {
              eventId = line.slice(4).trim();
            } else if (line.startsWith("event: ")) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              eventData += line.slice(6);
            } else if (line.startsWith(":")) {
              // Comment / heartbeat -- ignore
            }
          }

          if (eventId) {
            const parsed = parseInt(eventId, 10);
            if (!isNaN(parsed)) lastEventId = parsed;
          }

          // "done" event means the stream is complete
          if (eventType === "done") {
            return; // exit
          }

          if (eventType && eventData) {
            try {
              const data = JSON.parse(eventData);
              const message: WSMessage = {
                type: eventType as WSMessage["type"],
                task_id: taskId,
                data,
                timestamp: new Date().toISOString(),
              };
              onMessage?.(message);
            } catch {
              console.error("Failed to parse SSE data:", eventData);
            }
          }
        }
      }
    };

    // Retry loop with backoff
    let retries = 0;
    const maxRetries = 5;

    while (retries <= maxRetries && !signal.aborted) {
      try {
        await attemptStream();
        // Clean exit (stream ended normally via "done" event)
        break;
      } catch (err: unknown) {
        if (signal.aborted) break;

        retries++;
        if (retries > maxRetries) {
          console.error("SSE stream: max retries exceeded", err);
          break;
        }

        const delay = Math.min(1000 * 2 ** (retries - 1), 15000);
        console.warn(`SSE stream disconnected, retrying in ${delay}ms...`, err);
        setIsConnected(false);
        await new Promise((r) => setTimeout(r, delay));
      }
    }

    setIsConnected(false);
    onDisconnect?.();
    activeTaskRef.current = null;
  }

  return {
    /** Whether an SSE stream is currently open. */
    isConnected,
    /** Whether a start-task request is in flight. */
    isStarting,
    /** Currently active task ID (if any). */
    activeTaskId: activeTaskRef.current,
    /** Start a new task and open an SSE stream. */
    startTask,
    /** Cancel the active task. */
    cancelTask,
    /** Abort the SSE stream. */
    disconnect,
  };
}
