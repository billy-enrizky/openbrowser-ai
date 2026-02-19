import { API_BASE_URL } from "@/lib/config";
import type {
  BackendChatConversation,
  BackendChatMessage,
  ChatConversation,
  ChatDetailApiResponse,
  ChatListApiResponse,
  FileAttachment,
  Message,
} from "@/types";

function authHeaders(token: string | null): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function mapConversation(conversation: BackendChatConversation): ChatConversation {
  return {
    id: conversation.id,
    title: conversation.title,
    status: conversation.status,
    createdAt: new Date(conversation.created_at),
    updatedAt: new Date(conversation.updated_at),
    lastMessageAt: conversation.last_message_at ? new Date(conversation.last_message_at) : null,
  };
}

function mapMessage(message: BackendChatMessage): Message {
  const rawMetadata = (message.metadata || {}) as Record<string, unknown>;
  const attachments = Array.isArray(rawMetadata.attachments)
    ? (rawMetadata.attachments as Array<Partial<FileAttachment>>).map((attachment, index) => ({
        id: attachment.id || `${message.id}-attachment-${index}`,
        name: attachment.name || `attachment-${index + 1}`,
        type: attachment.type || "unknown",
        mimeType: attachment.mimeType,
        size: attachment.size,
        url: attachment.url,
        content: attachment.content,
        previewContent: attachment.previewContent,
      }))
    : undefined;
  const metadata = {
    stepNumber:
      (rawMetadata.stepNumber as number | undefined) ??
      (rawMetadata.step_number as number | undefined),
    isThinking:
      (rawMetadata.isThinking as boolean | undefined) ??
      (rawMetadata.is_thinking as boolean | undefined),
    isError:
      (rawMetadata.isError as boolean | undefined) ??
      (rawMetadata.is_error as boolean | undefined),
    screenshot: rawMetadata.screenshot as string | undefined,
    attachments,
  };

  return {
    id: message.id,
    role: message.role,
    content: message.content,
    timestamp: new Date(message.created_at),
    taskId: message.task_id || undefined,
    metadata,
  };
}

export async function fetchChatList(idToken: string | null): Promise<{
  conversations: ChatConversation[];
  activeConversationId: string | null;
}> {
  const response = await fetch(`${API_BASE_URL}/api/v1/chats`, {
    headers: authHeaders(idToken),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to fetch chats: ${text}`);
  }
  const data = (await response.json()) as ChatListApiResponse;
  return {
    conversations: data.conversations.map(mapConversation),
    activeConversationId: data.active_conversation_id || null,
  };
}

export async function fetchConversationDetail(
  idToken: string | null,
  conversationId: string,
): Promise<{ conversation: ChatConversation; messages: Message[] }> {
  const response = await fetch(`${API_BASE_URL}/api/v1/chats/${conversationId}`, {
    headers: authHeaders(idToken),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to fetch conversation: ${text}`);
  }
  const data = (await response.json()) as ChatDetailApiResponse;
  return {
    conversation: mapConversation(data.conversation),
    messages: data.messages.map(mapMessage),
  };
}

export async function createConversation(idToken: string | null, title?: string): Promise<ChatConversation> {
  const response = await fetch(`${API_BASE_URL}/api/v1/chats`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(idToken),
    },
    body: JSON.stringify({ title }),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to create conversation: ${text}`);
  }
  const data = (await response.json()) as BackendChatConversation;
  return mapConversation(data);
}

export async function deleteConversation(idToken: string | null, conversationId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/chats/${conversationId}`, {
    method: "DELETE",
    headers: authHeaders(idToken),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to delete conversation: ${text}`);
  }
}

export async function setActiveConversation(idToken: string | null, conversationId: string | null): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/v1/chats/active`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(idToken),
    },
    body: JSON.stringify({ conversation_id: conversationId }),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to set active conversation: ${text}`);
  }
}
