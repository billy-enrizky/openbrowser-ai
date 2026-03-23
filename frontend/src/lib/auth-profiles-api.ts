import { API_BASE_URL } from "@/lib/config";
import type { AuthProfile, BackendAuthProfile } from "@/types";

function authHeaders(token: string | null): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function mapProfile(p: BackendAuthProfile): AuthProfile {
  return {
    id: p.id,
    domain: p.domain,
    label: p.label,
    status: p.status as AuthProfile["status"],
    lastVerifiedAt: p.last_verified_at,
    createdAt: p.created_at,
    updatedAt: p.updated_at,
  };
}

export async function fetchAuthProfiles(
  token: string | null,
): Promise<AuthProfile[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/auth/profiles`, {
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to fetch auth profiles: ${await res.text()}`);
  const data = await res.json();
  return (data.profiles as BackendAuthProfile[]).map(mapProfile);
}

export async function startAuthSession(
  token: string | null,
  domain: string,
  label: string,
): Promise<{ task_id: string; vnc_url: string | null }> {
  const res = await fetch(`${API_BASE_URL}/api/v1/auth/profiles/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ domain, label }),
  });
  if (!res.ok) throw new Error(`Failed to start auth session: ${await res.text()}`);
  return res.json();
}

export async function saveAuthProfile(
  token: string | null,
  taskId: string,
  domain: string,
  label: string,
): Promise<AuthProfile> {
  const res = await fetch(`${API_BASE_URL}/api/v1/auth/profiles/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ task_id: taskId, domain, label }),
  });
  if (!res.ok) throw new Error(`Failed to save auth profile: ${await res.text()}`);
  const data = await res.json();
  return mapProfile(data);
}

export async function deleteAuthProfile(
  token: string | null,
  profileId: string,
): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/v1/auth/profiles/${profileId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to delete auth profile: ${await res.text()}`);
}

export async function updateAuthProfileLabel(
  token: string | null,
  profileId: string,
  label: string,
): Promise<AuthProfile> {
  const res = await fetch(`${API_BASE_URL}/api/v1/auth/profiles/${profileId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ label }),
  });
  if (!res.ok) throw new Error(`Failed to update auth profile: ${await res.text()}`);
  const data = await res.json();
  return mapProfile(data);
}
