# Frontend UI Configuration

This document describes how to configure the UI elements in the frontend application.

## Environment Variables

All UI configuration is done via environment variables. Create or update your `.env.local` file for local development, or set these in your production environment.

### App Information

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_APP_NAME` | Application name shown in header | `OpenBrowser` |
| `NEXT_PUBLIC_APP_VERSION` | Application version shown in header | `1.0` |

### Quick Actions (Chat Input)

Quick action buttons can be configured for the chat input area.

**Variable:** `NEXT_PUBLIC_QUICK_ACTIONS`

**Format:** JSON array of QuickAction objects

```json
[
  {
    "icon": "Presentation",
    "label": "Create slides",
    "prompt": "Create a presentation about...",
    "color": "from-purple-500/20 to-violet-500/20",
    "enabled": true
  }
]
```

**Properties:**
- `icon`: Lucide icon component name
- `label`: Button text
- `prompt` (optional): Message to send when clicked. If not provided, the label is sent.
- `color`: Tailwind gradient classes for the button background
- `enabled`: Whether to show this action

**Default:** Empty array (no quick actions shown)

### Integrations Panel

**Variable:** `NEXT_PUBLIC_INTEGRATIONS`

**Format:** JSON array of Integration objects

```json
[
  {
    "name": "Chrome",
    "enabled": true,
    "url": "https://example.com/chrome-integration"
  }
]
```

**Properties:**
- `name`: Integration display name
- `enabled`: Whether to show this integration
- `url` (optional): Link when clicking the integration icon

**Default:** Empty array (no integrations shown)

### Integration Panel Text

**Variable:** `NEXT_PUBLIC_INTEGRATION_TEXT`

**Description:** Text shown in the integrations panel.

**Default:** `Connect your tools to OpenBrowser`

### Plan Information

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_DEFAULT_PLAN` | Default plan name | `Free plan` |

## User Credits and Notifications

These values are fetched from the backend API and displayed in the UI. They are not configured via environment variables.

### Store Integration

The application uses Zustand for state management. The user info is stored in the global app store:

```typescript
interface UserInfo {
  planName: string;
  planType?: "free" | "pro" | "enterprise";
  credits: number;
  creditsUsed: number;
  notifications: number;
  displayName?: string;
  avatarUrl?: string;
}
```

To update user info from the backend:

```typescript
import { useAppStore } from "@/store";

const { setUserInfo, updateCredits } = useAppStore();

// Set full user info
setUserInfo({
  planName: "Pro plan",
  planType: "pro",
  credits: 10000,
  creditsUsed: 7500,
  notifications: 3,
});

// Update just credits
updateCredits(10000, 7500);
```

## Notes

- All `NEXT_PUBLIC_*` variables are available in the browser
- Changes to environment variables require a development server restart
- JSON arrays should be properly formatted (no single quotes, use double quotes for strings)
