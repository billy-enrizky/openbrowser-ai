# Frontend UI Configuration

This document describes how to configure the hard-coded UI elements in the frontend application.

## Environment Variables

All UI configuration is done via environment variables. Create or update your `.env.local` file for local development, or set these in your production environment.

### App Information

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_APP_NAME` | Application name shown in header | `OpenBrowser` |
| `NEXT_PUBLIC_APP_VERSION` | Application version shown in header | `1.0` |

### Quick Actions (Chat Input)

The quick action buttons shown below the chat input when no messages exist.

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

**To hide all quick actions:** Set to empty string or empty array: `NEXT_PUBLIC_QUICK_ACTIONS=""`

### Integrations Panel

The integrations shown when clicking the link icon in chat input.

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

**To hide integrations panel:** Set to empty string or empty array: `NEXT_PUBLIC_INTEGRATIONS=""`

### Integration Panel Text

**Variable:** `NEXT_PUBLIC_INTEGRATION_TEXT`

**Description:** Text shown next to the link icon in the integrations panel.

**Default:** `Connect your tools to OpenBrowser`

### Plan Information

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_DEFAULT_PLAN` | Plan name shown in header | `Free plan` |
| `NEXT_PUBLIC_TRIAL_CTA` | Text for trial upgrade button | `Start free trial` |

## User Credits and Notifications

These values are fetched from the backend API and displayed in the header. They are not configured via environment variables.

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

## Available Lucide Icons

Any icon name from [lucide-react](https://lucide.dev/) can be used for quick actions:

- Presentation, Globe, Smartphone, Palette, MoreHorizontal
- Code, Terminal, Database, Server, Cloud
- File, Folder, Image, Video, Music
- And many more...

See [lucide.dev](https://lucide.dev/) for the full list.

## Example: Minimal Configuration

To remove all quick actions and integrations, set empty arrays:

```bash
NEXT_PUBLIC_QUICK_ACTIONS=[]
NEXT_PUBLIC_INTEGRATIONS=[]
```

## Example: Custom Quick Actions

```bash
NEXT_PUBLIC_QUICK_ACTIONS=[
  {"icon":"Code","label":"Write code","color":"from-purple-500/20 to-violet-500/20","enabled":true},
  {"icon":"Database","label":"Analyze data","color":"from-cyan-500/20 to-blue-500/20","enabled":true}
]
```

## Notes

- All `NEXT_PUBLIC_*` variables are available in the browser
- Changes to environment variables require a development server restart
- JSON arrays should be properly formatted (no single quotes, use double quotes for strings)
