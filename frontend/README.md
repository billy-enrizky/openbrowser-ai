# OpenBrowser Frontend

A modern chat interface for OpenBrowser AI. Built with Next.js, TypeScript, and Tailwind CSS, this frontend provides a sleek dark-themed UI for interacting with browser automation agents.

## Features

- Real-time chat interface with WebSocket communication
- Sidebar navigation with projects and task history
- Support for both Browser Agent and Code Agent
- Screenshot display for browser automation tasks
- File attachment preview (CSV, JSON, text, code, images)
- Real-time backend log streaming
- Responsive design with dark theme

## Tech Stack

- **Next.js 16** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS 4** - Styling
- **Framer Motion** - Animations
- **Zustand** - State management
- **Lucide React** - Icons

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Export static files for GitHub Pages
npm run export
```

## Deployment

This frontend is deployed to GitHub Pages at [openbrowser.me](https://openbrowser.me).

### Architecture

```
+-------------------+          +-------------------+
|   openbrowser.me  |  HTTPS   | api.openbrowser.me|
|  (GitHub Pages)   | -------> |    (Backend)      |
|    Frontend       |   WSS    |   FastAPI + WS    |
+-------------------+          +-------------------+
```

### GitHub Pages Deployment

The frontend is automatically deployed via GitHub Actions when you push to the main branch.

**Manual deployment:**

```bash
# Build and export static files
npm run export

# Deploy to gh-pages branch
npm run deploy
```

### Environment Variables

For **production** (set in GitHub Actions secrets or `.env.production`):

```env
NEXT_PUBLIC_API_URL=https://api.openbrowser.me
NEXT_PUBLIC_WS_URL=wss://api.openbrowser.me/ws
```

For **local development** (create `.env.local`):

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

### Backend Deployment

The backend must be deployed separately to a platform that supports:
- Python/Docker containers
- WebSocket connections
- Persistent processes (for browser automation)

Recommended platforms:
- **Railway** - Easy Docker deployment
- **Render** - Free tier available
- **DigitalOcean App Platform** - Good for production
- **AWS EC2/ECS** - Full control

See `/backend/README.md` and `/DEPLOYMENT.md` for backend deployment instructions.

## Project Structure

```
src/
  app/
    globals.css       # Global styles
    layout.tsx        # Root layout
    page.tsx          # Main chat page
  components/
    chat/
      ChatInput.tsx   # Message input with quick actions
      ChatMessage.tsx # Individual message display
      ChatMessages.tsx # Message list container
      FileAttachment.tsx # File preview component
      LogPanel.tsx    # Backend logs panel
    layout/
      Header.tsx      # Top navigation bar
      Sidebar.tsx     # Left sidebar navigation
    ui/
      Button.tsx      # Button component
      Input.tsx       # Input component
      Textarea.tsx    # Textarea component
  hooks/
    useWebSocket.ts   # WebSocket connection hook
  lib/
    config.ts         # Configuration constants
    utils.ts          # Utility functions
  store/
    index.ts          # Zustand store
  types/
    index.ts          # TypeScript types
```

## Connecting to Backend

The frontend connects to the OpenBrowser backend via WebSocket for real-time communication.

1. Deploy the backend (see `/backend/README.md`)
2. Set environment variables to point to your backend URL
3. The WebSocket connection will be established automatically on page load

## License

MIT
