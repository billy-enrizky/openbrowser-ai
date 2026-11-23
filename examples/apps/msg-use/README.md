# Msg-Use

AI-powered message scheduler using browser agents and Gemini. Schedule personalized messages in natural language and let AI compose them intelligently.

> [!WARNING]
> This demo requires openbrowser v0.1.57+.

## Features

1. Agent logs into WhatsApp Web automatically
2. Parses natural language scheduling instructions
3. Composes personalized messages using AI
4. Schedules messages for future delivery or sends immediately
5. Reuses the saved browser profile when WhatsApp keeps the session valid

## Setup

Make sure the newest version of openbrowser is installed:
```bash
pip install -U openbrowser-ai
```

Export your Gemini API key, get it from: [Google AI Studio](https://makersuite.google.com/app/apikey)
```
export GOOGLE_API_KEY='your-gemini-api-key-here'
```

Clone the repo and cd into the app folder
```bash
git clone https://github.com/billy-enrizky/openbrowser-ai.git
cd openbrowser-ai/examples/apps/msg-use
```

## Initial Login

WhatsApp Web may require a QR scan on first use or when the saved session expires:
```bash
python login.py
```
- Scan the QR code when the browser opens
- The browser profile is reused on later runs when WhatsApp keeps the session valid

## Normal Usage

1. **Create `messages.txt`** in the app directory:
```bash
cat > messages.txt <<'EOF'
- Send "Hi" to Magnus on the 09.09 at 18:15
- Tell Camila at 20:00 that I miss her
- Remind mom to pick up the car next tuesday
EOF
```

2. **Test mode** - See what will be sent:
```bash
python scheduler.py --test
```

3. **Run scheduler**:
```bash
python scheduler.py

# Debug Mode - See the browser in action
python scheduler.py --debug

# Auto Mode - Respond to unread messages every ~30 minutes
python scheduler.py --auto
```

## Programmatic Usage

```python
import asyncio
from scheduler import parse_messages, send_message

async def main():
    for item in await parse_messages():
        await send_message(item["contact"], item["message"])

asyncio.run(main())
```

## Output

Example scheduling output:
```json
[
  {
    "contact": "Magnus",
    "original_message": "Hi",
    "composed_message": "Hi",
    "scheduled_time": "2025-06-13 18:15"
  },
  {
    "contact": "Camila",
    "original_message": "I miss her",
    "composed_message": "I miss you ❤️",
    "scheduled_time": "2025-06-14 20:00"
  }
]
```

## Files

- `scheduler.py` - Main scheduler script
- `login.py` - One-time login setup  
- `messages.txt` - Your message schedule in natural language

## License

MIT
