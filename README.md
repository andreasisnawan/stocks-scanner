# 📊 Telegram Bot with GitHub Actions

This Telegram bot automatically sending a report to 1 target chat within a schedule using GitHub Actions.

## 🚀 Fitur

Send a message & attach CSV file to Telegram.

Scheduled for:

- 08:00 WIB (before market open)
- 15:00 WIB (before market close)

You can also run this manually through GitHub Actions.

## Repo Structure

```
.
├── scripts
│   └── ...
├── scanner.py            # Main Python
├── requirements.txt      # Dependency Python
├── .github/workflows/
│   └── config.yml        # Workflow GitHub Actions
└── README.md
```

## Setup .env

```
TELEGRAM_BOT_TOKEN='...'
TELEGRAM_CHAT_ID='...'
```
