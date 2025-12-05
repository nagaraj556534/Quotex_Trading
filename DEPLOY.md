# Deployment Guide for Ubuntu 22.04

This guide explains how to deploy and run the Quotex Trading Bot on your Ubuntu server.

## Prerequisites

*   Ubuntu 22.04 LTS
*   Python 3.10 or higher (Ubuntu 22.04 comes with Python 3.10 by default)
*   Git

## Step 1: Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/nagaraj556534/Quotex_Trading.git
cd Quotex_Trading
```

## Step 2: Install Dependencies

It is recommended to use a virtual environment.

```bash
# Install pip and venv if not already installed
sudo apt update
sudo apt install python3-pip python3-venv -y

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Step 3: Configure Credentials

Since sensitive files were excluded from the repository, you need to recreate them.

### 1. Create `user_config.py`

```bash
nano user_config.py
```

Paste the following content (replace with your actual details):

```python
# User Configuration for Quotex Bot

# Quotex Credentials
QUOTEX_EMAIL = "your_email@example.com"
QUOTEX_PASSWORD = "your_password"

# Telegram Credentials
TELEGRAM_API_ID = 12345678  # Replace with your API ID
TELEGRAM_API_HASH = "your_api_hash"
TELEGRAM_PHONE = "your_phone_number"
TELEGRAM_GROUP = "your_group_id"

# Strategy 19 Settings
S19_MIN_FORECAST = 70
S19_STAKE_AMOUNT = 5
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

### 2. Create `settings/config.ini`

```bash
mkdir -p settings
nano settings/config.ini
```

Paste the following content:

```ini
[settings]
email=your_email@example.com
password=your_password
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

### 3. Create `saved_credentials.json` (Optional)

If you have a saved session, you can create this file, otherwise the bot might create it.

```bash
nano saved_credentials.json
```

Content:
```json
{}
```

## Step 4: Run the Bot

Ensure your virtual environment is active (`source .venv/bin/activate`), then run:

```bash
python app/main.py
```

## Running in Background (Optional)

To keep the bot running after you disconnect, use `nohup` or `screen`.

**Using nohup:**
```bash
nohup python app/main.py > bot.log 2>&1 &
```

**Using screen:**
```bash
screen -S quotexbot
python app/main.py
# Press Ctrl+A then D to detach
```
