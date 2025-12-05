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

## Step 2: Install Google Chrome

The bot uses a browser to log in securely. You must install Google Chrome.

```bash
# Download Google Chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

# Install it
sudo apt install ./google-chrome-stable_current_amd64.deb -y

# Verify installation
google-chrome --version
```

## Step 3: Install Python 3.12 and Dependencies

The trading library requires **Python 3.12** or higher. Ubuntu 22.04 comes with Python 3.10, so we need to install the newer version.

```bash
# Add the deadsnakes PPA to get newer Python versions
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Install Python 3.12 and venv
sudo apt install python3.12 python3.12-venv -y

# Remove old virtual environment if it exists
rm -rf .venv

# Create a virtual environment using Python 3.12
python3.12 -m venv .venv

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

## Step 4: Install Xvfb (Virtual Display)

Since we need to run the browser in "headed" mode (visible) to bypass Cloudflare, but you are on a server without a monitor, you need a virtual display.

```bash
sudo apt install xvfb -y
```

## Step 5: Run the Bot

Use `xvfb-run` to start the bot with a virtual display:

```bash
xvfb-run -a python app/main.py
```

## Running in Background

```bash
xvfb-run -a nohup python app/main.py > bot.log 2>&1 &
```

**Using screen:**
```bash
screen -S quotexbot
python app/main.py
# Press Ctrl+A then D to detach
```
