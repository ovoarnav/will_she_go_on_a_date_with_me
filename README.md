# will_she_go_on_a_date_with_me
# DM Temperature / Date Odds (HMM Demo)

A small Python tool that:
1) takes chat **screenshots** (PNG/JPG),
2) uses an OpenAI vision model to extract **behavioral features** + an **interest_score** (0–1),
3) updates a running belief over hidden states (**cold / warm / hot**),
4) outputs probabilities for outcomes (**show / resched / no**).

> Privacy note: This tool is intended for **personal/internal use**. Avoid storing or sharing raw screenshots or identifiable content.

---

## Requirements
- Python 3.10+ (recommended)
- Windows / macOS / Linux
- An OpenAI API key with access to a vision-capable model

---

## 1) Get an OpenAI API Key
1. Create an account / sign in to OpenAI.
2. Go to the API keys page.
3. Create a new secret key and copy it.

**Important:** Treat your key like a password. Don’t paste it into code or commit it to Git.

---

## 2) Set the API Key (Windows PowerShell)
Open PowerShell (or PyCharm terminal set to PowerShell) and run:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY_HERE"
