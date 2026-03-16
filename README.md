# LoraImageFormatter Pro

A quick vibe coded Python tool to automate the gathering and formatting of images for LoRA training. Not pretty, but very practical.

## Features
- **URL Scraping:** Download images directly from a website.
- **Auto-Formatting:** Resizes and pads images to square (512, 768, or 1024).
- **AI Captioning:** Uses the BLIP model to describe images automatically.
- **LoRA Ready:** Prepends your trigger word to text files.

## Installation
1. Clone this repo.
2. Install requirements: `pip install -r requirements.txt`
3. Run: `python lora_formatter.py`