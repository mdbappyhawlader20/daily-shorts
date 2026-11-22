import os
import re
import io
import json
import time
import asyncio
import textwrap
from datetime import datetime, timezone
import numpy as np
from google import genai
from google.genai import types
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from PIL import Image, ImageDraw, ImageFont
import edge_tts
from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips

# 1. Load encrypted secrets from GitHub Actions
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GDRIVE_FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"]
SERVICE_ACCOUNT_INFO = json.loads(os.environ["GDRIVE_SERVICE_ACCOUNT_JSON"])

client = genai.Client(api_key=GEMINI_API_KEY)

# 2. Generate a Money Psychology / Everyday Economics 2D Animated Script (with auto-retry)
prompt = """
Pick ONE timeless, counterintuitive money psychology trap, pricing illusion, cognitive bias, or simple wealth math concept (e.g., Decoy Effect, Diderot Effect, Rule of 72, Lifestyle Creep, Sunk Cost Fallacy, Anchoring Bias, Loss Aversion).
Do NOT mention current dates, stock prices, or volatile crypto/market news.
Return ONLY valid JSON with these exact keys:
- "youtube_title": punchy curiosity-gap Shorts title under 58 characters ending with #Shorts
- "scenes": list of 4 objects, each containing:
  - "narration": 1-2 crisp spoken sentences (max 18 words) that hook fast
  - "visual_prompt": detailed description for a clean flat 2D vector illustration (e.g. "minimalist flat 2D editorial vector illustration of a golden brain trapped inside a maze of price tags, dark navy background #090d16, bold gold #facc15 and cyan #38bdf8 accents, clean geometric shapes, no text")
"""

models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-3.0-flash"]
response = None

for attempt in range(5):
    model_name = models_to_try[attempt % len(models_to_try)]
    try:
        print(f"Generating script with {model_name} (attempt {attempt + 1})...")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        break
    except Exception as e:
        print(f"Temporary API busy error on {model_name}: {e}")
        time.sleep(10)

if response is None:
    raise RuntimeError("All Gemini script attempts failed after retries.")

data = json.loads(response.text)
