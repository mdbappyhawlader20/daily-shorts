import os
import re
import io
import json
import asyncio
import textwrap
from datetime import datetime, timezone
import numpy as np
from google import genai
from google.genai import types
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import edge_tts
from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips

# 1. Load encrypted secrets from GitHub Actions
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GDRIVE_FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"]
SERVICE_ACCOUNT_INFO = json.loads(os.environ["GDRIVE_SERVICE_ACCOUNT_JSON"])

client = genai.Client(api_key=GEMINI_API_KEY)

# 2. Generate a Money Psychology / Everyday Economics 2D Animated Script
prompt = """
Pick ONE timeless, counterintuitive money psychology trap, pricing illusion, cognitive bias, or simple wealth math concept (e.g., Decoy Effect, Diderot Effect, Rule of 72, Lifestyle Creep, Sunk Cost Fallacy, Anchoring Bias, Loss Aversion).
Do NOT mention current dates, stock prices, or volatile crypto/market news.
Return ONLY valid JSON with these exact keys:
- "youtube_title": punchy curiosity-gap Shorts title under 58 characters ending with #Shorts
- "scenes": list of 4 objects, each containing:
  - "narration": 1-2 crisp spoken sentences (max 18 words) that hook fast
  - "visual_prompt": detailed description for a clean flat 2D vector illustration (e.g. "minimalist flat 2D editorial vector illustration of a golden brain trapped inside a maze of price tags, dark navy background #090d16, bold gold #facc15 and cyan #38bdf8 accents, clean geometric shapes, no text")
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={"response_mime_type": "application/json"}
)
data = json.loads(response.text)

# Load bold font for synced subtitles
def get_sub_font(size=68):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

sub_font = get_sub_font(68)

# 3. Generate custom 2D vector artwork per scene
def generate_2d_illustration(visual_prompt, save_path):
    try:
        result = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=f"Flat 2D vector animation style, Kurzgesagt and Vox explainer inspired, dark slate navy background #090d16, vibrant gold and electric blue accents, clean vector lines, NO text, NO letters, NO numbers: {visual_prompt}",
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="9:16",
                output_mime_type="image/png"
            )
        )
        for generated_image in result.generated_images:
            img = Image.open(io.BytesIO(generated_image.image.image_bytes)).convert("RGB")
            img = img.resize((1180, 2098), Image.Resampling.LANCZOS)
            img.save(save_path)
            return
    except Exception as e:
        print("Image model fallback triggered:", e)

    # Fallback: crisp 2D geometric vector-style canvas
    img = Image.new("RGB", (1180, 2098), "#090d16")
    draw = ImageDraw.Draw(img)
    draw.ellipse([240, 520, 940, 1220], outline="#facc15", width=12)
    draw.ellipse([340, 620, 840, 1120], outline="#38bdf8", width=8)
    img.save(save_path)

# 4. Generate neural voiceover + exact word-by-word timestamps
async def generate_voice_and_timings(text, audio_path):
    communicate = edge_tts.Communicate(text, "en-US-AndrewNeural", rate="+6%")
    word_timings = []
    with open(audio_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # Convert 100-nanosecond ticks to seconds
                start_sec = chunk["offset"] / 10_000_000
                dur_sec = chunk["duration"] / 10_000_000
                word_timings.append((chunk["text"], start_sec, start_sec + dur_sec))
    return word_timings

# 5. Render transparent PNG subtitle badge for each spoken phrase chunk
def make_caption_frame(words_text, out_png):
    img = Image.new("RGBA", (1080, 320), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    wrapped = "\n".join(textwrap.wrap(words_text.upper(), width=18))

    bbox = draw.multiline_textbbox((0, 0), wrapped, font=sub_font, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (1080 - text_w) // 2
    y = (320 - text_h) // 2

    # Dark pill backdrop + thick black outline for mobile readability
    pad = 26
    draw.rounded_rectangle(
        [x - pad, y - pad, x + text_w + pad, y + text_h + pad],
        radius=24,
        fill=(9, 13, 22, 210)
    )
    draw.multiline_text(
        (x, y),
        wrapped,
        font=sub_font,
        fill="#facc15",
        stroke_width=4,
        stroke_fill="#000000",
        align="center"
    )
    img.save(out_png)

# 6. Assemble 2D animated scene with smooth camera zoom + synced subtitles
async def build_animated_scenes(scenes):
    scene_clips = []
    for i, scene in enumerate(scenes):
        audio_path = f"scene_{i}.mp3"
        art_path = f"art_{i}.png"

        generate_2d_illustration(scene["visual_prompt"], art_path)
        word_timings = await generate_voice_and_timings(scene["narration"], audio_path)

        audio_clip = AudioFileClip(audio_path)
        dur = audio_clip.duration

        # Smooth 2D Ken Burns camera zoom & pan on 1080x1920 stage
        base_art = ImageClip(art_path).with_duration(dur)
        animated_bg = (
            base_art
            .resized(lambda t: 1.0 + 0.08 * (t / max(dur, 0.1)))
            .with_position(("center", "center"))
        )
        stage = CompositeVideoClip([animated_bg], size=(1080, 1920)).with_duration(dur)

        # Group words into 3-word punchy subtitle bursts synced to voice
        caption_layers = [stage]
        chunk_size = 3
        for idx in range(0, len(word_timings), chunk_size):
            group = word_timings[idx:idx + chunk_size]
            phrase = " ".join(w[0] for w in group)
            c_start = group[0][1]
            c_end = group[-1][2]
            if idx + chunk_size < len(word_timings):
                c_end = word_timings[idx + chunk_size][1]
            else:
                c_end = dur

            c_dur = max(0.15, c_end - c_start)
            cap_png = f"cap_{i}_{idx}.png"
            make_caption_frame(phrase, cap_png)

            cap_clip = (
                ImageClip(cap_png)
                .with_start(c_start)
                .with_duration(c_dur)
                .with_position(("center", 1320))
            )
            caption_layers.append(cap_clip)

        full_scene = CompositeVideoClip(caption_layers, size=(1080, 1920)).with_audio(audio_clip)
        scene_clips.append(full_scene)

    return scene_clips

clips = asyncio.run(build_animated_scenes(data["scenes"]))

# 7. Render final 9:16 MP4
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
output_file = f"short_{timestamp}.mp4"
final_video = concatenate_videoclips(clips, method="compose")
final_video.write_videofile(output_file, fps=24, codec="libx264", audio_codec="aac")

# 8. Upload directly to your Zapier-watched Google Drive folder
creds = service_account.Credentials.from_service_account_info(
    SERVICE_ACCOUNT_INFO,
    scopes=["https://www.googleapis.com/auth/drive.file"]
)
drive_service = build("drive", "v3", credentials=creds)

safe_title = re.sub(r'[\\/*?:"<>|]', "", data["youtube_title"]).strip()
file_metadata = {
    "name": f"{safe_title}.mp4",
    "parents": [GDRIVE_FOLDER_ID]
}
media = MediaFileUpload(output_file, mimetype="video/mp4", resumable=True)
uploaded = drive_service.files().create(
    body=file_metadata,
    media_body=media,
    fields="id"
).execute()

print("Uploaded 2D Animated Short to Google Drive! File ID:", uploaded.get("id"))
