import anthropic
import base64
import json
import os
import glob

client = anthropic.Anthropic()

IMAGE_DIR = "images"
OUTPUT_FILE = "image_features.json"

PROMPT = """look at this image and return ONLY a JSON object (no other text) with these fields:

- num_people: integer
- age_brackets: list of strings, any of ["child", "adult", "older_adult"] present
- setting: "indoor" or "outdoor"
- color_temperature: "warm", "cool", or "neutral"
- brightness: "bright", "dim", or "moderate"
- nature_present: list of strings, any of ["water", "plants", "animals"] present, or empty list
- time_of_day: your best guess, or "unclear"
- scene_type: short description like "beach", "city street", "indoor portrait"

return valid JSON only, no markdown formatting, no explanation."""

image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.JPG"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

print(f"Found {len(image_paths)} images")

results = {}

for path in image_paths:
    filename = os.path.basename(path)
    print(f"Analyzing {filename}...")

    with open(path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    ext = filename.rsplit(".", 1)[-1].lower()
    media_type = "image/png" if ext == "png" else "image/jpeg"

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )

    raw = message.content[0].text
    try:
        features = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  WARNING: couldn't parse JSON for {filename}, storing raw text")
        features = {"_raw": raw, "_parse_error": True}

    results[filename] = features

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone — wrote features for {len(results)} images to {OUTPUT_FILE}")
