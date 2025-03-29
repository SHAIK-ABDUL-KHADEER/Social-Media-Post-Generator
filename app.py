from flask import Flask, render_template, request, jsonify
import pathlib
import os
import glob
import json
from google import genai
from google.genai import types
from refiner import refine_prompt  # Import refine function

app = Flask(__name__)
IMAGE_FOLDER = "static/images"
DATA_FOLDER = "data"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

METADATA_FILE = os.path.join(DATA_FOLDER, "image_metadata.json")

# Google GenAI API Configuration
GENAI_API_KEY = "AIzaSyAGV6dPt9miQCHD4PEqP1P5-nZ85lUofRI"
client = genai.Client(api_key=GENAI_API_KEY)
MODEL_ID = "gemini-2.0-flash-exp"


def load_metadata():
    """Load image metadata (filenames and prompts)."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as file:
            return json.load(file)
    return []


def save_metadata(metadata):
    """Save image metadata, ensuring the latest image is on top."""
    with open(METADATA_FILE, "w") as file:
        json.dump(metadata, file, indent=4)


def get_next_image_filename():
    """Find the next image filename (image_1.png, image_2.png, etc.)."""
    existing_images = glob.glob(os.path.join(IMAGE_FOLDER, "image_*.png"))
    next_number = len(existing_images) + 1
    return f"image_{next_number}.png"


def generate_image(user_prompt):
    """Generate an image using Google Gemini AI based on refined prompt."""
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=user_prompt,
        config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
    )

    image_filename = get_next_image_filename()
    image_path = os.path.join(IMAGE_FOLDER, image_filename)

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            pathlib.Path(image_path).write_bytes(part.inline_data.data)

            # Save metadata with the latest image first
            metadata = load_metadata()
            metadata.insert(0, {"filename": image_filename, "prompt": user_prompt})  # Insert at index 0
            save_metadata(metadata)

            return image_filename  # Return saved image filename
    return None


@app.route("/")
def home():
    """Render the homepage with all generated images and prompts (latest first)."""
    metadata = load_metadata()
    return render_template("index.html", images=metadata)


@app.route("/generate", methods=["POST"])
def generate():
    """Handle user prompt, refine it, generate an image, and return the image URL."""
    user_prompt = request.json.get("prompt", "").strip()

    if not user_prompt:
        return jsonify({"success": False, "error": "Prompt cannot be empty"})

    refined_prompt = refine_prompt(user_prompt)

    image_filename = generate_image(refined_prompt)
    if image_filename:
        image_url = f"/static/images/{image_filename}"
        return jsonify({"success": True, "image_url": image_url, "refined_prompt": refined_prompt})
    else:
        return jsonify({"success": False, "error": "Image generation failed"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)