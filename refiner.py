import google.generativeai as genai

# Configure Google Generative AI
genai.configure(api_key="AIzaSyAgmlM70rVc9g-lMtu8NIBD9hYqVRk0dVI")

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)


def refine_prompt(user_prompt):
    """Refine a user prompt for better image generation."""
    chat_session = model.start_chat(history=[
        {"role": "user", "parts": [
            "You are a Professional Prompt Engineer. Refine the prompt I send for generating high-quality images. Only return the refined prompt, nothing else. Ensure the image includes the given text."
        ]},
        {"role": "model", "parts": ["Okay, I'm ready!"]},
    ])

    response = chat_session.send_message(user_prompt)
    return response.text.strip()


if __name__ == "__main__":
    user_input = input("Enter your prompt: ")
    print("Refined Prompt:", refine_prompt(user_input))
