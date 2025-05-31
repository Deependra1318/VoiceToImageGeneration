# 🎤 Voice-to-Image Generator (Diffusion + GAN)

This project is an interactive **Voice-to-Image generation app** that combines **speech recognition**, **punctuation restoration**, and **AI-powered image generation**. Users can record or upload a voice prompt, which is transcribed and converted into a detailed image using **Stable Diffusion XL**. A simple **GAN** module is also defined, ready for experimentation and extension.

---

## 🚀 Features

- 🎙️ Voice transcription using OpenAI’s **Whisper** ASR.
- 📝 Automatic punctuation restoration using **Deep Multilingual Punctuation** model.
- 🎨 High-quality image generation using **Stable Diffusion XL (SDXL)**.
- 🧠 GAN module (`SimpleGANGenerator`) for experimentation.
- 🌐 **Gradio** web interface for ease of use.
- ✅ Optional **negative prompts** to refine image outputs.

---

## 📦 Installation

Make sure you're using Python 3.8+ with pip. Then run the following:

```bash
pip install --upgrade diffusers
pip install invisible_watermark transformers accelerate safetensors
pip install gradio
pip install git+https://github.com/openai/whisper.git
sudo apt install -y ffmpeg  # for audio support
pip install deepmultilingualpunctuation

🛠️ Usage
Run the main Python file to start the app:

bash
Copy
Edit
python app.py
Once running, the Gradio interface will launch (with a public link if share=True), allowing you to:

🎙️ Upload or record a voice input.

📝 View and edit the transcribed text.

❌ Optionally add a negative prompt to exclude certain features.

🖼️ Click Generate Image to get an AI-generated image.

📁 Project Structure
bash
Copy
Edit
.
├── app.py              # Main application logic
├── README.md           # This file
├── requirements.txt    # Optional: for packaging dependencies
⚙️ Configuration
You can configure the following options at the top of app.py:

python
Copy
Edit
ENABLE_GAN_MODULE = True               # Toggle GAN module (currently not in UI)
LATENT_DIMENSION = 100                # Latent space for GAN
DEVICE = "cuda" or "cpu"              # Automatically set
WHISPER_MODEL_SIZE = "small"          # Whisper model size
STABLE_DIFFUSION_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
🧪 GAN Module
A simple GAN generator (SimpleGANGenerator) is defined in the code but not yet integrated into the UI. You can generate a synthetic image using:

python
Copy
Edit
fake_image = generate_fake_image()
Feel free to extend the app to show both GAN and Diffusion results side-by-side.

✨ Examples
"A fantasy castle in the clouds" → 🏰 AI-generated visual.

"A futuristic robot holding flowers" → 🤖💐 Stylized render.

🙏 Credits
OpenAI Whisper

Stable Diffusion XL

Deep Multilingual Punctuation

Gradio

📜 License
This project is for research and educational purposes only. Check licenses of individual models and tools used.
