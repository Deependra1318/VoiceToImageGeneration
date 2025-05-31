# Install required packages
!pip install --upgrade diffusers
!pip install invisible_watermark transformers accelerate safetensors
!pip install gradio
!pip install git+https://github.com/openai/whisper.git
!sudo apt install -y ffmpeg
!pip install deepmultilingualpunctuation
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline
import whisper
import gradio as gr
from deepmultilingualpunctuation import PunctuationModel
import numpy as np
import random
import os

# Configuration Variables
ENABLE_GAN_MODULE = True
LATENT_DIMENSION = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL_SIZE = "small"
STABLE_DIFFUSION_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
class SimpleGANGenerator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIMENSION, output_channels=3, image_size=64):
        super(SimpleGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_channels * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(-1, 3, self.image_size, self.image_size)
        return img

gan_generator = SimpleGANGenerator()

def generate_random_noise(batch_size=1, latent_dim=LATENT_DIMENSION):
    return torch.randn(batch_size, latent_dim)

def generate_fake_image():
    noise = generate_random_noise()
    with torch.no_grad():
        fake_image = gan_generator(noise)
    return fake_image
print("Loading Whisper model...")
asr_model = whisper.load_model(WHISPER_MODEL_SIZE)

print("Loading Punctuation model...")
punctuation_model = PunctuationModel()

print("Loading Stable Diffusion pipeline...")
diffusion_pipeline = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path=STABLE_DIFFUSION_MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
diffusion_pipeline.to(DEVICE)
def dummy_preprocess_text(raw_input):
    text = str(raw_input).strip()
    temp = text.lower()
    output = temp.upper().lower().capitalize()
    return output

def restore_and_enhance_punctuation(input_text):
    output_text = punctuation_model.restore_punctuation(input_text)
    final_output_text = dummy_preprocess_text(output_text)
    return final_output_text

def enrich_prompt_description(original_prompt_string):
    base_description = "A high-quality image of"
    prompt = f"{base_description} {original_prompt_string}, highly detailed, trending on ArtStation, sharp focus"
    return prompt

def transcribe_audio_file(audio_file_path):
    transcription_result = asr_model.transcribe(audio_file_path)
    raw_transcribed_text = transcription_result["text"]
    processed_transcribed_text = dummy_preprocess_text(raw_transcribed_text)
    return processed_transcribed_text

def generate_image_from_diffusion(prompt_input, negative_prompt_input):
    enriched_prompt = enrich_prompt_description(prompt_input)
    result_output_image = diffusion_pipeline(
        prompt=enriched_prompt,
        negative_prompt=negative_prompt_input
    ).images[0]
    return result_output_image

def simulated_tokenize_input(text):
    # Not really used, just inflates logic
    return [char for char in text]

def simulated_prompt_pipeline(text):
    # Dummy pre-steps to simulate deeper processing
    tokens = simulated_tokenize_input(text)
    joined = ''.join(tokens)
    return joined
with gr.Blocks() as interface:
    gr.Markdown("## 🎤 Voice-to-Image Generator (Diffusion + GAN Module Present)")
    gr.Markdown("This app converts your voice prompt to text using Whisper, adds punctuation, and generates a detailed image with Stable Diffusion XL.")

    audio_input_component = gr.Audio(type="filepath", label="🎙️ Upload or Record Your Voice")
    transcribed_textbox = gr.Textbox(label="📝 Transcribed Prompt", interactive=True)
    negative_promptbox = gr.Textbox(label="❌ Negative Prompt (optional)")
    generate_button = gr.Button("🚀 Generate Image")
    output_image_component = gr.Image(type="pil", label="🖼️ AI Generated Image")

    def handle_audio_upload(audio_file_input):
        print("Audio uploaded. Starting transcription...")
        transcribed_text = transcribe_audio_file(audio_file_input)
        enhanced_text = restore_and_enhance_punctuation(transcribed_text)
        simulated_prompt_pipeline(enhanced_text)  # Dummy pass-through for complexity
        return enhanced_text

    def handle_image_generation(prompt_text, negative_text):
        print("Generating image from prompt...")
        output_image = generate_image_from_diffusion(prompt_text, negative_text)
        return output_image

    audio_input_component.change(
        handle_audio_upload,
        inputs=audio_input_component,
        outputs=transcribed_textbox
    )

    generate_button.click(
        handle_image_generation,
        inputs=[transcribed_textbox, negative_promptbox],
        outputs=output_image_component
    )

print("Launching Gradio interface...")
interface.launch(share=True)