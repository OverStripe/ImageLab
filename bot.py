# app.py
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import gradio as gr
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import os
import random
import threading

# ----------- CONFIGURATION -----------

# Model Configuration
MODEL_NAME = "stabilityai/stable-diffusion-2-1"
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Telegram Configuration
TELEGRAM_TOKEN = "7566276695:AAFNQmj3GR8FzJDUCNe7J0OrBweLL4V4qk4"
BOT_USERNAME = "@ImagesGeneratorToBot"

# Default Parameters
DEFAULT_STEPS = 50
DEFAULT_SCALE = 7.5
DEFAULT_SEED = 42
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

# ----------- LOAD MODEL -----------

print("🚀 Loading Stable Diffusion Model...")
pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    scheduler=EulerAncestralDiscreteScheduler.from_config(MODEL_NAME),
    safety_checker=None
).to("cuda")

# ----------- IMAGE GENERATION FUNCTION -----------

def generate_image(
    prompt: str,
    negative_prompt: str = "",
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_SCALE,
    seed: int = DEFAULT_SEED,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT
):
    try:
        torch.manual_seed(seed)
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height
        ).images[0]

        file_path = os.path.join(OUTPUT_DIR, f"{random.randint(1000, 9999)}_generated.png")
        image.save(file_path)
        return file_path
    except Exception as e:
        print(f"❌ Error during image generation: {e}")
        return None

# ----------- GRADIO INTERFACE -----------

def gradio_generate(prompt, negative_prompt, steps, guidance_scale, seed, width, height):
    file_path = generate_image(prompt, negative_prompt, steps, guidance_scale, seed, width, height)
    if file_path:
        return file_path
    else:
        return "Error generating image. Please try again."

with gr.Blocks(title="ImageLab") as demo:
    gr.Markdown("# 🚀 **ImageLab AI Image Generator**")
    gr.Markdown("### Generate stunning AI images from text prompts.")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="A futuristic city at sunset")
        negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Low quality, blurry, distorted")

    with gr.Row():
        steps = gr.Slider(10, 150, value=DEFAULT_STEPS, step=1, label="Steps")
        guidance_scale = gr.Slider(1.0, 20.0, value=DEFAULT_SCALE, step=0.5, label="Guidance Scale")

    with gr.Row():
        seed = gr.Number(value=random.randint(0, 999999), label="Seed")
        width = gr.Slider(256, 1024, value=DEFAULT_WIDTH, step=64, label="Width")
        height = gr.Slider(256, 1024, value=DEFAULT_HEIGHT, step=64, label="Height")

    output = gr.Image(label="Generated Image")
    btn = gr.Button("✨ Generate Image")

    btn.click(
        gradio_generate,
        inputs=[prompt, negative_prompt, steps, guidance_scale, seed, width, height],
        outputs=output
    )

# ----------- TELEGRAM BOT -----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Welcome to ImageLab Bot!* 🎨\n\n"
        "Send me a text prompt using `/generate [your prompt]` to create an AI image.",
        parse_mode='Markdown'
    )

async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ Please provide a prompt. Example: `/generate A sunset over mountains`")
        return

    prompt = " ".join(context.args)
    await update.message.reply_text(f"🎨 *Generating image for:* `{prompt}`", parse_mode='Markdown')

    file_path = generate_image(prompt)
    if file_path:
        await update.message.reply_photo(photo=open(file_path, 'rb'))
    else:
        await update.message.reply_text("❌ Error generating image. Please try again later.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *I only respond to commands!* Try `/generate A beautiful landscape`",
        parse_mode='Markdown'
    )

# Telegram Bot Handlers
app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("generate", generate))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

# ----------- RUN BOTH SERVICES -----------

def run_gradio():
    print("🌐 Starting Gradio Interface...")
    demo.launch(share=True)

def run_telegram():
    print("🤖 Starting Telegram Bot...")
    app.run_polling()

if __name__ == "__main__":
    thread1 = threading.Thread(target=run_gradio)
    thread2 = threading.Thread(target=run_telegram)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
  
