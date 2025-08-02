"""RunPod handler for FLUX image generation."""
import os
import torch
from diffusers import DiffusionPipeline
import base64

# Authenticate by setting HF_HUB_TOKEN
os.environ["HF_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

# Load the model (auto-authenticated)
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Krea-dev",
    torch_dtype=torch.float16,
    use_auth_token=True  # critical for gated model access
)
pipe.to("cuda")
print("✅ Model loaded to CUDA.")

# Main RunPod handler function
def handler(event):
    try:
        prompt = event["input"].get("prompt", "Astronaut on a desert island")
        print(f"🚀 Prompt received: {prompt}")
        
        image = pipe(prompt).images[0]
        output_path = "/tmp/generated_image.png"
        image.save(output_path)
        print("✅ Image saved to /tmp.")

        # OPTIONAL: return base64 instead of file path
        with open(output_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {"image_base64": img_b64}

    except Exception as e:
        print("❌ Error in handler:", str(e))
        return {"error": str(e)}
