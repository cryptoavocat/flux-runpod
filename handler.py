"""RunPod handler for FLUX image generation."""
import os
from huggingface_hub import login
from diffusers import DiffusionPipeline
import torch

# Authenticate with Hugging Face
token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable not found.")
login(token=token, new_session=False)
print("✅ Hugging Face login successful.")

# Load model (once)
# Authenticate with Hugging Face by setting HF_TOKEN directly in the env
os.environ["HF_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

# Load the model (will use the token from env automatically)
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Krea-dev",
    torch_dtype=torch.float16,
    use_auth_token=True  # 👈 critical for gated model access
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

        return {"output": output_path}
    
    except Exception as e:
        print("❌ Error in handler:", str(e))
        return {"error": str(e)}
