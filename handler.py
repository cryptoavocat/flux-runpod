"""Example handler file."""
from diffusers import DiffusionPipeline
import torch
import os

# Load model once when container boots
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Krea-dev",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

# Required function for RunPod
def handler(event):
    prompt = event["input"].get("prompt", "Astronaut in a jungle")
    image = pipe(prompt).images[0]

    # Save to file
    image_path = "/tmp/generated.png"
    image.save(image_path)

    # Return path so RunPod can expose it
    return {"output": image_path}
