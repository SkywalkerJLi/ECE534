# save as precache.py, run on login node with internet
from transformers import AutoProcessor, AutoModel
import torchvision.models as models

# Cache ResNet18 backbone (needed for ACT)
print("Caching ResNet18...")
model = models.resnet18(weights="DEFAULT")

# Cache SmolVLM processor just in case something needs it
print("Caching SmolVLM processor...")
proc = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

print("Done — all cached")
