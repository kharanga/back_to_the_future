import torch
import open_clip
from PIL import Image
import numpy as np
from tag_analyzer import LIBRARY_DIR

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model.eval()

def embed_image(image_path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")

    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        features = model.encode_image(tensor)

    vector = features[0].numpy().astype(np.float32)

    return vector / np.linalg.norm(vector)

if __name__ == "__main__":
    photos = sorted(LIBRARY_DIR.glob("*.jpg"))
    russells = [p for p in photos if p.name.startswith("russell_70s")]
    champions = [p for p in photos if p.name.startswith("champion")]

    v_r1 = embed_image(russells[0])
    v_r2 = embed_image(russells[1])
    v_c1 = embed_image(champions[0])

    print("russell vs russell:", np.dot(v_r1, v_r2))
    print("russell vs champion:", np.dot(v_r1, v_c1))
    print("self vs self:", np.dot(v_r1, v_r1))