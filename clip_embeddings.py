from transformers import CLIPProcessor, CLIPTextModel, CLIPModel
import torch
from PIL import Image
import glob
import os
import base64
import hashlib


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_text_embeddings(text):
    inputs = processor(text=text, return_tensors="pt", padding=True)
    return model.get_text_features(**inputs)

def compute_image_embeddings(image_path):
    image = Image.open(image_path)
    image = processor(text=None, images=image, return_tensors='pt')['pixel_values']
    return model.get_image_features(image)


def get_embedding_on_images(folder_path):
    embedings = []
    
    images_file_paths =  glob.glob(os.path.join(folder_path, '*.png')) + \
              glob.glob(os.path.join(folder_path, '*.jpg')) + \
              glob.glob(os.path.join(folder_path, '*.jpeg')) 
    
    for image_file in images_file_paths:
        embedings.append(
            {
                'embedding' : compute_image_embeddings(image_file),
                'filePath' : image_file,
                'id':  base64.b64encode(hashlib.sha256(image_file.encode()).digest()).decode()
            }
        )

    return embedings

# image_embeddings = get_embedding_on_images('images')

# embeddings=[e["embedding"][0] for e in image_embeddings]
# metadatas=[{k: e[k] for k in ["filePath"]} for e in image_embeddings]
# ids=[e["id"] for e in image_embeddings]

